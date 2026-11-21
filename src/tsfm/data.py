"""Data loading for batch-wise multivariate time series.

Input format: one CSV where every row is one time step, with a `batch_id`
column identifying the batch, one column per monitored physical variable and
(optionally) a `label` column ("accepted" / "rejected").

Each batch becomes one sample of shape [C, T] (channels x time). Batches may
have different lengths T, and different datasets may have a different number
of channels C; the collate function pads both and returns masks.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

LABELS = {"accepted": 0, "rejected": 1}
NON_FEATURE_COLS = {"batch_id", "label"}


def load_batches(csv_path: str) -> tuple[list[str], list[np.ndarray], list[int], list[str]]:
    """Read a CSV and split it into one [C, T] array per batch (row order kept)."""
    df = pd.read_csv(csv_path)
    channels = [c for c in df.columns if c not in NON_FEATURE_COLS]
    ids, series, labels = [], [], []
    for bid, g in df.groupby("batch_id", sort=True):
        ids.append(bid)
        series.append(g[channels].to_numpy(dtype=np.float32).T)  # [C, T]
        labels.append(LABELS[g["label"].iloc[0]] if "label" in g else -1)
    return ids, series, labels, channels


@dataclass
class ChannelStats:
    """Per-channel mean/std fitted on training data only.

    Global (dataset-level) standardisation is used instead of per-sample
    normalisation on purpose: absolute offsets between batches carry label
    information (see the process A EDA), which per-sample norm would erase.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, series: list[np.ndarray]) -> "ChannelStats":
        cat = np.concatenate(series, axis=1)
        std = cat.std(axis=1)
        return cls(cat.mean(axis=1), np.where(std < 1e-8, 1.0, std))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[:, None]) / self.std[:, None]


class Augmenter:
    """Label-preserving augmentations for standardised [C, T] batch series.

    Every strength is calibrated per variable from the *training* batches, because
    the variables behave very differently (process A EDA):
      * high-frequency noise ranges from ~0.01 sd (temperature_2) to ~0.9 sd (pressure_2/4)
      * batch-to-batch level differences of accepted batches are tiny for some variables
        (flow_rate_1, temperature_3/4: ~0.01 sd) while rejected batches show clear offsets
        there, so level shifts must stay far below that spread or they fake a "rejected" look.

    Each transform is applied independently with probability `p`:
      1. jitter        add N(0, (jitter * noise_c)^2) per time step
      2. level shift   add N(0, (shift * between_c)^2) once per channel
      3. amplitude     scale fluctuations around the channel mean by N(1, scale^2)
      4. time stretch  resample to length * U(1 - stretch, 1 + stretch)
      5. crop          keep a random contiguous window of >= crop_min of the series

    Deliberately NOT used: time reversal or flips (the process is directional: start-up
    ramp, shut-down), mixup across labels, channel permutation (variable identity is
    embedded), and dropping whole channels (could delete the only evidence of a fault).
    """

    def __init__(self, noise: np.ndarray, between: np.ndarray, p: float = 0.5, jitter: float = 0.25,
                 shift: float = 0.25, scale: float = 0.05, stretch: float = 0.1, crop_min: float = 0.9):
        self.noise, self.between = noise.astype(np.float32), between.astype(np.float32)
        self.p, self.jitter, self.shift, self.scale = p, jitter, shift, scale
        self.stretch, self.crop_min = stretch, crop_min

    @classmethod
    def fit(cls, series: list[np.ndarray], labels: list[int], **kw) -> "Augmenter":
        """series: standardised [C, T] arrays of the training split."""
        noise = np.median([np.diff(s, axis=1).std(axis=1) / np.sqrt(2) for s in series], axis=0)
        ref = [s for s, y in zip(series, labels) if y == 0] or series   # accepted batches if labelled
        med = np.stack([np.median(s, axis=1) for s in ref])
        between = 1.4826 * np.median(np.abs(med - np.median(med, axis=0)), axis=0)   # robust sd
        return cls(noise, between, **kw)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng = np.random
        C, T = x.shape
        x = x.copy()
        if rng.rand() < self.p:
            x += rng.randn(C, T).astype(np.float32) * (self.jitter * self.noise)[:, None]
        if rng.rand() < self.p:
            x += (rng.randn(C) * self.shift * self.between).astype(np.float32)[:, None]
        if rng.rand() < self.p:
            mu = x.mean(axis=1, keepdims=True)
            x = mu + (x - mu) * (1 + self.scale * rng.randn(C, 1)).astype(np.float32)
        if rng.rand() < self.p:
            new_T = max(8, int(round(T * rng.uniform(1 - self.stretch, 1 + self.stretch))))
            src, dst = np.linspace(0, 1, T), np.linspace(0, 1, new_T)
            x = np.stack([np.interp(dst, src, row) for row in x]).astype(np.float32)
            T = new_T
        if rng.rand() < self.p:
            keep = int(T * rng.uniform(self.crop_min, 1.0))
            start = rng.randint(0, T - keep + 1)
            x = x[:, start:start + keep]
        return np.ascontiguousarray(x)


class BatchSeriesDataset(Dataset):
    def __init__(self, ids, series, labels, stats: ChannelStats, max_len: int | None = None,
                 random_crop: bool = False, augment: Augmenter | None = None):
        self.ids, self.labels = ids, labels
        self.series = [stats.apply(s).astype(np.float32) for s in series]
        self.max_len, self.random_crop = max_len, random_crop
        self.augment = augment

    def __len__(self):
        return len(self.series)

    def __getitem__(self, i):
        x = self.series[i]
        if self.augment is not None:
            x = self.augment(x)
        if self.max_len is not None and x.shape[1] > self.max_len:
            start = np.random.randint(0, x.shape[1] - self.max_len + 1) if self.random_crop else 0
            x = x[:, start:start + self.max_len]
        return {"x": torch.from_numpy(x), "label": self.labels[i], "batch_id": self.ids[i]}


def make_collate(patch_size: int):
    """Pad channels and time (to a multiple of `patch_size`) and build masks."""

    def collate(items):
        C = max(it["x"].shape[0] for it in items)
        T = max(it["x"].shape[1] for it in items)
        T = int(np.ceil(T / patch_size) * patch_size)
        B = len(items)
        x = torch.zeros(B, C, T)
        time_mask = torch.zeros(B, T, dtype=torch.bool)      # True = real data
        channel_mask = torch.zeros(B, C, dtype=torch.bool)   # True = real channel
        for b, it in enumerate(items):
            c, t = it["x"].shape
            x[b, :c, :t] = it["x"]
            time_mask[b, :t] = True
            channel_mask[b, :c] = True
        return {
            "x": x,
            "time_mask": time_mask,
            "channel_mask": channel_mask,
            "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
            "batch_id": [it["batch_id"] for it in items],
        }

    return collate


class BatchSeriesDataModule(L.LightningDataModule):
    """Train CSV is split (stratified, by batch) into train/val; test CSV is held out."""

    def __init__(self, train_csv: str, test_csv: str | None = None, patch_size: int = 32,
                 batch_size: int = 8, val_fraction: float = 0.2, max_len: int | None = 4096,
                 seed: int = 42, num_workers: int = 0, augment: dict | None = None):
        """augment: None = off, or a dict of Augmenter strengths ({} = defaults).
        Augmentation is applied to the training split only."""
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if hasattr(self, "train_ds"):
            return
        h = self.hparams
        ids, series, labels, self.channels = load_batches(h.train_csv)
        idx = np.arange(len(ids))
        strat = labels if min(labels) >= 0 else None
        if h.val_fraction > 0:
            tr, va = train_test_split(idx, test_size=h.val_fraction, random_state=h.seed, stratify=strat)
        else:   # use every training batch for training (tiny datasets); no validation set
            tr, va = idx, idx[:0]
        pick = lambda lst, ix: [lst[i] for i in ix]
        self.stats = ChannelStats.fit(pick(series, tr))
        self.augmenter = None
        if h.augment is not None:
            std_train = [self.stats.apply(s) for s in pick(series, tr)]
            self.augmenter = Augmenter.fit(std_train, pick(labels, tr), **h.augment)
        self.train_ds = BatchSeriesDataset(pick(ids, tr), pick(series, tr), pick(labels, tr),
                                           self.stats, h.max_len, random_crop=True, augment=self.augmenter)
        self.val_ds = BatchSeriesDataset(pick(ids, va), pick(series, va), pick(labels, va), self.stats)
        if h.test_csv:
            t_ids, t_series, t_labels, t_channels = load_batches(h.test_csv)
            assert t_channels == self.channels, "test CSV has different channels than train CSV"
            self.test_ds = BatchSeriesDataset(t_ids, t_series, t_labels, self.stats)

    @property
    def train_labels(self) -> list[int]:
        return self.train_ds.labels

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          collate_fn=make_collate(self.hparams.patch_size),
                          num_workers=self.hparams.num_workers)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)
