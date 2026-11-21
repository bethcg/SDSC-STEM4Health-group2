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


class BatchSeriesDataset(Dataset):
    def __init__(self, ids, series, labels, stats: ChannelStats, max_len: int | None = None,
                 random_crop: bool = False):
        self.ids, self.labels = ids, labels
        self.series = [stats.apply(s).astype(np.float32) for s in series]
        self.max_len, self.random_crop = max_len, random_crop

    def __len__(self):
        return len(self.series)

    def __getitem__(self, i):
        x = self.series[i]
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
                 seed: int = 42, num_workers: int = 0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if hasattr(self, "train_ds"):
            return
        h = self.hparams
        ids, series, labels, self.channels = load_batches(h.train_csv)
        idx = np.arange(len(ids))
        strat = labels if min(labels) >= 0 else None
        tr, va = train_test_split(idx, test_size=h.val_fraction, random_state=h.seed, stratify=strat)
        pick = lambda lst, ix: [lst[i] for i in ix]
        self.stats = ChannelStats.fit(pick(series, tr))
        self.train_ds = BatchSeriesDataset(pick(ids, tr), pick(series, tr), pick(labels, tr),
                                           self.stats, h.max_len, random_crop=True)
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
