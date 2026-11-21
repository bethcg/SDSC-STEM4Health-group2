"""Export a finetuned classifier to one standalone file, and load it again.

The exported file is a plain `torch.save` dict (no Lightning needed to load it) with:
  * state_dict      encoder + classification-head weights
  * encoder_kwargs  architecture needed to rebuild the encoder
  * channel_names   the variables, in the order the model expects them
  * norm_mean/std   per-variable standardisation fitted on the training split.
                    New data must be normalised with these, not with its own statistics.
  * labels          class index -> name
  * metadata        source checkpoint, test metrics, ...

Usage:
    from tsfm.export import load_exported, predict_csv
    model, info = load_exported("model_process_A.pt")
    preds = predict_csv(model, info, "test_process_A.csv")
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from tsfm.data import LABELS, NON_FEATURE_COLS, make_collate
from tsfm.model import ClassificationHead, TimeSeriesEncoder


class ExportedClassifier(nn.Module):
    def __init__(self, encoder_kwargs: dict, channel_names: list[str], vocab: list[str] | None, n_classes: int = 2):
        super().__init__()
        kw = dict(encoder_kwargs)
        kw.pop("use_channel_embedding", None)
        self.encoder = TimeSeriesEncoder(**kw, n_channel_ids=len(vocab) if vocab else 0)
        self.head = ClassificationHead(self.encoder.hidden, n_classes)
        self.channel_names, self.vocab = channel_names, vocab

    def forward(self, x, time_mask, channel_mask):
        cids = None
        if self.vocab:
            cids = torch.tensor([self.vocab.index(c) for c in self.channel_names], device=x.device)
            cids = cids[None].expand(len(x), -1)
        z, patch_valid = self.encoder(x, time_mask, channel_mask, channel_ids=cids)
        return self.head(z, patch_valid, channel_mask)


def export(lit_model, stats, out_path: str, metadata: dict | None = None):
    """lit_model: a trained ClassifierModule; stats: the data module's ChannelStats."""
    enc_kw = dict(lit_model.hparams.encoder_kwargs)
    payload = {
        "format": "tsfm-classifier-v1",
        "state_dict": {k: v.cpu() for k, v in lit_model.state_dict().items()
                       if k.startswith(("encoder.", "head."))},
        "encoder_kwargs": enc_kw,
        "channel_names": list(lit_model.hparams.channel_names),
        "vocab": list(lit_model.vocab) if lit_model.vocab is not None else None,
        "n_classes": lit_model.hparams.n_classes,
        "labels": {v: k for k, v in LABELS.items()},
        "norm_mean": torch.tensor(stats.mean, dtype=torch.float32),
        "norm_std": torch.tensor(stats.std, dtype=torch.float32),
        "metadata": metadata or {},
    }
    torch.save(payload, out_path)
    return payload


def load_exported(path: str, map_location="cpu"):
    info = torch.load(path, map_location=map_location, weights_only=False)
    model = ExportedClassifier(info["encoder_kwargs"], info["channel_names"], info["vocab"], info["n_classes"])
    model.load_state_dict(info["state_dict"])
    return model.eval(), info


@torch.no_grad()
def predict_csv(model, info, csv_path: str, batch_size: int = 8, threshold: float = 0.5) -> pd.DataFrame:
    """Predict one label per batch_id of a CSV in the training format (label column optional)."""
    df = pd.read_csv(csv_path)
    names = info["channel_names"]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing variables the model needs: {missing}")
    mean, std = info["norm_mean"].numpy(), info["norm_std"].numpy()
    items = []
    for bid, g in df.groupby("batch_id", sort=True):
        x = (g[names].to_numpy(np.float32).T - mean[:, None]) / std[:, None]
        items.append({"x": torch.from_numpy(x.astype(np.float32)), "label": -1, "batch_id": bid,
                      "true": g["label"].iloc[0] if "label" in g else None})
    collate = make_collate(info["encoder_kwargs"]["patch_size"])
    rows = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        b = collate(chunk)
        p = model(b["x"], b["time_mask"], b["channel_mask"]).softmax(-1)[:, 1].numpy()
        for it, pr in zip(chunk, p):
            rows.append({"batch_id": it["batch_id"], "p_rejected": float(pr),
                         "predicted": info["labels"][int(pr >= threshold)], "label": it["true"]})
    return pd.DataFrame(rows)
