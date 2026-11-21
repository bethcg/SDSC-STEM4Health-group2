"""Lightning modules for the two stages.

1. MaskedReconstructionModule (pretraining, no labels needed)
   Random (channel, patch) blocks are hidden from the encoder; a linear
   decoder reconstructs them. Loss = MSE on the hidden, non-padded values.

2. ClassifierModule (finetuning)
   Same encoder (weights loaded from the pretraining checkpoint), decoder
   swapped for a classification head. The encoder can be frozen (linear
   probe) or trained with a smaller learning rate.
"""
from __future__ import annotations

import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from tsfm.model import ClassificationHead, ReconstructionHead, TimeSeriesEncoder


def channel_ids_for(channels: list[str], vocab: list[str], batch_size: int, device) -> torch.Tensor:
    missing = [c for c in channels if c not in vocab]
    if missing:
        raise ValueError(f"channels not in the encoder's vocabulary: {missing}")
    ids = torch.tensor([vocab.index(c) for c in channels], device=device)
    return ids[None].expand(batch_size, -1)


class MaskedReconstructionModule(L.LightningModule):
    def __init__(self, channel_names: list[str], hidden: int = 64, depth: int = 3, heads: int = 4,
                 patch_size: int = 32, dropout: float = 0.1, mask_ratio: float = 0.4,
                 lr: float = 1e-3, weight_decay: float = 0.05, use_channel_embedding: bool = True,
                 plot_every_n_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = TimeSeriesEncoder(hidden, depth, heads, patch_size, dropout,
                                         n_channel_ids=len(channel_names) if use_channel_embedding else 0)
        self.decoder = ReconstructionHead(hidden, patch_size)
        self._plot_batch = None

    def random_patch_mask(self, B, C, P, patch_valid, channel_mask):
        """Hide each valid (channel, patch) block independently with prob. mask_ratio."""
        m = torch.rand(B, C, P, device=self.device) < self.hparams.mask_ratio
        m &= patch_valid[:, None, :] & channel_mask[:, :, None]
        return m

    def forward_batch(self, batch):
        x, tmask, cmask = batch["x"], batch["time_mask"], batch["channel_mask"]
        B, C, T = x.shape
        p = self.hparams.patch_size
        patch_valid = self.encoder.patch_mask(tmask)
        pm = self.random_patch_mask(B, C, T // p, patch_valid, cmask)
        value_mask = pm.repeat_interleave(p, dim=2)                        # [B, C, T]
        cids = channel_ids_for(self.hparams.channel_names, self.hparams.channel_names, B, self.device)[:, :C]
        z, patch_valid = self.encoder(x, tmask, cmask, value_mask, cids)
        recon = self.decoder(z)
        target_w = (value_mask & tmask[:, None, :] & cmask[:, :, None]).float()
        loss = ((recon - x) ** 2 * target_w).sum() / target_w.sum().clamp(min=1.0)
        return loss, recon, value_mask

    def training_step(self, batch, _):
        loss, _, _ = self.forward_batch(batch)
        self.log("train/recon_loss", loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=len(batch["x"]))
        return loss

    def validation_step(self, batch, batch_idx):
        torch.manual_seed(batch_idx)   # same masks every epoch -> comparable val loss
        loss, recon, value_mask = self.forward_batch(batch)
        self.log("val/recon_loss", loss, prog_bar=True, on_epoch=True, batch_size=len(batch["x"]))
        if batch_idx == 0:
            self._plot_batch = (batch, recon.detach().cpu(), value_mask.cpu())

    def on_validation_epoch_end(self):
        if self._plot_batch is None or self.current_epoch % self.hparams.plot_every_n_epochs:
            return
        batch, recon, vmask = self._plot_batch
        T = int(batch["time_mask"][0].sum())
        names = self.hparams.channel_names
        show = list(range(min(4, len(names))))
        fig, axes = plt.subplots(len(show), 1, figsize=(10, 2.2 * len(show)), sharex=True)
        for ax, c in zip(np.atleast_1d(axes), show):
            m = vmask[0, c, :T].numpy()
            ax.fill_between(np.arange(T), 0, 1, where=m, color="orange", alpha=0.2,
                            transform=ax.get_xaxis_transform(), label="masked")
            ax.plot(batch["x"][0, c, :T].cpu(), lw=0.8, color="k", label="input")
            ax.plot(np.where(m, recon[0, c, :T].numpy(), np.nan), lw=0.8, color="tab:red", label="reconstruction")
            ax.set_ylabel(names[c], fontsize=8)
        np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7)
        fig.suptitle(f"{batch['batch_id'][0]}  epoch {self.current_epoch}")
        fig.tight_layout()
        self.logger.experiment.add_figure("val/reconstruction", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.trainer.max_epochs))
        return {"optimizer": opt, "lr_scheduler": sched}


class ClassifierModule(L.LightningModule):
    def __init__(self, channel_names: list[str], pretrained_ckpt: str | None = None,
                 freeze_encoder: bool = False, n_classes: int = 2, lr_head: float = 1e-3,
                 lr_encoder: float = 1e-4, weight_decay: float = 0.05,
                 class_weights: list[float] | None = None, encoder_kwargs: dict | None = None):
        super().__init__()
        self.save_hyperparameters()
        if pretrained_ckpt:
            pre = MaskedReconstructionModule.load_from_checkpoint(pretrained_ckpt, map_location="cpu")
            self.encoder = pre.encoder
            self.vocab = pre.hparams.channel_names if pre.encoder.channel_emb is not None else None
            # store the real encoder config, so this checkpoint can be reloaded without the pretrain ckpt
            ph = pre.hparams
            self.hparams.encoder_kwargs = dict(hidden=ph.hidden, depth=ph.depth, heads=ph.heads,
                                               patch_size=ph.patch_size, dropout=ph.dropout,
                                               use_channel_embedding=ph.use_channel_embedding)
            if self.vocab is not None and list(self.vocab) != list(channel_names):
                raise ValueError("finetune data has different variables than the pretrained encoder")
        else:   # train from scratch (baseline to measure what pretraining adds)
            kw = dict(encoder_kwargs or {})
            use_ce = kw.pop("use_channel_embedding", True)
            self.encoder = TimeSeriesEncoder(**kw, n_channel_ids=len(channel_names) if use_ce else 0)
            self.vocab = channel_names if use_ce else None
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.head = ClassificationHead(self.encoder.hidden, n_classes)
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", w)
        self._outputs = {"val": [], "test": []}

    def forward(self, batch):
        x = batch["x"]
        cids = None
        if self.vocab is not None:
            cids = channel_ids_for(self.hparams.channel_names, self.vocab, len(x), self.device)
        z, patch_valid = self.encoder(x, batch["time_mask"], batch["channel_mask"], channel_ids=cids)
        return self.head(z, patch_valid, batch["channel_mask"])

    def _step(self, batch, stage):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch["label"], weight=self.class_weights)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=stage == "train",
                 batch_size=len(logits))
        if stage in self._outputs:
            self._outputs[stage].append((logits.softmax(-1)[:, 1].detach().cpu(), batch["label"].cpu(),
                                         list(batch["batch_id"])))
        return loss

    def training_step(self, batch, _):
        if self.hparams.freeze_encoder:
            self.encoder.eval()   # keep dropout off in a frozen encoder
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def _epoch_metrics(self, stage):
        if not self._outputs[stage]:
            return
        p = torch.cat([o[0] for o in self._outputs[stage]]).numpy()
        y = torch.cat([o[1] for o in self._outputs[stage]]).numpy()
        ids = [i for o in self._outputs[stage] for i in o[2]]
        self._outputs[stage].clear()
        pred = (p >= 0.5).astype(int)
        # positive class = 1 = "rejected" (the minority class we want to catch)
        metrics = {
            "acc": float((pred == y).mean()),
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
        if len(np.unique(y)) == 2:
            metrics["auroc"] = float(roc_auc_score(y, p))
        for k, v in metrics.items():
            self.log(f"{stage}/{k}", v, prog_bar=k in ("balanced_acc", "f1", "auroc"))
        if stage == "test":
            self.test_results = {"metrics": metrics, "predictions": pd.DataFrame(
                {"batch_id": ids, "label": y, "p_rejected": p, "predicted": pred})}
        if stage == "test" or self.current_epoch % 5 == 0:
            cm = confusion_matrix(y, pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, v, ha="center", va="center")
            ax.set_xticks([0, 1], ["accepted", "rejected"]); ax.set_yticks([0, 1], ["accepted", "rejected"])
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout()
            self.logger.experiment.add_figure(f"{stage}/confusion_matrix", fig, self.current_epoch)
            plt.close(fig)

    def on_validation_epoch_end(self):
        self._epoch_metrics("val")

    def on_test_epoch_end(self):
        self._epoch_metrics("test")

    def configure_optimizers(self):
        groups = [{"params": self.head.parameters(), "lr": self.hparams.lr_head}]
        if not self.hparams.freeze_encoder:
            groups.append({"params": self.encoder.parameters(), "lr": self.hparams.lr_encoder})
        opt = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.trainer.max_epochs))
        return {"optimizer": opt, "lr_scheduler": sched}
