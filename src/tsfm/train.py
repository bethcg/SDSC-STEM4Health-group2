"""CLI for pretraining (masked reconstruction) and finetuning (classification).

Run from the repository root:

    # 1) self-supervised pretraining (labels are not used)
    python src/tsfm/train.py pretrain --epochs 100

    # 2) finetune: swap the decoder for a classifier, start from the pretrained encoder
    python src/tsfm/train.py finetune --pretrained lightning_logs/pretrain/version_0/checkpoints/best.ckpt

    # baseline without pretraining, or a frozen-encoder linear probe
    python src/tsfm/train.py finetune
    python src/tsfm/train.py finetune --pretrained <ckpt> --freeze-encoder

    # score a finetuned checkpoint on the test CSV (F1, precision, recall, ...)
    python src/tsfm/train.py evaluate --checkpoint lightning_logs/finetune/version_0/checkpoints/best.ckpt

    # plot decoder reconstructions of a pretrained checkpoint
    python src/tsfm/train.py reconstruct --checkpoint lightning_logs/pretrain/version_0/checkpoints/best.ckpt

    # export a finetuned model to one standalone file (default: model_process_A.pt)
    python src/tsfm/train.py export --checkpoint lightning_logs/finetune/version_0/checkpoints/best.ckpt

    # monitor
    tensorboard --logdir lightning_logs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make `tsfm` importable

import lightning as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from tsfm.data import BatchSeriesDataModule
from tsfm.lit import ClassifierModule, MaskedReconstructionModule


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["pretrain", "finetune", "evaluate", "reconstruct", "export"])
    p.add_argument("--train-csv", default="training_process_A.csv")
    p.add_argument("--test-csv", default="test_process_A.csv")
    p.add_argument("--logdir", default="lightning_logs")
    p.add_argument("--name", default=None, help="run name in TensorBoard (default: the stage)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=4096, help="random crop length for long training series")
    p.add_argument("--seed", type=int, default=42)
    # model (pretrain, or finetune from scratch)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-channel-embedding", action="store_true",
                   help="fully channel-agnostic encoder (no per-variable identity embedding)")
    # pretrain
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    # finetune
    p.add_argument("--pretrained", default=None, help="checkpoint from the pretrain stage")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--lr-encoder", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--layer-decay", type=float, default=None,
                   help="layer-wise LR decay, e.g. 0.5: head gets --lr, each lower encoder level x0.5")
    # data augmentation (training split only; strengths are relative to per-variable stats)
    p.add_argument("--augment", action="store_true", help="enable label-preserving augmentation")
    p.add_argument("--aug-p", type=float, default=0.5, help="probability of each augmentation")
    p.add_argument("--aug-jitter", type=float, default=0.25, help="noise, x the variable's own noise level")
    p.add_argument("--aug-shift", type=float, default=0.25, help="level shift, x the accepted batch-to-batch spread")
    p.add_argument("--aug-scale", type=float, default=0.05, help="sd of the fluctuation amplitude factor")
    p.add_argument("--aug-stretch", type=float, default=0.1, help="max relative time stretch")
    p.add_argument("--aug-crop-min", type=float, default=0.9, help="min fraction of the series kept by cropping")
    p.add_argument("--fast-dev-run", action="store_true", help="run 1 batch per stage to smoke-test")
    # evaluate / reconstruct
    p.add_argument("--checkpoint", default=None, help="finetune ckpt (evaluate) or pretrain ckpt (reconstruct)")
    p.add_argument("--out", default=None, help="output folder (default: next to the checkpoint)")
    p.add_argument("--split", choices=["train", "val", "test"], default="test", help="reconstruct: data split")
    p.add_argument("--batch-ids", nargs="*", default=None, help="reconstruct: batch ids to plot (default: first 4)")
    return p


def save_test_results(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res = model.test_results
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(res["metrics"], f, indent=2)
    res["predictions"].to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    print("\nTest metrics (positive class = rejected):")
    for k, v in res["metrics"].items():
        print(f"  {k:>13}: {v:.3f}")
    print(f"Saved test_metrics.json and test_predictions.csv to {out_dir}")


def default_out(ckpt, sub):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), sub)


def evaluate(a):
    if not a.checkpoint:
        raise SystemExit("--checkpoint (a finetune checkpoint) is required")
    # pretrained_ckpt=None: the encoder weights come from this checkpoint itself
    model = ClassifierModule.load_from_checkpoint(a.checkpoint, map_location="cpu", pretrained_ckpt=None)
    dm = BatchSeriesDataModule(a.train_csv, a.test_csv, patch_size=model.encoder.patch_size,
                               batch_size=a.batch_size, max_len=a.max_len, seed=a.seed)
    trainer = L.Trainer(logger=TensorBoardLogger(a.logdir, name="evaluate"), accelerator="auto")
    trainer.test(model, datamodule=dm)
    save_test_results(model, a.out or default_out(a.checkpoint, "evaluation"))


def reconstruct(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from tsfm.data import make_collate

    if not a.checkpoint:
        raise SystemExit("--checkpoint (a pretrain checkpoint) is required")
    model = MaskedReconstructionModule.load_from_checkpoint(a.checkpoint, map_location="cpu").eval()
    if a.mask_ratio != parser_default("mask_ratio"):
        model.hparams.mask_ratio = a.mask_ratio
    dm = BatchSeriesDataModule(a.train_csv, a.test_csv, patch_size=model.hparams.patch_size, seed=a.seed)
    dm.setup()
    ds = {"train": dm.train_ds, "val": dm.val_ds, "test": getattr(dm, "test_ds", None)}[a.split]
    ids = a.batch_ids or ds.ids[:4]
    unknown = [b for b in ids if b not in ds.ids]
    if unknown:
        raise SystemExit(f"batch ids not in the {a.split} split: {unknown}. Available: {', '.join(ds.ids)}")
    out = a.out or default_out(a.checkpoint, f"reconstructions_{a.split}")
    os.makedirs(out, exist_ok=True)
    collate = make_collate(model.hparams.patch_size)
    names = model.hparams.channel_names
    std = dm.stats.std
    for bid in ids:
        item = ds[ds.ids.index(bid)]
        batch = collate([item])
        torch.manual_seed(a.seed)                            # reproducible mask per run
        with torch.no_grad():
            loss, recon, vmask = model.forward_batch(batch)
        T = item["x"].shape[1]
        x, r, m = batch["x"][0, :, :T].numpy(), recon[0, :, :T].numpy(), vmask[0, :, :T].numpy()
        all_visible = not m.any()                            # mask_ratio 0 -> plot the full reconstruction
        err = float(((r - x) ** 2).mean()) if all_visible else loss.item()
        err_name = "MSE, nothing masked" if all_visible else "masked MSE"
        fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.6 * len(names)), sharex=True)
        for c, ax in enumerate(np.atleast_1d(axes)):
            # back to the original physical units
            xc = x[c] * std[c] + dm.stats.mean[c]
            rc = r[c] * std[c] + dm.stats.mean[c]
            if not all_visible:
                ax.fill_between(np.arange(T), 0, 1, where=m[c], color="orange", alpha=0.2,
                                transform=ax.get_xaxis_transform(), label="masked")
            ax.plot(xc, lw=0.7, color="k", label="input")
            ax.plot(rc if all_visible else np.where(m[c], rc, np.nan), lw=0.8, color="tab:red", label="reconstruction")
            ax.set_ylabel(names[c], fontsize=7, rotation=0, ha="right")
        np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7, ncol=3)
        np.atleast_1d(axes)[-1].set_xlabel("time step")
        label = {0: "accepted", 1: "rejected", -1: "?"}[item["label"]]
        fig.suptitle(f"{bid} ({label}) | {err_name} (standardised units) = {err:.3f}")
        fig.tight_layout()
        path = os.path.join(out, f"{bid}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"{bid:>8} {label:>9}  {err_name} = {err:.3f}  -> {path}")


def export_model(a):
    """Write a finetuned checkpoint to one standalone file (weights + config + normalisation stats)."""
    from tsfm.export import export

    if not a.checkpoint:
        raise SystemExit("--checkpoint (a finetune checkpoint) is required")
    out = a.out or "model_process_A.pt"
    model = ClassifierModule.load_from_checkpoint(a.checkpoint, map_location="cpu", pretrained_ckpt=None)
    # rebuild the training split with the same seed -> identical normalisation statistics
    dm = BatchSeriesDataModule(a.train_csv, None, patch_size=model.encoder.patch_size, seed=a.seed)
    dm.setup()
    if list(dm.channels) != list(model.hparams.channel_names):
        raise SystemExit("--train-csv has different variables than the checkpoint")
    meta = {"source_checkpoint": os.path.abspath(a.checkpoint), "train_csv": a.train_csv, "seed": a.seed}
    metrics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.checkpoint))), "test_metrics.json")
    if os.path.exists(metrics_file):
        meta["test_metrics"] = json.load(open(metrics_file))
    export(model, dm.stats, out, meta)
    print(f"Exported {a.checkpoint} -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def parser_default(name):
    return build_parser().get_default(name)


def main(argv=None):
    a = build_parser().parse_args(argv)
    L.seed_everything(a.seed, workers=True)
    if a.stage == "evaluate":
        return evaluate(a)
    if a.stage == "reconstruct":
        return reconstruct(a)
    if a.stage == "export":
        return export_model(a)

    augment = (dict(p=a.aug_p, jitter=a.aug_jitter, shift=a.aug_shift, scale=a.aug_scale,
                    stretch=a.aug_stretch, crop_min=a.aug_crop_min) if a.augment else None)
    dm = BatchSeriesDataModule(a.train_csv, a.test_csv, patch_size=a.patch_size,
                               batch_size=a.batch_size, max_len=a.max_len, seed=a.seed, augment=augment)
    dm.setup()
    logger = TensorBoardLogger(a.logdir, name=a.name or a.stage)
    callbacks = [LearningRateMonitor("epoch")]

    if a.stage == "pretrain":
        model = MaskedReconstructionModule(
            dm.channels, hidden=a.hidden, depth=a.depth, heads=a.heads, patch_size=a.patch_size,
            dropout=a.dropout, mask_ratio=a.mask_ratio, lr=a.lr,
            use_channel_embedding=not a.no_channel_embedding)
        callbacks.append(ModelCheckpoint(monitor="val/recon_loss", mode="min", filename="best", save_last=True))
    else:
        y = np.array(dm.train_labels)
        weights = (len(y) / (2 * np.bincount(y, minlength=2))).tolist()   # "balanced" class weights
        encoder_kwargs = dict(hidden=a.hidden, depth=a.depth, heads=a.heads, patch_size=a.patch_size,
                              dropout=a.dropout, use_channel_embedding=not a.no_channel_embedding)
        model = ClassifierModule(dm.channels, pretrained_ckpt=a.pretrained, freeze_encoder=a.freeze_encoder,
                                 lr_head=a.lr, lr_encoder=a.lr_encoder, class_weights=weights,
                                 encoder_kwargs=encoder_kwargs, layer_decay=a.layer_decay)
        if model.encoder.patch_size != a.patch_size:
            raise ValueError(f"--patch-size must match the pretrained encoder ({model.encoder.patch_size})")
        callbacks += [ModelCheckpoint(monitor="val/loss", mode="min", filename="best", save_last=True),
                      EarlyStopping(monitor="val/loss", mode="min", patience=a.patience)]

    if a.fast_dev_run:   # smoke test: no logger, so don't write checkpoints into the repo root
        callbacks = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
    trainer = L.Trainer(max_epochs=a.epochs, logger=logger, callbacks=callbacks, accelerator="auto",
                        log_every_n_steps=5, fast_dev_run=a.fast_dev_run, deterministic=False)
    trainer.fit(model, datamodule=dm)

    if a.stage == "finetune" and a.test_csv:
        trainer.test(model, datamodule=dm, ckpt_path=None if a.fast_dev_run else "best")
        if not a.fast_dev_run:
            save_test_results(model, logger.log_dir)


if __name__ == "__main__":
    main()
