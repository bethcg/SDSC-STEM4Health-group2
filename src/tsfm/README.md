# tsfm: masked-reconstruction pretraining → classification

A minimal base for a time-series foundation model on the batch process data.

## Data format
One CSV per split. Each row is one time step, `batch_id` identifies the batch, and every other column except `label` is a monitored variable.
The data module groups rows by `batch_id` into one `[channels, time]` sample per batch. Row order within a batch is kept as the time axis.
Variables are standardised per channel, using statistics from the training split only.
This is global standardisation, not per-sample, so offsets between batches (which matter for process A) are kept.

The training CSV is split again into train/val (80/20, stratified by label, by batch, seed 42). The test CSV is only used at the end of finetuning.

## Model
```
x [B, C, T] ──► per-channel conv stem (shared weights, 1 channel in) ──► tokens [B, C, P, H]
             ──► + time position (sinusoidal, any P) + optional variable-ID embedding
             ──► depth × [attention over time per channel → attention over channels per patch]
             ──► z [B, C, P, H]
                     ├─ pretrain: ReconstructionHead  = 1 linear layer (H → patch_size values)
                     └─ finetune: ClassificationHead  = masked mean-pool + 1 linear layer
```
* **Any length T:** batches are padded to a multiple of `patch_size`, and padding is masked out of the attention and the loss.
* **Any number of channels C:** the conv stem runs per channel, and channel attention has no fixed size. Padded channels are masked.
* **Masking:** each (channel, patch) block is hidden with probability `mask_ratio` (default 0.4). The hidden values are zeroed and flagged through a mask-indicator input channel. The loss is MSE on the hidden values only.
* **Variable-ID embedding:** this lets the model tell `temperature_1` apart from `pressure_2`, but ties the encoder to a list of variable names. Use `--no-channel-embedding` for a fully channel-agnostic encoder, e.g. to pretrain across processes A and B.

## Usage (from the repo root)
```bash
python src/tsfm/train.py pretrain --epochs 100
python src/tsfm/train.py finetune --pretrained lightning_logs/pretrain/version_0/checkpoints/best.ckpt
python src/tsfm/train.py finetune --pretrained <ckpt> --freeze-encoder   # linear probe
python src/tsfm/train.py finetune --name scratch                         # no-pretraining baseline
tensorboard --logdir lightning_logs                                      # monitor
```
```bash
# score a finetuned checkpoint on the test CSV -> test_metrics.json + test_predictions.csv
python src/tsfm/train.py evaluate --checkpoint lightning_logs/finetune/version_0/checkpoints/best.ckpt
# PNG reconstructions (original units) from a pretrained checkpoint; --split train|val|test,
# --batch-ids A_005 A_060, --mask-ratio 0 to reconstruct the whole series with nothing hidden
python src/tsfm/train.py reconstruct --checkpoint lightning_logs/pretrain/version_0/checkpoints/best.ckpt
```
Test metrics use "rejected" as the positive class for precision, recall and F1.

### Transfer to another process (e.g. A -> B)
```bash
python src/tsfm/train.py finetune --train-csv training_process_B.csv --test-csv test_process_B.csv \
    --init-from model_process_A.pt --no-validation --epochs 60 --batch-size 4 --max-len 5024 --name finetune_B_from_A
```
`--init-from` starts from an exported model.
* All shared weights are reused: conv stem, transformer and head.
* Embeddings are rebuilt for the new variables. They start at the mean of the source embeddings and train at the head learning rate.
* Normalisation statistics are refitted on the new training split.

`--no-validation` trains on every training batch for a fixed number of epochs and tests the final weights. This is meant for very small datasets. `--max-len 5024` avoids randomly cropping the 5001-step B batches.

### Exporting a model
```bash
python src/tsfm/train.py export --checkpoint lightning_logs/finetune/version_0/checkpoints/best.ckpt   # -> model_process_A.pt
```
This writes one standalone file (`--out` changes the name). It holds the encoder and head weights, the architecture, the variable order, and the training normalisation means and standard deviations.
Loading it needs only PyTorch, not Lightning:
```python
import sys; sys.path.insert(0, "src")
from tsfm.export import load_exported, predict_csv
model, info = load_exported("model_process_A.pt")
predictions = predict_csv(model, info, "test_process_A.csv")   # one row per batch_id
```
`--fast-dev-run` runs one batch per stage as a smoke test. `python src/tsfm/train.py -h` lists all options.

## Logged to TensorBoard
* **Pretrain:** train/val reconstruction loss, learning rate, and every 5 epochs a figure of input vs. reconstruction with the masked blocks shaded.
* **Finetune:** loss, accuracy, balanced accuracy, AUROC (val + test), and confusion matrices.
  Uses balanced class weights, early stopping and checkpointing on `val/loss`. The test set is evaluated with the best checkpoint.
