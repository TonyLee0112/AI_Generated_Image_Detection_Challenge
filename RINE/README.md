# RINE Module

This directory adds a RINE-based training pipeline to the existing challenge repository.
Baseline code under `baseline/` is unchanged.

## Included Files

- `RINE/src/`
- `RINE/train_rine.py`
- `RINE/test.py`
- `RINE/requirements_min.txt`
- `RINE/README.md`

Excluded by design:
- datasets
- checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- output folders and caches

## What This Code Does

- Trains either:
  - baseline CLIP head (`--model baseline`)
  - RINE detector (`--model rine`)
- Uses shard CSV train data for training.
- Uses official validation sets only:
  - `val/val_images` + `val_labels.csv`
  - `val/val_images_hard` + `val_hard_labels.csv`
- Logs epoch metrics and elapsed time.

## Environment

Use the existing environment that already has PyTorch/CUDA.
Install minimal missing packages:

```bash
pip install -r RINE/requirements_min.txt
```

## Server Usage (w2)

Data is not part of this repository.
Use server-local dataset paths via `--data-root`.

Example:

```bash
python RINE/train_rine.py \
  --data-root /path/to/local/dataset \
  --train-shards 0 1 \
  --model rine \
  --epochs 1 \
  --batch-size 256 \
  --num-workers 4 \
  --amp \
  --output-dir RINE/outputs/exp_shard01 \
  --checkpoint-prefix exp_shard01
```

Single-shard example:

```bash
python RINE/train_rine.py \
  --data-root /path/to/local/dataset \
  --train-shards 0 \
  --model rine \
  --epochs 1 \
  --batch-size 256 \
  --num-workers 4 \
  --amp \
  --output-dir RINE/outputs/exp_shard0 \
  --checkpoint-prefix exp_shard0
```

Verbose test run:

```bash
python RINE/test.py \
  --data-root /path/to/local/dataset \
  --train-shards 0 \
  --batch-size 256 \
  --num-workers 4 \
  --amp \
  --output-dir RINE/outputs/test_verbose \
  --checkpoint-prefix test_verbose
```
