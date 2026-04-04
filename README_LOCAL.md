# NTIRE 2026 DeepFake Detection — Local Training Guide

One-time setup, then `python train.py` with no extra arguments.

## Quick Start

```bash
# 1. Create conda environment (one-time)
conda env create -f environment.yml
conda activate AIGID_baseline

# 2. Prepare dataset (one-time, after downloading from CodaBench)
python prepare_data.py --zip_dir /path/to/your/downloads

# 3. Train
python train.py
```

## Requirements

- CUDA-capable GPU (tested: RTX A5000 24GB × 3)
- Miniconda / Anaconda
- Dataset downloaded from CodaBench (see docs/DATASET.md)

## Directory Structure

```
NTIRE2026/
├── train.py                # ← Run this to train
├── config.yaml             # ← Edit this to change settings
├── prepare_data.py         # ← Run once after download
├── smoke_test.py           # ← Verify environment without data
├── environment.yml         # ← Conda env definition
├── baseline/               # Original baseline code (unmodified logic)
│   ├── train_resnet.py     # ResNet-50 baseline (default model)
│   ├── train_vit.py        # ViT baseline
│   └── aug_utils_train/    # Augmentation utilities
├── data/ntire2026/
│   ├── train/shard_0~5/    # Training data (download + prepare_data.py)
│   ├── val_official/       # Official val set (no labels, optional)
│   └── manifests/
└── outputs/
    ├── checkpoints/        # Model checkpoints (auto-created)
    ├── logs/               # CSV training logs (auto-created)
    └── predictions/        # Test predictions
```

## Default Training Setup

| Setting | Value |
|---------|-------|
| Model | ResNet-50 |
| Train shards | 0, 1, 2, 3, 4 |
| Val shards | 5 |
| Batch size | 16 |
| Max epochs | 96 |
| GPU | index 0 |

To change any setting, edit `config.yaml`.

## Smoke Test (No Data Needed)

```bash
conda activate AIGID_baseline
python smoke_test.py
```

This runs imports, model build, synthetic dataloader, forward/backward, and Lightning fast_dev_run — all without real data.

## Resume Training

Training resumes automatically from `outputs/checkpoints/last.ckpt` if it exists.

## Multi-GPU Training

Edit `config.yaml`:
```yaml
training:
  gpus: [0, 1, 2]
```

## Outputs

- `outputs/checkpoints/` — top-3 + last checkpoints
- `outputs/logs/resnet/` — CSV logs (loss, accuracy, AUROC per epoch)
- `outputs/predictions/` — test submission CSV (when using inference scripts)
