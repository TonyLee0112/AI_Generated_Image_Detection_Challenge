# Local Training Guide

## Prerequisites

- CUDA-capable GPU (tested on NVIDIA RTX A5000 24GB)
- Conda environment with PyTorch + CUDA

## Quick Start

```bash
# 1. Clone
git clone https://github.com/TonyLee0112/AI_Generated_Image_Detection_Challenge.git
cd AI_Generated_Image_Detection_Challenge

# 2. Environment
conda create -n AIGID_baseline python=3.11
conda activate AIGID_baseline
pip install -r RINE/requirements_min.txt

# 3. Download data (see docs/DATASET.md)
# Place shards under data/ntire2026/train/

# 4. Train baseline (ResNet)
python train.py

# 5. Train RINE
python RINE/train_rine.py \
  --data-root data/ntire2026/train \
  --train-shards 0 1 2 3 4 \
  --model rine \
  --epochs 5 \
  --batch-size 256 \
  --num-workers 16 \
  --amp \
  --output-dir RINE/outputs/my_exp \
  --checkpoint-prefix my_exp
```

## Server Paths (w2)

| Resource | Path |
|---|---|
| NVMe dataset | `/home/w2/suho/datasets/ntire2026/train` |
| HDD dataset (backup) | `/media/drive/datasets/ntire2026/train` |
| CLIP weights (local) | `models/clip-vit-base-patch32` |
| Conda env | `AIGID_baseline` |

## Resume Training

```bash
python RINE/train_rine.py \
  --data-root data/ntire2026/train \
  --train-shards 0 1 2 3 4 \
  --model rine \
  --epochs 4 \
  --batch-size 256 \
  --num-workers 16 \
  --amp \
  --resume-from RINE/outputs/prev_exp/prev_exp_last_model.pt \
  --output-dir RINE/outputs/new_exp \
  --checkpoint-prefix new_exp
```
