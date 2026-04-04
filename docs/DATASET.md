# Dataset Guide — NTIRE 2026 DeepFake Detection

## Source

Competition: **NTIRE 2026 DeepFake Detection**
URL: https://www.codabench.org/competitions/12761/

## Download Instructions

1. Register / log in on CodaBench
2. Join the competition
3. Go to **Participate → Files** and download:
   - `shard_0.zip` through `shard_5.zip` (training data)
   - `val_official.zip` (official validation, **no labels**)

## Dataset Structure

Each training shard contains:
```
shard_N/
├── labels.csv          # columns: image_name, label (0=real, 1=fake)
└── images/
    ├── img_0000.jpg
    ├── img_0001.jpg
    └── ...
```

The official validation set contains:
```
val_official/
├── clear/
│   └── images/         # original resolution images (no labels)
└── distorted/
    └── images/         # distorted images (no labels)
```

## Setup (after download)

```bash
python prepare_data.py --zip_dir /path/to/downloads
```

This extracts zips into:
```
data/ntire2026/
├── train/
│   ├── shard_0/
│   ├── shard_1/
│   ├── shard_2/
│   ├── shard_3/
│   ├── shard_4/
│   └── shard_5/       ← used as local validation (has labels)
├── val_official/       ← official val (no labels, optional)
└── manifests/
    ├── train_shards.txt
    └── val_info.txt
```

## Local Validation Strategy

Since `val_official` has **no labels**, supervised local validation uses **shard_5** from the training data.

Default split in `config.yaml`:
```yaml
data:
  train_shards: [0, 1, 2, 3, 4]
  val_shards: [5]
```

## Label Format

`labels.csv` has two columns:
- `image_name`: filename without extension (or with `.jpg`)
- `label`: `0` = real image, `1` = AI-generated image

## Statistics (from baseline README)

- ~6 shards of training data
- Mix of real and AI-generated images
- Class imbalance → default class weights `[1.7, 1.0]` in config
