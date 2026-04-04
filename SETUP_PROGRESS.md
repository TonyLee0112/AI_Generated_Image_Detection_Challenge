# Setup Progress Log — NTIRE 2026 DeepFake Detection

## 2026-04-04

### Completed

- [x] **Repo cloned** from https://github.com/msu-video-group/NTIRE-2026-DeepFake-Detection/
  - Located at: `/home/w2/suho/NTIRE2026/`
  - Contains: `baseline/train_resnet.py`, `baseline/train_vit.py`, `baseline/aug_utils_train/`

- [x] **Repo analysis**
  - Two baselines: ResNet-50 (no pretrained) and ViT medium (pretrained timm)
  - Both had hardcoded `/root/users/deepfake_bench/...` paths → removed
  - Dataset format: `shard_N/labels.csv` + `shard_N/images/`
  - PyProject requires Python 3.11.14, torch==2.8.0

- [x] **Conda environment defined** (`environment.yml`)
  - Name: `AIGID_baseline`
  - Python 3.11.14, PyTorch 2.8.0, pytorch-lightning>=2.5, timm, kornia, etc.
  - Created via: `conda env create -f environment.yml`

- [x] **Directory structure created**
  ```
  data/ntire2026/train/        ← user fills with shards after download
  data/ntire2026/val_official/ ← optional official val (no labels)
  data/ntire2026/manifests/
  outputs/checkpoints/
  outputs/logs/
  outputs/predictions/
  ```

- [x] **`prepare_data.py`** written
  - Extracts downloaded zip files → correct directory layout
  - Verifies `labels.csv` + `images/` in each shard
  - Usage: `python prepare_data.py --zip_dir /path/to/downloads`

- [x] **`config.yaml`** written
  - All settings in one place (no hardcoded paths anywhere)
  - Default: ResNet-50, train shards 0-4, val shard 5, batch=16

- [x] **Root `train.py`** written
  - Zero extra CLI arguments required
  - Project-relative paths via `Path(__file__).parent`
  - Friendly error if data is missing (points to docs)
  - CSVLogger → `outputs/logs/`, checkpoints → `outputs/checkpoints/`
  - Auto-resumes from `last.ckpt` if present

- [x] **`smoke_test.py`** written — runs without real data
  - Tests: imports, model forward/backward, synthetic dataloader, output dirs, config load, Lightning fast_dev_run

- [x] **`README_LOCAL.md`** written
- [x] **`docs/DATASET.md`** written

### In Progress

- [ ] **Dataset download** (requires manual CodaBench login)

### Completed (continued)

- [x] **Conda environment created** successfully
  - `conda env create -f environment.yml` → AIGID_baseline
  - torch=2.6.0+cu124, pytorch-lightning=2.6.1, timm=1.0.26, kornia=0.8.2

- [x] **Smoke test PASSED (6/6)** — `python smoke_test.py`
  - imports: torch, pl, timm, kornia, pandas, yaml
  - model forward/backward on synthetic data (loss=0.72)
  - DataLoader with synthetic shard (1024×1024 images)
  - output directory creation
  - config.yaml parsing
  - Lightning fast_dev_run (1 step train+val, 25.7M params)

- [x] **train.py friendly error verified** — prints exact path and CodaBench URL when data missing

### Environment Details

- OS: Linux (Ubuntu, kernel 5.15)
- GPUs: 3× NVIDIA RTX A5000 (24GB each)
- CUDA: 12.x (driver), nvcc 10.1 (toolkit)
- Conda: 24.11.1

### Notes

- Official validation has no labels → using shard_5 for local supervised val
- ResNet-50 chosen as default (no internet download during training)
- ViT baseline available in `baseline/train_vit.py` — set `model: vit` in config.yaml
- Absolute paths fully removed from training pipeline
