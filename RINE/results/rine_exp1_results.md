# rine_exp1 Training Results

## Overview

| | |
|---|---|
| Model | RINE (CLIP ViT-B/32 backbone, frozen) |
| Total epochs | 9 (rine_exp1: 1–5, rine_exp1_cont: 6–9) |
| Total training time | ~9h 12m |
| Best checkpoint | rine_exp1_cont epoch 9 |
| Selection metric | val_auc + val_ap |

### rine_exp1 (epochs 1–5)

- Data: HDD `/media/drive/datasets/ntire2026/train`
- Shards: 0–4 (250,000 samples)
- LR: 2e-4, batch 256, AMP

### rine_exp1_cont (epochs 6–9)

- Resumed from: `rine_exp1/rine_exp1_last_model.pt` (epoch 5)
- Data: NVMe `/home/w2/suho/datasets/ntire2026/train` (moved from HDD for I/O speed)
- Same hyperparameters, 4 additional epochs

---

## Full 9-Epoch Metrics

### val (official validation, 10,000 samples)

| Epoch | Run | train_loss | AUC | AP | Acc | EER |
|-------|-----|-----------|-----|----|-----|-----|
| 1 | rine_exp1 | 0.6331 | 0.8901 | 0.8976 | 0.8068 | 0.1918 |
| 2 | rine_exp1 | 0.5250 | 0.8995 | 0.9098 | 0.8144 | 0.1817 |
| 3 | rine_exp1 | 0.5034 | 0.8968 | 0.9084 | 0.8134 | 0.1891 |
| 4 | rine_exp1 | 0.4929 | 0.9028 | 0.9144 | 0.8218 | 0.1814 |
| 5 | rine_exp1 | 0.4823 | 0.9026 | 0.9145 | 0.8214 | 0.1812 |
| 6 | rine_exp1_cont | 0.4713 | 0.8955 | 0.9100 | 0.8188 | 0.1890 |
| 7 | rine_exp1_cont | 0.4627 | 0.9008 | 0.9147 | 0.8201 | 0.1815 |
| 8 | rine_exp1_cont | 0.4579 | 0.8954 | 0.9103 | 0.8132 | 0.1898 |
| **9** | **rine_exp1_cont** | **0.4569** | **0.9008** | **0.9152** | **0.8190** | **0.1827** |

### val_hard (hard validation, 2,500 samples)

| Epoch | Run | AUC | AP | Acc | EER |
|-------|-----|-----|----|-----|-----|
| 1 | rine_exp1 | 0.7720 | 0.7959 | 0.7180 | 0.2956 |
| 2 | rine_exp1 | 0.7932 | 0.8177 | 0.7396 | 0.2804 |
| 3 | rine_exp1 | 0.7917 | 0.8243 | 0.7344 | 0.2848 |
| 4 | rine_exp1 | 0.8074 | 0.8375 | 0.7452 | 0.2680 |
| 5 | rine_exp1 | 0.8049 | 0.8345 | 0.7488 | 0.2688 |
| 6 | rine_exp1_cont | 0.7984 | 0.8297 | 0.7452 | 0.2736 |
| 7 | rine_exp1_cont | 0.8079 | 0.8384 | 0.7484 | 0.2716 |
| 8 | rine_exp1_cont | 0.8001 | 0.8350 | 0.7440 | 0.2748 |
| **9** | **rine_exp1_cont** | **0.8132** | **0.8460** | **0.7580** | **0.2628** |

---

## Top TIE Blocks (epoch 9)

Highest attention weight blocks for both val and val_hard:

| Rank | Block | Weight |
|------|-------|--------|
| 1 | 7 | 0.09311 |
| 2 | 8 | 0.08950 |
| 3 | 9 | 0.08595 |
| 4 | 6 | 0.08534 |
| 5 | 10 | 0.08335 |

---

## Notes

- Epoch 6 shows a slight drop from epoch 5 on both val sets, likely due to different data loading order after HDD→NVMe migration (different shuffle state).
- val_hard metrics improve monotonically from epoch 6 onward, reaching the overall best at epoch 9.
- Best model saved by `val_auc + val_ap` criterion → `rine_exp1_cont_best_model.pt` (epoch 9).
