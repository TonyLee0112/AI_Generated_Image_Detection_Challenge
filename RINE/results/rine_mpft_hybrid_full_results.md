# RINE + MPFT Hybrid (Full Shards) — Experiment Results

Full in-depth analysis: [`experiments/exp3_rine_mpft_hybrid_full/report.md`](../../experiments/exp3_rine_mpft_hybrid_full/report.md)

## Overview

| | |
|---|---|
| Model | RINE TIE pooling on CLIP ViT-B/32 with last-4 blocks fine-tuned |
| Backbone | `openai/clip-vit-base-patch32` (12 blocks) |
| Training shards | shards 0–5 (277,643 samples) |
| Total epochs | 9 (phase 1: 1–2 frozen head/TIE; phase 2: 3–9 last-4 FT) |
| Total training time | 6 h 18 m 46 s |
| Hardware | Single RTX A5000 (CUDA:0), AMP, 16 dataloader workers |
| Best checkpoint | epoch 7 (selection: val_auc + val_ap) |

## Headline metrics

| Split | AUC | AP | ACC | EER |
|---|---|---|---|---|
| val | **0.9227** | **0.9331** | 0.8321 | — |
| val_hard (ep 7) | **0.8339** | 0.8621 | 0.7628 | — |
| val_hard (ep 9) | **0.8418** | 0.8681 | 0.7652 | — |
| test_public — ALL | 0.7721 | 0.8023 | 0.6888 | 0.3084 |
| test_public — clean | **0.8951** | 0.9086 | 0.7984 | 0.1848 |
| test_public — distorted | 0.6506 | 0.6746 | 0.5792 | 0.4048 |

## Comparison with all prior experiments

| Experiment | Train | val AUC | val_hard AUC |
|---|---|---|---|
| exp1 RINE frozen (5 shards) | 250K | 0.9008 | 0.8132 |
| MPFT last-4 FT (paper) | 250K | 0.8479 | 0.7370 |
| MPFT whole + feat-interp | 250K | 0.8228 | 0.6975 |
| exp2 hybrid (1 shard) | 50K | 0.9215 | 0.8157 |
| **exp3 hybrid (6 shards)** | **277K** | **0.9227** | **0.8339** (0.8418 @ ep 9) |

Δ vs. best prior (RINE frozen): val **+0.0219**, val_hard **+0.0207** (+0.0286 @ ep 9).
Δ vs. exp2 (same recipe, 5.5× data): val +0.0012 (saturated), val_hard **+0.0182** (still gaining).

## Full 9-epoch metrics

### val (10,000)

| Epoch | Phase | Train Loss | AUC | AP | Acc |
|---|---|---|---|---|---|
| 1 | 1 | 0.6201 | 0.8945 | 0.9032 | 0.8130 |
| 2 | 1 | 0.5206 | 0.9004 | 0.9090 | 0.8128 |
| 3 | 2 | 0.4859 | 0.9163 | 0.9255 | 0.8274 |
| 4 | 2 | 0.4549 | 0.9209 | 0.9291 | 0.8310 |
| 5 | 2 | 0.4398 | 0.9220 | 0.9317 | 0.8404 |
| 6 | 2 | 0.4306 | 0.9185 | 0.9297 | 0.8247 |
| **7** | **2** | **0.4215** | **0.9227** | **0.9331** | **0.8321** |
| 8 | 2 | 0.4146 | 0.9225 | 0.9330 | 0.8329 |
| 9 | 2 | 0.4096 | 0.9226 | 0.9329 | 0.8354 |

### val_hard (2,500)

| Epoch | Phase | AUC | AP | Acc |
|---|---|---|---|---|
| 1 | 1 | 0.7826 | 0.8091 | 0.7228 |
| 2 | 1 | 0.7965 | 0.8203 | 0.7364 |
| 3 | 2 | 0.8236 | 0.8505 | 0.7552 |
| 4 | 2 | 0.8292 | 0.8522 | 0.7596 |
| 5 | 2 | 0.8274 | 0.8567 | 0.7668 |
| 6 | 2 | 0.8253 | 0.8592 | 0.7592 |
| 7 | 2 | 0.8339 | 0.8621 | 0.7628 |
| 8 | 2 | 0.8377 | 0.8658 | 0.7652 |
| 9 | 2 | **0.8418** | **0.8681** | 0.7652 |

## Key findings (TL;DR)

1. **Hybrid is state-of-the-art on our stack** — beats RINE frozen baseline on both clean and hard val with the same ViT-B/32 backbone.
2. **Data scaling saturates on clean val, still pays on val_hard** — 5.5× more data buys +0.018 val_hard AUC but only +0.001 val.
3. **Phase-1 warm-up is load-bearing** — epoch-3 unfreeze gives the single biggest per-epoch gain (+0.016 val, +0.027 val_hard).
4. **TIE continues to prefer frozen middle blocks** — block 7 is the top-weighted block every epoch, refuting "only top blocks matter."
5. **Test clean (0.8951) matches val (0.9227) closely**; distorted test (0.6506) is the bottleneck.
6. **Specific distortion failure modes** (full breakdown in report.md): JPEG-AI / repeated recompression, adversarial CLIP embeddings, wmforger, shotnoise — all AUC ≤ 0.57.

## Artifacts

Under `RINE/outputs/rine_mpft_hybrid_full/`:
- `rine_mpft_hybrid_full_best_model.pt` (epoch 7, 352 MB)
- `rine_mpft_hybrid_full_last_model.pt` (epoch 9, 352 MB)
- `history.json`, `run_args.json`, `train.log`
- `test_predictions.csv`, `test_eval.log`
- `test_breakdown.md`, `test_breakdown.json`

Mirrored (without checkpoints) under `experiments/exp3_rine_mpft_hybrid_full/`.
