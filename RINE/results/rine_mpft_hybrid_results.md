# RINE + MPFT Hybrid — Experiment Results

## Overview

| | |
|---|---|
| Model | RINE-style TIE pooling on CLIP ViT-B/32 with **last-4 blocks fine-tuned** |
| Backbone | `openai/clip-vit-base-patch32` (12 blocks, 88M backbone params) |
| Training shards | shard_0 only (50,000 samples) |
| Total epochs | 9 (Phase 1: epochs 1–2 frozen; Phase 2: epochs 3–9 last-4-blocks FT) |
| Total training time | 2h 19m |
| Hardware | Single RTX A5000 (CUDA:1), AMP |
| Best checkpoint | epoch 8 (selected by val AUC + AP) |

### Pipeline

```
Image → CLIP ViT-B/32
   (blocks 1–8 frozen, blocks 9–12 trainable w/ lr=1e-5)
   → 12 intermediate CLS tokens
   → Q1 (shared MLP, d=256) → TIE block-wise softmax weighting
   → Q2 → Classification head (MLP → logit)
Loss: BCE + 0.05 · SupCon
```

### Hyperparameters

| Setting | Value |
|---|---|
| Phase-1 epochs (backbone frozen) | 2 |
| Phase-2 epochs (last-4 blocks FT) | 7 |
| Head LR | 2e-4 |
| Backbone LR (phase 2) | 1e-5 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| AMP | on |
| Dropout | 0.5 |
| proj_dim | 256 |
| supcon_weight | 0.05 |

---

## Full 9-Epoch Metrics

### val (10,000 samples)

| Epoch | Phase | Train Loss | AUC | AP | Acc | EER |
|---|---|---|---|---|---|---|
| 1 | 1 (frozen) | 0.8257 | 0.8393 | 0.8567 | 0.7669 | — |
| 2 | 1 (frozen) | 0.6311 | 0.8787 | 0.8821 | 0.7994 | — |
| 3 | 2 (FT) | 0.5719 | 0.8974 | 0.9051 | 0.8101 | — |
| 4 | 2 (FT) | 0.5229 | 0.9093 | 0.9162 | 0.8220 | — |
| 5 | 2 (FT) | 0.4979 | 0.9118 | 0.9201 | 0.8257 | — |
| 6 | 2 (FT) | 0.4862 | 0.9098 | 0.9197 | 0.8234 | — |
| 7 | 2 (FT) | 0.4715 | 0.9154 | 0.9249 | 0.8332 | — |
| **8** | **2 (FT)** | **0.4677** | **0.9215** | **0.9296** | **0.8356** | — |
| 9 | 2 (FT) | 0.4533 | 0.9173 | 0.9260 | 0.8331 | — |

### val_hard (2,500 samples)

| Epoch | Phase | AUC | AP | Acc |
|---|---|---|---|---|
| 1 | 1 | 0.7085 | 0.7309 | 0.6556 |
| 2 | 1 | 0.7478 | 0.7681 | 0.6876 |
| 3 | 2 | 0.7786 | 0.8076 | 0.7136 |
| 4 | 2 | 0.7987 | 0.8232 | 0.7324 |
| 5 | 2 | 0.7986 | 0.8269 | 0.7328 |
| 6 | 2 | 0.8038 | 0.8343 | 0.7420 |
| 7 | 2 | 0.8099 | 0.8400 | 0.7472 |
| **8** | **2** | **0.8157** | **0.8413** | **0.7444** |
| 9 | 2 | 0.8139 | 0.8390 | 0.7416 |

---

## Comparison with Prior Experiments (from paper Table 11)

| Split | Model | AUC | AP | Acc | EER |
|---|---|---|---|---|---|
| val | RINE (frozen ViT-B/32, 5 shards × 9ep) | 0.9008 | 0.9152 | 0.8190 | 0.1827 |
| val | MPFT (ViT-L/14, last-4-blocks FT) | 0.8479 | 0.8512 | 0.7595 | 0.2410 |
| val | MPFT (ViT-L/14, whole + feat_interp α=0.4) | 0.8228 | 0.8238 | 0.7351 | 0.2622 |
| **val** | **RINE+MPFT hybrid (ours, shard_0 × 9ep, epoch 8)** | **0.9215** | **0.9296** | **0.8356** | — |
| val_hard | RINE (5 shards × 9ep) | 0.8132 | 0.8460 | 0.7580 | 0.2628 |
| val_hard | MPFT (last-4-blocks) | 0.7370 | 0.7522 | 0.6676 | 0.3288 |
| val_hard | MPFT (whole + feat_interp) | 0.6975 | 0.6974 | 0.6420 | 0.3608 |
| **val_hard** | **RINE+MPFT hybrid (ours, epoch 8)** | **0.8157** | **0.8413** | **0.7444** | — |

### Delta vs. prior RINE best

| Metric | Prior RINE best | Hybrid (epoch 8) | Δ |
|---|---|---|---|
| val AUC | 0.9008 | **0.9215** | **+0.0207** |
| val AP  | 0.9152 | **0.9296** | **+0.0144** |
| val_hard AUC | 0.8132 | **0.8157** | **+0.0025** |
| val_hard AP | 0.8460 | 0.8413 | −0.0047 |

Note: prior RINE used **5 training shards (250K samples)** while this hybrid used **only shard_0 (50K samples)**. Despite the 5× smaller training corpus, the hybrid outperforms on val and marginally surpasses on val_hard AUC.

---

## Key Findings

1. **Phase-1 warmup effective**: head/TIE converged to val AUC 0.879 in 2 epochs with backbone frozen, providing a stable signal before unfreezing.
2. **Phase-2 FT gives the big jump**: val AUC +0.019 and val_hard +0.031 immediately after unfreezing at epoch 3 (0.8787→0.8974 val, 0.7478→0.7786 val_hard).
3. **Val peaks at epoch 8, not the last epoch**: slight overfit at epoch 9 (val −0.004, val_hard −0.002). Early-stop by val selection metric correctly picks epoch 8.
4. **Beats RINE baseline on val by wide margin**: +2.07 AUC points on val even with 1/5 the training data.
5. **val_hard AP regression**: small drop (−0.0047). Training data size (1 shard) may be the bottleneck — likely to improve with more shards.

---

## Next Steps Suggested

1. **Scale to 5 shards**: directly compare against RINE baseline on the same data budget.
2. **Alpha sweep for Phase-2 LR**: backbone_lr ∈ {5e-6, 1e-5, 2e-5} to confirm 1e-5 is optimal.
3. **Longer phase-1 warmup**: try 3 warmup epochs to see if head converges more cleanly.
4. **Add TAM module** (Transformation Augmentation) from MPFT for extra robustness on val_hard.
5. **Ensemble**: combine epoch 8 hybrid with prior RINE best checkpoint for potential test-set gains.

---

## Artifacts

- Checkpoint: `RINE/outputs/rine_mpft_hybrid/rine_mpft_hybrid_best_model.pt` (epoch 8)
- Last checkpoint: `rine_mpft_hybrid_last_model.pt` (epoch 9)
- Full history: `history.json`
- Training log: `train.log`
- Script: `RINE/train_rine_mpft_hybrid.py`
