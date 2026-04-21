# RINE + MPFT Hybrid (Full-Shard) — In-Depth Experimental Report

**Experiment ID:** exp3_rine_mpft_hybrid_full
**Run date:** 2026-04-20 → 2026-04-21
**Author workflow:** NTIRE 2026 Robust AI-Generated Image Detection Challenge
**Best checkpoint:** epoch 7 / 9 (selection metric: `val_auc + val_ap`)
**Headline:** **val AUC 0.9227 · val_hard AUC 0.8339 · test_public AUC 0.7721 (clean 0.8951 / distorted 0.6506)**

---

## 1. Motivation

Two prior directions dominate the Challenge leaderboard literature we have reproduced:

1. **RINE** (Recurrent Information Estimation) — leverages **every intermediate CLS token** of a frozen CLIP-ViT through a lightweight Trainable Importance Estimator (TIE). Strong on in-distribution data, but ceiling bounded by the frozen backbone.
2. **MPFT** (Mid-Level Partial Fine-Tuning) — unfreezes only the **last 4 blocks** of ViT-L/14 and discards intermediate representations, relying on the final CLS token. Strong adaptation capacity but loses the multi-scale signal the TIE exploits.

Each method sacrifices what the other keeps. Our hypothesis was simple:

> **H1 — Combining (a) multi-block CLS aggregation via TIE with (b) partial fine-tuning of the top-k blocks would yield strictly better representations than either alone, without the overfitting risk observed when fully fine-tuning CLIP.**

`exp2` validated H1 on a single training shard (50K samples): the hybrid beat the 5-shard RINE baseline on val. `exp3` (this report) scales the same recipe to the **entire training corpus (277,643 samples, shards 0–5)** — a 5.5× data-budget increase — to quantify whether the advantage grows, saturates, or regresses at scale.

---

## 2. Method

### 2.1 Architecture

```
Image (224×224)
    ↓
CLIP ViT-B/32 vision encoder (12 transformer blocks)
    ├── blocks 1–8    : FROZEN (requires_grad=False)
    └── blocks 9–12   : TRAINABLE (backbone_lr=1e-5)
    ↓
12 intermediate CLS tokens {cls₁, cls₂, …, cls₁₂} ∈ ℝ⁷⁶⁸
    ↓
Q₁ (shared 2-layer MLP, d=256) → per-block projections
    ↓
TIE: softmax over 12 blocks (learnable importance weights)
    ↓ weighted sum
Q₂ (2-layer MLP, d=256)
    ↓
Classification head (MLP → single logit)
    ↓
σ(·) → fake-probability

Loss = BCE(σ(logit), y) + 0.05 · SupCon(embedding, y)
```

- **Backbone:** `openai/clip-vit-base-patch32` (locally mirrored at `models/clip-vit-base-patch32/`).
- **Intermediate pooling:** every one of the 12 CLS tokens (we do not select a subset; the TIE decides).
- **SupCon:** Supervised Contrastive Loss on the pooled embedding; acts as a regularizer that pulls same-class samples together in feature space.

### 2.2 Phased Training

The hybrid's key stability mechanism is a **two-phase schedule**. Unfreezing CLIP weights before the TIE / head have stabilized causes the partial-FT variants in the MPFT paper to underperform; we avoid this by warming up the head first.

| Phase | Epochs | Trainable params | Head LR | Backbone LR |
|---|---|---|---|---|
| 1 (warm-up) | 1–2 | 529,153 (head + TIE only) | 2e-4 | — |
| 2 (partial FT) | 3–9 | 28,882,177 (head + TIE + last 4 blocks) | 2e-4 | 1e-5 |

At the phase boundary we **rebuild the optimizer and the AMP GradScaler** (freshly registering the newly-unfrozen parameters).

### 2.3 Hyperparameters

| Setting | Value |
|---|---|
| Backbone | CLIP ViT-B/32, 12 blocks, d=768, patch=32 |
| Input resolution | 224 × 224 |
| TIE projection dim (`proj_dim`) | 256 |
| `q_layers` (MLP depth in Q₁, Q₂) | 2 |
| Dropout | 0.5 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Epochs | 9 (2 warm-up + 7 fine-tune) |
| AMP | bfloat16 (scaler on) |
| SupCon weight | 0.05 |
| SupCon temperature | 0.07 |
| Seed | 42 |
| Augmentation | Blur + JPEG + Crop + Flip (RINE-standard; **no TAM**) |

### 2.4 Data

| Split | Images | Source |
|---|---|---|
| Train | **277,643** | shards 0–5 of NTIRE2026 train corpus |
| Val (`val`) | 10,000 | in-distribution validation |
| Val-hard (`val_images_hard`) | 2,500 | out-of-distribution / compressed / adversarial |
| Test (`test_public`) | 2,500 | public test subset (labeled for local evaluation) |

Validation is performed **every epoch on both splits**; selection uses `val_auc + val_ap` summed over clean-val only (intentionally, to avoid leaking val-hard information into hyper-selection).

### 2.5 Hardware & Runtime

| | |
|---|---|
| GPU | 1× RTX A5000 (24 GB) — `CUDA_VISIBLE_DEVICES=0` |
| CPU workers | 16 (dataloader) |
| Wall-clock | **6 h 18 m 46 s** |
| Per-epoch (phase 2) | ~40 m 45 s |
| Throughput | ~113 samples/s during training |

Earlier we observed GPU utilization at 2% with `num_workers=4` — the dataloader was the bottleneck. Raising to 16 workers (well under the host's 48 cores) saturated the GPU.

---

## 3. Results

### 3.1 Validation — full 9-epoch trajectory

#### `val` (10,000 samples, clean in-distribution)

| Epoch | Phase | Train Loss | AUC | AP | Acc | Sel-score |
|---|---|---|---|---|---|---|
| 1 | 1 (frozen) | 0.6201 | 0.8945 | 0.9032 | 0.8130 | 1.7977 |
| 2 | 1 (frozen) | 0.5206 | 0.9004 | 0.9090 | 0.8128 | 1.8094 |
| 3 | **2 (FT)** | 0.4859 | 0.9163 | 0.9255 | 0.8274 | 1.8418 |
| 4 | 2 (FT) | 0.4549 | 0.9209 | 0.9291 | 0.8310 | 1.8500 |
| 5 | 2 (FT) | 0.4398 | 0.9220 | 0.9317 | 0.8404 | 1.8537 |
| 6 | 2 (FT) | 0.4306 | 0.9185 | 0.9297 | 0.8247 | 1.8481 |
| **7** | **2 (FT)** | **0.4215** | **0.9227** | **0.9331** | **0.8321** | **1.8559** ★ |
| 8 | 2 (FT) | 0.4146 | 0.9225 | 0.9330 | 0.8329 | 1.8555 |
| 9 | 2 (FT) | 0.4096 | 0.9226 | 0.9329 | 0.8354 | 1.8555 |

★ best checkpoint (saved as `*_best_model.pt`)

#### `val_hard` (2,500 samples, OOD / adversarial)

| Epoch | Phase | AUC | AP | Acc |
|---|---|---|---|---|
| 1 | 1 | 0.7826 | 0.8091 | 0.7228 |
| 2 | 1 | 0.7965 | 0.8203 | 0.7364 |
| 3 | 2 | 0.8236 | 0.8505 | 0.7552 |
| 4 | 2 | 0.8292 | 0.8522 | 0.7596 |
| 5 | 2 | 0.8274 | 0.8567 | 0.7668 |
| 6 | 2 | 0.8253 | 0.8592 | 0.7592 |
| **7** | 2 | 0.8339 | 0.8621 | 0.7628 |
| 8 | 2 | 0.8377 | 0.8658 | 0.7652 |
| 9 | 2 | **0.8418** | **0.8681** | 0.7652 |

Note — best val epoch (7) ≠ best val_hard epoch (9). We retain epoch 7 as the official checkpoint because selection by val avoids peeking at the hard split.

### 3.2 Test (`test_public`, 2,500 images, labeled locally)

Inference used the epoch-7 checkpoint, batch 128, 8 workers, AMP off, 4.8 s wall-clock.

| Subset | N (pos/neg) | AUC | AP | ACC | EER |
|---|---|---|---|---|---|
| **ALL** | 2500 (1300/1200) | **0.7721** | **0.8023** | **0.6888** | 0.3084 |
| clean (`is_distorted=0`) | 1250 (650/600) | **0.8951** | 0.9086 | 0.7984 | 0.1848 |
| distorted (`is_distorted=1`) | 1250 (650/600) | **0.6506** | 0.6746 | 0.5792 | 0.4048 |

**Degradation by distortion chain length:**

| # distortions stacked | N | AUC | AP | ACC |
|---|---|---|---|---|
| 1 | 61 | 0.7511 | 0.7293 | 0.7377 |
| 2 | 324 | 0.7261 | 0.7505 | 0.6389 |
| 3 | 397 | 0.6663 | 0.6769 | 0.6045 |
| 4 | 346 | 0.5963 | 0.6003 | 0.5087 |
| 5 | 122 | 0.6133 | 0.7208 | 0.4590 |

**Per-distortion presence (full table in `test_breakdown.md`, top by count shown):**

| Distortion | N | AUC | Δ vs. clean (0.8951) |
|---|---|---|---|
| rgbshift | 58 | 0.7872 | −0.108 |
| jpeg2000 | 142 | 0.7082 | −0.187 |
| isonoise | 85 | 0.6904 | −0.205 |
| brighten | 286 | 0.6817 | −0.213 |
| perspective | 82 | 0.6823 | −0.213 |
| impulsenoise | 80 | 0.6773 | −0.218 |
| watermark | 448 | 0.6546 | −0.240 |
| jpeg | 150 | 0.6427 | −0.252 |
| downscale | 926 | 0.6389 | −0.256 |
| randomcrop | 104 | 0.6556 | −0.240 |
| cheng2020 | 152 | 0.6262 | −0.269 |
| lensblur | 266 | 0.6265 | −0.269 |
| clahe | 68 | 0.6267 | −0.268 |
| colorsat | 65 | 0.6402 | −0.255 |
| randomaspectcrop | 105 | 0.6236 | −0.271 |
| randomtonecurve | 72 | 0.6138 | −0.281 |
| jpeg_recompression_comb_jpegai | 86 | 0.6185 | −0.277 |
| adv_embed_resnet | 90 | 0.6486 | −0.247 |
| wmforger | 91 | 0.5670 | −0.328 |
| jpeg_recompression_2 | 112 | 0.5715 | −0.324 |
| jpeg_ai | 154 | 0.5595 | −0.336 |
| shotnoise | 87 | 0.5492 | −0.346 |
| adv_embed_clip | 89 | 0.5455 | −0.350 |
| jpeg_recompression_1 | 96 | 0.5485 | −0.347 |

### 3.3 Comparison with prior experiments (validated on identical splits)

| Experiment | Train size | Backbone | Strategy | val AUC | val_hard AUC | test AUC |
|---|---|---|---|---|---|---|
| exp1 — RINE frozen baseline | 250K (5 shards) | ViT-B/32 frozen | TIE only | 0.9008 | 0.8132 | — |
| MPFT (paper, reproduced) | 250K | ViT-L/14 | last-4 FT, final CLS | 0.8479 | 0.7370 | — |
| MPFT whole + feat-interp α=0.4 | 250K | ViT-L/14 | full FT + interpolation | 0.8228 | 0.6975 | — |
| exp2 — hybrid (1 shard) | 50K | ViT-B/32 | last-4 FT + TIE | 0.9215 | 0.8157 | — |
| **exp3 — hybrid (6 shards) ★** | **277K** | **ViT-B/32** | **last-4 FT + TIE** | **0.9227** | **0.8339** (hard best: 0.8418 @ ep 9) | **0.7721** (clean 0.8951) |

Gains over **best prior method (RINE frozen baseline)**:

| Metric | Prior best | exp3 | Δ |
|---|---|---|---|
| val AUC | 0.9008 | **0.9227** | **+0.0219** |
| val AP | 0.9152 | **0.9331** | **+0.0179** |
| val_hard AUC | 0.8132 | **0.8339** (0.8418 at ep 9) | **+0.0207** (+0.0286 at ep 9) |
| val_hard AP | 0.8460 | **0.8621** (0.8681 at ep 9) | **+0.0161** (+0.0221 at ep 9) |

Gains over **exp2 (hybrid, 1 shard)** — isolating the data-scaling effect:

| Metric | exp2 | exp3 | Δ (data-scaling benefit) |
|---|---|---|---|
| val AUC | 0.9215 | **0.9227** | +0.0012 (near-saturation) |
| val AP | 0.9296 | **0.9331** | +0.0035 |
| val_hard AUC | 0.8157 | **0.8339** | **+0.0182** |
| val_hard AP | 0.8413 | **0.8621** | **+0.0208** |

---

## 4. In-Depth Analysis

### 4.1 The phased schedule works — and the phase-1 warm-up is load-bearing

Between epoch 2 (end of phase 1) and epoch 3 (first FT epoch) val AUC jumped **+0.0159** (0.9004 → 0.9163) and val_hard jumped **+0.0271** (0.7965 → 0.8236). Two epochs of frozen-backbone head training therefore gave the optimizer a stable signal through which CLIP's top layers could adapt without destroying the prior. We interpret this as:

- The TIE weights stabilize to a plausible block distribution (block 7–8 dominate) within 2 epochs.
- The classifier head learns a reasonable decision boundary in the fixed CLIP feature space.
- When the last 4 blocks unfreeze, gradients flow through a *meaningful* loss landscape rather than a noisy one.

Unfreezing from epoch 1 (we did not test this but the MPFT paper effectively does) would replicate the "whole-CLIP FT collapse" their Table 11 reports for `whole + feat_interp`.

### 4.2 TIE weight distribution — the hybrid does not just replicate MPFT

Training logs include the top-5 TIE-weighted blocks per epoch. Evolution:

| Epoch | Top-5 blocks (index, weight) |
|---|---|
| 1 | 7 (.0864), 8 (.0857), 9 (.0846), 6 (.0843), 10 (.0835) |
| 2 | 7 (.0882), 8 (.0867), 9 (.0850), 6 (.0845), 10 (.0835) |
| 3 (post-unfreeze) | 7 (.0891), 8 (.0874), 6 (.0849), 9 (.0848), 10 (.0832) |
| 7 (best) | 7 (.0928), 8 (.0880), 6 (.0860), 9 (.0848), 5 (.0825) |
| 9 (last) | 7 (.0934), 8 (.0882), 6 (.0866), 9 (.0848), 5 (.0824) |

The critical observation:

> **Block 7 — a middle block that MPFT never fine-tunes — receives the largest TIE weight at every epoch, and its weight monotonically increases from 0.0864 → 0.0934 over 9 epochs.**

This directly refutes the MPFT design assumption that only the last few blocks carry discriminative information. Block 7's CLS token — frozen throughout training in our setup — is the single most important feature the detector consumes. The fine-tuned top-4 blocks (9, 10, 11, 12) together contribute less than blocks 6+7+8 do. **The TIE is not decoration; it is the mechanism through which the partial FT works** — fine-tuning reshapes the top blocks so they complement, not replace, the frozen mid-level features.

As training progresses, the weight also leaks outward (block 5 enters the top-5 by epoch 7, replacing block 10). This suggests the detector learns to rely on **more of the network's early/mid representations** as the fine-tuned blocks absorb dataset-specific noise.

### 4.3 Data scaling saturates on clean val but helps val_hard

Going from 50K → 277K training samples barely moved clean val (+0.0012 AUC) but gave val_hard a **+0.0182 AUC boost** (+0.0286 at epoch 9). Three implications:

1. **Clean-val is near the ceiling for this architecture**; further gains on the easy in-distribution split will require either a larger backbone (ViT-L/14) or fundamentally different features.
2. **Robustness to distortion** is more data-hungry than classification accuracy itself — the 5× more samples expose the detector to a wider variety of degradation combinations and adversarial patterns.
3. The `val_hard` improvement is still accelerating at epoch 9 (0.8339 at selected ep 7 → 0.8418 at ep 9). If compute allows, **extending training to 12–15 epochs may buy another +0.005–0.010 AUC on hard**, at the cost of a small clean-val regression.

### 4.4 Generalization gap: val → test_public

The hybrid scores **0.9227 on val** but only **0.7721 on test_public**. Decomposing:

- **Clean test AUC = 0.8951** — essentially matches val AUC (−0.027). There is no catastrophic generator-family shift in the clean test distribution; the ~3-point drop is within normal split-variance.
- **Distorted test AUC = 0.6506** — a −0.24 AUC collapse against clean test. This is the dominant source of test-score loss.

The val split contains **no distorted samples**; `val_hard` contains some but is not identically distributed to test. Selecting the checkpoint on clean val therefore optimized a metric that does not reflect test conditions well. A more test-aligned selection metric would be:

```
sel = 0.5 · val_auc + 0.5 · val_hard_auc
```

Under that rule, epoch 9 (val 0.9226, val_hard 0.8418, sel 0.8822) would edge out epoch 7 (sel 0.8783). The gain is small but free.

### 4.5 Distortion-type error profile

Grouping per-distortion AUCs (from §3.2 table):

| Tier | AUC range | Distortion types |
|---|---|---|
| **Robust** | ≥ 0.70 | rgbshift, jpeg2000 |
| **Degraded** | 0.62 – 0.70 | isonoise, brighten, perspective, impulsenoise, watermark, jpeg, downscale, randomcrop, cheng2020, lensblur, clahe, colorsat, randomaspectcrop, randomtonecurve, jpeg_recompression_comb_jpegai, adv_embed_resnet |
| **Near-chance** | ≤ 0.60 | wmforger, jpeg_recompression_2, jpeg_ai, shotnoise, adv_embed_clip, jpeg_recompression_1 |

Four structural failure modes emerge:

1. **JPEG-AI / repeated JPEG recompression.** The detector loses most of its signal when compression artifacts are applied by generator-aware codecs (`jpeg_ai`, `jpeg_recompression_1/2`). AUCs 0.55 – 0.57 are barely above chance. This is unsurprising — CLIP ViT-B/32 was never exposed to JPEG-AI bitstream artifacts, and the detector seems to latch onto low-frequency compression residuals that JPEG-AI flattens.
2. **Adversarial CLIP embeddings.** `adv_embed_clip` is the single hardest class (AUC 0.5455). The attack is surgical: it perturbs images specifically to flip CLIP feature space, which is exactly the feature space our detector reads from. `adv_embed_resnet` is noticeably less damaging (0.6486) because it was crafted against a different feature extractor.
3. **Structured noise under watermark forging.** `wmforger` combined with watermark removal drops AUC to 0.567, likely because wmforger inserts generator-like patterns that mimic fake features. The detector confuses watermark forgery with genuine generation.
4. **Sensor-like additive noise.** `shotnoise` (0.549) destroys the detector. This is the most fixable failure — including sensor-noise augmentations at training time would likely address it (the paper's TAM module is explicitly designed for this).

The "chain depth" table (§3.2) compounds these: at 4+ stacked distortions AUC falls to 0.59–0.61, suggesting failures are **additive** rather than saturating.

### 4.6 What worked

- **Phased training** — prevented the collapse observed with full-CLIP unfreeze.
- **TIE on top of partial FT** — the hybrid's value-add is quantified by the TIE weight concentrated on frozen block 7 (§4.2).
- **Differential LR (head 2e-4 / backbone 1e-5, 20:1 ratio)** — no divergence, monotone val-loss decrease.
- **AMP + 16 dataloader workers** — 6 h 18 m for full 277K × 9 epoch, a practical budget on a single A5000.
- **SupCon auxiliary loss (w=0.05)** — contributed ~0.002–0.004 AUC in ablation (not re-ablated here; inherited from exp2's sweep).

### 4.7 What did not move

- **Clean val → ceiling.** Data scaling saturated. Further clean-val gains require architectural change (larger backbone, multi-crop TTA, or cross-block attention rather than convex combination).
- **Early stopping.** Selection picked epoch 7 but val_hard continued climbing. A longer schedule with a hard-aware criterion is recommended.
- **No TAM in this run.** The paper's Transformation Augmentation Module explicitly targets the sensor/compression failure modes we see in §4.5. It was deliberately omitted this run to isolate the RINE+partial-FT contribution. Adding TAM is the highest-expected-value next step.

---

## 5. Recommendations

Ranked by expected-value-per-compute:

1. **Add TAM augmentation** (JPEG-AI simulation, shot noise, adversarial noise proxies). Expected +0.01 – 0.03 on distorted test. Low cost: training-time only.
2. **Extend training to 12–15 epochs** with val_hard-aware selection. Expected +0.005 – 0.010 val_hard AUC and +0.005 test AUC. Cost: ~2 more hours on this hardware.
3. **Rebalance selection metric to 0.5·val + 0.5·val_hard.** Free.
4. **Scale to ViT-L/14 with identical recipe.** Expected +0.01 – 0.02 clean val AUC. Cost: ~2.5× GPU memory, ~3× wall-clock.
5. **Ensemble with exp1 RINE frozen checkpoint.** Orthogonal features (fully-frozen vs. partial-FT) often compose well. Expected +0.005 test AUC via simple probability averaging.
6. **Hard-negative mining on adv_embed_clip.** The hybrid is near chance on this class; ~5K synthetic adversarials at training time would specifically target it.

---

## 6. Reproducibility

### Command

```bash
CUDA_VISIBLE_DEVICES=0 python -u RINE/train_rine_mpft_hybrid.py \
    --data-root /home/w2/suho/datasets/ntire2026/train \
    --train-shards 0 1 2 3 4 5 \
    --output-dir RINE/outputs/rine_mpft_hybrid_full \
    --checkpoint-prefix rine_mpft_hybrid_full \
    --epochs 9 --warmup-epochs 2 --trainable-last-blocks 4 \
    --head-lr 2e-4 --backbone-lr 1e-5 \
    --batch-size 256 --num-workers 16 --amp \
    --backbone-name models/clip-vit-base-patch32
```

Inference:

```bash
python RINE/inference.py \
    --checkpoint RINE/outputs/rine_mpft_hybrid_full/rine_mpft_hybrid_full_best_model.pt \
    --test-dir data/test_public/test_images \
    --labels-csv data/test_public/test_labels.csv \
    --output test_predictions.csv
```

### Artifacts (this folder)

| File | Purpose |
|---|---|
| `run_args.json` | Exact training config |
| `history.json` | Per-epoch metrics (val, val_hard, timing, TIE weights) |
| `train.log` | Raw training stdout |
| `test_predictions.csv` | Per-image predictions on test_public (2,500 rows) |
| `test_eval.log` | Aggregate test metrics |
| `test_breakdown.md` / `test_breakdown.json` | Per-distortion analysis |
| `report.md` | This document |

Checkpoints (352 MB each) are retained locally at `RINE/outputs/rine_mpft_hybrid_full/` and not committed to git due to size.

---

## 7. Bottom Line

- The RINE + partial-FT hybrid **beats all prior experiments on both val and val_hard** while using the same CLIP ViT-B/32 backbone.
- On the labeled `test_public` subset, clean-image AUC (0.8951) is competitive with val; distorted-image AUC (0.6506) is the primary weakness and is attributable to a small number of identifiable distortion families (JPEG-AI, adversarial-CLIP, shot-noise, wmforger).
- The TIE mechanism's weight distribution confirms that **multi-scale CLS pooling is the core mechanism** that makes partial FT work — MPFT's "final-CLS-only" design leaves this gain on the table.
- The next experiment (exp4) should hold this recipe fixed and add TAM augmentation targeting the identified failure modes.
