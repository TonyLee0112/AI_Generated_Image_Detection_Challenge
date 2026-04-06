# Method: RINE Detector

## Overview

RINE (Representation INtErpolation) is a CLIP-based AI-generated image detector that uses multi-layer feature extraction with learnable attention-weighted aggregation.

## Architecture

```
Input Image (224x224)
    │
    ▼
CLIP ViT-B/32 Backbone (frozen)
    │
    ├── Block 1 hidden state ──► Linear Proj (→ 256d)
    ├── Block 2 hidden state ──► Linear Proj (→ 256d)
    ├── ...
    └── Block 12 hidden state ─► Linear Proj (→ 256d)
                                       │
                                       ▼
                             TIE Attention Pooling
                            (learnable block weights)
                                       │
                                       ▼
                               Q-Layer MLP Head
                                       │
                                  ┌────┴────┐
                                  ▼         ▼
                            BCE logit   SupCon embedding
```

### Key Components

1. **Frozen CLIP Backbone**: ViT-B/32 extracts features from all transformer blocks without fine-tuning.
2. **Per-block Projection**: Each block's hidden state is projected to a shared 256-d space.
3. **TIE (Token Interpolation via Experts) Attention**: Learnable weights decide how much each block contributes. Top blocks typically: 7 > 8 > 9 > 6 > 10.
4. **Q-Layer Head**: 2-layer MLP with dropout (0.5) for binary classification.
5. **Supervised Contrastive Loss**: Auxiliary loss (weight=0.05, temp=0.07) encourages real/fake separation in embedding space.

## Loss Function

```
L_total = L_bce + 0.05 * L_supcon
```

- `L_bce`: Binary cross-entropy on the classification logit
- `L_supcon`: Supervised contrastive loss on the embedding vector

## Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | CLIP ViT-B/32 (frozen) |
| Projection dim | 256 |
| Q-layers | 2 |
| Dropout | 0.5 |
| SupCon weight | 0.05 |
| SupCon temperature | 0.07 |
| Learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| AMP | Enabled (fp16) |
| Trainable parameters | 529,153 |

## Selection Metric

Best model is saved by `val_auc + val_ap` (sum of validation AUC and Average Precision).
