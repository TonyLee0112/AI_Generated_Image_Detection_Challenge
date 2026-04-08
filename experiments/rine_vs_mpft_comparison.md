# RINE vs MPFT Comparison

This file summarizes:
- **RINE** results from `RINE/results/rine_exp1_results.md`
- **MPFT** results from the shared comparison screenshot, where the teammate noted:
  - *20 epoch training finished*
  - *the 8-epoch trained MPFT checkpoint was the best*

## Best checkpoints

| Model | Backbone | Training style | Best checkpoint |
|---|---|---|---|
| RINE (ours) | CLIP ViT-B/32 | frozen CLIP + multi-layer TIE pooling + BCE/SupCon | epoch 9 |
| MPFT (teammate) | CLIP ViT-L/14 | texture-rich area masking + partial CLIP fine-tuning | 8-epoch checkpoint |

## Main comparison table

| Split | Model | AUC | AP | Accuracy | EER |
|---|---|---:|---:|---:|---:|
| val | RINE | **0.9008** | **0.9152** | **0.8190** | **0.1827** |
| val | MPFT | 0.8479 | 0.8512 | 0.7595 | 0.2410 |
| val_hard | RINE | **0.8132** | **0.8460** | **0.7580** | **0.2628** |
| val_hard | MPFT | 0.7370 | 0.7522 | 0.6676 | 0.3288 |
| test | RINE | N/A | N/A | N/A | N/A |
| test | MPFT | **0.6548** | **0.6746** | **0.6032** | **0.3942** |

## Gap table (RINE - MPFT)

> For AUC / AP / Accuracy, higher is better.  
> For EER, lower is better.

| Split | AUC gap | AP gap | Accuracy gap | EER improvement |
|---|---:|---:|---:|---:|
| val | +0.0529 | +0.0640 | +0.0595 | 0.0583 lower |
| val_hard | +0.0762 | +0.0938 | +0.0904 | 0.0660 lower |

## Takeaway

- On **val**, RINE outperforms MPFT on all reported metrics.
- On **val_hard**, RINE also outperforms MPFT on all reported metrics.
- MPFT has a reported **test** score in the shared table, but RINE test results are not yet listed in this repository, so **test-side final ranking cannot be concluded here**.
- Based on the currently confirmed validation results, **RINE is stronger than the teammate's best MPFT checkpoint on both validation splits**.

## Source notes

- RINE values: `RINE/results/rine_exp1_results.md`
- MPFT values: shared screenshot table discussed in chat
