# Experiment Registry

| ID | Name | Model | Epochs | Best val AUC | Best hard AUC | Status | Details |
|----|------|-------|--------|-------------|---------------|--------|---------|
| 1 | rine_exp1 + rine_exp1_cont | RINE (ViT-B/32, frozen) | 9 | 0.9008 | 0.8132 | Done | [results](../RINE/results/rine_exp1_results.md) |
| 2 | rine_mpft_hybrid | RINE + MPFT (ViT-B/32, last-4 blocks FT) | 9 | **0.9215** | **0.8157** | Done | [results](../RINE/results/rine_mpft_hybrid_results.md) |
| 3 | rine_mpft_hybrid_full | RINE + MPFT full shards (ViT-B/32, last-4 blocks FT, 277K samples) | 9 | **0.9227** | **0.8339** (ep9: 0.8418) | Done · test AUC 0.7721 (clean 0.8951) | [report](exp3_rine_mpft_hybrid_full/report.md) |
