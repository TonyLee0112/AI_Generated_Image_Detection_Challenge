# Experiment Protocol

## Naming Convention

```
{model}_{expN}[_{suffix}]
```

Examples:
- `rine_exp1` — first RINE experiment
- `rine_exp1_cont` — continuation of rine_exp1 from a checkpoint
- `rine_exp2_lr1e3` — second experiment with LR=1e-3

## Directory Structure

Each experiment produces:
```
RINE/outputs/{experiment_name}/
├── {prefix}_best_model.pt    # best checkpoint (by selection metric)
├── {prefix}_last_model.pt    # last epoch checkpoint
├── history.json              # per-epoch metrics (machine-readable)
└── run_args.json             # full argument snapshot
```

## Metrics Tracked

| Metric | Description |
|---|---|
| train_loss | Total loss (BCE + SupCon) |
| val_auc | ROC AUC on official validation set |
| val_ap | Average Precision on official validation set |
| val_acc | Accuracy at threshold 0.5 |
| val_eer | Equal Error Rate |
| val_hard_* | Same metrics on the hard validation set |

## Validation Sets

- **val** (official): 10,000 images, balanced real/fake, with transformations
- **val_hard**: 2,500 images, harder generators and stronger transformations

## Workflow

1. Define experiment name and hyperparameters
2. Run training with `train_rine.py`
3. Results are auto-saved to `RINE/outputs/{name}/`
4. Record summary in `experiments/index.md`
5. Best result details go in `RINE/results/{name}_results.md`

## Resume / Continuation

When resuming from a checkpoint:
- Use `--resume-from` pointing to a `_last_model.pt`
- Create a new experiment name with `_cont` suffix
- The continued run gets its own `outputs/` directory
- In reporting, renumber epochs continuously (e.g., epochs 6–9 if continuing from epoch 5)
