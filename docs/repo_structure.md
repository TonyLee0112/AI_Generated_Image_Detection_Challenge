# Repository Structure

```
AI_Generated_Image_Detection_Challenge/
│
├── README.md                    # Challenge overview (upstream)
├── README_LOCAL.md              # Local training guide
├── config.yaml                  # Baseline training config
├── train.py                     # Baseline training entry point
├── prepare_data.py              # Dataset download/extraction
├── smoke_test.py                # Quick sanity check
├── environment.yml              # Conda environment spec
├── pyproject.toml               # Python project metadata
│
├── baseline/                    # Upstream baseline models
│   ├── train_resnet.py
│   ├── train_vit.py
│   ├── inference_resnet.py
│   ├── inference_vit.py
│   ├── aug_utils_train/         # Data augmentation utilities
│   └── checkpoints/             # Baseline pretrained weights
│
├── RINE/                        # RINE detector module
│   ├── README.md                # RINE module guide + results summary
│   ├── train_rine.py            # RINE training script
│   ├── test.py                  # RINE inference/test script
│   ├── requirements_min.txt     # Minimal pip dependencies
│   ├── src/
│   │   ├── __init__.py
│   │   ├── rine_detector.py     # Model architecture
│   │   └── data.py              # Dataset and transforms
│   ├── results/                 # Experiment result reports (tracked)
│   │   └── rine_exp1_results.md
│   └── outputs/                 # Checkpoints and logs (gitignored)
│
├── docs/                        # Documentation
│   ├── DATASET.md               # Dataset download and structure
│   ├── method.md                # RINE method description
│   ├── experiment_protocol.md   # Naming, metrics, workflow
│   └── repo_structure.md        # This file
│
├── experiments/                  # Experiment registry
│   ├── index.md                 # All experiments listing
│   ├── summary.csv              # Machine-readable metrics
│   └── templates/
│       └── experiment_template.md
│
└── data/                        # Local dataset (gitignored)
    └── ntire2026/
        └── train/
            ├── shard_0/ ... shard_5/
            └── val/
```

## What Is Tracked in Git

- All source code (`*.py`)
- Documentation (`docs/`, `*.md`)
- Experiment results (`RINE/results/`)
- Experiment registry (`experiments/`)
- Config files (`config.yaml`, `pyproject.toml`, etc.)

## What Is Gitignored

- Dataset files (`data/`, `dataset/`)
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- Output directories (`outputs/`, `RINE/outputs/`)
- Logs (`*.log`)
- Python caches (`__pycache__/`)
- Local model weights (`models/`)
