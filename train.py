"""
Root-level train.py for NTIRE 2026 DeepFake Detection.
Default model: ResNet-50 baseline.

Usage:
    conda activate AIGID_baseline
    python train.py

All paths are project-relative. Edit config.yaml to change settings.
"""

import os
import sys
import yaml
from pathlib import Path

# ── project root (directory containing this file) ──────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline"))

# ── imports ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision.models import resnet50
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from train_resnet import (
    AIGenDetDataset,
    BaselineDetector,
    TrainingModule,
    collate,
)
from aug_utils_train import distort_images


class LocalTrainingModule(TrainingModule):
    """Thin subclass that overrides configure_callbacks to remove EarlyStopping.

    The baseline TrainingModule.configure_callbacks() includes EarlyStopping,
    which we disable here. Checkpoint dir is injected at construction time so
    ModelCheckpoint saves to outputs/checkpoints regardless of trainer root.
    """

    def __init__(self, *args, checkpoint_dir: str = "outputs/checkpoints", **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_dir = checkpoint_dir

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                dirpath=self._checkpoint_dir,
                filename="resnet-{epoch:02d}-{val_loss:.4f}",
                monitor=self.monitor_key,   # "val_loss" from base class
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True,
                enable_version_counter=False,
            )
        ]


# ── helpers ─────────────────────────────────────────────────────────────────

def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_data(cfg: dict) -> None:
    """Friendly error if required data directories are missing."""
    shard_dir = ROOT / cfg["data"]["shard_dir"]
    val_shards = cfg["data"]["val_shards"]
    train_shards = cfg["data"]["train_shards"]

    missing = []
    for s in train_shards + val_shards:
        d = shard_dir / f"shard_{s}"
        if not d.is_dir():
            missing.append(str(d))

    if missing:
        print("\n[ERROR] Missing dataset directories:")
        for m in missing:
            print(f"  {m}")
        print(
            "\nPlease download the NTIRE 2026 training shards from CodaBench "
            "(https://www.codabench.org/competitions/12761/) and run:\n"
            "  python prepare_data.py --zip_dir <path_to_downloads>\n"
            "See docs/DATASET.md for details."
        )
        sys.exit(1)

    # Also check each shard has labels.csv + images/
    for s in train_shards + val_shards:
        d = shard_dir / f"shard_{s}"
        if not (d / "labels.csv").is_file():
            print(f"[ERROR] {d}/labels.csv not found. Run prepare_data.py first.")
            sys.exit(1)
        if not (d / "images").is_dir():
            print(f"[ERROR] {d}/images/ not found. Run prepare_data.py first.")
            sys.exit(1)

    print(f"[OK] Data check passed. Using shards: train={train_shards}, val={val_shards}")


def make_dataloaders(cfg: dict):
    shard_dir = str(ROOT / cfg["data"]["shard_dir"])
    train_shards = cfg["data"]["train_shards"]
    val_shards = cfg["data"]["val_shards"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    image_size = cfg["training"].get("image_size", 1024)

    train_dataset = AIGenDetDataset.read_from_shards(
        shard_dir, shard_nums=train_shards, transform=distort_images,
        image_size=image_size,
    )
    val_dataset = AIGenDetDataset.read_from_shards(
        shard_dir, shard_nums=val_shards,
        image_size=image_size,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    train_dl = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(dataset=val_dataset, shuffle=False, drop_last=False, **common)
    return train_dl, val_dl


def build_training_module(cfg: dict, checkpoint_dir: str) -> LocalTrainingModule:
    model = BaselineDetector()
    if cfg["training"].get("compile_model", False):
        model = torch.compile(model)
    tcfg = cfg["training"]
    module = LocalTrainingModule(
        model=model,
        class_weights=tcfg["class_weights"],
        lr=tcfg["lr"],
        min_lr=tcfg["min_lr"],
        submission_file=None,
        checkpoint_dir=checkpoint_dir,
    )
    return module


def build_trainer(cfg: dict) -> pl.Trainer:
    tcfg = cfg["training"]
    ocfg = cfg["output"]

    ckpt_dir = ROOT / ocfg["checkpoint_dir"]
    log_dir = ROOT / ocfg["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = tcfg.get("gpus", [0])
    precision = tcfg.get("precision", "32-true")
    logger = CSVLogger(save_dir=str(log_dir), name="resnet")

    # For multi-GPU without NVLink: use DDP with static graph and
    # gradient_as_bucket_view to reduce PCIe overhead
    if len(gpu_ids) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=gpu_ids if torch.cuda.is_available() else 1,
        strategy=strategy,
        precision=precision,
        min_epochs=tcfg["min_epochs"],
        max_epochs=tcfg["max_epochs"],
        gradient_clip_val=tcfg["gradient_clip_val"],
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=str(ckpt_dir),
    )
    return trainer


def find_resume_ckpt(cfg: dict) -> str | None:
    ckpt_dir = ROOT / cfg["output"]["checkpoint_dir"]
    last = ckpt_dir / "last.ckpt"
    return str(last) if last.is_file() else None


def main():
    # Prevent numpy/torch worker threads from competing for CPU cores
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("medium")

    cfg = load_config()
    check_data(cfg)

    print(f"[INFO] Training with ResNet-50 baseline")
    print(f"[INFO] Train shards: {cfg['data']['train_shards']}")
    print(f"[INFO] Val shards:   {cfg['data']['val_shards']}")
    print(f"[INFO] Image size:   {cfg['training'].get('image_size', 1024)}")
    print(f"[INFO] Batch size:   {cfg['training']['batch_size']}")
    print(f"[INFO] Precision:    {cfg['training'].get('precision', '32-true')}")
    print(f"[INFO] Compile:      {cfg['training'].get('compile_model', False)}")
    print(f"[INFO] GPUs:         {cfg['training']['gpus']}")

    train_dl, val_dl = make_dataloaders(cfg)
    ckpt_dir = str(ROOT / cfg["output"]["checkpoint_dir"])
    module = build_training_module(cfg, checkpoint_dir=ckpt_dir)
    trainer = build_trainer(cfg)

    resume = find_resume_ckpt(cfg)
    if resume:
        print(f"[INFO] Resuming from checkpoint: {resume}")

    trainer.fit(
        module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=resume,
    )

    print("\n[DONE] Training complete.")
    print(f"  Checkpoints: {ROOT / cfg['output']['checkpoint_dir']}")
    print(f"  Logs:        {ROOT / cfg['output']['log_dir']}")


if __name__ == "__main__":
    main()
