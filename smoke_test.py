"""
Smoke test for NTIRE 2026 DeepFake Detection pipeline.

Tests (without real data):
  1. imports
  2. model build + forward/backward on synthetic data
  3. output directory creation
  4. checkpoint save (via trainer fast_dev_run)

Usage:
  python smoke_test.py
"""

import sys
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline"))

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_imports():
    print("1. Testing imports...")
    try:
        import torch
        import torchvision
        import pytorch_lightning as pl
        import timm
        import kornia
        import pandas as pd
        import yaml
        print(f"   {PASS} torch={torch.__version__}, pl={pl.__version__}, timm={timm.__version__}")
        return True
    except ImportError as e:
        print(f"   {FAIL} Import error: {e}")
        return False


def test_model_forward_backward():
    print("2. Testing model build + forward/backward...")
    try:
        import torch
        from train_resnet import BaselineDetector

        model = BaselineDetector()
        model.train()

        # Synthetic batch: B=2, C=3, H=224, W=224 (smaller than 1024 for speed)
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 2, (2,))

        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        assert logits.shape == (2, 2), f"Wrong logit shape: {logits.shape}"
        print(f"   {PASS} forward OK, loss={loss.item():.4f}, logits shape={logits.shape}")
        return True
    except Exception as e:
        print(f"   {FAIL} {e}")
        import traceback; traceback.print_exc()
        return False


def test_dataloader_synthetic():
    print("3. Testing dataloader with synthetic (fake) shard data...")
    try:
        import torch
        import pandas as pd
        import numpy as np
        from PIL import Image
        from train_resnet import AIGenDetDataset, collate
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Create a fake shard
            shard = tmp / "shard_0"
            images_dir = shard / "images"
            images_dir.mkdir(parents=True)

            n_images = 4
            for i in range(n_images):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(images_dir / f"img_{i:04d}.jpg")

            labels = pd.DataFrame({
                "image_name": [f"img_{i:04d}" for i in range(n_images)],
                "label": [0, 1, 0, 1],
            })
            labels.to_csv(shard / "labels.csv")

            dataset = AIGenDetDataset(
                root_dir=str(tmp),
                subset_dirs=["shard_0"],
            )
            dl = DataLoader(dataset, batch_size=2, collate_fn=collate, num_workers=0)
            batch = next(iter(dl))

            assert batch["image"].shape[0] == 2
            assert batch["label"].shape[0] == 2
            print(f"   {PASS} DataLoader OK, batch image shape={batch['image'].shape}")
        return True
    except Exception as e:
        print(f"   {FAIL} {e}")
        import traceback; traceback.print_exc()
        return False


def test_output_dirs():
    print("4. Testing output directory creation...")
    try:
        dirs = [
            ROOT / "outputs" / "checkpoints",
            ROOT / "outputs" / "logs",
            ROOT / "outputs" / "predictions",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            assert d.is_dir()
        print(f"   {PASS} Output dirs OK")
        return True
    except Exception as e:
        print(f"   {FAIL} {e}")
        return False


def test_config_load():
    print("5. Testing config.yaml load...")
    try:
        import yaml
        cfg_path = ROOT / "config.yaml"
        assert cfg_path.is_file(), "config.yaml not found"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert "data" in cfg
        assert "training" in cfg
        assert "output" in cfg
        print(f"   {PASS} config.yaml OK, model={cfg.get('model')}")
        return True
    except Exception as e:
        print(f"   {FAIL} {e}")
        return False


def test_lightning_fast_dev_run():
    print("6. Testing Lightning fast_dev_run (1 batch train+val)...")
    try:
        import torch
        import pandas as pd
        import numpy as np
        from PIL import Image
        from train_resnet import AIGenDetDataset, BaselineDetector, TrainingModule, collate
        from torch.utils.data import DataLoader
        import pytorch_lightning as pl
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Two shards: train (shard_0) + val (shard_1)
            for shard_name in ["shard_0", "shard_1"]:
                shard = tmp / shard_name
                images_dir = shard / "images"
                images_dir.mkdir(parents=True)
                for i in range(4):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    )
                    img.save(images_dir / f"img_{i:04d}.jpg")
                labels = pd.DataFrame({
                    "image_name": [f"img_{i:04d}" for i in range(4)],
                    "label": [0, 1, 0, 1],
                })
                labels.to_csv(shard / "labels.csv")

            common = dict(batch_size=2, num_workers=0, collate_fn=collate)
            train_dl = DataLoader(
                AIGenDetDataset(str(tmp), ["shard_0"]),
                shuffle=True, drop_last=False, **common
            )
            val_dl = DataLoader(
                AIGenDetDataset(str(tmp), ["shard_1"]),
                shuffle=False, **common
            )

            ckpt_dir = tmp / "ckpts"
            ckpt_dir.mkdir()

            model = BaselineDetector()
            module = TrainingModule(model=model, class_weights=[1.0, 1.0])

            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                fast_dev_run=True,
                enable_progress_bar=False,
                logger=False,
                default_root_dir=str(ckpt_dir),
            )
            trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
            print(f"   {PASS} Lightning fast_dev_run completed successfully")
        return True
    except Exception as e:
        print(f"   {FAIL} {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("NTIRE 2026 DeepFake Detection - Smoke Test")
    print("=" * 60)

    results = [
        test_imports(),
        test_model_forward_backward(),
        test_dataloader_synthetic(),
        test_output_dirs(),
        test_config_load(),
        test_lightning_fast_dev_run(),
    ]

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n[ALL PASS] Pipeline is ready. Provide data and run: python train.py")
    else:
        print("\n[SOME FAILED] Check errors above and install missing dependencies.")
        print("  conda activate AIGID_baseline && pip install -e baseline/")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
