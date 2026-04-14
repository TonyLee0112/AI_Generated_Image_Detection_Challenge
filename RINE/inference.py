"""
Inference script for RINE model on unlabeled test images.
Produces submission.csv with columns: image_name, pred
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.data import build_rine_eval_transform
from src.rine_detector import RINECLIPDetector, RINEConfig


class TestImageDataset(Dataset):
    """Dataset for unlabeled test images."""

    def __init__(self, image_dir: Path, transform=None):
        self.image_paths = sorted(
            [p for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path.name


def main():
    parser = argparse.ArgumentParser(description="RINE inference on test images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--labels-csv", type=str, default=None, help="Optional labels CSV for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint and reconstruct model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt["args"]

    config = RINEConfig(
        backbone_name=saved_args["backbone_name"],
        selected_layers=saved_args.get("selected_layers"),
        proj_dim=saved_args.get("proj_dim", 256),
        q_layers=saved_args.get("q_layers", 2),
        dropout=saved_args.get("dropout", 0.5),
        supcon_temperature=saved_args.get("supcon_temperature", 0.07),
        supcon_weight=saved_args.get("supcon_weight", 0.05),
        freeze_backbone=True,
        local_files_only=True,
    )
    model = RINECLIPDetector(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataset and loader
    transform = build_rine_eval_transform(crop_size=args.crop_size)
    dataset = TestImageDataset(Path(args.test_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Test images: {len(dataset)}")

    # Inference
    all_names = []
    all_preds = []
    start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            output = model(images)
            probs = output.probabilities.cpu().numpy()
            all_names.extend(names)
            all_preds.extend(probs.tolist())
            if batch_idx % 5 == 0 or batch_idx == len(loader):
                print(f"  batch {batch_idx}/{len(loader)} done")

    elapsed = time.perf_counter() - start
    print(f"Inference done in {elapsed:.1f}s")

    # Save submission
    # Strip extension from image_name to match expected format (image_name = stem)
    df = pd.DataFrame({
        "image_name": [Path(n).stem for n in all_names],
        "pred": all_preds,
    })
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions to {args.output}")
    print(f"Pred stats: mean={np.mean(all_preds):.4f} std={np.std(all_preds):.4f} "
          f"min={np.min(all_preds):.4f} max={np.max(all_preds):.4f}")

    # Evaluate if labels are provided
    if args.labels_csv:
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve

        labels_df = pd.read_csv(args.labels_csv)
        # Merge on image_name
        if "image_name" in labels_df.columns:
            # Strip extension if present in labels
            labels_df["image_name"] = labels_df["image_name"].apply(lambda x: Path(x).stem)
        merged = df.merge(labels_df[["image_name", "label"]], on="image_name", how="inner")
        print(f"\nEvaluation on {len(merged)} labeled samples:")

        y_true = merged["label"].values
        y_score = merged["pred"].values
        y_pred = (y_score >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fnr[eer_idx] + fpr[eer_idx]) / 2.0

        print(f"  AUC:  {auc:.4f}")
        print(f"  AP:   {ap:.4f}")
        print(f"  ACC:  {acc:.4f}")
        print(f"  EER:  {eer:.4f}")


if __name__ == "__main__":
    main()
