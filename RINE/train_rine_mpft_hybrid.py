"""
RINE + MPFT hybrid detector training.

Pipeline: Image -> CLIP ViT-B/32 (last K blocks fine-tuned) -> intermediate CLS tokens
         -> TIE block-wise weighting -> MLP head

Two-phase schedule:
  Phase 1 (epochs 1..warmup_epochs): backbone fully frozen, head/TIE warm-up.
  Phase 2 (remaining epochs): unfreeze last --trainable-last-blocks blocks with
                              differential LR (lower on backbone).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data import (
    PathLabelDataset,
    build_rine_eval_transform,
    build_rine_train_transform,
    load_labeled_image_folder,
    load_shard_samples,
)
from src.rine_detector import (
    DetectorDataParallel,
    RINECLIPDetector,
    RINEConfig,
    SupervisedContrastiveLoss,
    compute_detector_loss,
    count_trainable_parameters,
    summarize_tie_weights,
    unwrap_model,
)

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RINE + MPFT hybrid: RINE pooling + last-K-block fine-tuning.")
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--train-shards", type=int, nargs="*", default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="RINE/outputs/rine_mpft_hybrid")
    parser.add_argument("--checkpoint-prefix", type=str, default="rine_mpft_hybrid")
    parser.add_argument("--backbone-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--selected-layers", type=int, nargs="*", default=None)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--q-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)

    # Phased training
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Epochs before unfreezing the last K blocks.")
    parser.add_argument("--trainable-last-blocks", type=int, default=4,
                        help="Number of last encoder blocks to unfreeze in phase 2.")
    parser.add_argument("--head-lr", type=float, default=2e-4,
                        help="LR for RINE head/TIE/projections.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="LR for unfrozen backbone blocks in phase 2.")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Phase-1 LR (head-only). Falls back to head-lr if not set.")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--gpus", type=int, nargs="*", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fnr[idx] + fpr[idx]) / 2.0)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    metrics = {"auc": float("nan"), "ap": float("nan"), "acc": float("nan"), "eer": float("nan")}
    if len(y_true) == 0:
        return metrics
    pred = (y_score >= 0.5).astype(np.int64)
    metrics["acc"] = float(accuracy_score(y_true, pred))
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
        metrics["ap"] = float(average_precision_score(y_true, y_score))
        metrics["eer"] = float(compute_eer(y_true, y_score))
    return metrics


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def maybe_stratified_subsample(samples, max_samples, seed):
    sample_list = list(samples)
    if max_samples <= 0 or len(sample_list) <= max_samples:
        return sample_list
    labels = [int(label) for _, label in sample_list]
    can_stratify = len(set(labels)) >= 2 and max_samples >= len(set(labels))
    if can_stratify:
        kept, _ = train_test_split(sample_list, train_size=max_samples, random_state=seed, stratify=labels)
    else:
        kept, _ = train_test_split(sample_list, train_size=max_samples, random_state=seed, stratify=None)
    return list(kept)


def resolve_dataset_root(path_like):
    root = Path(path_like)
    if any(path.is_dir() for path in root.glob("shard_*")):
        return root
    nested = root / "dataset"
    if nested.is_dir() and any(path.is_dir() for path in nested.glob("shard_*")):
        return nested
    raise FileNotFoundError(f"Could not find shard_* under {root} or {nested}.")


def format_seconds(seconds):
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_dataloaders(args):
    data_root = resolve_dataset_root(args.data_root)
    eval_transform = build_rine_eval_transform(crop_size=args.crop_size)
    train_transform = build_rine_train_transform(crop_size=args.crop_size)

    train_samples = load_shard_samples(data_root=data_root, shard_indices=args.train_shards)
    train_samples = maybe_stratified_subsample(train_samples, args.max_train_samples, args.seed)
    train_dataset = PathLabelDataset(train_samples, transform=train_transform)

    val_images_dir = (data_root / "val" / "val_images").resolve()
    val_label_csv = (data_root / "val" / "val_labels.csv").resolve()
    val_hard_images_dir = (data_root / "val" / "val_images_hard").resolve()
    val_hard_label_csv = (data_root / "val" / "val_hard_labels.csv").resolve()

    val_samples = load_labeled_image_folder(val_images_dir, val_label_csv)
    val_hard_samples = load_labeled_image_folder(val_hard_images_dir, val_hard_label_csv)

    val_dataset = PathLabelDataset(val_samples, transform=eval_transform)
    val_hard_dataset = PathLabelDataset(val_hard_samples, transform=eval_transform)

    print(f"data_root={data_root.resolve()}")
    print(f"train_samples={len(train_dataset)} val={len(val_dataset)} val_hard={len(val_hard_dataset)}")

    pin_memory = torch.cuda.is_available()
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    val_hard_loader = DataLoader(val_hard_dataset, shuffle=False, **common)

    dataset_paths = {
        "data_root": str(data_root.resolve()),
        "val_images_dir": str(val_images_dir),
        "val_label_csv": str(val_label_csv),
        "val_hard_images_dir": str(val_hard_images_dir),
        "val_hard_label_csv": str(val_hard_label_csv),
    }
    return train_loader, val_loader, val_hard_loader, dataset_paths


def build_model(args):
    config = RINEConfig(
        backbone_name=args.backbone_name,
        selected_layers=args.selected_layers,
        proj_dim=args.proj_dim,
        q_layers=args.q_layers,
        dropout=args.dropout,
        supcon_temperature=args.supcon_temperature,
        supcon_weight=args.supcon_weight,
        freeze_backbone=True,
        trainable_last_blocks=0,
        local_files_only=(not args.online),
    )
    return RINECLIPDetector(config)


def build_optimizer_for_phase(raw_model, phase, args):
    if phase == 1:
        params = [p for p in raw_model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=args.head_lr, weight_decay=args.weight_decay)
    groups = raw_model.get_partial_ft_param_groups(
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
    )
    return optim.AdamW(groups, weight_decay=args.weight_decay)


def train_one_epoch(model, loader, optimizer, scaler, supcon, device, amp_enabled, supcon_weight):
    model.train()
    running = {"loss_total": 0.0, "loss_bce": 0.0, "loss_supcon": 0.0, "n": 0}
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            pixel_values, _unused, labels = batch
        else:
            pixel_values, labels = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            output = model(pixel_values)
            loss, stats = compute_detector_loss(
                output=output, labels=labels,
                supcon=supcon, supcon_weight=supcon_weight,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running["loss_total"] += float(loss.detach().cpu()) * bs
        running["loss_bce"] += stats["loss_bce"] * bs
        running["loss_supcon"] += stats["loss_supcon"] * bs
        running["n"] += bs
    n = max(running["n"], 1)
    return {k: v / n for k, v in running.items() if k != "n"}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(pixel_values)
        all_scores.append(output.probabilities.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    y_score = np.concatenate(all_scores) if all_scores else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    metrics = compute_metrics(y_true, y_score)

    raw = unwrap_model(model)
    if hasattr(raw, "tie") and hasattr(raw, "selected_layers"):
        tie_w = torch.softmax(raw.tie.importance.detach().cpu(), dim=0)
        metrics["top_tie_blocks"] = summarize_tie_weights(tie_w, raw.selected_layers, top_k=5)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA required. Use --allow-cpu to override.")
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_loader, val_loader, val_hard_loader, dataset_paths = make_dataloaders(args)
    model = build_model(args).to(device)
    raw_model = model  # keep reference before DP wrap

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed from: {args.resume_from}")

    supcon = SupervisedContrastiveLoss(temperature=args.supcon_temperature) if args.supcon_weight > 0 else None

    # Phase 1 optimizer (backbone frozen)
    optimizer = build_optimizer_for_phase(raw_model, phase=1, args=args)
    scaler = GradScaler(enabled=amp_enabled)

    if args.gpus and len(args.gpus) > 1 and device.type == "cuda":
        model = DetectorDataParallel(model, device_ids=args.gpus)
        print(f"Multi-GPU: {args.gpus}")
    elif args.gpus and len(args.gpus) == 1:
        torch.cuda.set_device(args.gpus[0])

    print(f"Phase 1 trainable params: {count_trainable_parameters(raw_model):,}")
    print(f"warmup_epochs={args.warmup_epochs} trainable_last_blocks={args.trainable_last_blocks}")
    print(f"head_lr={args.head_lr} backbone_lr={args.backbone_lr}")

    history = []
    best_score = -math.inf
    run_start = time.perf_counter()
    last_ckpt_name = f"{args.checkpoint_prefix}_last_model.pt" if args.checkpoint_prefix else "last_model.pt"
    best_ckpt_name = f"{args.checkpoint_prefix}_best_model.pt" if args.checkpoint_prefix else "best_model.pt"

    current_phase = 1

    for epoch in range(1, args.epochs + 1):
        # Phase transition
        if epoch == args.warmup_epochs + 1 and args.trainable_last_blocks > 0:
            raw_model.set_trainable_last_blocks(args.trainable_last_blocks)
            optimizer = build_optimizer_for_phase(raw_model, phase=2, args=args)
            scaler = GradScaler(enabled=amp_enabled)  # reset scaler on phase change
            current_phase = 2
            print(f"[phase-transition] epoch {epoch}: unfroze last {args.trainable_last_blocks} blocks. "
                  f"Trainable params: {count_trainable_parameters(raw_model):,}")

        t0 = time.perf_counter()
        train_stats = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scaler=scaler,
            supcon=supcon, device=device, amp_enabled=amp_enabled,
            supcon_weight=args.supcon_weight,
        )
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        val_stats = evaluate(model, val_loader, device)
        t_val = time.perf_counter() - t0

        t0 = time.perf_counter()
        val_hard_stats = evaluate(model, val_hard_loader, device)
        t_valh = time.perf_counter() - t0

        score = np.nan_to_num(val_stats.get("auc", 0)) + np.nan_to_num(val_stats.get("ap", 0))
        record = {
            "epoch": epoch, "phase": current_phase,
            "train": train_stats,
            "val_images": val_stats,
            "val_images_hard": val_hard_stats,
            "selection_score": float(score),
            "timing": {
                "train_seconds": t_train, "val_seconds": t_val, "val_hard_seconds": t_valh,
                "train_hms": format_seconds(t_train), "val_hms": format_seconds(t_val),
                "val_hard_hms": format_seconds(t_valh),
            },
        }
        history.append(record)

        print(
            f"[epoch {epoch:02d} | phase {current_phase}] "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"bce={train_stats['loss_bce']:.4f} supcon={train_stats['loss_supcon']:.4f} "
            f"| val_auc={val_stats['auc']:.4f} val_ap={val_stats['ap']:.4f} val_acc={val_stats['acc']:.4f} "
            f"| val_hard_auc={val_hard_stats['auc']:.4f} val_hard_ap={val_hard_stats['ap']:.4f} "
            f"val_hard_acc={val_hard_stats['acc']:.4f} "
            f"| time train={format_seconds(t_train)} val={format_seconds(t_val)} val_hard={format_seconds(t_valh)}"
        )
        if "top_tie_blocks" in val_stats:
            print(f"  top_tie(val)={val_stats['top_tie_blocks']}")

        state = unwrap_model(model).state_dict()
        torch.save({"model_state": state, "args": vars(args)}, output_dir / last_ckpt_name)
        if score > best_score:
            best_score = score
            torch.save(
                {"model_state": state, "args": vars(args), "best_record": record},
                output_dir / best_ckpt_name,
            )

        save_json(
            output_dir / "history.json",
            {
                "history": history,
                "best_score": best_score,
                "selection_metric": "val_auc_plus_ap",
                "dataset_paths": dataset_paths,
            },
        )

    total = time.perf_counter() - run_start
    save_json(output_dir / "run_args.json", vars(args))
    print(f"Total training time: {format_seconds(total)}")
    print(f"Artifacts: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
