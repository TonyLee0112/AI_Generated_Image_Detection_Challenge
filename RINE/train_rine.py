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
    build_baseline_clip_transform,
    build_rine_eval_transform,
    build_rine_train_transform,
    load_labeled_image_folder,
    load_shard_samples,
)
from src.rine_detector import (
    BaselineCLIPMLPDetector,
    RINECLIPDetector,
    RINEConfig,
    SupervisedContrastiveLoss,
    compute_detector_loss,
    count_trainable_parameters,
    summarize_tie_weights,
)

try:
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for validation metrics. Please install it with `pip install scikit-learn`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CLIP baseline or a RINE-style detector.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root containing shard_* and val/ subdirectories. (Default: ./dataset)",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        nargs="*",
        default=None,
        help="Shard ids to use for challenge format, e.g. --train-shards 0 1 2",
    )
    parser.add_argument(
        "--val-images-dir",
        type=str,
        default=None,
        help="Official validation image folder. Default: <data-root>/val/val_images",
    )
    parser.add_argument(
        "--val-label-csv",
        type=str,
        default=None,
        help="Official validation CSV. Default: <data-root>/val/val_labels.csv",
    )
    parser.add_argument(
        "--val-hard-images-dir",
        type=str,
        default=None,
        help="Official hard validation image folder. Default: <data-root>/val/val_images_hard",
    )
    parser.add_argument(
        "--val-hard-label-csv",
        type=str,
        default=None,
        help="Official hard validation CSV. Default: <data-root>/val/val_hard_labels.csv",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap for train shard samples. No train->val split is used. 0 means no cap.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Optional cap for official val_images samples. 0 means no cap.",
    )
    parser.add_argument(
        "--max-val-hard-samples",
        type=int,
        default=0,
        help="Optional cap for official val_images_hard samples. 0 means no cap.",
    )
    parser.add_argument("--output-dir", type=str, default="RINE/outputs/default_official_run")
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="",
        help="Optional prefix for checkpoint filenames, e.g. exp1 -> exp1_best_model.pt.",
    )
    parser.add_argument("--model", choices=["baseline", "rine"], default="rine")
    parser.add_argument("--backbone-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--selected-layers", type=int, nargs="*", default=None, help="1-based CLIP block indices.")
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--q-layers", type=int, default=2)
    parser.add_argument("--baseline-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", action="store_true", help="If set, override and train CLIP as well.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--persistent-workers",
        choices=["auto", "on", "off"],
        default="off",
        help="DataLoader persistent_workers policy when num_workers>0.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable. By default, this script requires GPU.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow online model lookup/download from Hugging Face. Default uses local cache only.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a .pt checkpoint (saved by this script) to load model weights from before training.",
    )
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


def maybe_stratified_subsample(
    samples: Iterable[tuple[Path, int]],
    max_samples: int,
    seed: int,
) -> list[tuple[Path, int]]:
    sample_list = list(samples)
    if max_samples <= 0 or len(sample_list) <= max_samples:
        return sample_list

    labels = [int(label) for _, label in sample_list]
    can_stratify = len(set(labels)) >= 2 and max_samples >= len(set(labels))
    if can_stratify:
        kept, _ = train_test_split(
            sample_list,
            train_size=max_samples,
            random_state=seed,
            stratify=labels,
        )
    else:
        kept, _ = train_test_split(
            sample_list,
            train_size=max_samples,
            random_state=seed,
            stratify=None,
        )
    return list(kept)


def checkpoint_filename(prefix: str, stem: str) -> str:
    cleaned = prefix.strip()
    if cleaned:
        return f"{cleaned}_{stem}.pt"
    return f"{stem}.pt"


def resolve_persistent_workers(num_workers: int, mode: str) -> bool:
    if num_workers <= 0:
        return False
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto
    return True


def resolve_dataset_root(path_like: str | Path) -> Path:
    root = Path(path_like)
    if any(path.is_dir() for path in root.glob("shard_*")):
        return root

    nested = root / "dataset"
    if nested.is_dir() and any(path.is_dir() for path in nested.glob("shard_*")):
        return nested

    raise FileNotFoundError(
        f"Could not find shard_* directories under {root} or {nested}. "
        "Set --data-root to your dataset root."
    )


def format_seconds(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, str]]:
    data_root = resolve_dataset_root(args.data_root)

    if args.model == "rine":
        train_transform = build_rine_train_transform(crop_size=args.crop_size)
        eval_transform = build_rine_eval_transform(crop_size=args.crop_size)
    else:
        train_transform = build_baseline_clip_transform(image_size=args.crop_size)
        eval_transform = build_baseline_clip_transform(image_size=args.crop_size)

    train_samples = load_shard_samples(data_root=data_root, shard_indices=args.train_shards)
    train_samples = maybe_stratified_subsample(
        samples=train_samples,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    train_dataset = PathLabelDataset(train_samples, transform=train_transform)

    official_val_images_dir = (data_root / "val" / "val_images").resolve()
    official_val_label_csv = (data_root / "val" / "val_labels.csv").resolve()
    official_val_hard_images_dir = (data_root / "val" / "val_images_hard").resolve()
    official_val_hard_label_csv = (data_root / "val" / "val_hard_labels.csv").resolve()

    # Hard-enforce official validation sets. Train-internal or arbitrary val paths are disallowed.
    if args.val_images_dir and Path(args.val_images_dir).resolve() != official_val_images_dir:
        raise ValueError(
            "Validation is fixed to official data only. "
            f"Expected val_images_dir={official_val_images_dir}, got {Path(args.val_images_dir).resolve()}."
        )
    if args.val_label_csv and Path(args.val_label_csv).resolve() != official_val_label_csv:
        raise ValueError(
            "Validation is fixed to official data only. "
            f"Expected val_label_csv={official_val_label_csv}, got {Path(args.val_label_csv).resolve()}."
        )
    if args.val_hard_images_dir and Path(args.val_hard_images_dir).resolve() != official_val_hard_images_dir:
        raise ValueError(
            "Validation is fixed to official data only. "
            f"Expected val_hard_images_dir={official_val_hard_images_dir}, "
            f"got {Path(args.val_hard_images_dir).resolve()}."
        )
    if args.val_hard_label_csv and Path(args.val_hard_label_csv).resolve() != official_val_hard_label_csv:
        raise ValueError(
            "Validation is fixed to official data only. "
            f"Expected val_hard_label_csv={official_val_hard_label_csv}, "
            f"got {Path(args.val_hard_label_csv).resolve()}."
        )

    val_images_dir = official_val_images_dir
    val_label_csv = official_val_label_csv
    val_hard_images_dir = official_val_hard_images_dir
    val_hard_label_csv = official_val_hard_label_csv

    for required_path in (val_images_dir, val_label_csv, val_hard_images_dir, val_hard_label_csv):
        if not Path(required_path).exists():
            raise FileNotFoundError(
                "Official validation paths are required but missing: "
                f"{required_path}. Expected under {data_root.resolve()}\\val."
            )

    val_samples = load_labeled_image_folder(image_dir=val_images_dir, labels_csv=val_label_csv)
    val_hard_samples = load_labeled_image_folder(image_dir=val_hard_images_dir, labels_csv=val_hard_label_csv)
    val_samples = maybe_stratified_subsample(
        samples=val_samples,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
    )
    val_hard_samples = maybe_stratified_subsample(
        samples=val_hard_samples,
        max_samples=args.max_val_hard_samples,
        seed=args.seed + 2,
    )
    val_dataset = PathLabelDataset(val_samples, transform=eval_transform)
    val_hard_dataset = PathLabelDataset(val_hard_samples, transform=eval_transform)

    shard_text = "all" if not args.train_shards else ",".join(str(value) for value in args.train_shards)
    print("Dataset mode: shardcsv + official validation")
    print(f"  data_root={data_root.resolve()}")
    print(f"  train_shards={shard_text} train_samples={len(train_dataset)}")
    print(
        "  val_images="
        f"{val_images_dir.resolve()} labels={val_label_csv.resolve()} samples={len(val_dataset)}"
    )
    print(
        "  val_images_hard="
        f"{val_hard_images_dir.resolve()} labels={val_hard_label_csv.resolve()} samples={len(val_hard_dataset)}"
    )
    print("  note=train/val split from train shards is disabled; only official validation sets are used.")

    persistent_workers = resolve_persistent_workers(args.num_workers, args.persistent_workers)
    pin_memory = torch.cuda.is_available()

    if os.name == "nt" and args.num_workers > 0:
        print(
            "[windows-loader-note] num_workers>0 can be unstable on some Windows setups. "
            "If you see worker/spawn errors, retry with --num-workers 0 and increase --batch-size."
        )
    print(
        "DataLoader config: "
        f"num_workers={args.num_workers} "
        f"persistent_workers={persistent_workers} "
        f"pin_memory={pin_memory}"
    )

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if args.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    val_hard_loader = DataLoader(
        val_hard_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    dataset_paths = {
        "data_root": str(data_root.resolve()),
        "val_images_dir": str(val_images_dir.resolve()),
        "val_label_csv": str(val_label_csv.resolve()),
        "val_hard_images_dir": str(val_hard_images_dir.resolve()),
        "val_hard_label_csv": str(val_hard_label_csv.resolve()),
    }
    return train_loader, val_loader, val_hard_loader, dataset_paths


def build_model(args: argparse.Namespace):
    freeze_backbone = False if args.unfreeze_backbone else args.freeze_backbone
    if args.model == "rine":
        config = RINEConfig(
            backbone_name=args.backbone_name,
            selected_layers=args.selected_layers,
            proj_dim=args.proj_dim,
            q_layers=args.q_layers,
            dropout=args.dropout,
            supcon_temperature=args.supcon_temperature,
            supcon_weight=args.supcon_weight,
            freeze_backbone=freeze_backbone,
            local_files_only=(not args.online),
        )
        model = RINECLIPDetector(config)
    else:
        model = BaselineCLIPMLPDetector(
            backbone_name=args.backbone_name,
            hidden_dim=args.baseline_hidden_dim,
            dropout=min(args.dropout, 0.5),
            freeze_backbone=freeze_backbone,
            local_files_only=(not args.online),
        )
    return model


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    supcon,
    device: torch.device,
    amp_enabled: bool,
    supcon_weight: float,
) -> Dict[str, float]:
    model.train()
    running = {"loss_total": 0.0, "loss_bce": 0.0, "loss_supcon": 0.0, "n": 0}
    gpu_checked = False

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda" and not gpu_checked:
            model_device = next(model.parameters()).device
            if (
                model_device.type != "cuda"
                or pixel_values.device.type != "cuda"
                or labels.device.type != "cuda"
            ):
                raise RuntimeError(
                    "GPU enforcement failed: model/input/labels are not all on CUDA. "
                    f"model={model_device}, input={pixel_values.device}, labels={labels.device}"
                )
            allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            print(
                "[gpu-check] "
                f"model={model_device} input={pixel_values.device} labels={labels.device} "
                f"allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB"
            )
            gpu_checked = True

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            output = model(pixel_values)
            loss, loss_stats = compute_detector_loss(
                output=output,
                labels=labels,
                supcon=supcon,
                supcon_weight=supcon_weight,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running["loss_total"] += loss_stats["loss_total"] * batch_size
        running["loss_bce"] += loss_stats["loss_bce"] * batch_size
        running["loss_supcon"] += loss_stats["loss_supcon"] * batch_size
        running["n"] += batch_size

    denom = max(running["n"], 1)
    return {
        "loss_total": running["loss_total"] / denom,
        "loss_bce": running["loss_bce"] / denom,
        "loss_supcon": running["loss_supcon"] / denom,
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    all_scores = []
    all_labels = []

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(pixel_values)
        scores = output.probabilities.detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.detach().cpu().numpy())

    y_score = np.concatenate(all_scores, axis=0) if all_scores else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)
    metrics = compute_metrics(y_true=y_true, y_score=y_score)

    if hasattr(model, "tie") and hasattr(model, "selected_layers"):
        tie_weights = torch.softmax(model.tie.importance.detach().cpu(), dim=0)
        metrics["top_tie_blocks"] = summarize_tie_weights(tie_weights, model.selected_layers, top_k=5)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("Model loading mode: offline/local-cache (use --online to allow downloads).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "CUDA is not available. This script is configured to require GPU. "
            "If you explicitly want CPU fallback, pass --allow-cpu."
        )
    amp_enabled = bool(args.amp and device.type == "cuda")

    train_loader, val_loader, val_hard_loader, dataset_paths = make_dataloaders(args)
    model = build_model(args).to(device)
    if device.type == "cuda":
        current_device = torch.cuda.current_device()
        print(f"CUDA device: {torch.cuda.get_device_name(current_device)}")
        print(f"Torch CUDA build: {torch.version.cuda}")
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Model parameters are not on CUDA after model.to(device).")

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed model weights from: {args.resume_from}")

    supcon = None
    if args.model == "rine" and args.supcon_weight > 0:
        supcon = SupervisedContrastiveLoss(temperature=args.supcon_temperature)

    trainable_params = count_trainable_parameters(model)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=amp_enabled)

    history = []
    best_score = -math.inf
    best_payload = None
    last_ckpt_name = checkpoint_filename(args.checkpoint_prefix, "last_model")
    best_ckpt_name = checkpoint_filename(args.checkpoint_prefix, "best_model")
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            supcon=supcon,
            device=device,
            amp_enabled=amp_enabled,
            supcon_weight=(args.supcon_weight if args.model == "rine" else 0.0),
        )
        train_elapsed = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_stats = evaluate(model=model, loader=val_loader, device=device)
        val_elapsed = time.perf_counter() - val_start

        val_hard_start = time.perf_counter()
        val_hard_stats = evaluate(model=model, loader=val_hard_loader, device=device)
        val_hard_elapsed = time.perf_counter() - val_hard_start
        epoch_elapsed = time.perf_counter() - epoch_start

        # Select best checkpoint by official val_images metrics.
        score = np.nan_to_num(val_stats.get("auc", np.nan)) + np.nan_to_num(val_stats.get("ap", np.nan))
        record = {
            "epoch": epoch,
            "train": train_stats,
            "val_images": val_stats,
            "val_images_hard": val_hard_stats,
            "selection_score": float(score),
            "timing": {
                "train_seconds": float(train_elapsed),
                "val_seconds": float(val_elapsed),
                "val_hard_seconds": float(val_hard_elapsed),
                "epoch_seconds": float(epoch_elapsed),
                "train_hms": format_seconds(train_elapsed),
                "val_hms": format_seconds(val_elapsed),
                "val_hard_hms": format_seconds(val_hard_elapsed),
                "epoch_hms": format_seconds(epoch_elapsed),
            },
        }
        history.append(record)

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"val_auc={val_stats['auc']:.4f} "
            f"val_ap={val_stats['ap']:.4f} "
            f"val_acc={val_stats['acc']:.4f} "
            f"val_eer={val_stats['eer']:.4f} "
            f"| val_hard_auc={val_hard_stats['auc']:.4f} "
            f"val_hard_ap={val_hard_stats['ap']:.4f} "
            f"val_hard_acc={val_hard_stats['acc']:.4f} "
            f"val_hard_eer={val_hard_stats['eer']:.4f} "
            f"| time train={format_seconds(train_elapsed)} "
            f"val={format_seconds(val_elapsed)} "
            f"val_hard={format_seconds(val_hard_elapsed)} "
            f"epoch={format_seconds(epoch_elapsed)}"
        )
        if "top_tie_blocks" in val_stats:
            print(f"  top_tie_blocks(val)={val_stats['top_tie_blocks']}")
        if "top_tie_blocks" in val_hard_stats:
            print(f"  top_tie_blocks(val_hard)={val_hard_stats['top_tie_blocks']}")

        torch.save({"model_state": model.state_dict(), "args": vars(args)}, output_dir / last_ckpt_name)
        if score > best_score:
            best_score = score
            best_payload = {"model_state": model.state_dict(), "args": vars(args), "best_record": record}
            torch.save(best_payload, output_dir / best_ckpt_name)

    total_elapsed = time.perf_counter() - run_start
    save_json(
        output_dir / "history.json",
        {
            "history": history,
            "best_score": best_score,
            "selection_metric": "val_images_auc_plus_ap",
            "dataset_paths": dataset_paths,
            "timing": {
                "total_training_seconds": float(total_elapsed),
                "total_training_hms": format_seconds(total_elapsed),
            },
        },
    )
    save_json(output_dir / "run_args.json", vars(args))
    print(f"Total training time: {format_seconds(total_elapsed)} ({total_elapsed:.1f}s)")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
