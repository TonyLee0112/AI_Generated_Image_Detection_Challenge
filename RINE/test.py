from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from train_rine import checkpoint_filename, compute_metrics, make_dataloaders, parse_args, save_json, set_seed
from src.rine_detector import SupervisedContrastiveLoss, compute_detector_loss, count_trainable_parameters, summarize_tie_weights


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def train_one_epoch_verbose(
    model,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    supcon,
    device: torch.device,
    amp_enabled: bool,
    supcon_weight: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running = {"loss_total": 0.0, "loss_bce": 0.0, "loss_supcon": 0.0, "n": 0}
    total_batches = len(loader)

    log(f"[train][epoch {epoch:02d}] start | batches={total_batches}")
    for step, (pixel_values, labels) in enumerate(loader, start=1):
        step_start = time.perf_counter()
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda" and step == 1:
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

        elapsed = time.perf_counter() - step_start
        if device.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            mem_text = f"cuda_alloc={allocated_mb:.1f}MB cuda_reserved={reserved_mb:.1f}MB"
        else:
            mem_text = "cpu_mode"

        log(
            f"[train][epoch {epoch:02d}] batch {step:05d}/{total_batches:05d} "
            f"loss_total={loss_stats['loss_total']:.4f} "
            f"loss_bce={loss_stats['loss_bce']:.4f} "
            f"loss_supcon={loss_stats['loss_supcon']:.4f} "
            f"time={elapsed:.3f}s "
            f"{mem_text}"
        )

    denom = max(running["n"], 1)
    summary = {
        "loss_total": running["loss_total"] / denom,
        "loss_bce": running["loss_bce"] / denom,
        "loss_supcon": running["loss_supcon"] / denom,
    }
    log(
        f"[train][epoch {epoch:02d}] done | "
        f"loss_total={summary['loss_total']:.4f} "
        f"loss_bce={summary['loss_bce']:.4f} "
        f"loss_supcon={summary['loss_supcon']:.4f}"
    )
    return summary


@torch.no_grad()
def evaluate_verbose(model, loader: DataLoader, device: torch.device, epoch: int) -> Dict[str, float]:
    model.eval()
    total_batches = len(loader)
    all_scores = []
    all_labels = []

    log(f"[eval][epoch {epoch:02d}] start | batches={total_batches}")
    for step, (pixel_values, labels) in enumerate(loader, start=1):
        step_start = time.perf_counter()
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(pixel_values)
        scores = output.probabilities.detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.detach().cpu().numpy())

        elapsed = time.perf_counter() - step_start
        if device.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
            mem_text = f"cuda_alloc={allocated_mb:.1f}MB cuda_reserved={reserved_mb:.1f}MB"
        else:
            mem_text = "cpu_mode"

        log(
            f"[eval][epoch {epoch:02d}] batch {step:05d}/{total_batches:05d} "
            f"score_mean={float(np.mean(scores)):.4f} "
            f"time={elapsed:.3f}s "
            f"{mem_text}"
        )

    y_score = np.concatenate(all_scores, axis=0) if all_scores else np.array([], dtype=np.float32)
    y_true = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)
    metrics = compute_metrics(y_true=y_true, y_score=y_score)

    if hasattr(model, "tie") and hasattr(model, "selected_layers"):
        tie_weights = torch.softmax(model.tie.importance.detach().cpu(), dim=0)
        metrics["top_tie_blocks"] = summarize_tie_weights(tie_weights, model.selected_layers, top_k=5)

    log(
        f"[eval][epoch {epoch:02d}] done | "
        f"auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
        f"acc={metrics['acc']:.4f} eer={metrics['eer']:.4f}"
    )
    if "top_tie_blocks" in metrics:
        log(f"[eval][epoch {epoch:02d}] top_tie_blocks={metrics['top_tie_blocks']}")
    return metrics


def main() -> None:
    args = parse_args()
    log("test.py started (verbose mode)")
    log("args=" + json.dumps(vars(args), ensure_ascii=False))
    set_seed(args.seed)
    log(f"seed set to {args.seed}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"output_dir={output_dir.resolve()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "CUDA is not available. test.py requires GPU by default. "
            "If CPU fallback is explicitly needed, pass --allow-cpu."
        )
    amp_enabled = bool(args.amp and device.type == "cuda")

    log("building dataloaders...")
    train_loader, val_loader, val_hard_loader, dataset_paths = make_dataloaders(args)
    log(
        "dataloaders ready | "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} val_hard_batches={len(val_hard_loader)} "
        f"batch_size={args.batch_size}"
    )

    log("building model...")
    from train_rine import build_model  # local import to keep startup logs clear

    model = build_model(args).to(device)
    if device.type == "cuda":
        current_device = torch.cuda.current_device()
        log(f"CUDA device={torch.cuda.get_device_name(current_device)}")
        log(f"Torch CUDA build={torch.version.cuda}")
        param_device = next(model.parameters()).device
        if param_device.type != "cuda":
            raise RuntimeError(f"Model parameters are not on CUDA: {param_device}")
        log(f"model_device={param_device}")

    supcon = None
    if args.model == "rine" and args.supcon_weight > 0:
        supcon = SupervisedContrastiveLoss(temperature=args.supcon_temperature)
        log(
            f"supcon enabled | temperature={args.supcon_temperature} "
            f"weight={args.supcon_weight}"
        )
    else:
        log("supcon disabled")

    trainable_params = count_trainable_parameters(model)
    log(f"model={args.model} device={device} trainable_params={trainable_params:,}")

    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=amp_enabled)
    log(f"optimizer=AdamW lr={args.lr} weight_decay={args.weight_decay} amp={amp_enabled}")

    history = []
    best_score = -math.inf
    last_ckpt_name = checkpoint_filename(args.checkpoint_prefix, "last_model")
    best_ckpt_name = checkpoint_filename(args.checkpoint_prefix, "best_model")

    for epoch in range(1, args.epochs + 1):
        log(f"[epoch {epoch:02d}] ===== start =====")
        train_stats = train_one_epoch_verbose(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            supcon=supcon,
            device=device,
            amp_enabled=amp_enabled,
            supcon_weight=(args.supcon_weight if args.model == "rine" else 0.0),
            epoch=epoch,
        )
        val_stats = evaluate_verbose(model=model, loader=val_loader, device=device, epoch=epoch)
        val_hard_stats = evaluate_verbose(model=model, loader=val_hard_loader, device=device, epoch=epoch)

        score = np.nan_to_num(val_stats.get("auc", np.nan)) + np.nan_to_num(val_stats.get("ap", np.nan))
        record = {
            "epoch": epoch,
            "train": train_stats,
            "val_images": val_stats,
            "val_images_hard": val_hard_stats,
            "selection_score": float(score),
        }
        history.append(record)

        log(
            f"[epoch {epoch:02d}] summary | "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"val_auc={val_stats['auc']:.4f} "
            f"val_ap={val_stats['ap']:.4f} "
            f"val_acc={val_stats['acc']:.4f} "
            f"val_eer={val_stats['eer']:.4f} "
            f"| val_hard_auc={val_hard_stats['auc']:.4f} "
            f"val_hard_ap={val_hard_stats['ap']:.4f} "
            f"val_hard_acc={val_hard_stats['acc']:.4f} "
            f"val_hard_eer={val_hard_stats['eer']:.4f}"
        )

        torch.save({"model_state": model.state_dict(), "args": vars(args)}, output_dir / last_ckpt_name)
        log(f"[epoch {epoch:02d}] saved {last_ckpt_name}")
        if score > best_score:
            best_score = score
            best_payload = {"model_state": model.state_dict(), "args": vars(args), "best_record": record}
            torch.save(best_payload, output_dir / best_ckpt_name)
            log(f"[epoch {epoch:02d}] new best saved {best_ckpt_name} score={float(score):.6f}")

        log(f"[epoch {epoch:02d}] ===== done =====")

    save_json(
        output_dir / "history.json",
        {
            "history": history,
            "best_score": best_score,
            "selection_metric": "val_images_auc_plus_ap",
            "dataset_paths": dataset_paths,
        },
    )
    save_json(output_dir / "run_args.json", vars(args))
    log(f"artifacts saved to {output_dir.resolve()}")
    log("test.py finished")


if __name__ == "__main__":
    main()
