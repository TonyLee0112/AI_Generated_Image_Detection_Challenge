from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Standard CLIP normalization statistics.
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BinaryImageFolder(Dataset):
    """
    Thin wrapper around torchvision.datasets.ImageFolder that forces:
        real -> 0
        fake -> 1

    Expected structure:
        root/
          real/
            *.png|jpg|jpeg|webp
          fake/
            *.png|jpg|jpeg|webp

    If your challenge dataset is organized differently, adapt this class and keep
    the rest of the training code unchanged.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        real_name: str = "real",
        fake_name: str = "fake",
    ) -> None:
        self.root = Path(root)
        self.dataset = datasets.ImageFolder(str(self.root), transform=transform)
        self.real_name = real_name
        self.fake_name = fake_name

        classes = set(self.dataset.class_to_idx.keys())
        expected = {real_name, fake_name}
        if not expected.issubset(classes):
            raise ValueError(
                f"Expected ImageFolder classes {sorted(expected)} under {self.root}, "
                f"but found {sorted(classes)}"
            )

        self.real_idx = self.dataset.class_to_idx[real_name]
        self.fake_idx = self.dataset.class_to_idx[fake_name]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, raw_target = self.dataset[index]
        if raw_target == self.real_idx:
            target = 0
        elif raw_target == self.fake_idx:
            target = 1
        else:
            raise RuntimeError(f"Unexpected class index: {raw_target}")
        return image, torch.tensor(target, dtype=torch.long)


def _coerce_binary_label(raw_value: str, csv_path: Path, row_number: int) -> int:
    try:
        label = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid label '{raw_value}' in {csv_path} at row {row_number}") from exc
    if label not in (0, 1):
        raise ValueError(f"Expected binary label 0/1 in {csv_path} at row {row_number}, got: {label}")
    return label


def _read_labels_csv(csv_path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []

    # Primary path: robust to an extra unnamed index column (common in challenge shards).
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        dict_reader = csv.DictReader(fp)
        fieldnames = [name.strip() for name in (dict_reader.fieldnames or [])]
        if "image_name" in fieldnames and "label" in fieldnames:
            for row_number, row in enumerate(dict_reader, start=2):
                image_name = (row.get("image_name") or "").strip()
                label_text = (row.get("label") or "").strip()
                if not image_name:
                    continue
                label = _coerce_binary_label(label_text, csv_path=csv_path, row_number=row_number)
                rows.append((image_name, label))
            if rows:
                return rows

    # Fallback path for unusual CSV header formatting.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty labels.csv: {csv_path}")

        normalized = [column.strip().lower() for column in header]
        if "image_name" in normalized and "label" in normalized:
            image_idx = normalized.index("image_name")
            label_idx = normalized.index("label")
        elif len(header) >= 2:
            image_idx = len(header) - 2
            label_idx = len(header) - 1
        else:
            raise ValueError(f"Could not parse labels.csv header in {csv_path}: {header}")

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            max_idx = max(image_idx, label_idx)
            if len(row) <= max_idx:
                raise ValueError(f"Malformed row in {csv_path} at row {row_number}: {row}")

            image_name = row[image_idx].strip()
            label_text = row[label_idx].strip()
            if not image_name:
                continue

            label = _coerce_binary_label(label_text, csv_path=csv_path, row_number=row_number)
            rows.append((image_name, label))

    if not rows:
        raise ValueError(f"No valid samples parsed from {csv_path}")
    return rows


def _shard_sort_key(path: Path) -> Tuple[int, str]:
    suffix = path.name.split("_", maxsplit=1)[-1]
    if suffix.isdigit():
        return (0, f"{int(suffix):08d}")
    return (1, path.name)


def load_shard_samples(data_root: str | Path, shard_indices: Optional[Sequence[int]] = None) -> list[tuple[Path, int]]:
    """
    Load sample list from challenge-style shard folders:
        shard_i/
          images/
          labels.csv
    """
    root = Path(data_root)
    if shard_indices:
        shard_dirs = [root / f"shard_{idx}" for idx in shard_indices]
    else:
        shard_dirs = sorted([path for path in root.glob("shard_*") if path.is_dir()], key=_shard_sort_key)

    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard directories found under {root}. Expected folders like shard_0, shard_1, ..."
        )

    missing_shards = [path for path in shard_dirs if not path.is_dir()]
    if missing_shards:
        raise FileNotFoundError(f"Requested shards not found: {missing_shards}")

    samples: list[tuple[Path, int]] = []
    missing_images: list[Path] = []

    for shard_dir in shard_dirs:
        csv_path = shard_dir / "labels.csv"
        image_dir = shard_dir / "images"
        if not csv_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(
                f"Shard format mismatch in {shard_dir}. Expected files: {csv_path} and folder: {image_dir}"
            )

        for image_name, label in _read_labels_csv(csv_path):
            image_path = image_dir / image_name
            if image_path.is_file():
                samples.append((image_path, label))
            else:
                missing_images.append(image_path)

    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:5])
        suffix = "" if len(missing_images) <= 5 else f" (+{len(missing_images) - 5} more)"
        raise FileNotFoundError(f"Missing images referenced by labels.csv: {preview}{suffix}")

    if not samples:
        raise ValueError(f"No samples found under shards in {root}")
    return samples


def load_labeled_image_folder(image_dir: str | Path, labels_csv: str | Path) -> list[tuple[Path, int]]:
    """
    Load labeled samples from an image folder + CSV pair.

    Expected:
      image_dir/
        *.jpg|png|...
      labels_csv:
        image_name,label,(optional extra columns...)
    """
    image_dir_path = Path(image_dir)
    labels_csv_path = Path(labels_csv)
    if not image_dir_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir_path}")
    if not labels_csv_path.is_file():
        raise FileNotFoundError(f"Label CSV not found: {labels_csv_path}")

    parsed = _read_labels_csv(labels_csv_path)
    samples: list[tuple[Path, int]] = []
    missing_images: list[Path] = []
    for image_name, label in parsed:
        image_path = image_dir_path / image_name
        if image_path.is_file():
            samples.append((image_path, label))
        else:
            missing_images.append(image_path)

    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:5])
        suffix = "" if len(missing_images) <= 5 else f" (+{len(missing_images) - 5} more)"
        raise FileNotFoundError(f"Missing images referenced by CSV {labels_csv_path}: {preview}{suffix}")
    if not samples:
        raise ValueError(f"No samples found for CSV {labels_csv_path}")
    return samples


class PathLabelDataset(Dataset):
    """Generic dataset from explicit (image_path, label) samples."""

    def __init__(self, samples: Sequence[tuple[str | Path, int]], transform: Optional[Callable] = None) -> None:
        self.samples = [(Path(image_path), int(label)) for image_path, label in samples]
        if not self.samples:
            raise ValueError("PathLabelDataset received no samples.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class PadToMinSize:
    """
    Symmetric padding so crop-based pipelines do not crash on small images.
    This does not resize the content.
    """

    def __init__(self, min_size: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> None:
        self.min_size = min_size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        pad_w = max(0, self.min_size - width)
        pad_h = max(0, self.min_size - height)
        if pad_w == 0 and pad_h == 0:
            return image

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)


class RandomJPEGCompression:
    """
    JPEG re-encoding in PIL space.

    Useful when robustness to common distribution shifts matters and when one wants
    to follow the blur+JPEG style of augmentation often used in generalized SID.
    """

    def __init__(self, p: float = 0.5, quality_range: Tuple[int, int] = (35, 95)) -> None:
        self.p = p
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image

        image = image.convert("RGB")
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")
        compressed.load()  # decouple from buffer
        buffer.close()
        return compressed


class ToRGB:
    """Pickle-safe RGB converter for Windows DataLoader workers."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def build_baseline_clip_transform(image_size: int = 224) -> transforms.Compose:
    """
    Generic baseline transform for a CLIP+MLP setup.

    This intentionally includes resizing, because many simple CLIP baselines do.
    The RINE path below keeps the content crop-based instead of resize-based.
    """
    return transforms.Compose([
        ToRGB(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_rine_train_transform(
    crop_size: int = 224,
    blur_probability: float = 0.5,
    jpeg_probability: float = 0.5,
) -> transforms.Compose:
    """
    RINE-inspired training transform:
        - no resize
        - Gaussian blur with p=0.5
        - JPEG re-encode with p=0.5
        - random crop to 224
        - horizontal flip with p=0.5
    """
    blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=blur_probability)
    jpeg = RandomJPEGCompression(p=jpeg_probability)

    return transforms.Compose([
        ToRGB(),
        PadToMinSize(crop_size),
        blur,
        jpeg,
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_rine_eval_transform(crop_size: int = 224) -> transforms.Compose:
    """
    RINE-inspired evaluation transform:
        - no resize
        - center crop to 224
    """
    return transforms.Compose([
        ToRGB(),
        PadToMinSize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
