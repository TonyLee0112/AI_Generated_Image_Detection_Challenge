"""RINE-style drop-in modules for CLIP-based AI-generated image detection."""
from .rine_detector import (
    RINEConfig,
    DetectorOutput,
    BaselineCLIPMLPDetector,
    RINECLIPDetector,
    SupervisedContrastiveLoss,
    compute_detector_loss,
    count_trainable_parameters,
    summarize_tie_weights,
)
from .data import (
    BinaryImageFolder,
    PathLabelDataset,
    build_baseline_clip_transform,
    build_rine_train_transform,
    build_rine_eval_transform,
    load_labeled_image_folder,
    load_shard_samples,
)
