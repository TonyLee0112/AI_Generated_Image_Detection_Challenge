from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionModel


@dataclass
class RINEConfig:
    """
    Configuration for a RINE-style detector.

    Notes:
        - `selected_layers` is 1-based and refers to Transformer block indices.
          Example: [1, 3, 6, 9, 12] uses the outputs of blocks 1/3/6/9/12.
        - If `selected_layers` is None, all encoder blocks are used.
    """
    backbone_name: str = "openai/clip-vit-large-patch14"
    selected_layers: Optional[List[int]] = None
    proj_dim: int = 256
    q_layers: int = 2
    dropout: float = 0.5
    supcon_temperature: float = 0.07
    supcon_weight: float = 0.05
    freeze_backbone: bool = True
    local_files_only: bool = True


@dataclass
class DetectorOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    features: torch.Tensor
    tie_weights: Optional[torch.Tensor] = None
    cls_stack: Optional[torch.Tensor] = None


class LayerwiseProjection(nn.Module):
    """
    Shared MLP applied on the last dimension.

    Input can be [B, D] or [B, L, D]. nn.Linear naturally applies to the
    trailing dimension, so the same block works for both Q1 and Q2.
    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, output_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = output_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """
    Paper-inspired classification head:
        d' -> d' -> d' -> 1
    with ReLU between dense layers.
    """

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TrainableImportanceEstimator(nn.Module):
    """
    TIE module from the RINE design.

    A learnable matrix A in R^{L x d'} is turned into per-feature softmax weights
    over transformer blocks. Each feature dimension chooses how strongly to rely
    on each block.
    """

    def __init__(self, num_blocks: int, feature_dim: int) -> None:
        super().__init__()
        self.importance = nn.Parameter(torch.zeros(num_blocks, feature_dim))
        nn.init.normal_(self.importance, mean=0.0, std=0.02)

    def forward(self, q1_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q1_features: [B, L, d']

        Returns:
            aggregated: [B, d']
            tie_weights: [L, d']
        """
        if q1_features.ndim != 3:
            raise ValueError(f"Expected q1_features to have shape [B, L, d'], got {tuple(q1_features.shape)}")

        tie_weights = torch.softmax(self.importance, dim=0)  # softmax across blocks
        aggregated = (q1_features * tie_weights.unsqueeze(0)).sum(dim=1)
        return aggregated, tie_weights


class BaselineCLIPMLPDetector(nn.Module):
    """
    Assumed existing baseline:
        image -> CLIP vision encoder -> pooled CLS -> MLP -> logit

    This is included so ablations can compare a conventional CLIP+MLP head against
    the RINE-style multi-block aggregation head with the same training script.
    """

    def __init__(
        self,
        backbone_name: str = "openai/clip-vit-large-patch14",
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.backbone = CLIPVisionModel.from_pretrained(backbone_name, local_files_only=local_files_only)
        self.hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def train(self, mode: bool = True) -> "BaselineCLIPMLPDetector":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> DetectorOutput:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=False)
        else:
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=False)

        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        features = self.head(pooled)
        logits = self.classifier(features).squeeze(-1)
        return DetectorOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            features=F.normalize(features, dim=-1),
            tie_weights=None,
            cls_stack=pooled.unsqueeze(1),
        )


class RINECLIPDetector(nn.Module):
    """
    RINE-style detector:

        image
          -> frozen CLIP vision encoder
          -> collect CLS from selected transformer blocks
          -> Q1 projection on each block feature
          -> TIE weighted aggregation across blocks
          -> Q2 projection
          -> classification head

    This is intentionally written as a drop-in module for CLIP-based detection
    codebases. It uses Hugging Face CLIPVisionModel because it exposes hidden
    states cleanly via `output_hidden_states=True`.
    """

    def __init__(self, config: Optional[RINEConfig] = None) -> None:
        super().__init__()
        self.config = config or RINEConfig()
        self.backbone = CLIPVisionModel.from_pretrained(
            self.config.backbone_name,
            local_files_only=self.config.local_files_only,
        )
        self.hidden_size = self.backbone.config.hidden_size
        self.num_hidden_layers = self.backbone.config.num_hidden_layers

        if self.config.selected_layers is None:
            self.selected_layers = list(range(1, self.num_hidden_layers + 1))
        else:
            self.selected_layers = list(self.config.selected_layers)

        invalid_layers = [idx for idx in self.selected_layers if idx < 1 or idx > self.num_hidden_layers]
        if invalid_layers:
            raise ValueError(
                f"selected_layers must be in [1, {self.num_hidden_layers}], got invalid entries: {invalid_layers}"
            )

        if self.config.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.q1 = LayerwiseProjection(
            input_dim=self.hidden_size,
            output_dim=self.config.proj_dim,
            num_layers=self.config.q_layers,
            dropout=self.config.dropout,
        )
        self.tie = TrainableImportanceEstimator(
            num_blocks=len(self.selected_layers),
            feature_dim=self.config.proj_dim,
        )
        self.q2 = LayerwiseProjection(
            input_dim=self.config.proj_dim,
            output_dim=self.config.proj_dim,
            num_layers=self.config.q_layers,
            dropout=self.config.dropout,
        )
        self.classifier = ClassificationHead(
            feature_dim=self.config.proj_dim,
            dropout=self.config.dropout,
        )

    def train(self, mode: bool = True) -> "RINECLIPDetector":
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def _extract_cls_stack(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        else:
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden states. Make sure output_hidden_states=True.")

        # hidden_states[0] is patch+positional embedding output; blocks start from index 1
        cls_tokens = [hidden_states[layer_idx][:, 0, :] for layer_idx in self.selected_layers]
        cls_stack = torch.stack(cls_tokens, dim=1)  # [B, L, D]
        return cls_stack

    def forward(self, pixel_values: torch.Tensor) -> DetectorOutput:
        cls_stack = self._extract_cls_stack(pixel_values)
        q1_features = self.q1(cls_stack)                   # [B, L, d']
        aggregated, tie_weights = self.tie(q1_features)    # [B, d'], [L, d']
        q2_features = self.q2(aggregated)                  # [B, d']
        logits = self.classifier(q2_features)              # [B]

        return DetectorOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            features=F.normalize(q2_features, dim=-1),
            tie_weights=tie_weights,
            cls_stack=cls_stack,
        )


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for one feature vector per sample.

    Features are L2-normalized internally. If a batch contains no positive pair
    for some sample, that sample is ignored in the final average.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"features must have shape [B, D], got {tuple(features.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}")

        features = F.normalize(features, dim=-1)
        labels = labels.contiguous().view(-1)
        batch_size = features.shape[0]

        similarity = torch.matmul(features, features.T) / self.temperature
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)

        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = label_mask & logits_mask

        # Stability
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        exp_similarity = torch.exp(similarity) * logits_mask.float()
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True) + self.eps)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return features.new_tensor(0.0)

        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp(min=1)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


def compute_detector_loss(
    output: DetectorOutput,
    labels: torch.Tensor,
    supcon: Optional[SupervisedContrastiveLoss] = None,
    supcon_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined detector loss.

    For the baseline model, `supcon_weight` can be left at 0.0 to recover BCE-only training.
    For the RINE model, a typical setup is BCE + xi * SupCon.
    """
    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(output.logits, labels)
    total = bce
    supcon_value = output.logits.new_tensor(0.0)

    if supcon is not None and supcon_weight > 0:
        supcon_value = supcon(output.features, labels.long())
        total = total + supcon_weight * supcon_value

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_bce": float(bce.detach().cpu()),
        "loss_supcon": float(supcon_value.detach().cpu()),
    }
    return total, stats


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def summarize_tie_weights(tie_weights: torch.Tensor, selected_layers: Sequence[int], top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Returns the most influential blocks according to mean TIE weight across feature dimensions.
    """
    if tie_weights.ndim != 2:
        raise ValueError(f"tie_weights must be [L, d'], got {tuple(tie_weights.shape)}")
    mean_per_block = tie_weights.mean(dim=1)
    top_k = min(top_k, mean_per_block.numel())
    values, indices = torch.topk(mean_per_block, k=top_k)
    return [(int(selected_layers[idx]), float(values[i].detach().cpu())) for i, idx in enumerate(indices)]
