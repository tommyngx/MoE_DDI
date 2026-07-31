from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = functional.log_softmax(logits, dim=-1)
        log_pt = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.label_smoothing > 0:
            nll = functional.cross_entropy(
                logits,
                targets,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
        else:
            nll = -log_pt
        if self.weight is not None:
            nll = nll * self.weight[targets]
        return (focal_factor * nll).mean()


def class_weights(
    counts: np.ndarray,
    strategy: str,
    *,
    effective_beta: float = 0.9999,
) -> torch.Tensor | None:
    counts = np.asarray(counts, dtype=np.float64)
    if strategy == "none":
        return None
    observed = counts > 0
    weights = np.zeros_like(counts)
    if strategy == "inverse":
        weights[observed] = 1.0 / counts[observed]
    elif strategy == "inverse_sqrt":
        weights[observed] = 1.0 / np.sqrt(counts[observed])
    elif strategy == "effective_number":
        weights[observed] = (1.0 - effective_beta) / (
            1.0 - np.power(effective_beta, counts[observed])
        )
    else:
        raise ValueError(f"Unknown class weighting strategy: {strategy}")
    if observed.any():
        weights[observed] /= weights[observed].mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_loss(config: dict, counts: np.ndarray, device: torch.device) -> nn.Module:
    loss_config = config["loss"]
    weights = class_weights(
        counts,
        loss_config["class_weighting"],
        effective_beta=loss_config.get("effective_beta", 0.9999),
    )
    if weights is not None:
        weights = weights.to(device)
    if loss_config["name"] == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=loss_config.get("label_smoothing", 0.0),
        )
    return FocalLoss(
        gamma=loss_config.get("gamma", 2.0),
        weight=weights,
        label_smoothing=loss_config.get("label_smoothing", 0.0),
    )

