from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from schema import DatasetSchema


def activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (),
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    activation_layer(activation),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    router_probabilities: torch.Tensor


class FamilyExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MoEDDI(nn.Module):
    """Feature-family experts with per-sample top-k routing."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        family_indices: dict[str, tuple[int, ...]],
        *,
        expert_hidden_dim: int = 512,
        expert_dim: int = 256,
        expert_dropout: float = 0.15,
        router_hidden_dim: int = 128,
        router_top_k: int | None = 2,
        include_generalist: bool = True,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        filtered = {name: values for name, values in family_indices.items() if values}
        if include_generalist:
            filtered["Generalist"] = tuple(range(num_features))
        if len(filtered) < 2:
            raise ValueError("MoEDDI requires at least two non-empty experts")

        self.family_names = tuple(filtered)
        self.router_top_k = router_top_k
        self.experts = nn.ModuleDict()
        for family, indices in filtered.items():
            buffer_name = f"indices_{len(self.experts)}"
            self.register_buffer(buffer_name, torch.tensor(indices, dtype=torch.long))
            self.experts[family] = FamilyExpert(
                len(indices), expert_hidden_dim, expert_dim, expert_dropout
            )

        router_input_dim = 2 * len(self.family_names)
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, len(self.family_names)),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def _indices(self, expert_index: int) -> torch.Tensor:
        return getattr(self, f"indices_{expert_index}")

    def forward(self, inputs: torch.Tensor) -> ModelOutput:
        expert_outputs = []
        router_statistics = []
        for index, family in enumerate(self.family_names):
            selected = torch.index_select(inputs, 1, self._indices(index))
            expert_outputs.append(self.experts[family](selected))
            router_statistics.extend(
                [selected.mean(dim=1), selected.square().mean(dim=1).sqrt()]
            )

        router_inputs = torch.stack(router_statistics, dim=1)
        router_logits = self.router(router_inputs)
        dense_probabilities = torch.softmax(router_logits, dim=-1)
        probabilities = dense_probabilities
        if self.router_top_k is not None and self.router_top_k < len(self.family_names):
            if self.router_top_k < 1:
                raise ValueError("router_top_k must be positive or null")
            top_values, top_indices = torch.topk(router_logits, self.router_top_k, dim=-1)
            sparse_logits = torch.full_like(router_logits, float("-inf"))
            sparse_logits.scatter_(1, top_indices, top_values)
            probabilities = torch.softmax(sparse_logits, dim=-1)

        stacked = torch.stack(expert_outputs, dim=1)
        mixture = torch.sum(stacked * probabilities.unsqueeze(-1), dim=1)
        logits = self.classifier(mixture)
        mean_probability = dense_probabilities.mean(dim=0)
        balance_loss = len(self.family_names) * mean_probability.square().sum() - 1.0
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        return ModelOutput(
            logits=logits,
            balance_loss=balance_loss,
            router_z_loss=router_z_loss,
            router_probabilities=probabilities,
        )


def build_model(config: dict, schema: DatasetSchema) -> nn.Module:
    model_config = config["model"]
    name = model_config["name"]
    num_classes = config["data"]["num_classes"]
    if name == "linear":
        return MLPClassifier(
            schema.num_features,
            num_classes,
            hidden_dims=(),
            layer_norm=model_config.get("layer_norm", True),
        )
    if name in {"mlp", "tddi_mlp"}:
        return MLPClassifier(
            schema.num_features,
            num_classes,
            hidden_dims=model_config.get("hidden_dims", [1024, 512]),
            activation=model_config.get("activation", "relu"),
            dropout=model_config.get("dropout", 0.0),
            layer_norm=model_config.get("layer_norm", True),
        )
    if name == "moeddi":
        return MoEDDI(
            schema.num_features,
            num_classes,
            schema.family_indices,
            expert_hidden_dim=model_config.get("expert_hidden_dim", 512),
            expert_dim=model_config.get("expert_dim", 256),
            expert_dropout=model_config.get("expert_dropout", 0.15),
            router_hidden_dim=model_config.get("router_hidden_dim", 128),
            router_top_k=model_config.get("router_top_k", 2),
            include_generalist=model_config.get("include_generalist", True),
            classifier_hidden_dim=model_config.get("classifier_hidden_dim", 512),
            classifier_dropout=model_config.get("classifier_dropout", 0.2),
        )
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}
