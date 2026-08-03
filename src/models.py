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


class TDDINumericalMLP(MLPClassifier):
    """Numerical-only path of the released T-DDI TabTransformer.

    T-DDI has no categorical columns, so its transformer branch is never used.
    With ``mlp_hidden_mults=(2, 2)``, the active prediction path is a LayerNorm
    followed by ``D -> 2D -> 2D -> C`` linear layers with ReLU activations.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_multipliers: list[int] | tuple[int, ...] = (2, 2),
    ) -> None:
        if not hidden_multipliers or any(value < 1 for value in hidden_multipliers):
            raise ValueError("hidden_multipliers must contain positive integers")
        super().__init__(
            input_dim,
            num_classes,
            hidden_dims=[input_dim * value for value in hidden_multipliers],
            activation="relu",
            dropout=0.0,
            layer_norm=True,
        )


@dataclass
class ModelOutput:
    logits: torch.Tensor
    balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    router_probabilities: torch.Tensor
    auxiliary_logits: torch.Tensor | None = None
    global_logits: torch.Tensor | None = None


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
        use_shared_trunk: bool = False,
        shared_trunk_hidden_dim: int = 512,
        use_tddi_backbone: bool = False,
        tddi_hidden_multipliers: list[int] | tuple[int, ...] = (2, 2),
        zero_init_moe_residual: bool = True,
        router_log_statistics: bool = False,
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
        self.router_log_statistics = router_log_statistics
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
        if use_shared_trunk and use_tddi_backbone:
            raise ValueError("use_shared_trunk and use_tddi_backbone are mutually exclusive")
        self.shared_trunk = None
        self.tddi_backbone = None
        self.tddi_projection = None
        self.tddi_classifier = None
        classifier_input_dim = expert_dim
        if use_shared_trunk:
            # T-DDI's strongest inductive bias is an always-on global numerical
            # path. It complements family experts by retaining interactions
            # spanning descriptors that the router would otherwise suppress.
            self.shared_trunk = nn.Sequential(
                nn.LayerNorm(num_features),
                nn.Linear(num_features, shared_trunk_hidden_dim),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(shared_trunk_hidden_dim, expert_dim),
                nn.GELU(),
            )
            classifier_input_dim += expert_dim
        if use_tddi_backbone:
            if not tddi_hidden_multipliers or any(
                value < 1 for value in tddi_hidden_multipliers
            ):
                raise ValueError("tddi_hidden_multipliers must contain positive integers")
            backbone_layers: list[nn.Module] = [nn.LayerNorm(num_features)]
            previous = num_features
            for multiplier in tddi_hidden_multipliers:
                hidden = num_features * multiplier
                backbone_layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
                previous = hidden
            self.tddi_backbone = nn.Sequential(*backbone_layers)
            self.tddi_projection = nn.Sequential(
                nn.Linear(previous, expert_dim),
                nn.GELU(),
            )
            self.tddi_classifier = nn.Linear(previous, num_classes)
            classifier_input_dim += expert_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )
        if use_tddi_backbone and zero_init_moe_residual:
            final_layer = self.classifier[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def _indices(self, expert_index: int) -> torch.Tensor:
        return getattr(self, f"indices_{expert_index}")

    def load_tddi_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize the global path from a TDDINumericalMLP checkpoint."""
        if self.tddi_backbone is None or self.tddi_classifier is None:
            raise ValueError("This MoEDDI instance has no T-DDI backbone")
        source_linear_prefixes = sorted(
            {
                ".".join(key.split(".")[:2])
                for key, value in state_dict.items()
                if key.startswith("network.") and key.endswith(".weight") and value.ndim == 2
            },
            key=lambda value: int(value.split(".")[1]),
        )
        target_linears = [
            module for module in self.tddi_backbone if isinstance(module, nn.Linear)
        ] + [self.tddi_classifier]
        if len(source_linear_prefixes) != len(target_linears):
            raise ValueError("T-DDI checkpoint depth does not match the configured backbone")
        try:
            with torch.no_grad():
                self.tddi_backbone[0].weight.copy_(state_dict["network.0.weight"])
                self.tddi_backbone[0].bias.copy_(state_dict["network.0.bias"])
                for prefix, target in zip(
                    source_linear_prefixes, target_linears, strict=True
                ):
                    target.weight.copy_(state_dict[f"{prefix}.weight"])
                    target.bias.copy_(state_dict[f"{prefix}.bias"])
        except (KeyError, RuntimeError) as error:
            raise ValueError("T-DDI checkpoint tensors do not match the global backbone") from error

    def set_tddi_trainable(self, trainable: bool) -> None:
        if self.tddi_backbone is None or self.tddi_classifier is None:
            return
        modules = (self.tddi_backbone, self.tddi_classifier)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    def tddi_parameters(self):
        if self.tddi_backbone is None or self.tddi_classifier is None:
            return iter(())
        return iter((*self.tddi_backbone.parameters(), *self.tddi_classifier.parameters()))

    def forward(self, inputs: torch.Tensor) -> ModelOutput:
        expert_outputs = []
        router_statistics = []
        for index, family in enumerate(self.family_names):
            selected = torch.index_select(inputs, 1, self._indices(index))
            expert_outputs.append(self.experts[family](selected))
            selected_mean = selected.mean(dim=1)
            selected_rms = selected.square().mean(dim=1).sqrt()
            if self.router_log_statistics:
                selected_mean = selected_mean.sign() * torch.log1p(selected_mean.abs())
                selected_rms = torch.log1p(selected_rms)
            router_statistics.extend([selected_mean, selected_rms])

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
        classifier_inputs = mixture
        tddi_logits = None
        if self.shared_trunk is not None:
            classifier_inputs = torch.cat((mixture, self.shared_trunk(inputs)), dim=-1)
        if self.tddi_backbone is not None:
            tddi_hidden = self.tddi_backbone(inputs)
            classifier_inputs = torch.cat(
                (mixture, self.tddi_projection(tddi_hidden)), dim=-1
            )
            tddi_logits = self.tddi_classifier(tddi_hidden)
        moe_logits = self.classifier(classifier_inputs)
        logits = moe_logits if tddi_logits is None else tddi_logits + moe_logits
        mean_probability = dense_probabilities.mean(dim=0)
        balance_loss = len(self.family_names) * mean_probability.square().sum() - 1.0
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        return ModelOutput(
            logits=logits,
            balance_loss=balance_loss,
            router_z_loss=router_z_loss,
            router_probabilities=probabilities,
            auxiliary_logits=moe_logits if tddi_logits is not None else None,
            global_logits=tddi_logits,
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
    if name == "tddi_mlp":
        return TDDINumericalMLP(
            schema.num_features,
            num_classes,
            hidden_multipliers=model_config.get("hidden_multipliers", [2, 2]),
        )
    if name == "mlp":
        return MLPClassifier(
            schema.num_features,
            num_classes,
            hidden_dims=model_config.get("hidden_dims", [1024, 512]),
            activation=model_config.get("activation", "relu"),
            dropout=model_config.get("dropout", 0.0),
            layer_norm=model_config.get("layer_norm", True),
        )
    if name == "bishop":
        # Keep the sizeable paper port isolated from the stable baseline/MoE
        # definitions. Importing lazily also avoids a circular model registry.
        from bishop import BiSHop

        return BiSHop(
            schema.num_features,
            num_classes,
            embedding_dim=model_config.get("embedding_dim", 32),
            output_dim=model_config.get("output_dim", 8),
            patch_dim=model_config.get("patch_dim", 8),
            factor=model_config.get("factor", 8),
            aggregation=model_config.get("aggregation", 4),
            model_dim=model_config.get("model_dim", 128),
            feedforward_dim=model_config.get("feedforward_dim", 256),
            num_heads=model_config.get("num_heads", 4),
            encoder_layers=model_config.get("encoder_layers", 2),
            decoder_layers=model_config.get(
                "decoder_layers", model_config.get("encoder_layers", 2) + 1
            ),
            dropout=model_config.get("dropout", 0.1),
            classifier_hidden_dims=model_config.get(
                "classifier_hidden_dims", [512]
            ),
            classifier_dropout=model_config.get("classifier_dropout", 0.2),
            quantile_sample_size=model_config.get("quantile_sample_size", 2048),
            quantile_max_rows=model_config.get("quantile_max_rows", 100_000),
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
            use_shared_trunk=model_config.get("use_shared_trunk", False),
            shared_trunk_hidden_dim=model_config.get("shared_trunk_hidden_dim", 512),
            use_tddi_backbone=model_config.get("use_tddi_backbone", False),
            tddi_hidden_multipliers=model_config.get(
                "tddi_hidden_multipliers", [2, 2]
            ),
            zero_init_moe_residual=model_config.get("zero_init_moe_residual", True),
            router_log_statistics=model_config.get("router_log_statistics", False),
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
