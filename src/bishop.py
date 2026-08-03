"""Numerical BiSHop integration for the MoE-DDI training pipeline.

This module is a compact port of ``Reference/BiSHop-main/models``.  It keeps
the paper's numerical quantile embedding, patch hierarchy, bi-directional GSH
blocks, learnable alpha-entmax sparsity, decoder, and classification head while
removing the original W&B/OpenML/einops training stack.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


class _EntmaxBisect(torch.autograd.Function):
    """Differentiable alpha-entmax copied from the authors' Apache-2.0 code."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        alpha: torch.Tensor,
        dim: int = -1,
        iterations: int = 24,
    ) -> torch.Tensor:
        alpha_shape = list(inputs.shape)
        alpha_shape[dim] = 1
        ctx.alpha_input_shape = alpha.shape
        expanded_alpha = alpha.expand(*alpha_shape)
        ctx.alpha = expanded_alpha
        ctx.dim = dim

        scaled = inputs * (expanded_alpha - 1)
        maximum = scaled.max(dim=dim, keepdim=True).values
        tau_low = maximum - 1
        tau_high = maximum - (1 / inputs.shape[dim]) ** (expanded_alpha - 1)
        f_low = cls_probability(scaled - tau_low, expanded_alpha).sum(dim) - 1
        delta = tau_high - tau_low

        probabilities = None
        for _ in range(iterations):
            delta = delta / 2
            tau_middle = tau_low + delta
            probabilities = cls_probability(scaled - tau_middle, expanded_alpha)
            f_middle = probabilities.sum(dim) - 1
            same_sign = (f_middle * f_low >= 0).unsqueeze(dim)
            tau_low = torch.where(same_sign, tau_middle, tau_low)

        if probabilities is None:  # pragma: no cover - iterations is validated positive
            raise RuntimeError("Entmax bisection did not run")
        probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (probabilities,) = ctx.saved_tensors
        generalized_derivative = torch.where(
            probabilities > 0,
            probabilities ** (2 - ctx.alpha),
            probabilities.new_zeros(()),
        )
        input_gradient = gradient * generalized_derivative
        normalizer = generalized_derivative.sum(ctx.dim).clamp_min(1e-12)
        correction = input_gradient.sum(ctx.dim) / normalizer
        input_gradient -= correction.unsqueeze(ctx.dim) * generalized_derivative

        alpha_gradient = None
        if ctx.needs_input_grad[1]:
            entropy_terms = torch.where(
                probabilities > 0,
                probabilities * probabilities.clamp_min(1e-12).log(),
                probabilities.new_zeros(()),
            )
            entropy = entropy_terms.sum(ctx.dim, keepdim=True)
            skewed = generalized_derivative / normalizer.unsqueeze(ctx.dim)
            alpha_gradient = gradient * (probabilities - skewed) / (ctx.alpha - 1) ** 2
            alpha_gradient -= gradient * (entropy_terms - skewed * entropy) / (
                ctx.alpha - 1
            )
            alpha_gradient = alpha_gradient.sum(ctx.dim, keepdim=True)
            alpha_gradient = alpha_gradient.sum_to_size(ctx.alpha_input_shape)
        return input_gradient, alpha_gradient, None, None


def cls_probability(values: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return values.clamp_min(0) ** (1 / (alpha - 1))


class LearnableEntmax(nn.Module):
    """Per-head alpha-entmax with alpha constrained to the stable (1, 2) range."""

    def __init__(self, num_heads: int, iterations: int = 24) -> None:
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.zeros(num_heads))
        self.iterations = iterations

    @property
    def alpha(self) -> torch.Tensor:
        return 1.01 + 0.98 * torch.sigmoid(self.raw_alpha)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        return _EntmaxBisect.apply(scores, alpha, -1, self.iterations)


class NumericalQuantileEmbedding(nn.Module):
    """Piecewise-linear numerical encoding used by the released BiSHop."""

    def __init__(self, num_features: int, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        initial = torch.linspace(0, 1, embedding_dim + 1).repeat(num_features, 1)
        self.register_buffer("quantiles", initial)
        self.register_buffer("bins_fitted", torch.tensor(False))

    def set_quantiles(self, quantiles: np.ndarray | torch.Tensor) -> None:
        values = torch.as_tensor(
            quantiles,
            dtype=self.quantiles.dtype,
            device=self.quantiles.device,
        )
        if values.shape != self.quantiles.shape:
            raise ValueError(
                f"Expected quantiles shaped {tuple(self.quantiles.shape)}, "
                f"received {tuple(values.shape)}"
            )
        self.quantiles.copy_(values)
        self.bins_fitted.fill_(True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.num_features:
            raise ValueError(f"Expected [batch, {self.num_features}] numerical inputs")
        left = self.quantiles[:, :-1]
        right = self.quantiles[:, 1:]
        denominator = (right - left).clamp_min(torch.finfo(inputs.dtype).eps)
        return ((inputs.unsqueeze(-1) - left) / denominator).clamp(0, 1)


class GSHAttention(nn.Module):
    """Multi-head Generalized Sparse Hopfield retrieval layer."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.normalizer = LearnableEntmax(num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, model_dim = queries.shape
        key_length = keys.shape[1]
        query = self.query_projection(queries).view(
            batch, query_length, self.num_heads, self.head_dim
        )
        key = self.key_projection(keys).view(
            batch, key_length, self.num_heads, self.head_dim
        )
        # The authors' Hopfield path retrieves values from key-space memories.
        value = self.key_projection(values)
        value = self.value_projection(value).view(
            batch, key_length, self.num_heads, self.head_dim
        )
        scores = torch.einsum("blhe,bshe->bhls", query, key)
        weights = self.normalizer(scores / math.sqrt(self.head_dim))
        retrieved = torch.einsum("bhls,bshd->blhd", self.dropout(weights), value)
        return self.output_projection(retrieved.reshape(batch, query_length, model_dim))


class BiSHopBlock(nn.Module):
    """Column-wise retrieval followed by bottlenecked row-wise retrieval."""

    def __init__(
        self,
        num_patches: int,
        factor: int,
        model_dim: int,
        feedforward_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_attention = GSHAttention(model_dim, num_heads, dropout)
        self.embedding_pooling = GSHAttention(model_dim, num_heads, dropout)
        self.embedding_attention = GSHAttention(model_dim, num_heads, dropout)
        self.pooling = nn.Parameter(torch.randn(num_patches, factor, model_dim))
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(model_dim) for _ in range(4))
        self.feedforward_feature = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.feedforward_embedding = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, embedding_dim, num_patches, model_dim = inputs.shape
        feature_input = inputs.reshape(batch * embedding_dim, num_patches, model_dim)
        feature_output = feature_input + self.dropout(
            self.feature_attention(feature_input, feature_input, feature_input)
        )
        feature_output = self.norms[0](feature_output)
        feature_output = feature_output + self.dropout(
            self.feedforward_feature(feature_output)
        )
        embedding_input = self.norms[1](feature_output)

        embedding_send = embedding_input.view(
            batch, embedding_dim, num_patches, model_dim
        ).permute(0, 2, 1, 3)
        embedding_send = embedding_send.reshape(
            batch * num_patches, embedding_dim, model_dim
        )
        pooling = self.pooling.unsqueeze(0).expand(batch, -1, -1, -1)
        pooling = pooling.reshape(batch * num_patches, self.pooling.shape[1], model_dim)
        buffer = self.embedding_pooling(pooling, embedding_send, embedding_send)
        embedding_output = embedding_send + self.dropout(
            self.embedding_attention(embedding_send, buffer, buffer)
        )
        embedding_output = self.norms[2](embedding_output)
        embedding_output = embedding_output + self.dropout(
            self.feedforward_embedding(embedding_output)
        )
        embedding_output = self.norms[3](embedding_output)
        return embedding_output.view(batch, num_patches, embedding_dim, model_dim).permute(
            0, 2, 1, 3
        )


class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim: int, model_dim: int) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.projection = nn.Linear(patch_dim, model_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, feature_dim, embedding_dim = inputs.shape
        num_patches = feature_dim // self.patch_dim
        patches = inputs.view(batch, num_patches, self.patch_dim, embedding_dim)
        patches = patches.permute(0, 3, 1, 2)
        return self.projection(patches)


class PatchMerge(nn.Module):
    def __init__(self, model_dim: int, aggregation: int) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.norm = nn.LayerNorm(aggregation * model_dim)
        self.projection = nn.Linear(aggregation * model_dim, model_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, embedding_dim, num_patches, model_dim = inputs.shape
        remainder = num_patches % self.aggregation
        if num_patches < self.aggregation:
            repeats = math.ceil(self.aggregation / num_patches)
            inputs = inputs.repeat(1, 1, repeats, 1)[:, :, : self.aggregation]
        elif remainder:
            padding = self.aggregation - remainder
            inputs = torch.cat((inputs, inputs[:, :, -padding:]), dim=2)
        merged_patches = inputs.shape[2] // self.aggregation
        merged = inputs.reshape(
            batch,
            embedding_dim,
            merged_patches,
            self.aggregation * model_dim,
        )
        return self.projection(self.norm(merged))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        aggregation: int,
        num_patches: int,
        factor: int,
        model_dim: int,
        feedforward_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.merge = (
            PatchMerge(model_dim, aggregation) if aggregation > 1 else nn.Identity()
        )
        output_patches = math.ceil(num_patches / aggregation)
        self.block = BiSHopBlock(
            output_patches,
            factor,
            model_dim,
            feedforward_dim,
            num_heads,
            dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.merge(inputs))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        num_patches: int,
        patch_dim: int,
        factor: int,
        model_dim: int,
        feedforward_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = BiSHopBlock(
            num_patches,
            factor,
            model_dim,
            feedforward_dim,
            num_heads,
            dropout,
        )
        self.cross_attention = GSHAttention(model_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.patch_prediction = nn.Linear(model_dim, patch_dim)

    def forward(
        self,
        queries: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, embedding_dim, num_patches, model_dim = queries.shape
        output = self.block(queries)
        flat_output = output.reshape(batch * embedding_dim, num_patches, model_dim)
        flat_encoder = encoder_output.reshape(
            batch * embedding_dim, encoder_output.shape[2], model_dim
        )
        flat_output = flat_output + self.dropout(
            self.cross_attention(flat_output, flat_encoder, flat_encoder)
        )
        normalized = self.norm1(flat_output)
        decoded = self.norm2(normalized + self.feedforward(normalized))
        output = decoded.view(batch, embedding_dim, num_patches, model_dim)
        prediction = self.patch_prediction(output)
        prediction = prediction.permute(0, 2, 3, 1).reshape(
            batch, num_patches * self.patch_prediction.out_features, embedding_dim
        )
        return output, prediction


class BiSHop(nn.Module):
    """BiSHop classifier adapted to all-numerical DDI descriptors.

    Quantiles are fitted from a bounded, uniformly sampled training stream by
    the shared training engine; they are persistent buffers and therefore
    travel with checkpoints.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        embedding_dim: int = 32,
        output_dim: int = 8,
        patch_dim: int = 8,
        factor: int = 8,
        aggregation: int = 4,
        model_dim: int = 128,
        feedforward_dim: int = 256,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 3,
        dropout: float = 0.1,
        classifier_hidden_dims: Iterable[int] = (512,),
        classifier_dropout: float = 0.2,
        quantile_sample_size: int = 2048,
        quantile_max_rows: int | None = 100_000,
    ) -> None:
        super().__init__()
        dimensions = (
            num_features,
            num_classes,
            embedding_dim,
            output_dim,
            patch_dim,
            factor,
            aggregation,
            model_dim,
            feedforward_dim,
            num_heads,
        )
        if min(dimensions) < 1:
            raise ValueError("BiSHop dimensions must be positive")
        if encoder_layers < 1 or decoder_layers < 0:
            raise ValueError("Invalid BiSHop encoder/decoder depth")
        if decoder_layers > encoder_layers + 1:
            raise ValueError("decoder_layers cannot exceed encoder_layers + 1")
        if quantile_sample_size < 2:
            raise ValueError("quantile_sample_size must be at least 2")
        if quantile_max_rows is not None and quantile_max_rows < 2:
            raise ValueError("quantile_max_rows must be at least 2 or null")

        self.quantile_sample_size = quantile_sample_size
        self.quantile_max_rows = quantile_max_rows
        self.numerical_embedding = NumericalQuantileEmbedding(
            num_features, embedding_dim
        )

        padded_embedding_dim = math.ceil(embedding_dim / patch_dim) * patch_dim
        self.embedding_padding = padded_embedding_dim - embedding_dim
        input_patches = padded_embedding_dim // patch_dim
        self.patch_embedding = PatchEmbedding(patch_dim, model_dim)
        self.encoder_position = nn.Parameter(
            torch.randn(1, num_features, input_patches, model_dim)
        )
        self.pre_norm = nn.LayerNorm(model_dim)

        self.encoder = nn.ModuleList()
        patch_counts = [input_patches]
        for index in range(encoder_layers):
            layer_aggregation = 1 if index == 0 else aggregation
            self.encoder.append(
                EncoderLayer(
                    aggregation=layer_aggregation,
                    num_patches=patch_counts[-1],
                    factor=factor,
                    model_dim=model_dim,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            patch_counts.append(math.ceil(patch_counts[-1] / layer_aggregation))

        padded_output_dim = math.ceil(output_dim / patch_dim) * patch_dim
        output_patches = padded_output_dim // patch_dim
        self.decoder_position = nn.Parameter(
            torch.randn(1, num_features, output_patches, model_dim)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(
                num_patches=output_patches,
                patch_dim=patch_dim,
                factor=factor,
                model_dim=model_dim,
                feedforward_dim=feedforward_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(decoder_layers)
        )
        self.encoder_only_projection = (
            nn.Linear(patch_counts[-1] * model_dim, padded_output_dim)
            if decoder_layers == 0
            else None
        )

        classifier_input = num_features * padded_output_dim
        layers: list[nn.Module] = [nn.LayerNorm(classifier_input)]
        previous = classifier_input
        for hidden in classifier_hidden_dims:
            if hidden < 1:
                raise ValueError("classifier_hidden_dims must be positive")
            layers.extend(
                [nn.Linear(previous, hidden), nn.GELU(), nn.Dropout(classifier_dropout)]
            )
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.classifier = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_quantiles(
        self,
        feature_batches: Iterable[np.ndarray],
        *,
        seed: int,
    ) -> int:
        """Fit quantile bins from a bounded uniform reservoir of training rows."""
        rng = np.random.default_rng(seed)
        reservoir: np.ndarray | None = None
        priorities: np.ndarray | None = None
        rows_seen = 0
        for batch in feature_batches:
            values = np.asarray(batch, dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != self.numerical_embedding.num_features:
                raise ValueError("Quantile batch has an incompatible feature shape")
            keys = rng.random(len(values))
            rows_seen += len(values)
            if reservoir is None:
                candidates = values
                candidate_keys = keys
            else:
                candidates = np.concatenate((reservoir, values), axis=0)
                candidate_keys = np.concatenate((priorities, keys))
            keep_count = min(self.quantile_sample_size, len(candidates))
            keep = np.argpartition(candidate_keys, keep_count - 1)[:keep_count]
            reservoir = candidates[keep].copy()
            priorities = candidate_keys[keep].copy()

        if reservoir is None or len(reservoir) < 2:
            raise ValueError("At least two training rows are required to fit BiSHop bins")
        levels = np.linspace(0, 1, self.numerical_embedding.embedding_dim + 1)
        quantiles = np.quantile(reservoir, levels, axis=0).T.astype(np.float32)
        self.numerical_embedding.set_quantiles(quantiles)
        return rows_seen

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.numerical_embedding(inputs)
        # Released BiSHop uses flip=True: quantile cells are patched and original
        # columns become the row-wise cellular dimension.
        embedded = embedded.transpose(1, 2)
        if self.embedding_padding:
            padding = embedded[:, :1].expand(-1, self.embedding_padding, -1)
            embedded = torch.cat((padding, embedded), dim=1)
        encoded = self.pre_norm(self.patch_embedding(embedded) + self.encoder_position)
        encoder_outputs = [encoded]
        for layer in self.encoder:
            encoded = layer(encoded)
            encoder_outputs.append(encoded)

        if self.decoder:
            decoded = self.decoder_position.expand(inputs.shape[0], -1, -1, -1)
            prediction = None
            for index, layer in enumerate(self.decoder):
                decoded, layer_prediction = layer(decoded, encoder_outputs[index])
                prediction = (
                    layer_prediction
                    if prediction is None
                    else prediction + layer_prediction
                )
            if prediction is None:  # pragma: no cover - guarded by self.decoder
                raise RuntimeError("BiSHop decoder produced no prediction")
        else:
            flattened = encoded.flatten(start_dim=2)
            prediction = self.encoder_only_projection(flattened).transpose(1, 2)
        return self.classifier(prediction.flatten(start_dim=1))
