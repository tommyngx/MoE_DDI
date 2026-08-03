import copy

import numpy as np
import torch

from bishop import BiSHop, LearnableEntmax


def _small_bishop() -> BiSHop:
    return BiSHop(
        12,
        5,
        embedding_dim=8,
        output_dim=4,
        patch_dim=4,
        factor=3,
        aggregation=2,
        model_dim=16,
        feedforward_dim=24,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=3,
        dropout=0.0,
        classifier_hidden_dims=[16],
        classifier_dropout=0.0,
        quantile_sample_size=8,
        quantile_max_rows=20,
    )


def test_bishop_forward_backward_and_learnable_sparsity():
    model = _small_bishop()
    rows = np.random.default_rng(7).normal(size=(20, 12)).astype(np.float32)
    assert model.fit_quantiles([rows[:9], rows[9:]], seed=11) == 20

    output = model(torch.from_numpy(rows[:3]))
    assert output.shape == (3, 5)
    output.square().mean().backward()

    entmax_parameters = [
        module.raw_alpha.grad
        for module in model.modules()
        if isinstance(module, LearnableEntmax)
    ]
    assert entmax_parameters
    used_gradients = [gradient for gradient in entmax_parameters if gradient is not None]
    assert used_gradients
    assert all(torch.isfinite(gradient).all() for gradient in used_gradients)


def test_bishop_serialization_and_deterministic_inference():
    torch.manual_seed(17)
    model = _small_bishop().eval()
    rows = np.random.default_rng(3).normal(size=(12, 12)).astype(np.float32)
    model.fit_quantiles([rows], seed=5)
    inputs = torch.from_numpy(rows[:4])
    expected = model(inputs)

    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(inputs), expected)
    torch.testing.assert_close(
        restored.numerical_embedding.quantiles,
        model.numerical_embedding.quantiles,
    )
