import numpy as np
import torch

from losses import FocalLoss, class_weights


def test_focal_loss_is_finite_and_backpropagates():
    logits = torch.randn(8, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_effective_number_weights_are_normalized_on_observed_classes():
    weights = class_weights(
        np.asarray([100, 10, 0]),
        "effective_number",
        effective_beta=0.99,
    )
    assert weights is not None
    np.testing.assert_allclose(weights[:2].mean().item(), 1.0, rtol=1e-5)
    assert weights[2].item() == 0

