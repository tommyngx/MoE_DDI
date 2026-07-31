import torch

from models import MLPClassifier, MoEDDI


def test_baseline_forward_shape():
    model = MLPClassifier(12, 5, hidden_dims=[8])
    assert model(torch.randn(4, 12)).shape == (4, 5)


def test_moeddi_forward_backward_and_router_normalization():
    model = MoEDDI(
        12,
        5,
        {
            "PEOE_VSA": (0, 1, 2, 3),
            "MR_VSA": (3, 4, 5, 6),
            "MTPSA": (7, 8),
        },
        expert_hidden_dim=8,
        expert_dim=6,
        router_hidden_dim=5,
        router_top_k=2,
        classifier_hidden_dim=7,
    )
    output = model(torch.randn(4, 12))
    assert output.logits.shape == (4, 5)
    assert output.router_probabilities.shape == (4, 4)
    torch.testing.assert_close(
        output.router_probabilities.sum(dim=1),
        torch.ones(4),
    )
    assert torch.all((output.router_probabilities > 0).sum(dim=1) == 2)
    (output.logits.mean() + output.balance_loss + output.router_z_loss).backward()
