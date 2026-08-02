import copy

import torch

from models import MLPClassifier, MoEDDI, TDDINumericalMLP, count_parameters


def test_baseline_forward_shape():
    model = MLPClassifier(12, 5, hidden_dims=[8])
    assert model(torch.randn(4, 12)).shape == (4, 5)


def test_tddi_numerical_path_matches_released_active_head():
    model = TDDINumericalMLP(3780, 178)
    linear_shapes = [
        tuple(module.weight.shape)
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert linear_shapes == [(7560, 3780), (7560, 7560), (178, 7560)]
    assert count_parameters(model)["trainable"] == 87_098_938


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
        use_shared_trunk=True,
        shared_trunk_hidden_dim=9,
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
    assert model.shared_trunk[1].weight.grad is not None

    restored = copy.deepcopy(model).eval()
    restored.load_state_dict(model.state_dict())
    inputs = torch.randn(3, 12)
    model.eval()
    torch.testing.assert_close(model(inputs).logits, restored(inputs).logits)


def test_tddi_residual_hybrid_starts_from_exact_global_predictions():
    torch.manual_seed(7)
    baseline = TDDINumericalMLP(12, 5, hidden_multipliers=[2, 2]).eval()
    hybrid = MoEDDI(
        12,
        5,
        {
            "PEOE_VSA": (0, 1, 2, 3),
            "MR_VSA": (4, 5, 6, 7),
            "MTPSA": (8, 9, 10, 11),
        },
        expert_hidden_dim=8,
        expert_dim=6,
        router_hidden_dim=5,
        router_top_k=None,
        include_generalist=False,
        use_tddi_backbone=True,
        tddi_hidden_multipliers=[2, 2],
        zero_init_moe_residual=True,
        classifier_hidden_dim=7,
        classifier_dropout=0.0,
    ).eval()
    hybrid.load_tddi_state_dict(baseline.state_dict())
    inputs = torch.randn(4, 12)

    assert torch.equal(hybrid(inputs).logits, baseline(inputs))
    hybrid.set_tddi_trainable(False)
    assert not hybrid.tddi_classifier.weight.requires_grad
    hybrid.set_tddi_trainable(True)
    assert hybrid.tddi_classifier.weight.requires_grad
