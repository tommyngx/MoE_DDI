import torch

from metrics import ClassificationMetrics


def test_metrics_perfect_predictions():
    tracker = ClassificationMetrics(3, top_k=[1, 2], calibration_bins=5)
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    targets = torch.tensor([0, 1, 2])
    tracker.update(logits, targets)
    aggregate, per_class = tracker.compute()
    assert aggregate["accuracy"] == 1.0
    assert aggregate["macro_f1"] == 1.0
    assert aggregate["top_2_accuracy"] == 1.0
    assert all(row["f1"] == 1.0 for row in per_class)


def test_metrics_accept_ensemble_probabilities():
    tracker = ClassificationMetrics(2, top_k=[1], calibration_bins=5)
    tracker.update_probabilities(
        torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        torch.tensor([0, 1]),
    )
    aggregate, _ = tracker.compute()
    assert aggregate["accuracy"] == 1.0
