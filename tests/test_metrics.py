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

