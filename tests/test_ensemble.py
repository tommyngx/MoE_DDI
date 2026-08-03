import numpy as np
import torch

from engine import evaluate_models
from preprocessing import TrainStatistics


class ConstantModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("values", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs):
        return self.values.unsqueeze(0).expand(len(inputs), -1)


def test_probability_ensemble_and_uncertainty_are_streamed():
    statistics = TrainStatistics(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        class_counts=np.asarray([2, 0]),
        label_values=np.asarray([1, 2]),
        count=2,
        schema_fingerprint="test",
    )
    config = {
        "data": {"num_classes": 2},
        "preprocessing": {"normalization": "none"},
        "evaluation": {
            "top_k": [1],
            "calibration_bins": 5,
            "confidence_thresholds": {"low": 0.1, "high": 0.9},
        },
    }
    stream = [
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
        )
    ]
    aggregate, _, _, _, _ = evaluate_models(
        [ConstantModel([5.0, 0.0]), ConstantModel([0.0, 4.0])],
        stream,
        statistics,
        torch.device("cpu"),
        config,
    )

    assert aggregate["ensemble_size"] == 2
    assert aggregate["accuracy"] == 1.0
    assert aggregate["mean_variance"] > 0
    assert aggregate["mean_mutual_information"] > 0
    assert aggregate["confidence_strata"]["low"]["coverage"] == 1.0
