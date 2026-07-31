from __future__ import annotations

import numpy as np
import torch


class ClassificationMetrics:
    def __init__(
        self,
        num_classes: int,
        *,
        top_k: list[int] | tuple[int, ...] = (1, 3, 5),
        calibration_bins: int = 15,
    ) -> None:
        self.num_classes = num_classes
        self.top_k = sorted(set(top_k))
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.top_correct = {value: 0 for value in self.top_k}
        self.total = 0
        self.bin_count = np.zeros(calibration_bins, dtype=np.int64)
        self.bin_confidence = np.zeros(calibration_bins, dtype=np.float64)
        self.bin_correct = np.zeros(calibration_bins, dtype=np.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = torch.softmax(logits.detach(), dim=-1)
        predictions = probabilities.argmax(dim=-1)
        target_values = targets.detach().cpu().numpy().astype(np.int64)
        predicted_values = predictions.cpu().numpy().astype(np.int64)
        np.add.at(self.confusion, (target_values, predicted_values), 1)
        self.total += len(target_values)

        max_k = min(max(self.top_k), self.num_classes)
        top_indices = probabilities.topk(max_k, dim=-1).indices
        for value in self.top_k:
            effective = min(value, self.num_classes)
            correct = top_indices[:, :effective].eq(targets.unsqueeze(1)).any(dim=1)
            self.top_correct[value] += int(correct.sum().item())

        confidence, _ = probabilities.max(dim=-1)
        confidence_values = confidence.cpu().numpy()
        correct_values = (predicted_values == target_values).astype(np.float64)
        bins = np.minimum(
            (confidence_values * len(self.bin_count)).astype(np.int64),
            len(self.bin_count) - 1,
        )
        np.add.at(self.bin_count, bins, 1)
        np.add.at(self.bin_confidence, bins, confidence_values)
        np.add.at(self.bin_correct, bins, correct_values)

    def compute(self) -> tuple[dict[str, float], list[dict[str, float | int]]]:
        true_positive = np.diag(self.confusion).astype(np.float64)
        support = self.confusion.sum(axis=1).astype(np.float64)
        predicted = self.confusion.sum(axis=0).astype(np.float64)
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros_like(true_positive),
            where=support > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        observed = support > 0
        macro_f1 = float(f1[observed].mean()) if observed.any() else 0.0
        weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0
        accuracy = float(true_positive.sum() / self.total) if self.total else 0.0

        expected_calibration_error = 0.0
        for index, count in enumerate(self.bin_count):
            if count == 0 or self.total == 0:
                continue
            mean_confidence = self.bin_confidence[index] / count
            mean_accuracy = self.bin_correct[index] / count
            expected_calibration_error += count / self.total * abs(
                mean_accuracy - mean_confidence
            )

        aggregate = {
            "num_samples": int(self.total),
            "accuracy": accuracy,
            "micro_f1": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "ece": float(expected_calibration_error),
        }
        aggregate.update(
            {
                f"top_{value}_accuracy": self.top_correct[value] / self.total
                if self.total
                else 0.0
                for value in self.top_k
            }
        )
        per_class = [
            {
                "class": class_index,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
            for class_index in range(self.num_classes)
        ]
        return aggregate, per_class

