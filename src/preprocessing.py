from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data import CsvBatchStream
from schema import DatasetSchema


@dataclass(frozen=True)
class TrainStatistics:
    mean: np.ndarray
    std: np.ndarray
    class_counts: np.ndarray
    label_values: np.ndarray
    count: int
    schema_fingerprint: str
    cache_fingerprint: str | None = None

    def normalize(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std

    def transform(self, features: np.ndarray, mode: str = "standardize") -> np.ndarray:
        if mode == "standardize":
            return self.normalize(features)
        if mode == "none":
            return features
        raise ValueError(f"Unknown normalization mode: {mode}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32),
            class_counts=self.class_counts.astype(np.int64),
            label_values=self.label_values.astype(np.int64),
            count=np.asarray(self.count, dtype=np.int64),
            schema_fingerprint=np.asarray(self.schema_fingerprint),
            cache_fingerprint=np.asarray(self.cache_fingerprint or ""),
        )

    @classmethod
    def load(cls, path: str | Path, schema: DatasetSchema | None = None) -> TrainStatistics:
        with np.load(path, allow_pickle=False) as archive:
            result = cls(
                mean=archive["mean"].astype(np.float32),
                std=archive["std"].astype(np.float32),
                class_counts=archive["class_counts"].astype(np.int64),
                label_values=archive["label_values"].astype(np.int64),
                count=int(archive["count"]),
                schema_fingerprint=str(archive["schema_fingerprint"]),
                cache_fingerprint=(
                    str(archive["cache_fingerprint"])
                    if "cache_fingerprint" in archive.files
                    and str(archive["cache_fingerprint"])
                    else None
                ),
            )
        if schema is not None and result.schema_fingerprint != schema.fingerprint:
            raise ValueError("Statistics schema fingerprint does not match the CSV header")
        return result


def compute_train_statistics(
    stream: CsvBatchStream,
    schema: DatasetSchema,
    *,
    num_classes: int,
    min_std: float = 1e-6,
    cache_fingerprint: str | None = None,
) -> TrainStatistics:
    count = 0
    mean = np.zeros(schema.num_features, dtype=np.float64)
    m2 = np.zeros(schema.num_features, dtype=np.float64)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for features, labels in stream:
        values = features.astype(np.float64, copy=False)
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        combined = count + batch_count
        mean += delta * (batch_count / combined)
        m2 += batch_m2 + np.square(delta) * count * batch_count / combined
        count = combined
        class_counts += np.bincount(labels, minlength=num_classes)

    if count < 2:
        raise ValueError("At least two training rows are required for statistics")
    variance = m2 / count
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std < min_std, 1.0, std)
    return TrainStatistics(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        class_counts=class_counts,
        label_values=np.asarray(stream.label_values, dtype=np.int64),
        count=count,
        schema_fingerprint=schema.fingerprint,
        cache_fingerprint=cache_fingerprint,
    )
