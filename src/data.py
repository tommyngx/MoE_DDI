from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from schema import DatasetSchema

ROLE_CODE = {"train": 0, "validation": 1, "test": 2}


def _arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError as exc:
        raise RuntimeError(
            "PyArrow is required for streaming CSV input. Install the project dependencies."
        ) from exc
    return pa, pacsv


def normalize_raw_labels(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Labels contain missing or non-finite values")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, atol=1e-8):
        raise ValueError("Labels must be integral, including when serialized as floats")
    return rounded.astype(np.int64)


def encode_labels(values: np.ndarray, label_values: np.ndarray) -> np.ndarray:
    raw = normalize_raw_labels(values)
    vocabulary = np.asarray(label_values, dtype=np.int64)
    if vocabulary.ndim != 1 or len(vocabulary) == 0:
        raise ValueError("label_values must be a non-empty one-dimensional array")
    positions = np.searchsorted(vocabulary, raw)
    valid = positions < len(vocabulary)
    if valid.any():
        valid[valid] &= vocabulary[positions[valid]] == raw[valid]
    if not valid.all():
        unknown = np.unique(raw[~valid]).tolist()
        raise ValueError(f"Labels are absent from the training vocabulary: {unknown[:20]}")
    return positions.astype(np.int64)


def normalize_labels(values: np.ndarray, num_classes: int) -> np.ndarray:
    """Validate labels that are already encoded as contiguous class indices."""
    labels = normalize_raw_labels(values)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}]")
    return labels


def discover_label_values(
    paths: Sequence[str | Path],
    target_column: str,
    *,
    expected_num_classes: int | None,
    block_size_mb: int = 64,
) -> np.ndarray:
    pa, pacsv = _arrow_modules()
    values: set[int] = set()
    options = pacsv.ConvertOptions(
        include_columns=[target_column],
        column_types={target_column: pa.float64()},
    )
    for path in paths:
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=block_size_mb * 1024 * 1024),
            convert_options=options,
        )
        for batch in reader:
            raw = normalize_raw_labels(batch.column(0).to_numpy())
            values.update(raw.tolist())
    vocabulary = np.asarray(sorted(values), dtype=np.int64)
    if expected_num_classes is not None and len(vocabulary) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} unique training labels, found {len(vocabulary)}"
        )
    return vocabulary


def load_split_manifest(path: str | Path, paths: Sequence[Path], role: str) -> list[np.ndarray]:
    code = ROLE_CODE[role]
    with np.load(path, allow_pickle=False) as archive:
        file_names = [str(item) for item in archive["file_names"]]
        by_name = {
            name: archive[f"assignments_{index}"] for index, name in enumerate(file_names)
        }
        masks = []
        for file_path in paths:
            if file_path.name not in by_name:
                raise ValueError(f"Split manifest has no assignment for {file_path.name}")
            masks.append(by_name[file_path.name] == code)
    return masks


class CsvBatchStream:
    """Re-openable, bounded batch stream over one or more wide CSV files."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        schema: DatasetSchema,
        *,
        num_classes: int,
        batch_size: int,
        block_size_mb: int = 64,
        max_rows: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
        row_masks: Sequence[np.ndarray] | None = None,
        label_values: np.ndarray | None = None,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.schema = schema
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.block_size = block_size_mb * 1024 * 1024
        self.max_rows = max_rows
        self.shuffle = shuffle
        self.seed = seed
        self.row_masks = row_masks
        self.label_values = label_values
        if row_masks is not None and len(row_masks) != len(self.paths):
            raise ValueError("row_masks must align with paths")

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        pa, pacsv = _arrow_modules()
        rng = np.random.default_rng(self.seed)
        yielded = 0
        selected_columns = list(self.schema.feature_columns) + [self.schema.target_column]
        column_types = {name: pa.float32() for name in self.schema.feature_columns}
        column_types[self.schema.target_column] = pa.float64()
        convert_options = pacsv.ConvertOptions(
            include_columns=selected_columns,
            column_types=column_types,
            strings_can_be_null=True,
        )

        for file_index, path in enumerate(self.paths):
            row_offset = 0
            reader = pacsv.open_csv(
                path,
                read_options=pacsv.ReadOptions(block_size=self.block_size, use_threads=True),
                convert_options=convert_options,
            )
            mask = None if self.row_masks is None else self.row_masks[file_index]
            for record_batch in reader:
                frame = record_batch.to_pandas()
                rows_in_batch = len(frame)
                if mask is not None:
                    batch_mask = mask[row_offset : row_offset + rows_in_batch]
                    if len(batch_mask) != rows_in_batch:
                        raise ValueError(f"Manifest row count does not match {path}")
                    frame = frame.loc[batch_mask]
                row_offset += rows_in_batch
                if frame.empty:
                    continue

                features = frame.loc[:, self.schema.feature_columns].to_numpy(
                    dtype=np.float32, copy=True
                )
                raw_labels = frame.loc[:, self.schema.target_column].to_numpy()
                labels = (
                    encode_labels(raw_labels, self.label_values)
                    if self.label_values is not None
                    else normalize_labels(raw_labels, self.num_classes)
                )
                if self.shuffle:
                    order = rng.permutation(len(labels))
                    features, labels = features[order], labels[order]

                for start in range(0, len(labels), self.batch_size):
                    end = min(start + self.batch_size, len(labels))
                    if self.max_rows is not None:
                        remaining = self.max_rows - yielded
                        if remaining <= 0:
                            return
                        end = min(end, start + remaining)
                    if end <= start:
                        return
                    x_batch = np.ascontiguousarray(features[start:end])
                    y_batch = np.ascontiguousarray(labels[start:end])
                    yielded += len(y_batch)
                    yield x_batch, y_batch
                    if self.max_rows is not None and yielded >= self.max_rows:
                        return


def iter_metadata(
    paths: Sequence[str | Path],
    schema: DatasetSchema,
    *,
    block_size_mb: int = 64,
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    pa, pacsv = _arrow_modules()
    required = [
        "drugid-drug_a",
        "drugid-drug_b",
        "drugsmiles-drug_a",
        "drugsmiles-drug_b",
        schema.target_column,
    ]
    missing = [name for name in required if name not in schema.all_columns]
    if missing:
        raise ValueError(f"Split generation requires missing columns: {missing}")
    types = {name: pa.string() for name in required[:-1]}
    types[schema.target_column] = pa.float64()
    options = pacsv.ConvertOptions(include_columns=required, column_types=types)
    for file_index, path in enumerate(paths):
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=block_size_mb * 1024 * 1024),
            convert_options=options,
        )
        for batch in reader:
            frame = batch.to_pandas()
            yield file_index, {name: frame[name].to_numpy(copy=True) for name in required}
