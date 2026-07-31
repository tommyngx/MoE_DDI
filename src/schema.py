from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_METADATA_COLUMNS = (
    "drugid-drug_a",
    "drugid-drug_b",
    "drugname-drug_a",
    "drugname-drug_b",
    "drugsmiles-drug_a",
    "drugsmiles-drug_b",
)

FAMILY_PATTERNS = {
    "PEOE_VSA": re.compile(r"peoe_?vsa", re.IGNORECASE),
    "VSA_EState": re.compile(r"vsa_?estate", re.IGNORECASE),
    "EState_VSA": re.compile(r"estate_?vsa", re.IGNORECASE),
    "SlogP_VSA": re.compile(r"slogp_?vsa", re.IGNORECASE),
    "LabuteASA": re.compile(r"labute_?asa", re.IGNORECASE),
    "MTPSA": re.compile(r"mtpsa", re.IGNORECASE),
    "MR_VSA": re.compile(r"mr_?vsa", re.IGNORECASE),
}


@dataclass(frozen=True)
class DatasetSchema:
    all_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    target_column: str
    family_indices: dict[str, tuple[int, ...]]
    feature_families: tuple[tuple[str, ...], ...]
    fingerprint: str

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def read_header(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.reader(handle))
    return [item.strip() for item in row]


def infer_families(feature_name: str) -> tuple[str, ...]:
    families = tuple(
        family for family, pattern in FAMILY_PATTERNS.items() if pattern.search(feature_name)
    )
    return families or ("Other",)


def infer_schema(
    path: str | Path,
    target_column: str = "class",
    expected_num_features: int | None = 3780,
) -> DatasetSchema:
    columns = read_header(path)
    if len(columns) != len(set(columns)):
        duplicates = sorted({name for name in columns if columns.count(name) > 1})
        raise ValueError(f"Duplicate CSV columns are not supported: {duplicates[:10]}")
    if target_column not in columns:
        raise ValueError(f"Target column {target_column!r} not found in {path}")

    metadata = tuple(name for name in DEFAULT_METADATA_COLUMNS if name in columns)
    excluded = set(metadata) | {target_column}
    features = tuple(name for name in columns if name not in excluded)
    if expected_num_features is not None and len(features) != expected_num_features:
        raise ValueError(
            f"Expected {expected_num_features} feature columns, found {len(features)} in {path}"
        )

    feature_families = tuple(infer_families(name) for name in features)
    family_indices: dict[str, list[int]] = {}
    for index, families in enumerate(feature_families):
        for family in families:
            family_indices.setdefault(family, []).append(index)
    frozen_indices = {key: tuple(value) for key, value in sorted(family_indices.items())}
    fingerprint = hashlib.sha256("\n".join(columns).encode()).hexdigest()
    return DatasetSchema(
        all_columns=tuple(columns),
        feature_columns=features,
        metadata_columns=metadata,
        target_column=target_column,
        family_indices=frozen_indices,
        feature_families=feature_families,
        fingerprint=fingerprint,
    )


def assert_matching_schema(reference: DatasetSchema, path: str | Path) -> None:
    candidate_columns = tuple(read_header(path))
    if candidate_columns != reference.all_columns:
        raise ValueError(f"Column order or names differ from the training schema: {path}")


def family_summary(schema: DatasetSchema) -> dict[str, int]:
    return {family: len(indices) for family, indices in schema.family_indices.items()}

