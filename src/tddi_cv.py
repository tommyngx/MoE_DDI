from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from config import load_config, resolve_project_path, split_paths
from data import encode_labels, normalize_raw_labels
from engine import (
    _save_checkpoint,
    evaluate_checkpoint,
    load_and_validate_schema,
    train,
)
from preprocessing import TrainStatistics
from utils import set_seed, write_json


def _load_label_array(path: Path, target_column: str, block_size_mb: int) -> np.ndarray:
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError as exc:
        raise RuntimeError("PyArrow is required for streaming CSV input") from exc

    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size_mb * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(
            include_columns=[target_column],
            column_types={target_column: pa.float64()},
        ),
    )
    parts = [normalize_raw_labels(batch.column(0).to_numpy()) for batch in reader]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def stratified_fold_ids(labels: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Match sklearn StratifiedKFold's round-robin class allocation."""
    labels = np.asarray(labels, dtype=np.int64)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if n_folds > np.iinfo(np.uint8).max:
        raise ValueError("n_folds must not exceed 255")
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a non-empty one-dimensional array")
    _, first_indices, inverse = np.unique(
        labels, return_index=True, return_inverse=True
    )
    _, class_order = np.unique(first_indices, return_inverse=True)
    encoded = class_order[inverse]
    num_classes = len(first_indices)
    counts = np.bincount(encoded, minlength=num_classes)
    if counts.min() < n_folds:
        raise ValueError("Every class must have at least n_folds development samples")

    ordered = np.sort(encoded)
    allocation = np.asarray(
        [
            np.bincount(ordered[fold::n_folds], minlength=num_classes)
            for fold in range(n_folds)
        ]
    ).T
    rng = np.random.RandomState(seed)
    fold_ids = np.empty(len(labels), dtype=np.uint8)
    for class_index in range(num_classes):
        class_folds = np.arange(n_folds, dtype=np.uint8).repeat(allocation[class_index])
        rng.shuffle(class_folds)
        fold_ids[encoded == class_index] = class_folds
    return fold_ids


def _write_fold_manifest(
    path: Path,
    all_paths: list[Path],
    dev_lengths: list[int],
    fold_ids: np.ndarray,
    held_out_fold: int,
    test_length: int,
) -> None:
    assignments = []
    offset = 0
    for length in dev_lengths:
        values = fold_ids[offset : offset + length]
        assignments.append(np.where(values == held_out_fold, 1, 0).astype(np.uint8))
        offset += length
    assignments.append(np.full(test_length, 2, dtype=np.uint8))
    payload = {"file_names": np.asarray([item.name for item in all_paths])}
    payload.update(
        {f"assignments_{index}": values for index, values in enumerate(assignments)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def train_tddi_cv(config: dict, *, n_folds: int = 3) -> tuple[Path, dict]:
    if config["model"]["name"] != "tddi_mlp":
        raise ValueError("T-DDI CV requires model.name=tddi_mlp")
    if config["preprocessing"].get("normalization", "standardize") != "none":
        raise ValueError("T-DDI CV requires preprocessing.normalization=none")

    schema = load_and_validate_schema(config)
    train_paths = split_paths(config, "train")
    validation_paths = split_paths(config, "validation")
    test_paths = split_paths(config, "test")
    if len(test_paths) != 1:
        raise ValueError("T-DDI CV currently requires exactly one held-out test file")
    dev_paths = train_paths + validation_paths
    all_paths = dev_paths + test_paths
    if len({path.name for path in all_paths}) != len(all_paths):
        raise ValueError("Fold manifests require unique CSV filenames")

    raw_train_labels = [
        _load_label_array(path, schema.target_column, config["data"]["block_size_mb"])
        for path in train_paths
    ]
    label_values = np.unique(np.concatenate(raw_train_labels))
    if len(label_values) != config["data"]["num_classes"]:
        raise ValueError(
            f"Expected {config['data']['num_classes']} train classes, found {len(label_values)}"
        )
    raw_validation_labels = [
        _load_label_array(path, schema.target_column, config["data"]["block_size_mb"])
        for path in validation_paths
    ]
    raw_dev_labels = raw_train_labels + raw_validation_labels
    dev_labels = np.concatenate(
        [encode_labels(values, label_values) for values in raw_dev_labels]
    )
    test_length = len(
        _load_label_array(
            test_paths[0], schema.target_column, config["data"]["block_size_mb"]
        )
    )
    fold_ids = stratified_fold_ids(dev_labels, n_folds, config["seed"])

    root = Path(config["training"]["run_dir"])
    if not root.is_absolute():
        root = resolve_project_path(config, root)
    info_dir = root / "info"
    manifest_dir = info_dir / "fold_manifests"
    root.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"])
    best_paths = []
    last_paths = []
    fold_report = []
    first_stats_path = None
    for fold in range(n_folds):
        fold_number = fold + 1
        fold_dir = root / "folds" / f"fold_{fold_number}"
        manifest_path = manifest_dir / f"fold_{fold_number}.npz"
        _write_fold_manifest(
            manifest_path,
            all_paths,
            [len(values) for values in raw_dev_labels],
            fold_ids,
            fold,
            test_length,
        )

        fold_config = copy.deepcopy(config)
        fold_config["run_name"] = f"{config['run_name']}_fold{fold_number}"
        fold_config["data"]["split_manifest"] = str(manifest_path)
        fold_config["data"]["stats_path"] = str(fold_dir / "info" / "train_stats.npz")
        fold_config["training"]["run_dir"] = str(fold_dir)

        train_mask = fold_ids != fold
        fold_stats = TrainStatistics(
            mean=np.zeros(schema.num_features, dtype=np.float32),
            std=np.ones(schema.num_features, dtype=np.float32),
            class_counts=np.bincount(
                dev_labels[train_mask], minlength=config["data"]["num_classes"]
            ),
            label_values=label_values,
            count=int(train_mask.sum()),
            schema_fingerprint=schema.fingerprint,
        )
        fold_stats.save(fold_config["data"]["stats_path"])
        if first_stats_path is None:
            first_stats_path = Path(fold_config["data"]["stats_path"])

        best_path = train(fold_config, reset_seed=False)
        best_paths.append(best_path)
        last_paths.append(fold_dir / "last.pt")
        fold_report.append(
            {
                "fold": fold_number,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int((~train_mask).sum()),
                "manifest": str(manifest_path),
                "best_checkpoint": str(best_path),
            }
        )

    ensemble_payload = {
        "checkpoint_type": "probability_ensemble",
        "member_checkpoints": [str(path.relative_to(root)) for path in best_paths],
        "schema_fingerprint": schema.fingerprint,
        "label_values": label_values,
        "config": config,
    }
    ensemble_path = root / "best.pt"
    _save_checkpoint(ensemble_path, ensemble_payload)
    _save_checkpoint(
        root / "last.pt",
        {
            **ensemble_payload,
            "member_checkpoints": [str(path.relative_to(root)) for path in last_paths],
        },
    )

    report = {
        "protocol": "stratified_kfold_train_plus_validation_probability_ensemble",
        "n_folds": n_folds,
        "seed": config["seed"],
        "development_rows": int(len(dev_labels)),
        "test_rows": int(test_length),
        "folds": fold_report,
        "ensemble_checkpoint": str(ensemble_path),
    }
    write_json(info_dir / "cv_summary.json", report)
    write_json(info_dir / "resolved_config.json", config)

    evaluation_config = copy.deepcopy(config)
    evaluation_config["data"]["split_manifest"] = None
    evaluation_config["data"]["stats_path"] = str(first_stats_path)
    evaluation_config["training"]["run_dir"] = str(root)
    metrics = evaluate_checkpoint(evaluation_config, ensemble_path)
    return ensemble_path, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the released T-DDI 3-fold probability-ensemble protocol"
    )
    parser.add_argument("--config", default="configs/tddi.yaml")
    parser.add_argument("--data", "--input-dir", dest="data_dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.data_dir:
        config["data"]["root"] = str(Path(args.data_dir).expanduser().resolve())
    if args.run_dir:
        config["training"]["run_dir"] = str(Path(args.run_dir).expanduser().resolve())
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.max_train_rows is not None:
        config["training"]["max_rows"] = args.max_train_rows
    if args.max_eval_rows is not None:
        config["evaluation"]["max_rows"] = args.max_eval_rows
    if args.device is not None:
        config["training"]["device"] = args.device
    checkpoint, metrics = train_tddi_cv(config, n_folds=args.n_folds)
    print(json.dumps({"checkpoint": str(checkpoint), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
