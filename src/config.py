from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 2026,
    "run_name": "moeddi",
    "data": {
        "root": "Dataset",
        "train_files": ["train_extracted.csv"],
        "validation_files": ["validation_extracted.csv"],
        "test_files": ["test_extracted.csv"],
        "target_column": "class",
        "expected_num_features": 3780,
        "num_classes": 178,
        "block_size_mb": 64,
        "batch_size": 256,
        "stats_path": "runs/_cache/train_stats_full.npz",
        "split_manifest": None,
    },
    "preprocessing": {
        "max_rows": None,
        "min_std": 1e-6,
        "normalization": "standardize",
    },
    "model": {"name": "moeddi"},
    "loss": {
        "name": "focal",
        "gamma": 2.0,
        "class_weighting": "none",
        "effective_beta": 0.9999,
        "label_smoothing": 0.0,
        "moe_auxiliary_weight": 0.0,
        "global_auxiliary_weight": 0.0,
        "moe_balance_weight": 0.01,
        "router_z_weight": 0.001,
    },
    "training": {
        "epochs": 50,
        "max_rows": None,
        "max_steps_per_epoch": None,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "pretrained_checkpoint": None,
        "tddi_pretrained_checkpoint": None,
        "freeze_tddi_epochs": 0,
        "tddi_backbone_lr_multiplier": 1.0,
        "gradient_clip_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "mixed_precision": True,
        "cosine_t_max": None,
        "early_stopping_patience": 8,
        "selection_metric": "macro_f1",
        "device": "auto",
        "run_dir": "runs/moeddi",
    },
    "evaluation": {
        "max_rows": None,
        "top_k": [1, 3, 5],
        "calibration_bins": 15,
        "confidence_thresholds": None,
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    config = _deep_merge(DEFAULT_CONFIG, user_config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent)
    validate_config(config)
    return config


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def shared_stats_path(config: dict[str, Any]) -> Path:
    """Return a reusable cache path without writing into the raw Dataset folder."""
    max_rows = config.get("preprocessing", {}).get("max_rows")
    scope = "full" if max_rows is None else f"rows_{max_rows}"
    return Path(config["_project_root"]) / "runs" / "_cache" / f"train_stats_{scope}.npz"


def split_paths(config: dict[str, Any], role: str) -> list[Path]:
    key = {"train": "train_files", "validation": "validation_files", "test": "test_files"}[role]
    data_root = resolve_project_path(config, config["data"]["root"])
    paths = [data_root / item for item in config["data"][key]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {role} data file(s): {missing}")
    return paths


def validate_config(config: dict[str, Any]) -> None:
    if config["data"]["num_classes"] < 2:
        raise ValueError("data.num_classes must be at least 2")
    if config["data"]["batch_size"] < 1:
        raise ValueError("data.batch_size must be positive")
    if config["model"]["name"] not in {"linear", "mlp", "tddi_mlp", "moeddi"}:
        raise ValueError(f"Unsupported model.name: {config['model']['name']}")
    if config["loss"]["name"] not in {"cross_entropy", "focal"}:
        raise ValueError(f"Unsupported loss.name: {config['loss']['name']}")
    if config["loss"]["class_weighting"] not in {
        "none",
        "inverse",
        "inverse_sqrt",
        "effective_number",
    }:
        raise ValueError("Unsupported loss.class_weighting")
    if config["preprocessing"].get("normalization", "standardize") not in {
        "none",
        "standardize",
    }:
        raise ValueError("preprocessing.normalization must be 'none' or 'standardize'")
    cosine_t_max = config["training"].get("cosine_t_max")
    if cosine_t_max is not None and cosine_t_max < 1:
        raise ValueError("training.cosine_t_max must be positive or null")
    if config["training"].get("freeze_tddi_epochs", 0) < 0:
        raise ValueError("training.freeze_tddi_epochs must not be negative")
    if config["training"].get("tddi_backbone_lr_multiplier", 1.0) <= 0:
        raise ValueError("training.tddi_backbone_lr_multiplier must be positive")
    ratios = config.get("split", {}).get("ratios")
    if ratios is not None and abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split.ratios must sum to 1")
