from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import resolve_project_path, split_paths
from data import CsvBatchStream, discover_label_values, load_split_manifest
from losses import build_loss
from metrics import ClassificationMetrics
from models import ModelOutput, build_model, count_parameters
from plotting import (
    plot_class_distribution,
    plot_evaluation_figures,
    plot_training_history,
)
from preprocessing import (
    TrainStatistics,
    compute_train_statistics,
)
from reporting import write_run_summary
from schema import DatasetSchema, assert_matching_schema, infer_schema
from utils import select_device, set_seed, write_json


def load_and_validate_schema(config: dict) -> DatasetSchema:
    train = split_paths(config, "train")
    schema = infer_schema(
        train[0],
        target_column=config["data"]["target_column"],
        expected_num_features=config["data"]["expected_num_features"],
    )
    for role in ("train", "validation", "test"):
        for path in split_paths(config, role):
            assert_matching_schema(schema, path)
    return schema


def _all_paths(config: dict) -> list[Path]:
    seen: set[Path] = set()
    result = []
    for role in ("train", "validation", "test"):
        for path in split_paths(config, role):
            if path not in seen:
                seen.add(path)
                result.append(path)
    return result


def make_stream(
    config: dict,
    schema: DatasetSchema,
    role: str,
    *,
    max_rows: int | None,
    shuffle: bool,
    seed: int,
    label_values: np.ndarray | None = None,
) -> CsvBatchStream:
    manifest_value = config["data"].get("split_manifest")
    if manifest_value:
        paths = _all_paths(config)
        manifest = resolve_project_path(config, manifest_value)
        masks = load_split_manifest(manifest, paths, role)
    else:
        paths = split_paths(config, role)
        masks = None
    return CsvBatchStream(
        paths,
        schema,
        num_classes=config["data"]["num_classes"],
        batch_size=config["data"]["batch_size"],
        block_size_mb=config["data"]["block_size_mb"],
        max_rows=max_rows,
        shuffle=shuffle,
        seed=seed,
        row_masks=masks,
        label_values=label_values,
    )


def _statistics_cache_fingerprint(config: dict, schema: DatasetSchema) -> str:
    manifest_value = config["data"].get("split_manifest")
    paths = _all_paths(config) if manifest_value else split_paths(config, "train")

    def identity(path: Path) -> dict:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    payload = {
        "schema": schema.fingerprint,
        "files": [identity(path) for path in paths],
        "num_classes": config["data"]["num_classes"],
        "max_rows": config["preprocessing"].get("max_rows"),
        "min_std": config["preprocessing"].get("min_std", 1e-6),
        "split_manifest": (
            identity(resolve_project_path(config, manifest_value))
            if manifest_value
            else None
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prepare_statistics(config: dict, schema: DatasetSchema, *, force: bool = False) -> Path:
    stats_path = resolve_project_path(config, config["data"]["stats_path"])
    cache_fingerprint = _statistics_cache_fingerprint(config, schema)
    if stats_path.exists() and not force:
        try:
            cached_statistics = TrainStatistics.load(stats_path, schema)
            if cached_statistics.cache_fingerprint in {None, cache_fingerprint}:
                print(f"[prepare] Reusing training statistics: {stats_path}", flush=True)
                return stats_path
            print(
                "[prepare] Dataset or preprocessing settings changed; rebuilding cache.",
                flush=True,
            )
        except (KeyError, OSError, ValueError) as error:
            print(f"[prepare] Statistics cache is invalid; rebuilding ({error}).", flush=True)
    if not stats_path.exists() and not force:
        for fallback_value in config["data"].get("stats_fallback_paths", []):
            fallback_path = resolve_project_path(config, fallback_value)
            if fallback_path == stats_path or not fallback_path.is_file():
                continue
            try:
                cached_statistics = TrainStatistics.load(fallback_path, schema)
            except (KeyError, OSError, ValueError):
                continue
            if cached_statistics.cache_fingerprint not in {None, cache_fingerprint}:
                continue
            cached_statistics = replace(
                cached_statistics,
                cache_fingerprint=cache_fingerprint,
            )
            cached_statistics.save(stats_path)
            print(
                f"[prepare] Migrated reusable statistics cache: {fallback_path} -> {stats_path}",
                flush=True,
            )
            return stats_path
    print("[prepare] Discovering the training label vocabulary...", flush=True)
    label_values = discover_label_values(
        split_paths(config, "train"),
        schema.target_column,
        expected_num_classes=config["data"]["num_classes"],
        block_size_mb=config["data"]["block_size_mb"],
    )
    stream = make_stream(
        config,
        schema,
        "train",
        max_rows=config["preprocessing"].get("max_rows"),
        shuffle=False,
        seed=config["seed"],
        label_values=label_values,
    )
    print(
        f"[prepare] Computing feature statistics for {len(label_values)} classes...",
        flush=True,
    )
    stats = compute_train_statistics(
        stream,
        schema,
        num_classes=config["data"]["num_classes"],
        min_std=config["preprocessing"].get("min_std", 1e-6),
        cache_fingerprint=cache_fingerprint,
    )
    stats.save(stats_path)
    print(f"[prepare] Saved statistics: {stats_path}", flush=True)
    return stats_path


def _forward(model: nn.Module, inputs: torch.Tensor) -> ModelOutput:
    raw_output = model(inputs)
    if isinstance(raw_output, ModelOutput):
        return raw_output
    zero = raw_output.new_zeros(())
    empty_router = raw_output.new_zeros((raw_output.shape[0], 0))
    return ModelOutput(
        logits=raw_output,
        balance_loss=zero,
        router_z_loss=zero,
        router_probabilities=empty_router,
    )


def evaluate_model(
    model: nn.Module,
    stream: CsvBatchStream,
    statistics: TrainStatistics,
    device: torch.device,
    config: dict,
    *,
    criterion: nn.Module | None = None,
) -> tuple[dict, list[dict], np.ndarray]:
    model.eval()
    metrics = ClassificationMetrics(
        config["data"]["num_classes"],
        top_k=config["evaluation"].get("top_k", [1, 3, 5]),
        calibration_bins=config["evaluation"].get("calibration_bins", 15),
    )
    total_loss = 0.0
    total_samples = 0
    router_sum: np.ndarray | None = None
    with torch.inference_mode():
        for features, labels in stream:
            normalized = statistics.normalize(features)
            inputs = torch.from_numpy(normalized).to(device)
            targets = torch.from_numpy(labels).to(device)
            output = _forward(model, inputs)
            metrics.update(output.logits, targets)
            if criterion is not None:
                loss = criterion(output.logits, targets)
                total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
            if output.router_probabilities.shape[1]:
                values = output.router_probabilities.sum(dim=0).cpu().numpy()
                router_sum = values if router_sum is None else router_sum + values

    aggregate, per_class = metrics.compute()
    if criterion is not None and total_samples:
        aggregate["loss"] = total_loss / total_samples
    if router_sum is not None and total_samples:
        aggregate["mean_router_probability"] = (router_sum / total_samples).tolist()
        aggregate["router_family_names"] = list(getattr(model, "family_names", ()))
    return aggregate, per_class, metrics.confusion.copy()


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_per_class_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "original_class_id",
                "precision",
                "recall",
                "f1",
                "support",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_run_dir(config: dict) -> Path:
    training_config = config["training"]
    value = training_config.get("run_dir") or training_config.get("output_dir")
    if not value:
        value = f"runs/{config['run_name']}"
    return resolve_project_path(config, value)


def _load_pretrained_weights(
    model: nn.Module,
    config: dict,
    schema: DatasetSchema,
    statistics: TrainStatistics,
    device: torch.device,
) -> Path | None:
    value = config["training"].get("pretrained_checkpoint")
    if not value:
        return None

    checkpoint_path = resolve_project_path(config, value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported pretrained checkpoint format: {checkpoint_path}")

    # Project checkpoints contain model_state and validation metadata. A plain
    # PyTorch state_dict is also accepted for interoperability.
    if "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
        checkpoint_fingerprint = checkpoint.get("schema_fingerprint")
        if checkpoint_fingerprint is not None and checkpoint_fingerprint != schema.fingerprint:
            raise ValueError("Pretrained checkpoint schema does not match current dataset")
        checkpoint_labels = checkpoint.get("label_values")
        if checkpoint_labels is not None and not np.array_equal(
            checkpoint_labels, statistics.label_values
        ):
            raise ValueError(
                "Pretrained checkpoint label vocabulary does not match training statistics"
            )
    else:
        model_state = checkpoint

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise ValueError(
            "Pretrained weights are incompatible with the configured model architecture: "
            f"{checkpoint_path}"
        ) from error
    print(f"[train] Loaded pretrained weights: {checkpoint_path}", flush=True)
    return checkpoint_path


def train(config: dict) -> Path:
    set_seed(config["seed"])
    schema = load_and_validate_schema(config)
    stats_path = prepare_statistics(config, schema)
    statistics = TrainStatistics.load(stats_path, schema)
    device = select_device(config["training"].get("device", "auto"))
    model = build_model(config, schema).to(device)
    pretrained_path = _load_pretrained_weights(model, config, schema, statistics, device)
    criterion = build_loss(config, statistics.class_counts, device)
    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_config["epochs"]),
    )
    use_scaler = bool(training_config.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    run_dir = resolve_run_dir(config)
    info_dir = run_dir / "info"
    run_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    write_json(info_dir / "resolved_config.json", config)
    schema.write_json(info_dir / "schema.json")
    write_json(
        info_dir / "label_mapping.json",
        {
            "internal_to_original": {
                str(index): int(value)
                for index, value in enumerate(statistics.label_values)
            }
        },
    )
    parameter_summary = count_parameters(model)
    plot_class_distribution(
        statistics.class_counts,
        statistics.label_values,
        run_dir,
    )
    print(
        f"[train] model={config['model']['name']} device={device} "
        f"parameters={parameter_summary['trainable']:,}",
        flush=True,
    )
    write_json(
        info_dir / "model_summary.json",
        {
            "model": config["model"]["name"],
            "parameters": count_parameters(model),
            "device": str(device),
            "pretrained_checkpoint": (
                str(pretrained_path) if pretrained_path is not None else None
            ),
        },
    )

    history: list[dict] = []
    best_value = float("-inf")
    patience = 0
    best_path = run_dir / "best.pt"
    accumulation = training_config.get("gradient_accumulation_steps", 1)
    max_steps = training_config.get("max_steps_per_epoch")
    started = time.time()

    for epoch in range(training_config["epochs"]):
        model.train()
        stream = make_stream(
            config,
            schema,
            "train",
            max_rows=training_config.get("max_rows"),
            shuffle=True,
            seed=config["seed"] + epoch,
            label_values=statistics.label_values,
        )
        optimizer.zero_grad(set_to_none=True)
        running = {"classification": 0.0, "balance": 0.0, "router_z": 0.0}
        samples = 0
        steps = 0
        for steps, (features, labels) in enumerate(stream, start=1):
            normalized = statistics.normalize(features)
            inputs = torch.from_numpy(normalized).to(device)
            targets = torch.from_numpy(labels).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_scaler,
            ):
                output = _forward(model, inputs)
                classification_loss = criterion(output.logits, targets)
                balance_term = (
                    config["loss"].get("moe_balance_weight", 0.0) * output.balance_loss
                )
                router_z_term = (
                    config["loss"].get("router_z_weight", 0.0) * output.router_z_loss
                )
                loss = (classification_loss + balance_term + router_z_term) / accumulation
            scaler.scale(loss).backward()
            if steps % accumulation == 0:
                if training_config.get("gradient_clip_norm") is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        training_config["gradient_clip_norm"],
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(labels)
            samples += batch_size
            running["classification"] += float(classification_loss.item()) * batch_size
            running["balance"] += float(output.balance_loss.item()) * batch_size
            running["router_z"] += float(output.router_z_loss.item()) * batch_size
            if max_steps is not None and steps >= max_steps:
                break

        if steps and steps % accumulation:
            if training_config.get("gradient_clip_norm") is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config["gradient_clip_norm"]
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if samples == 0:
            raise RuntimeError("Training stream produced no samples")

        validation_stream = make_stream(
            config,
            schema,
            "validation",
            max_rows=config["evaluation"].get("max_rows"),
            shuffle=False,
            seed=config["seed"],
            label_values=statistics.label_values,
        )
        validation, _, _ = evaluate_model(
            model,
            validation_stream,
            statistics,
            device,
            config,
            criterion=criterion,
        )
        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_samples": samples,
            "train_classification_loss": running["classification"] / samples,
            "train_balance_loss": running["balance"] / samples,
            "train_router_z_loss": running["router_z"] / samples,
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_record)
        write_json(info_dir / "history.json", history)
        plot_training_history(history, run_dir, epoch=epoch + 1)
        print(
            f"[epoch {epoch + 1}/{training_config['epochs']}] "
            f"train_loss={epoch_record['train_classification_loss']:.4f} "
            f"val_loss={validation.get('loss', float('nan')):.4f} "
            f"val_accuracy={validation['accuracy']:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "schema_fingerprint": schema.fingerprint,
            "label_values": statistics.label_values,
            "config": config,
            "validation": validation,
            "pretrained_checkpoint": (
                str(pretrained_path) if pretrained_path is not None else None
            ),
        }
        _save_checkpoint(run_dir / "last.pt", checkpoint)
        selection = float(validation[training_config["selection_metric"]])
        if selection > best_value:
            best_value = selection
            patience = 0
            _save_checkpoint(best_path, checkpoint)
        else:
            patience += 1
        write_run_summary(
            run_dir,
            config,
            schema,
            statistics,
            parameter_summary,
            str(device),
            history,
        )
        scheduler.step()
        if patience >= training_config["early_stopping_patience"]:
            break
    return best_path


def evaluate_checkpoint(config: dict, checkpoint_path: str | Path) -> dict:
    set_seed(config["seed"])
    schema = load_and_validate_schema(config)
    statistics = TrainStatistics.load(
        resolve_project_path(config, config["data"]["stats_path"]),
        schema,
    )
    device = select_device(config["training"].get("device", "auto"))
    model = build_model(config, schema).to(device)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_project_path(config, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["schema_fingerprint"] != schema.fingerprint:
        raise ValueError("Checkpoint schema does not match current dataset")
    if not np.array_equal(checkpoint["label_values"], statistics.label_values):
        raise ValueError("Checkpoint label vocabulary does not match training statistics")
    model.load_state_dict(checkpoint["model_state"])
    criterion = build_loss(config, statistics.class_counts, device)
    stream = make_stream(
        config,
        schema,
        "test",
        max_rows=config["evaluation"].get("max_rows"),
        shuffle=False,
        seed=config["seed"],
        label_values=statistics.label_values,
    )
    aggregate, per_class, confusion = evaluate_model(
        model,
        stream,
        statistics,
        device,
        config,
        criterion=criterion,
    )
    for row in per_class:
        row["original_class_id"] = int(statistics.label_values[row["class"]])
    info_dir = resolve_run_dir(config) / "info"
    write_json(info_dir / "test_metrics.json", aggregate)
    _write_per_class_csv(info_dir / "test_per_class.csv", per_class)
    np.save(info_dir / "test_confusion_matrix.npy", confusion)
    run_dir = resolve_run_dir(config)
    plot_class_distribution(statistics.class_counts, statistics.label_values, run_dir)
    plot_evaluation_figures(aggregate, per_class, confusion, run_dir)
    history_path = info_dir / "history.json"
    history = []
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    summary_config = config
    resolved_config_path = info_dir / "resolved_config.json"
    if resolved_config_path.is_file():
        summary_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        summary_config["data"]["root"] = config["data"]["root"]
        summary_config["training"]["run_dir"] = str(run_dir)
    write_run_summary(
        run_dir,
        summary_config,
        schema,
        statistics,
        count_parameters(model),
        str(device),
        history,
        test_metrics=aggregate,
    )
    return aggregate
