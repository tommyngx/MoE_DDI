from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import resolve_dataset_tag, resolve_project_path, split_paths
from data import CsvBatchStream, discover_label_values, load_split_manifest
from losses import build_loss
from metrics import ClassificationMetrics
from models import ModelOutput, build_model, count_parameters
from plotting import (
    plot_class_distribution,
    plot_evaluation_figures,
    plot_training_history,
    plot_tsne_top10_classes,
)
from preprocessing import (
    TrainStatistics,
    compute_train_statistics,
)
from reporting import write_run_summary
from schema import DatasetSchema, assert_matching_schema, infer_schema
from utils import select_device, set_seed, write_json, write_yaml


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
    data_root = resolve_project_path(config, config["data"]["root"])
    dataset_stats_path = data_root / "train_stats.npz"

    # Direct check: if train_stats.npz exists in data folder and not forced, run directly!
    if dataset_stats_path.is_file() and not force:
        try:
            TrainStatistics.load(dataset_stats_path, schema)
            print(
                "[prepare] Found train_stats.npz in dataset folder; "
                f"running directly: {dataset_stats_path}",
                flush=True,
            )
            return dataset_stats_path
        except Exception as error:
            print(f"[prepare] train_stats.npz invalid; recalculating ({error}).", flush=True)

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
    )
    stats.save(dataset_stats_path)
    print(f"[prepare] Saved train_stats.npz to dataset folder: {dataset_stats_path}", flush=True)
    return dataset_stats_path


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
        auxiliary_logits=None,
        global_logits=None,
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
    aggregate, per_class, confusion, _, _ = evaluate_models(
        [model],
        stream,
        statistics,
        device,
        config,
        criterion=criterion,
    )
    return aggregate, per_class, confusion


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(
        dim=-1
    )


def evaluate_models(
    models: list[nn.Module],
    stream: CsvBatchStream,
    statistics: TrainStatistics,
    device: torch.device,
    config: dict,
    *,
    criterion: nn.Module | None = None,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray, np.ndarray]:
    if not models:
        raise ValueError("At least one model is required for evaluation")
    for model in models:
        model.eval()
    metrics = ClassificationMetrics(
        config["data"]["num_classes"],
        top_k=config["evaluation"].get("top_k", [1, 3, 5]),
        calibration_bins=config["evaluation"].get("calibration_bins", 15),
    )
    total_loss = 0.0
    total_samples = 0
    router_sum: np.ndarray | None = None
    uncertainty_sums = {
        "entropy": 0.0,
        "variance": 0.0,
        "mutual_information": 0.0,
        "confidence": 0.0,
    }
    thresholds = config["evaluation"].get("confidence_thresholds")
    stratum_metrics = None
    if thresholds:
        stratum_metrics = {
            name: ClassificationMetrics(
                config["data"]["num_classes"],
                top_k=config["evaluation"].get("top_k", [1, 3, 5]),
                calibration_bins=config["evaluation"].get("calibration_bins", 15),
            )
            for name in ("high", "medium", "low")
        }
    collected_embeddings = []
    collected_labels = []

    with torch.inference_mode():
        for features, labels in stream:
            transformed = statistics.transform(
                features,
                config["preprocessing"].get("normalization", "standardize"),
            )
            inputs = torch.from_numpy(transformed).to(device)
            targets = torch.from_numpy(labels).to(device)
            collected_embeddings.append(transformed)
            collected_labels.append(labels)
            outputs = [_forward(member, inputs) for member in models]
            member_probabilities = torch.stack(
                [torch.softmax(output.logits, dim=-1) for output in outputs],
                dim=0,
            )
            probabilities = member_probabilities.mean(dim=0)
            metrics.update_probabilities(probabilities, targets)
            if criterion is not None:
                # log(p) is a valid logits representation because softmax(log(p)) == p.
                loss = criterion(probabilities.clamp_min(1e-8).log(), targets)
                total_loss += float(loss.item()) * len(labels)
            total_samples += len(labels)
            router_outputs = [
                output.router_probabilities
                for output in outputs
                if output.router_probabilities.shape[1]
            ]
            if router_outputs:
                values = torch.stack(router_outputs).mean(dim=0).sum(dim=0).cpu().numpy()
                router_sum = values if router_sum is None else router_sum + values

            predictive_entropy = _entropy(probabilities)
            member_entropy = _entropy(member_probabilities)
            variance = member_probabilities.var(dim=0, unbiased=False).mean(dim=-1)
            mutual_information = predictive_entropy - member_entropy.mean(dim=0)
            confidence = 1.0 - predictive_entropy / np.log(probabilities.shape[1])
            uncertainty_sums["entropy"] += float(predictive_entropy.sum().item())
            uncertainty_sums["variance"] += float(variance.sum().item())
            uncertainty_sums["mutual_information"] += float(
                mutual_information.sum().item()
            )
            uncertainty_sums["confidence"] += float(confidence.sum().item())

            if stratum_metrics is not None:
                high = confidence >= float(thresholds["high"])
                low = confidence < float(thresholds["low"])
                masks = {"high": high, "medium": ~(high | low), "low": low}
                for name, mask in masks.items():
                    if mask.any():
                        stratum_metrics[name].update_probabilities(
                            probabilities[mask], targets[mask]
                        )

    aggregate, per_class = metrics.compute()
    aggregate["ensemble_size"] = len(models)
    if criterion is not None and total_samples:
        aggregate["loss"] = total_loss / total_samples
    if total_samples:
        aggregate.update(
            {
                f"mean_{name}": value / total_samples
                for name, value in uncertainty_sums.items()
            }
        )
    if router_sum is not None and total_samples:
        aggregate["mean_router_probability"] = (router_sum / total_samples).tolist()
        aggregate["router_family_names"] = list(getattr(models[0], "family_names", ()))
    if stratum_metrics is not None:
        aggregate["confidence_thresholds"] = {
            "low": float(thresholds["low"]),
            "high": float(thresholds["high"]),
        }
        aggregate["confidence_strata"] = {}
        for name, tracker in stratum_metrics.items():
            values, _ = tracker.compute()
            values["coverage"] = values["num_samples"] / total_samples if total_samples else 0.0
            aggregate["confidence_strata"][name] = values

    all_embeddings = (
        np.concatenate(collected_embeddings, axis=0)
        if collected_embeddings
        else np.zeros((0, schema.num_features))
    )
    all_labels = (
        np.concatenate(collected_labels, axis=0)
        if collected_labels
        else np.zeros((0,), dtype=np.int64)
    )
    return aggregate, per_class, metrics.confusion.copy(), all_embeddings, all_labels


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


def _load_tddi_backbone_weights(
    model: nn.Module,
    config: dict,
    schema: DatasetSchema,
    statistics: TrainStatistics,
) -> Path | None:
    value = config["training"].get("tddi_pretrained_checkpoint")
    if not value:
        return None
    if config["training"].get("pretrained_checkpoint"):
        raise ValueError(
            "Use either pretrained_checkpoint or tddi_pretrained_checkpoint, not both"
        )
    if not hasattr(model, "load_tddi_state_dict"):
        raise ValueError("Configured model cannot import a T-DDI numerical checkpoint")

    checkpoint_path = resolve_project_path(config, value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"T-DDI checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        if checkpoint.get("member_checkpoints"):
            raise ValueError(
                "Use one fold/member checkpoint for T-DDI initialization, not an ensemble manifest"
            )
        raise ValueError(f"Unsupported T-DDI checkpoint format: {checkpoint_path}")
    if checkpoint.get("schema_fingerprint") not in {None, schema.fingerprint}:
        raise ValueError("T-DDI checkpoint schema does not match current dataset")
    checkpoint_labels = checkpoint.get("label_values")
    if checkpoint_labels is not None and not np.array_equal(
        checkpoint_labels, statistics.label_values
    ):
        raise ValueError("T-DDI checkpoint label vocabulary does not match current dataset")
    model.load_tddi_state_dict(checkpoint["model_state"])
    print(f"[train] Initialized global backbone from T-DDI: {checkpoint_path}", flush=True)
    return checkpoint_path


def train(config: dict, *, reset_seed: bool = True) -> Path:
    if reset_seed:
        set_seed(config["seed"])
    schema = load_and_validate_schema(config)
    stats_path = prepare_statistics(config, schema)
    statistics = TrainStatistics.load(stats_path, schema)
    device = select_device(config["training"].get("device", "auto"))
    model = build_model(config, schema).to(device)
    if hasattr(model, "fit_quantiles") and not config["training"].get(
        "pretrained_checkpoint"
    ):
        quantile_stream = make_stream(
            config,
            schema,
            "train",
            max_rows=model.quantile_max_rows,
            shuffle=False,
            seed=config["seed"],
            label_values=statistics.label_values,
        )
        normalization = config["preprocessing"].get("normalization", "standardize")
        feature_batches = (
            statistics.transform(features, normalization)
            for features, _ in quantile_stream
        )
        rows_seen = model.fit_quantiles(feature_batches, seed=config["seed"])
        print(
            f"[train] Fitted BiSHop quantile bins from {rows_seen:,} training rows "
            f"(reservoir={model.quantile_sample_size:,}).",
            flush=True,
        )
    pretrained_path = _load_pretrained_weights(model, config, schema, statistics, device)
    tddi_pretrained_path = _load_tddi_backbone_weights(
        model, config, schema, statistics
    )
    criterion = build_loss(config, statistics.class_counts, device)
    training_config = config["training"]
    optimizer_parameters = model.parameters()
    backbone_lr_multiplier = float(
        training_config.get("tddi_backbone_lr_multiplier", 1.0)
    )
    if tddi_pretrained_path is not None and backbone_lr_multiplier != 1.0:
        tddi_parameters = list(model.tddi_parameters())
        tddi_parameter_ids = {id(parameter) for parameter in tddi_parameters}
        residual_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in tddi_parameter_ids
        ]
        optimizer_parameters = [
            {"params": residual_parameters},
            {
                "params": tddi_parameters,
                "lr": training_config["learning_rate"] * backbone_lr_multiplier,
            },
        ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(
            training_config.get("cosine_t_max")
            or max(1, training_config["epochs"])
        ),
    )
    use_scaler = bool(training_config.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    dataset_tag = resolve_dataset_tag(config)
    run_dir = resolve_run_dir(config)
    info_dir = run_dir / f"info_{dataset_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(info_dir / "resolved_config.yaml", config)
    schema.write_yaml(info_dir / "schema.yaml")
    write_yaml(
        info_dir / "label_mapping.yaml",
        {
            "internal_to_original": {
                str(index): int(value)
                for index, value in enumerate(statistics.label_values)
            }
        },
    )
    parameter_summary = count_parameters(model)
    freeze_tddi_epochs = 0
    if tddi_pretrained_path is not None:
        freeze_tddi_epochs = int(training_config.get("freeze_tddi_epochs", 0))
        if freeze_tddi_epochs > 0:
            model.set_tddi_trainable(False)
    plot_class_distribution(
        statistics.class_counts,
        statistics.label_values,
        run_dir,
        dataset_tag=dataset_tag,
    )
    print(
        f"[train] model={config['model']['name']} device={device} "
        f"parameters={parameter_summary['trainable']:,}",
        flush=True,
    )
    write_yaml(
        info_dir / "model_summary.yaml",
        {
            "model": config["model"]["name"],
            "parameters": parameter_summary,
            "parameters_during_initial_freeze": count_parameters(model),
            "device": str(device),
            "pretrained_checkpoint": (
                str(pretrained_path) if pretrained_path is not None else None
            ),
            "tddi_pretrained_checkpoint": (
                str(tddi_pretrained_path) if tddi_pretrained_path is not None else None
            ),
            "freeze_tddi_epochs": freeze_tddi_epochs,
            "tddi_backbone_lr_multiplier": backbone_lr_multiplier,
        },
    )

    history: list[dict] = []
    best_value = float("-inf")
    patience = 0
    best_path = run_dir / f"best_{dataset_tag}.pt"
    accumulation = training_config.get("gradient_accumulation_steps", 1)
    max_steps = training_config.get("max_steps_per_epoch")
    started = time.time()

    # A zero-initialized MoE residual makes the imported T-DDI prediction an
    # exact member of the hybrid hypothesis class.  Evaluate and checkpoint
    # that point before optimization so validation-based selection can always
    # fall back to the imported baseline if residual training is harmful.
    if tddi_pretrained_path is not None:
        initial_stream = make_stream(
            config,
            schema,
            "validation",
            max_rows=config["evaluation"].get("max_rows"),
            shuffle=False,
            seed=config["seed"],
            label_values=statistics.label_values,
        )
        initial_validation, _, _ = evaluate_model(
            model,
            initial_stream,
            statistics,
            device,
            config,
            criterion=criterion,
        )
        initial_record = {
            "epoch": 0,
            "stage": "tddi_initialization",
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate_1st": optimizer.param_groups[0]["lr"],
            "learning_rate_2nd": (
                optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None
            ),
            "train_samples": 0,
            "train_classification_loss": None,
            "train_moe_auxiliary_loss": None,
            "train_global_auxiliary_loss": None,
            "train_balance_loss": None,
            "train_router_z_loss": None,
            "validation": initial_validation,
            "elapsed_seconds": time.time() - started,
        }
        history.append(initial_record)
        write_json(info_dir / "history.json", history)
        initial_checkpoint = {
            "epoch": 0,
            "stage": "tddi_initialization",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "schema_fingerprint": schema.fingerprint,
            "label_values": statistics.label_values,
            "config": config,
            "validation": initial_validation,
            "pretrained_checkpoint": None,
            "tddi_pretrained_checkpoint": str(tddi_pretrained_path),
        }
        best_value = float(
            initial_validation[training_config["selection_metric"]]
        )
        _save_checkpoint(best_path, initial_checkpoint)
        print(
            "[epoch 0/T-DDI] "
            f"val_loss={initial_validation.get('loss', float('nan')):.4f} "
            f"val_accuracy={initial_validation['accuracy']:.4f} "
            f"val_macro_f1={initial_validation['macro_f1']:.4f}",
            flush=True,
        )

    for epoch in range(training_config["epochs"]):
        if freeze_tddi_epochs and epoch == freeze_tddi_epochs:
            model.set_tddi_trainable(True)
            print("[train] Unfroze the T-DDI global backbone.", flush=True)
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
        running = {
            "classification": 0.0,
            "moe_auxiliary": 0.0,
            "global_auxiliary": 0.0,
            "balance": 0.0,
            "router_z": 0.0,
        }
        samples = 0
        steps = 0
        for steps, (features, labels) in enumerate(stream, start=1):
            transformed = statistics.transform(
                features,
                config["preprocessing"].get("normalization", "standardize"),
            )
            inputs = torch.from_numpy(transformed).to(device)
            targets = torch.from_numpy(labels).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_scaler,
            ):
                output = _forward(model, inputs)
                classification_loss = criterion(output.logits, targets)
                moe_auxiliary_loss = output.logits.new_zeros(())
                if output.auxiliary_logits is not None:
                    moe_auxiliary_loss = criterion(output.auxiliary_logits, targets)
                global_auxiliary_loss = output.logits.new_zeros(())
                if output.global_logits is not None:
                    global_auxiliary_loss = criterion(output.global_logits, targets)
                moe_auxiliary_term = (
                    config["loss"].get("moe_auxiliary_weight", 0.0)
                    * moe_auxiliary_loss
                )
                global_auxiliary_term = (
                    config["loss"].get("global_auxiliary_weight", 0.0)
                    * global_auxiliary_loss
                )
                balance_term = (
                    config["loss"].get("moe_balance_weight", 0.0) * output.balance_loss
                )
                router_z_term = (
                    config["loss"].get("router_z_weight", 0.0) * output.router_z_loss
                )
                loss = (
                    classification_loss
                    + moe_auxiliary_term
                    + global_auxiliary_term
                    + balance_term
                    + router_z_term
                ) / accumulation
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
            running["moe_auxiliary"] += float(moe_auxiliary_loss.item()) * batch_size
            running["global_auxiliary"] += float(global_auxiliary_loss.item()) * batch_size
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
            "learning_rate_1st": optimizer.param_groups[0]["lr"],
            "learning_rate_2nd": (
                optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else None
            ),
            "train_samples": samples,
            "train_classification_loss": running["classification"] / samples,
            "train_moe_auxiliary_loss": running["moe_auxiliary"] / samples,
            "train_global_auxiliary_loss": running["global_auxiliary"] / samples,
            "train_balance_loss": running["balance"] / samples,
            "train_router_z_loss": running["router_z"] / samples,
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_record)
        write_yaml(info_dir / "history.yaml", history)
        plot_training_history(history, run_dir, epoch=epoch + 1, dataset_tag=dataset_tag)
        print(
            f"[epoch {epoch + 1}/{training_config['epochs']}] "
            f"train_loss={epoch_record['train_classification_loss']:.4f} "
            f"val_loss={validation.get('loss', float('nan')):.4f} "
            f"val_accuracy={validation['accuracy']:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )

        selection = float(validation[training_config["selection_metric"]])
        if selection > best_value:
            best_value = selection
            patience = 0
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
                "tddi_pretrained_checkpoint": (
                    str(tddi_pretrained_path) if tddi_pretrained_path is not None else None
                ),
            }
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


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> dict:
    set_seed(config["seed"])
    schema = load_and_validate_schema(config)
    statistics = TrainStatistics.load(
        resolve_project_path(config, config["data"]["stats_path"]),
        schema,
    )
    device = select_device(config["training"].get("device", "auto"))
    requested_values = (
        list(checkpoint_path)
        if isinstance(checkpoint_path, (list, tuple))
        else [checkpoint_path]
    )
    checkpoint_paths = []
    models = []
    pending = [Path(value) for value in requested_values]
    while pending:
        resolved_path = pending.pop(0)
        if not resolved_path.is_absolute():
            resolved_path = resolve_project_path(config, resolved_path)
        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
        member_values = payload.get("member_checkpoints")
        if member_values:
            for member_value in member_values:
                member_path = Path(member_value)
                if not member_path.is_absolute():
                    member_path = resolved_path.parent / member_path
                pending.append(member_path)
            continue
        if payload["schema_fingerprint"] != schema.fingerprint:
            raise ValueError(f"Checkpoint schema does not match current dataset: {resolved_path}")
        if not np.array_equal(payload["label_values"], statistics.label_values):
            raise ValueError(
                f"Checkpoint label vocabulary does not match training statistics: {resolved_path}"
            )
        checkpoint_normalization = payload.get("config", {}).get(
            "preprocessing", {}
        ).get("normalization", "standardize")
        requested_normalization = config["preprocessing"].get(
            "normalization", "standardize"
        )
        if checkpoint_normalization != requested_normalization:
            raise ValueError(
                "Checkpoint normalization mode does not match evaluation config: "
                f"{resolved_path} uses {checkpoint_normalization!r}, requested "
                f"{requested_normalization!r}"
            )
        model = build_model(config, schema).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        models.append(model)
        checkpoint_paths.append(str(resolved_path))
        del payload
    if not models:
        raise ValueError("Ensemble checkpoint contains no member checkpoints")
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
    aggregate, per_class, confusion, embeddings, labels = evaluate_models(
        models,
        stream,
        statistics,
        device,
        config,
        criterion=criterion,
    )
    for row in per_class:
        row["original_class_id"] = int(statistics.label_values[row["class"]])
    dataset_tag = resolve_dataset_tag(config)
    run_dir = resolve_run_dir(config)
    info_dir = run_dir / f"info_{dataset_tag}"
    info_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(info_dir / "test_metrics.yaml", aggregate)
    _write_per_class_csv(info_dir / "test_per_class.csv", per_class)
    np.save(info_dir / "test_confusion_matrix.npy", confusion)

    plot_class_distribution(
        statistics.class_counts,
        statistics.label_values,
        run_dir,
        dataset_tag=dataset_tag,
    )
    plot_evaluation_figures(aggregate, per_class, confusion, run_dir, dataset_tag=dataset_tag)
    plot_tsne_top10_classes(
        embeddings, labels, statistics.class_counts, run_dir, dataset_tag=dataset_tag
    )

    history_path = info_dir / "history.yaml"
    if not history_path.is_file():
        history_path = info_dir / "history.json"
    history = []
    if history_path.is_file():
        if history_path.suffix == ".yaml":
            import yaml

            history = yaml.safe_load(history_path.read_text(encoding="utf-8")) or []
        else:
            history = json.loads(history_path.read_text(encoding="utf-8"))

    summary_config = config
    resolved_config_path = info_dir / "resolved_config.yaml"
    if not resolved_config_path.is_file():
        resolved_config_path = info_dir / "resolved_config.json"
    if resolved_config_path.is_file():
        if resolved_config_path.suffix == ".yaml":
            import yaml

            summary_config = (
                yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or config
            )
        else:
            summary_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        summary_config["data"]["root"] = config["data"]["root"]
        summary_config["training"]["run_dir"] = str(run_dir)
    write_run_summary(
        run_dir,
        summary_config,
        schema,
        statistics,
        count_parameters(models[0]),
        str(device),
        history,
        test_metrics=aggregate,
    )
    return aggregate
