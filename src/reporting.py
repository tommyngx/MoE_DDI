from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from preprocessing import TrainStatistics
from schema import DatasetSchema


def _section(lines: list[str], title: str) -> None:
    lines.extend(["", f"[{title}]", "-" * (len(title) + 2)])


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def write_run_summary(
    run_dir: str | Path,
    config: dict,
    schema: DatasetSchema,
    statistics: TrainStatistics,
    parameter_summary: dict[str, int],
    device: str,
    history: list[dict],
    *,
    test_metrics: dict | None = None,
) -> Path:
    """Write a paper-ready plain-text summary of setup, training, and results."""
    run_dir = Path(run_dir)
    lines = [
        "MoEDDI EXPERIMENT SUMMARY",
        "=" * 80,
        f"Generated UTC: {datetime.now(UTC).isoformat()}",
    ]

    _section(lines, "Run")
    lines.extend(
        [
            f"Run name: {config['run_name']}",
            f"Run directory: {run_dir}",
            f"Seed: {config['seed']}",
            f"Device: {device}",
        ]
    )

    data_config = config["data"]
    _section(lines, "Data")
    lines.extend(
        [
            f"Input directory: {data_config['root']}",
            f"Train files: {_format_value(data_config['train_files'])}",
            f"Validation files: {_format_value(data_config['validation_files'])}",
            f"Test files: {_format_value(data_config['test_files'])}",
            f"Feature count: {schema.num_features}",
            f"Class count: {data_config['num_classes']}",
            f"Rows used for train statistics: {statistics.count}",
            f"Schema fingerprint: {schema.fingerprint}",
            f"Original class-ID range: {statistics.label_values.min()}–"
            f"{statistics.label_values.max()}",
        ]
    )

    _section(lines, "Model")
    lines.extend(
        [
            f"Model: {config['model']['name']}",
            f"Total parameters: {parameter_summary['total']:,}",
            f"Trainable parameters: {parameter_summary['trainable']:,}",
        ]
    )
    for key, value in config["model"].items():
        if key != "name":
            lines.append(f"{key}: {_format_value(value)}")

    _section(lines, "Optimization")
    lines.append("Optimizer: AdamW")
    for key, value in config["training"].items():
        lines.append(f"{key}: {_format_value(value)}")
    for key, value in config["loss"].items():
        lines.append(f"loss.{key}: {_format_value(value)}")
    lines.append(f"batch_size: {data_config['batch_size']}")

    if history:
        selection_metric = config["training"]["selection_metric"]
        best = max(history, key=lambda row: float(row["validation"][selection_metric]))
        last = history[-1]
        _section(lines, "Training and validation result")
        lines.extend(
            [
                f"Completed epochs: {len(history)}",
                f"Best epoch: {best['epoch']}",
                f"Best validation {selection_metric}: "
                f"{best['validation'][selection_metric]:.8f}",
                f"Best validation accuracy: {best['validation']['accuracy']:.8f}",
                f"Best validation loss: {best['validation'].get('loss', float('nan')):.8f}",
                f"Last train loss: {last['train_classification_loss']:.8f}",
                f"Last validation loss: "
                f"{last['validation'].get('loss', float('nan')):.8f}",
                f"Elapsed seconds: {last['elapsed_seconds']:.2f}",
            ]
        )

    if test_metrics:
        _section(lines, "Held-out test result")
        preferred_order = [
            "num_samples",
            "loss",
            "accuracy",
            "micro_f1",
            "macro_f1",
            "weighted_f1",
            "top_1_accuracy",
            "top_3_accuracy",
            "top_5_accuracy",
            "ece",
        ]
        for key in preferred_order:
            if key in test_metrics:
                lines.append(f"{key}: {_format_value(test_metrics[key])}")

    _section(lines, "Artifacts")
    lines.extend(
        [
            f"Best checkpoint: {run_dir / 'best.pt'}",
            f"Last checkpoint: {run_dir / 'last.pt'}",
            f"Configuration: {run_dir / 'info' / 'resolved_config.json'}",
            f"Training history: {run_dir / 'info' / 'history.json'}",
            f"Test metrics: {run_dir / 'info' / 'test_metrics.json'}",
            f"Plots: {run_dir / 'plots'}",
        ]
    )
    lines.extend(
        [
            "",
            "Note: bounded smoke/debug runs must not be reported as full-dataset results.",
            "",
        ]
    )

    path = run_dir / "summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)
    return path
