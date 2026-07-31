from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def _series(history: list[dict], key: str, nested: str | None = None) -> list[float]:
    if nested is None:
        return [float(row[key]) for row in history]
    return [float(row[key][nested]) for row in history]


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
    os.replace(temporary, path)


def plot_training_history(history: list[dict], run_dir: str | Path, *, epoch: int) -> None:
    """Update the main training plot and save one immutable snapshot per epoch."""
    if not history:
        return
    epochs = _series(history, "epoch")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, _series(history, "train_classification_loss"), label="train")
    axes[0, 0].plot(epochs, _series(history, "validation", "loss"), label="validation")
    axes[0, 0].set_title("Classification loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, _series(history, "validation", "accuracy"), label="accuracy")
    axes[0, 1].plot(epochs, _series(history, "validation", "macro_f1"), label="macro-F1")
    axes[0, 1].set_title("Validation metrics")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, _series(history, "train_balance_loss"), label="balance")
    axes[1, 0].plot(epochs, _series(history, "train_router_z_loss"), label="router z")
    axes[1, 0].set_title("MoE auxiliary losses")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, _series(history, "learning_rate"), label="learning rate")
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.suptitle("MoEDDI training progress")
    figure.tight_layout()

    plot_dir = Path(run_dir) / "plots"
    _save_figure(figure, plot_dir / "training_curves.png")
    _save_figure(figure, plot_dir / "epochs" / f"epoch_{epoch:03d}.png")
    plt.close(figure)


def plot_class_distribution(
    class_counts: np.ndarray,
    label_values: np.ndarray,
    run_dir: str | Path,
) -> None:
    order = np.argsort(class_counts)[::-1]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(np.arange(len(order)), class_counts[order], width=1.0)
    axis.set_yscale("log")
    axis.set_xlabel("Classes sorted by training frequency")
    axis.set_ylabel("Training samples (log scale)")
    axis.set_title("Long-tail class distribution")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.99,
        0.96,
        f"Most frequent raw ID: {int(label_values[order[0]])}\n"
        f"Least frequent raw ID: {int(label_values[order[-1]])}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    _save_figure(figure, Path(run_dir) / "plots" / "paper" / "class_distribution.png")
    plt.close(figure)


def plot_evaluation_figures(
    aggregate: dict,
    per_class: list[dict],
    confusion: np.ndarray,
    run_dir: str | Path,
) -> None:
    paper_dir = Path(run_dir) / "plots" / "paper"

    support = np.asarray([row["support"] for row in per_class], dtype=np.float64)
    f1 = np.asarray([row["f1"] for row in per_class], dtype=np.float64)
    observed = support > 0
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        support[observed],
        f1[observed],
        c=f1[observed],
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Test support per class (log scale)")
    axis.set_ylabel("Per-class F1")
    axis.set_title("Long-tail performance: F1 versus class support")
    axis.grid(alpha=0.25)
    figure.colorbar(scatter, ax=axis, label="F1")
    _save_figure(figure, paper_dir / "per_class_f1_vs_support.png")
    plt.close(figure)

    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals > 0,
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xlabel("Predicted internal class")
    axis.set_ylabel("True internal class")
    axis.set_title("Row-normalized test confusion matrix")
    figure.colorbar(image, ax=axis, label="Fraction")
    _save_figure(figure, paper_dir / "confusion_matrix_normalized.png")
    plt.close(figure)

    metric_keys = [
        key
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "top_3_accuracy",
            "top_5_accuracy",
        )
        if key in aggregate
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    values = [float(aggregate[key]) for key in metric_keys]
    bars = axis.bar(metric_keys, values)
    axis.bar_label(bars, fmt="%.3f")
    axis.set_ylim(0, max(1.0, max(values, default=1.0) * 1.15))
    axis.set_ylabel("Score")
    axis.set_title("Held-out test metrics")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, paper_dir / "test_metrics.png")
    plt.close(figure)

    router_values = aggregate.get("mean_router_probability")
    router_names = aggregate.get("router_family_names")
    if router_values and router_names:
        figure, axis = plt.subplots(figsize=(10, 5))
        bars = axis.bar(router_names, router_values)
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
        axis.set_ylabel("Mean routing probability")
        axis.set_title("MoEDDI expert specialization")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
        _save_figure(figure, paper_dir / "router_specialization.png")
        plt.close(figure)
