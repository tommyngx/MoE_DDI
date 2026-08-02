from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def _series(history: list[dict], key: str, nested: str | None = None) -> list[float]:
    if nested is None:
        return [float(row[key]) if row[key] is not None else float("nan") for row in history]
    return [
        float(row[key][nested]) if row[key][nested] is not None else float("nan")
        for row in history
    ]


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    os.replace(temporary, path)


def _plot_series(
    axis: plt.Axes,
    epochs: list[float],
    values: list[float],
    name: str,
    color: str,
    *,
    mode: str = "min",
    linestyle: str = "-",
    linewidth: float = 2.0,
    is_lr: bool = False,
    mark_star: bool = True,
) -> None:
    valid_pairs = [(float(v), int(e)) for v, e in zip(values, epochs) if not np.isnan(v)]
    if not valid_pairs:
        return

    if mode == "min":
        best_v, best_e = min(valid_pairs, key=lambda x: x[0])
        prefix = "1st Min"
    else:
        best_v, best_e = max(valid_pairs, key=lambda x: x[0])
        prefix = "1st Max" if not is_lr else "Max"

    if is_lr:
        val_str = f"{best_v:.4e}"
    else:
        val_str = f"{best_v:.4f}"

    label = f"{name}\n{prefix}: {val_str} (E{best_e})"
    axis.plot(epochs, values, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    if mark_star and not is_lr:
        axis.scatter(
            [best_e],
            [best_v],
            color=color,
            marker="*",
            s=160,
            zorder=6,
            edgecolors="black",
            linewidths=0.8,
        )


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def plot_training_history(history: list[dict], run_dir: str | Path, *, epoch: int) -> None:
    """Update the main training plot with rich aesthetics, metric-specific 1st best values, star markers, multi-group LR, and elapsed timing."""
    if not history:
        return
    epochs = _series(history, "epoch")
    
    # Modern color palette
    c_train = "#1f77b4"      # Steel blue
    c_val = "#ff7f0e"        # Warm orange
    c_acc = "#2ca02c"        # Forest green
    c_f1 = "#9467bd"         # Elegant purple
    c_aux1 = "#d62728"       # Red
    c_aux2 = "#8c564b"       # Brown
    c_lr1 = "#008080"        # Teal (1st Group)
    c_lr2 = "#e377c2"        # Pink/Magenta (2nd Group)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=160, facecolor="white")

    # Subplot 1: Classification Loss (Best = Min)
    s_train_loss = _series(history, "train_classification_loss")
    s_val_loss = _series(history, "validation", "loss")
    _plot_series(axes[0, 0], epochs, s_train_loss, "Train Loss", c_train, mode="min")
    _plot_series(axes[0, 0], epochs, s_val_loss, "Val Loss", c_val, mode="min", linestyle="--")
    axes[0, 0].set_title("Classification Loss", fontsize=12, fontweight="bold", color="#111111")
    axes[0, 0].legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    # Subplot 2: Validation Metrics (Best = Max)
    s_val_acc = _series(history, "validation", "accuracy")
    s_val_f1 = _series(history, "validation", "macro_f1")
    _plot_series(axes[0, 1], epochs, s_val_acc, "Accuracy", c_acc, mode="max")
    _plot_series(axes[0, 1], epochs, s_val_f1, "Macro-F1", c_f1, mode="max")
    axes[0, 1].set_title("Validation Metrics", fontsize=12, fontweight="bold", color="#111111")
    axes[0, 1].legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    # Subplot 3: MoE Auxiliary Losses (Best = Min)
    s_balance = _series(history, "train_balance_loss")
    s_router_z = _series(history, "train_router_z_loss")
    _plot_series(axes[1, 0], epochs, s_balance, "Balance Loss", "#17becf", mode="min", linewidth=1.8)
    _plot_series(axes[1, 0], epochs, s_router_z, "Router Z Loss", "#bcbd22", mode="min", linewidth=1.8)
    if "train_moe_auxiliary_loss" in history[0] and history[0]["train_moe_auxiliary_loss"] is not None:
        s_moe_aux = _series(history, "train_moe_auxiliary_loss")
        _plot_series(axes[1, 0], epochs, s_moe_aux, "MoE Aux", c_aux1, mode="min", linewidth=1.8)
    if "train_global_auxiliary_loss" in history[0] and history[0]["train_global_auxiliary_loss"] is not None:
        s_global_aux = _series(history, "train_global_auxiliary_loss")
        _plot_series(axes[1, 0], epochs, s_global_aux, "Global Aux", c_aux2, mode="min", linewidth=1.8)
    axes[1, 0].set_title("MoE & Auxiliary Losses", fontsize=12, fontweight="bold", color="#111111")
    axes[1, 0].legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    # Subplot 4: Learning Rate (1st vs 2nd Group)
    lr_1st = _series(history, "learning_rate_1st") if "learning_rate_1st" in history[0] else _series(history, "learning_rate")
    _plot_series(axes[1, 1], epochs, lr_1st, "1st: MoE Head", c_lr1, mode="max", is_lr=True, mark_star=False)
    
    if "learning_rate_2nd" in history[0] and any(row.get("learning_rate_2nd") is not None for row in history):
        lr_2nd = _series(history, "learning_rate_2nd")
        _plot_series(axes[1, 1], epochs, lr_2nd, "2nd: T-DDI Backbone", c_lr2, mode="max", is_lr=True, linestyle="--", mark_star=False)

    axes[1, 1].set_title("Learning Rate Schedule (1st & 2nd Groups)", fontsize=12, fontweight="bold", color="#111111")
    axes[1, 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1, 1].legend(frameon=True, framealpha=0.95, facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    # Styling all subplots cleanly for headless/server compatibility
    for axis in axes.flat:
        axis.set_facecolor("white")
        axis.set_xlabel("Epoch", fontsize=10, color="#111111")
        axis.set_axisbelow(True)
        axis.grid(True, linestyle="--", linewidth=0.8, color="#d0d0d0", alpha=0.8)
        axis.tick_params(labelsize=9, labelcolor="#111111")
        for spine in axis.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(1.0)

    # Calculate timing stats
    elapsed_seconds = float(history[-1].get("elapsed_seconds", 0.0))
    time_info = ""
    if elapsed_seconds > 0:
        total_time_str = _format_time(elapsed_seconds)
        avg_per_epoch = elapsed_seconds / max(1, epoch)
        avg_time_str = f"{avg_per_epoch:.2f}s/epoch" if avg_per_epoch < 60 else _format_time(avg_per_epoch) + "/epoch"
        time_info = f" | Total: {total_time_str} ({avg_time_str})"

    figure.suptitle(f"MoEDDI Training Progress (Epoch {epoch}){time_info}", fontsize=13, fontweight="bold", y=0.99, color="#111111")
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
