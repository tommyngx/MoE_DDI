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
    mark_best: bool = True,
    best_label_prefix: str = "Val",
    best_marker_color: str = "blue",
) -> None:
    valid_pairs = [(float(v), int(e)) for v, e in zip(values, epochs) if not np.isnan(v)]
    if not valid_pairs:
        return

    # Main series curve
    axis.plot(epochs, values, label=name, color=color, linestyle=linestyle, linewidth=linewidth)

    # BB2 style scatter point for highlighted best epoch
    if mark_best and not is_lr:
        if mode == "min":
            best_v, best_e = min(valid_pairs, key=lambda x: x[0])
            tag = f"1st {best_label_prefix}"
        else:
            best_v, best_e = max(valid_pairs, key=lambda x: x[0])
            tag = f"1st {best_label_prefix}"

        val_str = f"{best_v:.4f}"
        scatter_label = f"{tag}: {val_str} (Epoch {best_e})"

        axis.scatter(
            [best_e],
            [best_v],
            s=180,
            color=best_marker_color,
            marker="*",
            zorder=6,
            edgecolors="#161A1F",
            linewidths=0.8,
            label=scatter_label,
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


def _format_model_title(model_name: str | None) -> str:
    if not model_name:
        return "Model"
    name_lower = model_name.lower()
    mapping = {
        "moeddi": "MoEDDI",
        "tabtransformer": "TabTransformer",
        "bishop": "BISHOP",
        "tddi_mlp": "TDDI",
        "linear": "Linear",
        "mlp": "MLP",
    }
    return mapping.get(name_lower, model_name.upper())


def plot_training_history(
    history: list[dict],
    run_dir: str | Path,
    *,
    epoch: int,
    dataset_tag: str | None = None,
    model_name: str | None = None,
) -> None:
    """Update the main training plot in BB2 style with thick dark spines, off-white background, navy grid, and highlighted scatter points."""
    if not history:
        return
    epochs = _series(history, "epoch")
    
    # BB2 / Modern Color Palette
    c_train = "#d9534f"      # BB2 Red
    c_val = "#22c55e"        # Green
    c_acc = "#22c55e"        # Green
    c_f1 = "#8b5cf6"         # Purple
    c_aux1 = "#ef4444"       # Light red
    c_aux2 = "#f59e0b"       # Amber
    c_lr1 = "#0284c7"        # Sky Blue (1st Group)
    c_lr2 = "#ec4899"        # Pink/Magenta (2nd Group)
    c_scatter = "#1d4ed8"    # BB2 Highlight Blue

    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), dpi=160)

    # BB2 Light background color (#f7f7f7)
    figure.patch.set_facecolor("#f7f7f7")

    # Subplot 1: Classification Loss
    s_train_loss = _series(history, "train_classification_loss")
    s_val_loss = _series(history, "validation", "loss")
    _plot_series(axes[0, 0], epochs, s_train_loss, "Training Loss", c_train, mode="min", mark_best=False)
    _plot_series(
        axes[0, 0],
        epochs,
        s_val_loss,
        "Val Loss",
        c_val,
        mode="min",
        linestyle="--",
        mark_best=True,
        best_label_prefix="Val Loss",
        best_marker_color=c_scatter,
    )
    axes[0, 0].set_title("Training and Validation Loss", color="#222831", fontsize=12, fontweight="bold")

    # Subplot 2: Validation Metrics
    s_val_acc = _series(history, "validation", "accuracy")
    s_val_f1 = _series(history, "validation", "macro_f1")
    _plot_series(
        axes[0, 1],
        epochs,
        s_val_acc,
        "Val Accuracy",
        c_acc,
        mode="max",
        mark_best=True,
        best_label_prefix="Accuracy",
        best_marker_color=c_scatter,
    )
    _plot_series(
        axes[0, 1],
        epochs,
        s_val_f1,
        "Val Macro-F1",
        c_f1,
        mode="max",
        mark_best=True,
        best_label_prefix="Macro-F1",
        best_marker_color="#b91c1c",
    )
    axes[0, 1].set_title("Validation Accuracy & Macro-F1", color="#222831", fontsize=12, fontweight="bold")

    # Subplot 3: Auxiliary & Regularization Losses
    s_balance = _series(history, "train_balance_loss")
    s_router_z = _series(history, "train_router_z_loss")
    _plot_series(axes[1, 0], epochs, s_balance, "Balance Loss", "#06b6d4", mode="min", linewidth=1.8, mark_best=False)
    _plot_series(axes[1, 0], epochs, s_router_z, "Router Z Loss", "#eab308", mode="min", linewidth=1.8, mark_best=False)
    if "train_moe_auxiliary_loss" in history[0] and history[0]["train_moe_auxiliary_loss"] is not None:
        s_moe_aux = _series(history, "train_moe_auxiliary_loss")
        _plot_series(
            axes[1, 0],
            epochs,
            s_moe_aux,
            "Aux Loss",
            c_aux1,
            mode="min",
            linewidth=1.8,
            mark_best=True,
            best_label_prefix="Aux Loss",
            best_marker_color=c_scatter,
        )
    if "train_global_auxiliary_loss" in history[0] and history[0]["train_global_auxiliary_loss"] is not None:
        s_global_aux = _series(history, "train_global_auxiliary_loss")
        _plot_series(axes[1, 0], epochs, s_global_aux, "Global Aux", c_aux2, mode="min", linewidth=1.8, mark_best=False)
    axes[1, 0].set_title("Auxiliary & Regularization Losses", color="#222831", fontsize=12, fontweight="bold")

    # Subplot 4: Learning Rate (1st vs 2nd Group)
    lr_1st = _series(history, "learning_rate_1st") if "learning_rate_1st" in history[0] else _series(history, "learning_rate")
    _plot_series(axes[1, 1], epochs, lr_1st, "1st: Primary Head LR", c_lr1, mode="max", is_lr=True, mark_best=False)
    
    if "learning_rate_2nd" in history[0] and any(row.get("learning_rate_2nd") is not None for row in history):
        lr_2nd = _series(history, "learning_rate_2nd")
        _plot_series(axes[1, 1], epochs, lr_2nd, "2nd: Backbone LR", c_lr2, mode="max", is_lr=True, linestyle="--", mark_best=False)

    axes[1, 1].set_title("Learning Rate Schedule (1st & 2nd Groups)", color="#222831", fontsize=12, fontweight="bold")
    axes[1, 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Apply BB2 styling to all subplots
    for axis in axes.flat:
        axis.set_facecolor("#f7f7f7")
        axis.set_xlabel("Epochs", color="#222831", fontsize=10)
        axis.tick_params(axis="x", colors="#222831", labelsize=9)
        axis.tick_params(axis="y", colors="#222831", labelsize=9)
        axis.grid(True, linestyle="--", alpha=0.5, color="navy")
        
        # BB2 thick dark spines around subplot
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color("#161A1F")

        # BB2 style legend box
        legend = axis.legend(frameon=True, fontsize=8.5)
        if legend:
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("#161A1F")
            legend.get_frame().set_linewidth(1.2)
            for text in legend.get_texts():
                text.set_color("#222831")

    # Calculate timing stats
    elapsed_seconds = float(history[-1].get("elapsed_seconds", 0.0))
    time_info = ""
    if elapsed_seconds > 0:
        total_time_str = _format_time(elapsed_seconds)
        avg_per_epoch = elapsed_seconds / max(1, epoch)
        avg_time_str = f"{avg_per_epoch:.2f}s/epoch" if avg_per_epoch < 60 else _format_time(avg_per_epoch) + "/epoch"
        time_info = f" | Total: {total_time_str} ({avg_time_str})"

    model_title = _format_model_title(model_name)
    figure.suptitle(f"{model_title} Training Progress (Epoch {epoch}){time_info}", color="#222831", fontsize=14, fontweight="bold", y=0.99)
    figure.tight_layout()

    run_path = Path(run_dir)
    tag = dataset_tag or "dataset"
    _save_figure(figure, run_path / f"training_{tag}.png")
    plt.close(figure)


def plot_class_distribution(
    class_counts: np.ndarray,
    label_values: np.ndarray,
    run_dir: str | Path,
    dataset_tag: str | None = None,
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
    run_path = Path(run_dir)
    tag = dataset_tag or "dataset"
    target_dir = run_path / f"info_{tag}" if dataset_tag else run_path
    target_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(figure, target_dir / f"datadist_{tag}.png")
    plt.close(figure)


def plot_evaluation_figures(
    aggregate: dict,
    per_class: list[dict],
    confusion: np.ndarray,
    run_dir: str | Path,
    dataset_tag: str | None = None,
) -> None:
    tag = dataset_tag or "dataset"
    target_dir = Path(run_dir) / f"info_{tag}"
    target_dir.mkdir(parents=True, exist_ok=True)

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
    _save_figure(figure, target_dir / "per_class_f1_vs_support.png")
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
    _save_figure(figure, target_dir / "confusion_matrix_normalized.png")
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
    _save_figure(figure, target_dir / "test_metrics.png")
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
        _save_figure(figure, target_dir / "router_specialization.png")
        plt.close(figure)


def plot_tsne_top10_classes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
    run_dir: str | Path,
    *,
    dataset_tag: str | None = None,
    max_samples_per_class: int = 300,
    seed: int = 42,
) -> Path | None:
    """Compute and save a 2D t-SNE scatter plot for the top 10 most frequent classes."""
    if len(embeddings) == 0 or len(labels) == 0:
        return None

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    target_dir = Path(run_dir)
    if dataset_tag:
        target_dir = target_dir / f"info_{dataset_tag}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find top 10 most frequent classes
    top10_classes = np.argsort(class_counts)[::-1][:10]

    rng = np.random.default_rng(seed)
    selected_indices = []
    selected_labels = []

    for cls in top10_classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        if len(cls_indices) > max_samples_per_class:
            cls_indices = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
        selected_indices.extend(cls_indices)
        selected_labels.extend([cls] * len(cls_indices))

    if not selected_indices:
        return None

    sub_embeddings = np.asarray(embeddings)[selected_indices]
    sub_labels = np.asarray(selected_labels)

    n_samples = len(sub_embeddings)
    perplexity = min(30, max(5, n_samples // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embedding_2d = tsne.fit_transform(sub_embeddings)

    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, cls in enumerate(top10_classes):
        mask = sub_labels == cls
        if not np.any(mask):
            continue
        axis.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            color=colors[idx % 10],
            label=f"Class {cls} (n={class_counts[cls]:,})",
            alpha=0.75,
            edgecolors="none",
            s=40,
        )

    title_text = "t-SNE Feature Representation (Top 10 DDI Classes)"
    if dataset_tag:
        title_text += f" - {dataset_tag}"
    axis.set_title(title_text, fontsize=13, fontweight="bold", pad=12)
    axis.set_xlabel("t-SNE Dimension 1", fontsize=11)
    axis.set_ylabel("t-SNE Dimension 2", fontsize=11)
    axis.legend(title="Top 10 Classes", bbox_to_anchor=(1.03, 1), loc="upper left", frameon=True)
    axis.grid(True, linestyle="--", alpha=0.3)

    output_path = target_dir / "tsne_top10_classes.png"
    _save_figure(figure, output_path)
    plt.close(figure)
    return output_path
