import numpy as np

from plotting import (
    plot_class_distribution,
    plot_evaluation_figures,
    plot_training_history,
    plot_tsne_top10_classes,
)


def test_plot_training_history_writes_tagged_files(tmp_path):
    history = [
        {
            "epoch": 1,
            "learning_rate": 0.001,
            "train_classification_loss": 2.0,
            "train_balance_loss": 0.1,
            "train_router_z_loss": 0.2,
            "validation": {"loss": 2.1, "accuracy": 0.2, "macro_f1": 0.1},
        }
    ]
    plot_training_history(history, tmp_path, epoch=1, dataset_tag="DDI2025")
    assert (tmp_path / "training_DDI2025.png").is_file()


def test_paper_plots_are_created(tmp_path):
    counts = np.asarray([100, 20, 2])
    labels = np.asarray([1, 4, 9])
    plot_class_distribution(counts, labels, tmp_path, dataset_tag="DDI2025")
    assert (tmp_path / "datadist_DDI2025.png").is_file()
    aggregate = {
        "accuracy": 0.7,
        "macro_f1": 0.6,
        "weighted_f1": 0.68,
        "top_3_accuracy": 0.9,
        "top_5_accuracy": 0.95,
        "mean_router_probability": [0.6, 0.4],
        "router_family_names": ["PEOE_VSA", "Generalist"],
    }
    per_class = [
        {"class": 0, "original_class_id": 1, "support": 10, "f1": 0.8},
        {"class": 1, "original_class_id": 4, "support": 5, "f1": 0.5},
        {"class": 2, "original_class_id": 9, "support": 1, "f1": 0.1},
    ]
    confusion = np.asarray([[8, 2, 0], [1, 3, 1], [0, 1, 0]])
    plot_evaluation_figures(aggregate, per_class, confusion, tmp_path, dataset_tag="DDI2025")
    info_dir = tmp_path / "info_DDI2025"
    assert (info_dir / "per_class_f1_vs_support.png").is_file()
    assert (info_dir / "confusion_matrix_normalized.png").is_file()
    assert (info_dir / "test_metrics.png").is_file()
    assert (info_dir / "router_specialization.png").is_file()


def test_plot_tsne_top10_classes_creates_image(tmp_path):
    embeddings = np.random.randn(50, 16)
    labels = np.random.randint(0, 5, size=50)
    class_counts = np.asarray([20, 15, 10, 3, 2])
    out_path = plot_tsne_top10_classes(
        embeddings, labels, class_counts, tmp_path, dataset_tag="DDI2025"
    )
    assert out_path is not None
    assert out_path.is_file()
    assert out_path.name == "tsne_top10_classes.png"
