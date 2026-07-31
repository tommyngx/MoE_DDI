import numpy as np

from plotting import (
    plot_class_distribution,
    plot_evaluation_figures,
    plot_training_history,
)


def test_plot_training_history_writes_main_and_epoch_snapshot(tmp_path):
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
    plot_training_history(history, tmp_path, epoch=1)
    assert (tmp_path / "plots" / "training_curves.png").is_file()
    assert (tmp_path / "plots" / "epochs" / "epoch_001.png").is_file()


def test_paper_plots_are_created(tmp_path):
    counts = np.asarray([100, 20, 2])
    labels = np.asarray([1, 4, 9])
    plot_class_distribution(counts, labels, tmp_path)
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
    plot_evaluation_figures(aggregate, per_class, confusion, tmp_path)
    paper = tmp_path / "plots" / "paper"
    assert (paper / "class_distribution.png").is_file()
    assert (paper / "per_class_f1_vs_support.png").is_file()
    assert (paper / "confusion_matrix_normalized.png").is_file()
    assert (paper / "test_metrics.png").is_file()
    assert (paper / "router_specialization.png").is_file()
