import numpy as np

from preprocessing import TrainStatistics
from reporting import write_run_summary
from schema import DatasetSchema


def test_summary_contains_hyperparameters_and_results(tmp_path):
    schema = DatasetSchema(
        all_columns=("PEOEVSA1", "class"),
        feature_columns=("PEOEVSA1",),
        metadata_columns=(),
        target_column="class",
        family_indices={"PEOE_VSA": (0,)},
        feature_families=(("PEOE_VSA",),),
        fingerprint="abc",
    )
    statistics = TrainStatistics(
        mean=np.asarray([0.0], dtype=np.float32),
        std=np.asarray([1.0], dtype=np.float32),
        class_counts=np.asarray([8, 2]),
        label_values=np.asarray([1, 4]),
        count=10,
        schema_fingerprint="abc",
    )
    config = {
        "run_name": "test",
        "seed": 7,
        "data": {
            "root": "Dataset",
            "train_files": ["train.csv"],
            "validation_files": ["validation.csv"],
            "test_files": ["test.csv"],
            "num_classes": 2,
            "batch_size": 4,
        },
        "model": {"name": "moeddi", "router_top_k": 2},
        "training": {
            "epochs": 1,
            "selection_metric": "macro_f1",
            "run_dir": str(tmp_path),
        },
        "loss": {"name": "focal", "gamma": 2.0},
    }
    history = [
        {
            "epoch": 1,
            "train_classification_loss": 1.2,
            "elapsed_seconds": 3.0,
            "validation": {
                "loss": 1.1,
                "accuracy": 0.6,
                "macro_f1": 0.5,
            },
        }
    ]
    path = write_run_summary(
        tmp_path,
        config,
        schema,
        statistics,
        {"total": 100, "trainable": 90},
        "cpu",
        history,
        test_metrics={"num_samples": 10, "accuracy": 0.7, "macro_f1": 0.6},
    )
    text = path.read_text(encoding="utf-8")
    assert "MoEDDI EXPERIMENT SUMMARY" in text
    assert "Best validation macro_f1: 0.50000000" in text
    assert "accuracy: 0.70000000" in text
