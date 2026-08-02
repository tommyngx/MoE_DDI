import numpy as np

from preprocessing import TrainStatistics, compute_train_statistics
from schema import DatasetSchema


def _schema() -> DatasetSchema:
    return DatasetSchema(
        all_columns=("PEOEVSA1", "MRVSA1", "class"),
        feature_columns=("PEOEVSA1", "MRVSA1"),
        metadata_columns=(),
        target_column="class",
        family_indices={"MR_VSA": (1,), "PEOE_VSA": (0,)},
        feature_families=(("PEOE_VSA",), ("MR_VSA",)),
        fingerprint="test",
    )


def test_streaming_statistics_and_roundtrip(tmp_path):
    stream = [
        (np.asarray([[1, 2], [3, 4]], dtype=np.float32), np.asarray([0, 1])),
        (np.asarray([[5, 6], [7, 8]], dtype=np.float32), np.asarray([1, 2])),
    ]
    stream = type("TestStream", (list,), {"label_values": np.asarray([1, 5, 9])})(stream)
    stats = compute_train_statistics(stream, _schema(), num_classes=3)
    np.testing.assert_allclose(stats.mean, [4, 5])
    np.testing.assert_allclose(stats.std, np.sqrt([5, 5]))
    np.testing.assert_array_equal(stats.class_counts, [1, 2, 1])
    np.testing.assert_array_equal(stats.label_values, [1, 5, 9])
    path = tmp_path / "stats.npz"
    stats.save(path)
    loaded = TrainStatistics.load(path, _schema())
    np.testing.assert_allclose(loaded.mean, stats.mean)
    assert loaded.count == 4
    values = np.asarray([[4, 7]], dtype=np.float32)
    np.testing.assert_allclose(loaded.transform(values, "none"), values)
    np.testing.assert_allclose(loaded.transform(values, "standardize"), [[0, 2 / np.sqrt(5)]])
