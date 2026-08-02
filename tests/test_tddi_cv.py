import numpy as np

from tddi_cv import stratified_fold_ids


def test_stratified_fold_ids_are_deterministic_and_balanced():
    labels = np.repeat(np.arange(4), [12, 15, 18, 21])
    first = stratified_fold_ids(labels, 3, 42)
    second = stratified_fold_ids(labels, 3, 42)
    np.testing.assert_array_equal(first, second)
    for class_value in np.unique(labels):
        counts = np.bincount(first[labels == class_value], minlength=3)
        assert counts.max() - counts.min() <= 1
