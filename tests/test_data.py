import numpy as np
import pytest

from data import encode_labels, normalize_labels


def test_normalize_labels_accepts_integral_float_strings():
    labels = normalize_labels(np.asarray(["0.0", "2", "177"]), 178)
    np.testing.assert_array_equal(labels, [0, 2, 177])


def test_normalize_labels_rejects_fractional_or_out_of_range_values():
    with pytest.raises(ValueError):
        normalize_labels(np.asarray([1.25]), 178)
    with pytest.raises(ValueError):
        normalize_labels(np.asarray([178]), 178)


def test_encode_labels_maps_non_contiguous_raw_ids():
    vocabulary = np.asarray([1, 4, 9, 218])
    np.testing.assert_array_equal(
        encode_labels(np.asarray(["218.0", "1", "9"]), vocabulary),
        [3, 0, 2],
    )
    with pytest.raises(ValueError):
        encode_labels(np.asarray([7]), vocabulary)
