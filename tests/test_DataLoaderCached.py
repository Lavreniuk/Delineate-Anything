import warnings

import numpy as np
import pytest

from methods.main.DataLoaderCached import DataLoaderCached


def test_normalize_local():
    band = np.array(
        [
            [0, 1, 2, 3, 4],
            [100, 200, 300, 400, 500],
            [1000, 2000, 3000, 4000, 5000],
            [6000, 7000, 8000, 9000, 10000],
            [11000, 12000, 13000, 14000, 15000],
        ],
        dtype=np.uint16,
    )

    norm = DataLoaderCached._normalize_part(band, percentiles=[1, 99])

    assert norm.dtype == np.uint8
    assert norm.min() >= 0
    assert norm.max() <= 255
    assert np.isclose(norm[0, 0], 0.0)
    assert np.isclose(norm[-1, -1], 255.0)


def test_normalize_local_fallback():
    band = np.full((4, 4), 42, dtype=np.uint16)

    norm = DataLoaderCached._normalize_part(band, percentiles=[1, 99])

    assert norm.dtype == np.uint8
    assert np.all(norm == 0)


def test_normalize_local_ignores_nodata_value():
    band = np.array([[10, 20], [30, 65535]], dtype=np.uint16)

    norm = DataLoaderCached._normalize_part(
        band,
        percentiles=[1, 99],
        nodata_value=65535,
    )

    assert norm[1, 1] == 255
    assert norm[0, 0] == 0
    assert norm[1, 0] > norm[0, 1]


def test_normalize_part_rejects_mixed_criteria():
    band = np.array([[0, 100], [200, 300]], dtype=np.uint16)

    with np.testing.assert_raises(ValueError):
        DataLoaderCached._normalize_part(
            band,
            lower=0.0,
            upper=300.0,
            percentiles=[1, 99],
        )

