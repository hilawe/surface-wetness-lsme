"""Unit tests for the two-estimator wetness combination (swi.two_estimator)."""

import numpy as np

from swi import two_estimator as te


def test_rank_normalize_spans_zero_to_one():
    f = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, np.nan]])
    mask = np.ones(f.shape, dtype=bool)
    r = te.rank_normalize(f, mask)
    assert np.isclose(r[0, 0], 0.0)        # smallest -> 0
    assert np.isclose(r[1, 1], 1.0)        # largest -> 1
    assert np.isclose(r[0, 2], 0.5)        # middle of five
    assert np.isnan(r[1, 2])               # NaN stays NaN


def test_wetness_index_inverts_for_low_is_wet():
    f = np.array([[10.0, 50.0]])
    mask = np.ones(f.shape, dtype=bool)
    w = te.wetness_index(f, mask, wet_is_low=True)
    assert w[0, 0] > w[0, 1]                # low field -> wetter
    w2 = te.wetness_index(f, mask, wet_is_low=False)
    assert w2[0, 0] < w2[0, 1]


def test_combine_mean_and_agreement():
    a = np.array([[0.2, 0.8, np.nan]])
    b = np.array([[0.4, 0.8, 0.5]])
    combined, agreement = te.combine(a, b)
    assert np.isclose(combined[0, 0], 0.3)        # mean of 0.2 and 0.4
    assert np.isclose(combined[0, 1], 0.8)
    assert np.isnan(combined[0, 2])               # one input missing
    assert np.isclose(agreement[0, 1], 1.0)       # identical indices
    assert np.isclose(agreement[0, 0], 0.8)       # 1 - |0.2 - 0.4|
