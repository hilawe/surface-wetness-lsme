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


def test_shared_support_excludes_one_sided_cells():
    """Cells where one estimator is missing must be excluded from the shared
    support; otherwise the F1 bug (different-population percentiles) creeps
    back in.
    """
    thermal = np.array([[1.0, 2.0, np.nan, 4.0]])
    micro = np.array([[5.0, np.nan, 7.0, 8.0]])
    land = np.ones_like(thermal, dtype=bool)
    shared = te.shared_support_mask(thermal, micro, land)
    # only positions where BOTH are finite AND land is True qualify
    assert shared.tolist() == [[True, False, False, True]]


def test_joint_indices_rank_over_shared_support():
    """Both estimator percentiles must reference the SAME shared-support
    population so a cell-wise comparison is meaningful. This is the F1 fix.
    """
    # 5 cells, but only 3 have BOTH estimators finite; the other 2 are
    # one-sided and must be NaN on output.
    thermal = np.array([10.0, 20.0, 30.0, np.nan, 50.0])
    micro = np.array([np.nan, 0.5, 0.7, 0.9, 0.8])
    land = np.ones(5, dtype=bool)
    t, m, shared = te.joint_indices(thermal, micro, land,
                                    thermal_wet_is_low=True,
                                    microwave_wet_is_low=True)
    # cells 0 and 3 are one-sided -> NaN in both outputs
    assert np.isnan(t[0]) and np.isnan(m[0])
    assert np.isnan(t[3]) and np.isnan(m[3])
    # cells 1, 2, 4 are the shared support; both indices defined there
    assert np.isfinite(t[1]) and np.isfinite(m[1])
    assert np.isfinite(t[2]) and np.isfinite(m[2])
    assert np.isfinite(t[4]) and np.isfinite(m[4])
    # rank normalization over the 3 shared cells, then wet-is-low inverts.
    # thermal values 20, 30, 50 -> ranks 0, 0.5, 1 -> wet: 1, 0.5, 0
    assert np.isclose(t[1], 1.0) and np.isclose(t[2], 0.5) and np.isclose(t[4], 0.0)
    # microwave values 0.5, 0.7, 0.8 -> ranks 0, 0.5, 1 -> wet: 1, 0.5, 0
    assert np.isclose(m[1], 1.0) and np.isclose(m[2], 0.5) and np.isclose(m[4], 0.0)
