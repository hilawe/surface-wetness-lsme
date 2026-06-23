"""Unit tests for the anomaly machinery (swi.anomaly)."""

import numpy as np

from swi import anomaly


def test_temporal_mean_ignores_nan_and_respects_min_valid():
    # cell (0,0): 10,20,30 -> mean 20 ; cell (0,1): only one finite -> NaN at min_valid=2
    stack = np.array([
        [[10.0, 5.0]],
        [[20.0, np.nan]],
        [[30.0, np.nan]],
    ])
    m = anomaly.temporal_mean(stack, min_valid=2)
    assert np.isclose(m[0, 0], 20.0)
    assert np.isnan(m[0, 1])
    # with min_valid=1 the single value is its own mean
    m1 = anomaly.temporal_mean(stack, min_valid=1)
    assert np.isclose(m1[0, 1], 5.0)


def test_temporal_anomaly_departures_sum_to_zero():
    stack = np.array([
        [[1.0, 4.0]],
        [[2.0, 6.0]],
        [[3.0, 8.0]],
    ])
    anom = anomaly.temporal_anomaly(stack)
    # each cell's anomalies sum to zero (mean removed)
    assert np.allclose(np.nansum(anom, axis=0), 0.0)
    # cell (0,0) mean is 2 -> anomalies -1, 0, +1
    assert np.allclose(anom[:, 0, 0], [-1.0, 0.0, 1.0])


def test_temporal_anomaly_nan_propagation_and_min_valid():
    stack = np.array([
        [[10.0, 7.0]],
        [[20.0, np.nan]],
        [[30.0, np.nan]],
    ])
    anom = anomaly.temporal_anomaly(stack, min_valid=2)
    # cell with only one finite month: undefined mean -> all-NaN anomaly
    assert np.all(np.isnan(anom[:, 0, 1]))
    # the present month stays NaN even though the mean cell is well defined
    assert np.allclose(anom[:, 0, 0], [-10.0, 0.0, 10.0])


def test_temporal_anomaly_is_shape_agnostic_over_channels():
    # (T, nlat, nlon, nchannel) per-channel stack
    rng = np.random.default_rng(0)
    stack = rng.normal(size=(4, 3, 5, 7))
    anom = anomaly.temporal_anomaly(stack)
    assert anom.shape == stack.shape
    # anomaly equals value minus per-cell mean, checked on one channel
    mean_c2 = stack[:, :, :, 2].mean(axis=0)
    assert np.allclose(anom[:, :, :, 2], stack[:, :, :, 2] - mean_c2)


def test_pattern_skill_perfect_and_masked():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = 2.0 * a + 1.0                      # perfectly correlated
    s = anomaly.pattern_skill(a, b)
    assert np.isclose(s["pearson_r"], 1.0)
    assert np.isclose(s["spearman_r"], 1.0)
    # masking restricts the cells used
    mask = np.array([[True, True, False], [False, False, False]])
    s2 = anomaly.pattern_skill(a, b, mask=mask)
    assert s2["n"] == 2


def test_anomaly_pattern_skill_per_time_and_aggregate():
    # build two anomaly stacks that agree, plus an anti-correlated one
    base = np.array([
        [[-1.0, 0.0, 1.0]],
        [[2.0, 0.0, -2.0]],
    ])
    per, agg = anomaly.anomaly_pattern_skill(base, base.copy())
    assert len(per) == 2
    assert all("t" in p for p in per)
    assert np.isclose(agg["pearson_r"], 1.0)           # identical -> r = 1

    per_neg, agg_neg = anomaly.anomaly_pattern_skill(base, -base)
    assert np.isclose(agg_neg["pearson_r"], -1.0)      # opposite -> r = -1


def test_block_mean_coarsens_and_ignores_nan():
    a = np.array([1.0, 3.0, 10.0, np.nan])             # factor 2 -> [2.0, 10.0]
    out = anomaly.block_mean(a, 2, axis=0)
    assert np.allclose(out, [2.0, 10.0])               # (1+3)/2 ; nanmean(10, nan)=10
    # n=1 is a no-op
    assert np.allclose(anomaly.block_mean(a[:3], 1, 0), a[:3])
    # 2-D coarsen over both axes by 2
    g = np.arange(16, dtype=float).reshape(4, 4)
    c = anomaly.block_mean(anomaly.block_mean(g, 2, 0), 2, 1)
    assert c.shape == (2, 2)
    assert np.isclose(c[0, 0], np.mean([0, 1, 4, 5]))  # 2.5
