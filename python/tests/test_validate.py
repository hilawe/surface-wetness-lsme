"""Validation metrics and regridding."""

import numpy as np

from swi import validate as val


def test_skill_scores_recovers_correlation():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, 5000)
    b = 0.5 * a + rng.normal(0, 5, 5000)          # strongly correlated
    s = val.skill_scores(a, b)
    assert s["n"] == 5000
    assert s["pearson_r"] > 0.9
    assert s["spearman_r"] > 0.9
    # anti-correlated -> negative
    s2 = val.skill_scores(a, -b)
    assert s2["pearson_r"] < -0.9


def test_skill_scores_degenerate():
    s = val.skill_scores(np.ones(10), np.arange(10.0))
    assert np.isnan(s["pearson_r"])          # zero-variance input
    assert s["n"] == 10


def test_categorical_perfect_and_random():
    # perfect agreement
    a = np.array([10, 80, 90, 5, 70])
    b = np.array([0.1, 0.5, 0.6, 0.05, 0.45])
    c = val.categorical(a, b, a_hi=50, b_hi=0.3)
    assert c["POD"] == 1.0 and c["FAR"] == 0.0 and c["CSI"] == 1.0
    assert c["HSS"] > 0.99


def test_detection_contrast():
    # index fires (>0) on cells whose reference is higher -> ratio > 1
    index = np.array([0, 0, 0, 5, 10, 20, 0, 8])
    ref = np.array([0.1, 0.2, 0.15, 0.4, 0.5, 0.45, 0.12, 0.42])
    dc = val.detection_contrast(index, ref, thr=0.0)
    assert dc["n_hi"] == 4 and dc["n_lo"] == 4
    assert dc["mean_hi"] > dc["mean_lo"]
    assert dc["ratio"] > 1.0


def test_detection_by_zone_splits_latitude():
    # the index fires only in the tropics, where the reference is higher; the
    # tropics also has a non-firing cell so the contrast ratio is defined
    lat = np.array([0, 5, 10, 40, 45, 70, 75], float)
    index = np.array([5, 0, 8, 0, 0, 0, 0], float)
    ref = np.array([0.5, 0.1, 0.6, 0.2, 0.2, 0.1, 0.1], float)
    z = val.detection_by_zone(index, ref, lat, thr=0.0)
    assert set(z) == {"tropics", "midlatitudes", "high latitudes"}
    assert z["tropics"]["n_hi"] == 2 and z["tropics"]["ratio"] > 1.0
    assert z["midlatitudes"]["n_hi"] == 0     # index never fires outside tropics
    assert z["high latitudes"]["n_hi"] == 0


def test_pattern_correlation():
    base = np.linspace(0, 1, 100).reshape(10, 10)
    assert val.pattern_correlation(base, 2 * base + 1) > 0.99
    assert val.pattern_correlation(base, -base) < -0.99


def test_temporal_anomaly_correlation():
    rng = np.random.default_rng(0)
    T, ny, nx = 12, 4, 5
    # cell (0,0): b tracks a (high corr); cell (1,1): b independent of a
    base = rng.uniform(0, 50, (T, ny, nx))
    A = base.copy()
    B = base.copy()
    B[:, 1, 1] = rng.uniform(0, 50, T)            # decorrelate this cell
    A[:, 1, 1] = rng.uniform(0, 50, T)
    r, n = val.temporal_anomaly_correlation(A, B, min_n=8)
    assert n[0, 0] == T
    assert r[0, 0] > 0.99                          # a==b elsewhere
    assert abs(r[1, 1]) < 0.9                      # decorrelated cell
    # a cell with too few months is NaN
    A2 = A.copy(); A2[2:, 2, 2] = np.nan
    r2, n2 = val.temporal_anomaly_correlation(A2, B, min_n=8)
    assert np.isnan(r2[2, 2])


def test_regrid_nearest_identity_and_coarsen():
    lat = np.linspace(-89, 89, 90); lon = np.linspace(1, 359, 180)
    field = np.outer(np.sin(np.radians(lat)), np.cos(np.radians(lon)))
    # identity when target == source grid
    out = val.regrid_nearest(lat, lon, field, lat, lon)
    assert np.allclose(out, field)
    # coarsen to half resolution -> shape matches, values plausible
    dlat = lat[::2]; dlon = lon[::2]
    out2 = val.regrid_nearest(lat, lon, field, dlat, dlon)
    assert out2.shape == (dlat.size, dlon.size)
    assert np.isfinite(out2).all()
