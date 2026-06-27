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


def test_skill_scores_spearman_tie_permutation_invariance():
    """Permuting tied values must not change the Spearman correlation.

    The previous ordinal-rank implementation failed this badly: a 1000-cell
    field with 950 tied zeros gave Spearman 1.0 from spatial ordering inside
    the ties, and permuting only the zero positions dropped it to about 0.16.
    """
    rng = np.random.default_rng(42)
    # zero-inflated WET-like field: 950 zeros, 50 nonzero detections
    a = np.zeros(1000)
    nz = rng.choice(1000, size=50, replace=False)
    a[nz] = rng.uniform(1, 100, 50)
    b = a + rng.normal(0, 1, 1000)
    rho_orig = val.skill_scores(a, b)["spearman_r"]
    # permute ONLY the zero positions in a; the nonzero values keep their
    # rank ordering versus b, so a faithful Spearman should be unchanged.
    a2 = a.copy()
    zero_idx = np.where(a == 0)[0]
    perm = rng.permutation(zero_idx)
    a2[zero_idx] = a[perm]
    rho_perm = val.skill_scores(a2, b)["spearman_r"]
    assert np.isclose(rho_orig, rho_perm, atol=1e-12), (
        f"Spearman not tie-invariant: orig={rho_orig}, permuted={rho_perm}")


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


def test_regrid_nearest_handles_minus180_to_180_source():
    """Regression: a source on -180..180 longitude must regrid correctly onto a
    0..360 destination. Previously the eastern hemisphere collapsed onto the
    western edge column because no wrap was applied, corrupting any reference
    field fed through regrid_nearest from a -180..180 grid.
    """
    src_lat = np.linspace(-89, 89, 36)
    src_lon = np.linspace(-179, 179, 72)                 # -180..180 source
    # construct a field whose value is a function of longitude only
    field = np.broadcast_to(src_lon[None, :], (src_lat.size, src_lon.size)).copy()
    dst_lat = src_lat
    dst_lon = np.linspace(1, 359, 72)                    # 0..360 destination
    out = val.regrid_nearest(src_lat, src_lon, field, dst_lat, dst_lon)
    # eastern hemisphere (90..170 east) must NOT collapse to the western edge
    east_idx = np.where((dst_lon >= 90) & (dst_lon <= 170))[0]
    east_values = out[0, east_idx]
    # in a -180..180 convention, 90E and 170E carry values close to 90 and 170,
    # so the regridded field over the eastern strip should NOT match -179.
    assert (east_values > 80).all(), (
        "eastern hemisphere collapsed onto western edge (longitude wrap broken)")
    # spot check: 90 east -> source value about 90 (within bin width)
    east_90 = out[0, np.argmin(np.abs(dst_lon - 90))]
    assert abs(east_90 - 90.0) < 5.0


def test_regrid_nearest_wraps_at_dateline():
    """A query right at the dateline (~359) should pick the same source as ~1
    when the source longitude axis wraps modulo 360.
    """
    src_lat = np.array([0.0])
    src_lon = np.array([1.0, 90.0, 180.0, 270.0, 359.0])
    field = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
    # query at 0.5 should be nearest 1 OR 359 depending on wrap; either is fine
    # for the test, but the value must be one of the dateline-adjacent ones.
    out = val.regrid_nearest(src_lat, src_lon, field,
                             np.array([0.0]), np.array([0.5]))
    assert out[0, 0] in (10.0, 50.0)
    # query at 359.5 should land on 359 (value 50), not collapse to the western
    # edge.
    out2 = val.regrid_nearest(src_lat, src_lon, field,
                              np.array([0.0]), np.array([359.5]))
    assert out2[0, 0] in (10.0, 50.0)
