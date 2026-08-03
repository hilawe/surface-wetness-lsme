"""Co-location and skill metrics for validating the Surface Wetness Index.

The wetness index (WET) is a 0 to 100 index, not a physical soil-moisture value,
so validation is about skill and monotonic association (rank correlation,
pattern correlation, wet/dry detection), not absolute agreement. These helpers
co-locate a reference field onto our grid, mask to common valid land cells, and
compute the metrics. They are reference-agnostic: the same code serves ESA CCI
soil moisture, ERA5-Land, in-situ point data (after gridding), or inundation.
"""

import numpy as np


def regrid_nearest(src_lat, src_lon, src, dst_lat, dst_lon):
    """Nearest-neighbor regrid of a 2-D field onto a target grid.

    src_lat, src_lon, dst_lat, dst_lon are 1-D ascending. Longitude is
    normalized to 0..360 internally so a source on -180..180 (for example ERA5)
    regrids correctly onto a 0..360 product grid. Previously the function
    assumed both grids shared a convention; against a 0..360 destination, every
    cell east of 180 degrees in a -180..180 source would collapse onto the
    western edge column, corrupting any reference fed through it.
    src has shape (src_lat, src_lon). Returns a (dst_lat, dst_lon) array. NaNs
    in src propagate.
    """
    src = np.asarray(src, dtype=np.float64)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    if src.shape != (src_lat.size, src_lon.size):
        raise ValueError("src shape must match (src_lat, src_lon)")
    # Normalize longitudes to 0..360 and sort, so source and destination share a
    # convention regardless of input. Rotate the source longitude axis to match.
    src_lon360 = src_lon % 360.0
    order = np.argsort(src_lon360)
    src_lon_n = src_lon360[order]
    src_n = src[:, order]
    dst_lon360 = dst_lon % 360.0
    ilat = np.clip(np.searchsorted(src_lat, dst_lat), 0, src_lat.size - 1)
    # Latitude refinement (non-circular).
    left_lat = np.clip(ilat - 1, 0, src_lat.size - 1)
    choose_left_lat = np.abs(src_lat[left_lat] - dst_lat) < np.abs(src_lat[ilat] - dst_lat)
    ilat[choose_left_lat] = left_lat[choose_left_lat]
    # Longitude refinement with wrap on BOTH ends. searchsorted returns an
    # insertion point in [0, N]; the two candidate neighbors are that point mod N
    # (which wraps to 0 when the destination is past the source's max) and one
    # less mod N (which wraps to N-1 when the destination is below the source's
    # min). Compare with great-circle longitude distance and pick the closer.
    def _lon_dist(a, b):
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)
    ilon_raw = np.searchsorted(src_lon_n, dst_lon360)
    right = ilon_raw % src_lon_n.size
    left = (ilon_raw - 1) % src_lon_n.size
    d_left = _lon_dist(src_lon_n[left], dst_lon360)
    d_right = _lon_dist(src_lon_n[right], dst_lon360)
    ilon = np.where(d_left <= d_right, left, right)
    return src_n[np.ix_(ilat, ilon)]


def _rank(x):
    """Average-rank assignment with proper tie handling.

    Tied values receive the average of the ranks they would have occupied. This
    is the standard definition of rank used in Spearman correlation. An ordinal
    rank (the previous behaviour here) was a real bug on zero-inflated fields:
    permuting the tied zeros could swing Spearman from 1.0 to 0.16, because rank
    skill leaked from the spatial ordering inside the ties rather than from the
    retrieval itself.
    """
    from scipy.stats import rankdata
    return rankdata(x, method="average").astype(np.float64)


def skill_scores(a, b):
    """Pointwise skill between two 1-D arrays (already co-located, finite).

    Returns n, pearson_r, spearman_r (rank correlation), bias (a-b), rmse.
    Spearman is the headline for an index-versus-physical comparison and uses
    scipy's average-rank Spearman so it is invariant to ordering inside ties.
    """
    from scipy.stats import spearmanr
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = a.size
    out = {"n": int(n), "pearson_r": np.nan, "spearman_r": np.nan,
           "bias": np.nan, "rmse": np.nan}
    if n < 3:
        return out
    if a.std() > 0 and b.std() > 0:
        out["pearson_r"] = float(np.corrcoef(a, b)[0, 1])
        # spearmanr returns nan if all values in either array are tied; that is
        # the correct undefined behaviour, propagate it.
        rho = spearmanr(a, b).statistic
        out["spearman_r"] = float(rho) if np.isfinite(rho) else np.nan
    out["bias"] = float((a - b).mean())
    out["rmse"] = float(np.sqrt(((a - b) ** 2).mean()))
    return out


def pattern_correlation(field_a, field_b, mask=None):
    """Spatial (pattern) correlation between two 2-D fields over valid cells."""
    a = np.asarray(field_a, np.float64); b = np.asarray(field_b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def categorical(a, b, a_hi, b_hi):
    """Wet-detection contingency skill: 'wet' = a > a_hi (ours), b > b_hi (ref).

    Returns POD (probability of detection), FAR (false alarm ratio), CSI
    (critical success index), and Heidke skill score.
    """
    a = np.asarray(a); b = np.asarray(b)
    ours = a > a_hi
    ref = b > b_hi
    hits = int((ours & ref).sum())
    miss = int((~ours & ref).sum())
    fa = int((ours & ~ref).sum())
    cn = int((~ours & ~ref).sum())
    n = hits + miss + fa + cn
    pod = hits / (hits + miss) if (hits + miss) else np.nan
    far = fa / (hits + fa) if (hits + fa) else np.nan
    csi = hits / (hits + miss + fa) if (hits + miss + fa) else np.nan
    # Heidke skill score
    exp = ((hits + miss) * (hits + fa) + (cn + miss) * (cn + fa)) / n if n else np.nan
    hss = (hits + cn - exp) / (n - exp) if n and (n - exp) else np.nan
    return {"n": n, "hits": hits, "misses": miss, "false_alarms": fa,
            "correct_negatives": cn, "POD": pod, "FAR": far, "CSI": csi, "HSS": hss}


def detection_contrast(index, ref, thr=0.0):
    """Mean reference where a detector index fires (index > thr) vs not.

    The right diagnostic for a zero-inflated detection index like WET: if the
    index has detection skill, the reference (for example soil moisture) is
    higher where the index fires. Returns the two means, their ratio, and counts.
    """
    index = np.asarray(index, np.float64); ref = np.asarray(ref, np.float64)
    hi = ref[index > thr]; lo = ref[index <= thr]
    mean_hi = float(hi.mean()) if hi.size else np.nan
    mean_lo = float(lo.mean()) if lo.size else np.nan
    ratio = mean_hi / mean_lo if (lo.size and mean_lo) else np.nan
    return {"n_hi": int(hi.size), "n_lo": int(lo.size),
            "mean_hi": mean_hi, "mean_lo": mean_lo, "ratio": ratio}


ZONES = (("tropics", 0.0, 23.5), ("midlatitudes", 23.5, 55.0),
         ("high latitudes", 55.0, 90.1))


def detection_by_zone(index, ref, lat, thr=0.0):
    """Detection contrast split by absolute-latitude zone.

    index, ref, and lat are 1-D arrays over the same co-located cells. Returns a
    dict mapping zone name to a detection_contrast result, for the tropics
    (abs(lat) < 23.5), the mid latitudes (23.5 to 55), and the high latitudes
    (above 55). This exposes where the detector is strong and where it weakens,
    in particular the high-latitude freeze-thaw zone that is a known blind spot.
    """
    index = np.asarray(index, np.float64)
    ref = np.asarray(ref, np.float64)
    a = np.abs(np.asarray(lat, np.float64))
    out = {}
    for name, lo, hi in ZONES:
        sel = (a >= lo) & (a < hi)
        out[name] = detection_contrast(index[sel], ref[sel], thr=thr)
    return out


def common_valid(field_a, field_b, land=None):
    """Boolean mask of cells where both fields are finite (and optionally land)."""
    m = np.isfinite(field_a) & np.isfinite(field_b)
    if land is not None:
        m &= land
    return m


def temporal_anomaly_correlation(stack_a, stack_b, min_n=8):
    """Per-cell temporal anomaly correlation between two (T, nlat, nlon) stacks.

    Anomalies are departures from each cell's temporal mean over the valid
    months. Returns (r_map, n_map): the per-cell Pearson correlation of the
    anomalies and the number of valid months. Cells with fewer than min_n valid
    months, or zero variance on either side, are NaN.
    """
    A = np.asarray(stack_a, np.float64)
    B = np.asarray(stack_b, np.float64)
    valid = np.isfinite(A) & np.isfinite(B)
    n = valid.sum(axis=0)

    Am = np.where(valid, A, np.nan)
    Bm = np.where(valid, B, np.nan)
    with np.errstate(invalid="ignore"):
        Aa = Am - np.nanmean(Am, axis=0)
        Ba = Bm - np.nanmean(Bm, axis=0)
    Aa = np.where(valid, Aa, 0.0)
    Ba = np.where(valid, Ba, 0.0)

    cov = (Aa * Ba).sum(axis=0)
    va = (Aa ** 2).sum(axis=0)
    vb = (Ba ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    r[(n < min_n) | (va == 0) | (vb == 0)] = np.nan
    return r, n


def weighted_pearson(a, b, w):
    """Pearson correlation of a and b with per-sample weights w."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    w = np.asarray(w, np.float64)
    sw = w.sum()
    if sw <= 0 or a.size < 3:
        return np.nan
    ma = (w * a).sum() / sw
    mb = (w * b).sum() / sw
    da, db = a - ma, b - mb
    va = (w * da * da).sum()
    vb = (w * db * db).sum()
    if va <= 0 or vb <= 0:
        return np.nan
    return float((w * da * db).sum() / np.sqrt(va * vb))


def weighted_spearman(a, b, w):
    """Rank correlation with per-sample weights, as weighted Pearson on ranks.

    On an equal-angle grid, cells at high latitude cover far less area than
    tropical ones. An unweighted global correlation therefore over-represents
    high latitudes. Weighting by the cosine of latitude makes the statistic
    represent area rather than cell count.

    The estimand is specific and worth naming: ranks are ordinary cell-count
    midranks, and the weights are applied to the correlation of those ranks.
    Building the ranks from the area-weighted empirical distribution instead is
    an equally defensible definition and gives slightly lower values. Neither is
    "the" weighted Spearman, so a comparison between two fields should use one
    definition throughout, as it does here.
    """
    return weighted_pearson(_rank(a), _rank(b), w)


def weighted_rank(x, w):
    """Midranks built from the weighted empirical distribution.

    Each sample's rank is the cumulative weight of all smaller values plus half
    its own group's weight, with ties sharing one rank. This is the alternative
    weighted-rank definition discussed in weighted_spearman's docstring.
    """
    x = np.asarray(x, np.float64); w = np.asarray(w, np.float64)
    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    change = np.r_[True, xs[1:] != xs[:-1]]
    gid = np.cumsum(change) - 1
    gw = np.bincount(gid, weights=ws)              # total weight per tied group
    below = np.concatenate([[0.0], np.cumsum(gw)[:-1]])
    r = (below + gw / 2.0)[gid]                    # weight below + half own group
    out = np.empty_like(r)
    out[order] = r
    return out


def weighted_spearman_rankdist(a, b, w):
    """Weighted Spearman with ranks built from the weighted distribution.

    The default weighted_spearman ranks by cell count and weights the
    correlation of those ranks. This variant builds the ranks themselves from
    the area-weighted empirical distribution, the other defensible definition.
    Reporting both shows how much the choice of definition carries.
    """
    return weighted_pearson(weighted_rank(a, w), weighted_rank(b, w), w)


def weighted_detection_contrast(index, ref, w, thr=0.0):
    """Area-weighted ratio of mean ref where the index fires to where it does not."""
    index = np.asarray(index, np.float64); ref = np.asarray(ref, np.float64)
    w = np.asarray(w, np.float64)
    hi, lo = index > thr, index <= thr
    swh, swl = w[hi].sum(), w[lo].sum()
    if swh <= 0 or swl <= 0:
        return {"mean_hi": np.nan, "mean_lo": np.nan, "ratio": np.nan,
                "n_hi": int(hi.sum())}
    mh = float((w[hi] * ref[hi]).sum() / swh)
    ml = float((w[lo] * ref[lo]).sum() / swl)
    return {"mean_hi": mh, "mean_lo": ml,
            "ratio": float(mh / ml) if ml != 0 else np.nan,
            "n_hi": int(hi.sum())}


def block_ids(lat, lon, deg=10.0):
    """Integer block label per sample for a spatial block bootstrap."""
    lat = np.asarray(lat, np.float64); lon = np.asarray(lon, np.float64)
    return (np.floor(lon / deg).astype(np.int64) * 100000
            + np.floor(lat / deg).astype(np.int64))


def block_bootstrap_ci(stat_fn, blocks, n_draws=2000, seed=0, alpha=0.05,
                       return_draws=False, null_value=0.0):
    """Percentile confidence interval for a statistic under block resampling.

    stat_fn takes an index array of sample positions and returns a scalar.
    Neighbouring grid cells are spatially correlated, so resampling individual
    cells understates the interval. Whole blocks are resampled instead.

    null_value sets the reference the draw fraction is reported against, which
    is 0 for a correlation or a difference and 1 for a ratio such as a detection
    contrast. The reported key is named for it, so a contrast bootstrap cannot
    be read as if zero were its null.
    """
    blocks = np.asarray(blocks)
    uniq = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in uniq}
    rng = np.random.RandomState(seed)
    vals = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        pick = uniq[rng.randint(0, uniq.size, uniq.size)]
        idx = np.concatenate([members[p] for p in pick])
        vals[i] = stat_fn(idx)
    good = vals[np.isfinite(vals)]
    if good.size == 0:
        out = {"lo": np.nan, "hi": np.nan, "n_blocks": int(uniq.size),
               "n_draws": int(n_draws), f"frac_gt_{null_value:g}": np.nan}
        if return_draws:
            out["draws"] = []
        return out
    lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out = {"lo": float(lo), "hi": float(hi), "n_blocks": int(uniq.size),
           "n_draws": int(n_draws),
           f"frac_gt_{null_value:g}": float((good > null_value).mean())}
    if return_draws:
        out["draws"] = [float(v) for v in vals]
    return out
