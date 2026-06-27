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
    ilon = np.clip(np.searchsorted(src_lon_n, dst_lon360), 0, src_lon_n.size - 1)
    # refine: searchsorted gives the insertion point, pick the closer neighbor.
    # For longitude, the closer neighbor includes the wrap-around case.
    for idx, coord, src_c in ((ilat, dst_lat, src_lat),):
        left = np.clip(idx - 1, 0, src_c.size - 1)
        choose_left = np.abs(src_c[left] - coord) < np.abs(src_c[idx] - coord)
        idx[choose_left] = left[choose_left]
    # Longitude refinement, with wrap-around distance.
    def _lon_dist(a, b):
        d = np.abs(a - b) % 360.0
        return np.minimum(d, 360.0 - d)
    left = (ilon - 1) % src_lon_n.size
    choose_left = _lon_dist(src_lon_n[left], dst_lon360) < _lon_dist(src_lon_n[ilon], dst_lon360)
    ilon[choose_left] = left[choose_left]
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
