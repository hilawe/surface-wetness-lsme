"""Combine the two independent GridSat-B1-anchored wetness estimators.

The joint GridSat-B1 plus SSM/I work has two physically independent estimators of
relative land surface wetness, both anchored on the same GridSat-B1 record:

  thermal     the clear-sky diurnal infrared range dT = Tmax - Tmin (swi.thermal).
              A high thermal inertia (wet or vegetated surface) damps the swing,
              so a SMALL dT is the wet end.
  microwave   the 19 GHz vertical-minus-horizontal polarization difference
              (V - H) from the derived emissivity. It is large over bare, dry,
              smooth surfaces and small over vegetated or wet ones, so a SMALL
              polarization difference is the wet end.

The two were shown to agree (about +0.25 correlation over land) on the
bare-versus-vegetated and wetness axis, which is what makes a combined product
meaningful. This module turns each into a 0-to-1 relative wetness index by
rank-normalizing over land (with the sign chosen so that 1 is wet for both), then
averages them and reports their agreement.

Honest scope: this is a RELATIVE, monthly, surface-wetness-and-vegetation index,
not absolute soil moisture, and it is most meaningful over low vegetation. The
absolute level carries surface-type structure; the time-varying wetness signal is
cleanest as an anomaly against a climatology, which needs a longer record than is
in hand yet.
"""

import numpy as np


def rank_normalize(field, mask):
    """Percentile rank in 0..1 of field over the masked cells, NaN elsewhere.

    Rank-normalizing (rather than a z-score) is robust to the skewed, bounded
    distributions of dT and the polarization difference. Uses average ranks for
    ties so the result is invariant to ordering inside tied values.
    """
    from scipy.stats import rankdata
    field = np.asarray(field, dtype=np.float64)
    out = np.full(field.shape, np.nan)
    sel = np.asarray(mask, dtype=bool) & np.isfinite(field)
    vals = field[sel]
    if vals.size == 0:
        return out
    # Average-rank percentile in [0, 1]. rankdata returns 1..N, shift to 0..N-1.
    ranks = rankdata(vals, method="average") - 1.0
    out[sel] = ranks / max(vals.size - 1, 1)
    return out


def wetness_index(field, mask, wet_is_low=True):
    """Relative wetness in 0..1 from a field (1 = wet). wet_is_low inverts the
    rank so that the LOW end of the field maps to wet (true for both dT and the
    polarization difference). NOTE: callers driving the joint product should
    use rank_normalize_shared so both estimators are ranked over the same
    cells.
    """
    r = rank_normalize(field, mask)
    return (1.0 - r) if wet_is_low else r


def shared_support_mask(thermal_field, microwave_field, land_mask):
    """Cells where BOTH estimators have finite data on land.

    The Codex review caught a real bug: rank-normalizing each estimator over
    its own clear-sky support and then comparing them cell by cell makes the
    percentiles on a shared cell reference different populations, biasing both
    `combined_wetness` and `agreement`. Ranking both estimators over the
    shared support is the fix.
    """
    return (np.asarray(land_mask, dtype=bool)
            & np.isfinite(np.asarray(thermal_field, dtype=np.float64))
            & np.isfinite(np.asarray(microwave_field, dtype=np.float64)))


def joint_indices(thermal_field, microwave_field, land_mask,
                  thermal_wet_is_low=True, microwave_wet_is_low=True):
    """Rank-normalize both estimators over a SHARED-SUPPORT mask, then return
    (thermal_wet, microwave_wet, shared_mask).

    Both percentiles reference the same cell population, so combining them or
    differencing them is now meaningful. Both default to wet-is-low because the
    diurnal range dT (thermal) and the 19 GHz V-H polarization difference
    (microwave) both fall on the wet end.
    """
    shared = shared_support_mask(thermal_field, microwave_field, land_mask)
    t = rank_normalize(thermal_field, shared)
    m = rank_normalize(microwave_field, shared)
    if thermal_wet_is_low:
        t = np.where(shared, 1.0 - t, np.nan)
    else:
        t = np.where(shared, t, np.nan)
    if microwave_wet_is_low:
        m = np.where(shared, 1.0 - m, np.nan)
    else:
        m = np.where(shared, m, np.nan)
    return t, m, shared


def combine(thermal_wet, micro_wet):
    """Combined wetness and agreement where both indices are valid.

    Returns (combined, agreement). Combined is the mean of the two 0-to-1
    indices. Agreement is 1 - |difference|, so 1 means the two estimators
    place the cell at the same point on the wet-to-dry scale and 0 means
    opposite ends.

    PRECONDITION: thermal_wet and micro_wet must have been rank-normalized
    over the SAME support (use joint_indices), otherwise the cell-wise
    comparison is biased. The function does not enforce this because the
    callers passing arbitrary fields are tested elsewhere, but the joint
    product driver is responsible for using joint_indices.
    """
    thermal_wet = np.asarray(thermal_wet, dtype=np.float64)
    micro_wet = np.asarray(micro_wet, dtype=np.float64)
    both = np.isfinite(thermal_wet) & np.isfinite(micro_wet)
    combined = np.full(thermal_wet.shape, np.nan)
    agreement = np.full(thermal_wet.shape, np.nan)
    combined[both] = 0.5 * (thermal_wet[both] + micro_wet[both])
    agreement[both] = 1.0 - np.abs(thermal_wet[both] - micro_wet[both])
    return combined, agreement


def write_product(path, lat, lon, fields, attrs):
    """Write the two-estimator wetness product as a CF-1.8 NetCDF file.

    fields is a dict name -> (array, units, long_name). attrs is a dict of global
    attributes. Float fields are stored as float32 with NaN fill.
    """
    import netCDF4

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlat.units = "degrees_north"; vlat.standard_name = "latitude"; vlat[:] = lat
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlon.units = "degrees_east"; vlon.standard_name = "longitude"; vlon[:] = lon
        for name, (arr, units, long_name) in fields.items():
            v = ds.createVariable(name, "f4", ("lat", "lon"),
                                  fill_value=np.float32(np.nan), zlib=True)
            v.units = units; v.long_name = long_name
            v[:] = np.asarray(arr, dtype=np.float32)
        for k, val in attrs.items():
            setattr(ds, k, val)
    finally:
        ds.close()
    return path
