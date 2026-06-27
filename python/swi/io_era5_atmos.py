"""Loader for the ERA5 single-level atmospheric fields used by swi.lsme.

Reads the file produced by scripts/fetch_era5_atmos.py (total column water
vapour, 2 m temperature, skin temperature; 0.25 degree; 00/06/12/18 UTC) and
delivers a field on the CSU F-13 grid for the first-order atmospheric
correction. Column water vapour (kg m-2 = mm) feeds swi.lsme.transmissivity.

Two design points matter for the emissivity method:

- The morning descending overpass is near 06 h LOCAL solar time. For each
  longitude the local hour of a given UTC time is (utc + lon/15) mod 24, so the
  default 'morning' reduction picks, per longitude column, the synoptic time
  whose local hour is closest to 06 h. 'daymean' (mean over the day) is the
  simpler fallback; column water vapour has only a weak diurnal cycle so the two
  are close, but the overpass pick is the right convention and matters more once
  the temperature fields are used.
- ERA5 latitude is stored north-first; we flip it to ascending so it matches the
  south-first CSU grid before the nearest-neighbour regrid.
"""

import re

import numpy as np

from . import validate

OVERPASS_LST_HOURS = 6.0          # local solar time of the morning descending pass


def date_from_csu_name(path):
    """Pull (year, month, day) from a CSU file name `..._D<YYYYMMDD>.nc`, or None."""
    m = re.search(r"_D(\d{4})(\d{2})(\d{2})", str(path))
    if not m:
        return None
    return tuple(int(g) for g in m.groups())


def _flip_to_ascending(lat, field):
    """Return (lat_asc, field_asc) so latitude increases along axis -2/0."""
    lat = np.asarray(lat, dtype=np.float64)
    if lat[0] > lat[-1]:
        return lat[::-1].copy(), field[..., ::-1, :].copy()
    return lat, field


def _day_indices(datetimes, day):
    """Indices of datetimes falling on `day` (a (y, m, d) tuple); all if None."""
    if day is None:
        return np.arange(len(datetimes))
    y, m, d = day
    return np.array([i for i, t in enumerate(datetimes)
                     if (t.year, t.month, t.day) == (y, m, d)], dtype=int)


def _morning_field(day_stack, hours, lon):
    """Per-longitude pick of the synoptic time nearest local OVERPASS_LST_HOURS.

    day_stack : (ntime, nlat, nlon); hours : (ntime,) UTC hours; lon : (nlon,).
    Returns (nlat, nlon).
    """
    hours = np.asarray(hours, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lst = (hours[:, None] + lon[None, :] / 15.0) % 24.0      # (ntime, nlon)
    dist = np.abs(lst - OVERPASS_LST_HOURS)
    dist = np.minimum(dist, 24.0 - dist)                     # circular
    best = np.argmin(dist, axis=0)                           # (nlon,)
    lon_idx = np.arange(lon.size)
    return day_stack[best, :, lon_idx].T                     # (nlat, nlon)


def reduce_field(stack, datetimes, lon, day=None, mode="morning"):
    """Reduce a (ntime, nlat, nlon) stack to one (nlat, nlon) field.

    mode 'morning' = per-longitude nearest-to-local-06h; 'daymean' = mean over
    the selected day's (or all) times.
    """
    idx = _day_indices(datetimes, day)
    if idx.size == 0:
        raise ValueError(f"no ERA5 times found for day {day}")
    sub = np.asarray(stack)[idx]
    if mode == "daymean":
        return np.asarray(sub.mean(axis=0), dtype=np.float64)
    if mode == "morning":
        hours = np.array([datetimes[i].hour for i in idx], dtype=np.float64)
        return _morning_field(sub, hours, lon)
    raise ValueError(f"unknown mode {mode!r}")


def field_on_grid(path, var, dst_lat, dst_lon, day=None, mode="morning"):
    """One ERA5 variable, overpass-reduced and regridded onto (dst_lat, dst_lon).

    var is the netCDF variable name: 'tcwv' (mm), 't2m' (K), or 'skt' (K). day is
    a (year, month, day) tuple selecting the overpass date; None uses the whole
    file. Returns a (dst_lat, dst_lon) float64 array.
    """
    import netCDF4 as nc

    ds = nc.Dataset(path)
    try:
        lat = np.asarray(ds["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds["longitude"][:], dtype=np.float64)
        vt = ds["valid_time"]
        datetimes = nc.num2date(vt[:], vt.units)
        data = np.ma.filled(ds[var][:], np.nan).astype(np.float64)
    finally:
        ds.close()

    lat, data = _flip_to_ascending(lat, data)
    # Normalize ERA5 longitude to 0..360 and rotate the data column-wise to
    # match. ERA5 commonly ships -180..180 while the product grid is 0..360, and
    # passing a -180..180 axis through regrid_nearest used to collapse the
    # eastern hemisphere onto the western edge column. regrid_nearest now also
    # handles this internally; explicit normalization here documents the
    # convention and makes downstream reduce_field calls unambiguous.
    lon360 = lon % 360.0
    order = np.argsort(lon360)
    lon = lon360[order]
    data = data[..., order]
    field = reduce_field(data, datetimes, lon, day=day, mode=mode)
    return validate.regrid_nearest(lat, lon, field,
                                   np.asarray(dst_lat, dtype=np.float64),
                                   np.asarray(dst_lon, dtype=np.float64))


def tcwv_on_grid(path, dst_lat, dst_lon, day=None, mode="morning"):
    """Total column water vapour (mm) on (dst_lat, dst_lon). See field_on_grid."""
    return field_on_grid(path, "tcwv", dst_lat, dst_lon, day=day, mode=mode)
