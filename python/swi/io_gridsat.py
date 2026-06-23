"""Reader for Ken Knapp's GridSat cloud-cleared product (the LSME Ts + mask).

Ken's 1998 delivery sits in two flat directories:

  isccp/  GRIDSAT-CLOUD.<YYYY>.<MM>.<DD>.<HH>.nc   (3-hourly, netCDF-4)
  ml/     ML-CLOUD.<YYYY>.<MM>.<DD>.<HH>.npz        (3-hourly, NumPy archive)

The ISCCP netCDF files carry both LSME plug-ins, on a 0.07 degree grid
(lat -70..70, lon -180..180, 2000 x 5143):

    clr = Estimated Clear Sky Brightness Temperature (K)  -> cloud-cleared Ts
    cld = Combined Cloud Flag Result                      -> ISCCP clear-sky mask
    irwin = 11 micron brightness temperature

The ML files hold the modern ESDS cloud mask as a single 2000 x 5143 uint8 array
under the key ``cloud_mask``: 0 = clear, 1 = cloud, 255 = fill (the clear fraction
matches the ISCCP cld==0 fraction to within a couple of percent on a sample, which
corroborates 0 = clear; confirm the exact convention with Ken).

This reader selects an overpass timestep, screens for clear sky (by the ISCCP cld
flag or the ML mask), converts longitude to the 0..360 convention of the CSU
microwave grid, and area-averages (downsamples) onto the F-13 0.25 degree grid,
returning the skin temperature and a clear-sky boolean mask ready for
``swi.lsme.derive_emissivity``.

The older `/store/isccp/gridsat` probe files used the name GRIDSAT-B1.<...>.v02r01.nc
under per-year subdirectories; gridsat_cld_path supports both layouts.
"""

import os

import numpy as np

from . import telsem

GRIDSAT_FILL = -31999
# Physical bounds on the clear-sky brightness temperature. Ken Knapp (2026-06-20)
# advised ignoring out-of-range clr where the underlying Tb is invalid and noted
# the ISCCP cld flag cannot be trusted as clear there, so the floor is 160 K (his
# suggested cutoff) and the ML mask is the primary screen rather than the cld flag.
CLR_MIN_K = 160.0
CLR_MAX_K = 350.0
CLEAR_FLAG = 0                 # ISCCP cld value taken as clear (ML mask preferred)
ML_CLEAR_VALUE = 0            # ML cloud_mask: 0 = clear, 1 = cloud, 255 = fill
OVERPASS_HOUR_UTC = 6
OVERPASS_LST = 6.0            # local solar time of the morning descending pass
GRIDSAT_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)


def gridsat_cld_path(root, year, month, day, hour=OVERPASS_HOUR_UTC,
                     prefix="GRIDSAT-CLOUD", suffix=".nc", year_subdir=False):
    """Path to a GridSat cloud netCDF file under ``root``.

    Defaults to Ken's flat 1998 layout (`GRIDSAT-CLOUD.<...>.nc`). Set
    prefix="GRIDSAT-B1", suffix=".v02r01.nc", year_subdir=True for the older
    `/store/isccp/gridsat` probe files.
    """
    name = f"{prefix}.{year:04d}.{month:02d}.{day:02d}.{hour:02d}{suffix}"
    if year_subdir:
        return os.path.join(root, f"{year:04d}", name)
    return os.path.join(root, name)


def ml_mask_path(ml_root, year, month, day, hour=OVERPASS_HOUR_UTC):
    """Path to the matching ML cloud-mask .npz under ``ml_root`` (flat layout)."""
    name = f"ML-CLOUD.{year:04d}.{month:02d}.{day:02d}.{hour:02d}.npz"
    return os.path.join(ml_root, name)


def read_ml_mask(npz_path):
    """Load the ML cloud mask (2000 x 5143 uint8: 0 clear, 1 cloud, 255 fill)."""
    z = np.load(npz_path)
    return np.asarray(z["cloud_mask"])


def read_clr_cld(path, clear_flag=CLEAR_FLAG, ml_path=None,
                 ml_clear_value=ML_CLEAR_VALUE):
    """Read one GridSat cloud file -> (lat, lon, clr_K, clear_mask).

    lon is returned in 0..360 ascending; clr is NaN at fill. The clear mask is
    True where clr is valid and the pixel is clear sky: by the ML mask
    (cloud_mask == ml_clear_value) when ml_path is given, otherwise by the ISCCP
    flag (cld == clear_flag).
    """
    import netCDF4 as nc

    d = nc.Dataset(path)
    try:
        lat = np.asarray(d["lat"][:], dtype=np.float64)
        lon = np.asarray(d["lon"][:], dtype=np.float64)
        clr = np.ma.filled(d["clr"][:], np.nan).astype(np.float64)
        cld = np.ma.filled(d["cld"][:], -999).astype(np.int64)
    finally:
        d.close()

    clr = np.squeeze(clr)
    cld = np.squeeze(cld)
    clr[clr <= GRIDSAT_FILL + 1] = np.nan
    # Reject unphysical clr regardless of the cloud flag: some cells carry bad
    # clear-sky-temperature values (seen near -130 K) that the ISCCP cld flag
    # still marks clear. The ML mask excludes most, but clamp to be safe.
    clr[(clr < CLR_MIN_K) | (clr > CLR_MAX_K)] = np.nan

    if ml_path is not None:
        ml = np.squeeze(read_ml_mask(ml_path))
        clear = (ml == ml_clear_value) & np.isfinite(clr)
    else:
        clear = (cld == clear_flag) & np.isfinite(clr)

    lon360 = lon % 360.0
    order = np.argsort(lon360)
    return lat, lon360[order], clr[:, order], clear[:, order]


def ts_on_grid(path, dst_lat, dst_lon, clear_flag=CLEAR_FLAG, ml_path=None):
    """Cloud-cleared Ts and clear mask from one GridSat file on (dst_lat, dst_lon).

    Returns (ts, clear) where ts is the clear-sky brightness temperature (K)
    area-averaged from the 0.07 degree GridSat cells into each target cell, NaN
    where no clear GridSat pixel fell, and clear is the boolean finite(ts) mask.
    Pass ml_path to screen with the ML mask instead of the ISCCP cld flag.
    """
    lat, lon, clr, clear = read_clr_cld(path, clear_flag=clear_flag,
                                        ml_path=ml_path)
    lon2d, lat2d = np.meshgrid(lon, lat)
    sel = clear
    cell_lat = lat2d[sel]
    cell_lon = lon2d[sel]
    cell_val = clr[sel][:, np.newaxis]                 # (N, 1) for resampler
    grid = telsem.resample_cells_to_grid(cell_lat, cell_lon, cell_val,
                                         dst_lat, dst_lon)
    ts = grid[:, :, 0]
    return ts, np.isfinite(ts)


def overpass_hour_per_lon(hours, lon, lst_target=OVERPASS_LST):
    """Index (into hours) of the UTC time nearest local lst_target, per longitude.

    For longitude lon the local solar time of a UTC hour h is (h + lon/15) mod 24.
    Returns a length-nlon array of indices into the hours sequence.
    """
    hours = np.asarray(hours, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lst = (hours[:, None] + lon[None, :] / 15.0) % 24.0
    dist = np.abs(lst - lst_target)
    dist = np.minimum(dist, 24.0 - dist)
    return np.argmin(dist, axis=0)


def day_overpass_on_grid(root, year, month, day, dst_lat, dst_lon,
                         hours=GRIDSAT_HOURS, clear_flag=CLEAR_FLAG,
                         ml_root=None):
    """Per-longitude morning-overpass Ts and clear mask from a day of GridSat files.

    Loads each available 3-hourly file for the date, regrids it, and for each
    longitude keeps the timestep whose local hour is closest to OVERPASS_LST. This
    matches the descending ~06h overpass better than a single UTC file. Pass
    ml_root to screen with the ML mask (ml/ alongside the isccp/ files). Missing
    hours are skipped; raises FileNotFoundError if the whole day is absent.
    """
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    ts_stack, clear_stack, have = [], [], []
    for h in hours:
        p = gridsat_cld_path(root, year, month, day, h)
        if not os.path.exists(p):
            continue
        mlp = None
        if ml_root is not None:
            cand = ml_mask_path(ml_root, year, month, day, h)
            mlp = cand if os.path.exists(cand) else None
        ts, clear = ts_on_grid(p, dst_lat, dst_lon, clear_flag=clear_flag,
                               ml_path=mlp)
        ts_stack.append(ts)
        clear_stack.append(clear)
        have.append(h)
    if not have:
        raise FileNotFoundError(
            f"no GridSat cloud files for {year}-{month:02d}-{day:02d} under {root}")

    ts_stack = np.asarray(ts_stack)
    clear_stack = np.asarray(clear_stack)
    best = overpass_hour_per_lon(have, dst_lon)
    lon_idx = np.arange(dst_lon.size)
    out_ts = ts_stack[best, :, lon_idx].T
    out_clear = clear_stack[best, :, lon_idx].T
    return out_ts, out_clear
