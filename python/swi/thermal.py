"""Thermal-inertia wetness estimator from the GridSat-B1 diurnal infrared cycle.

This is the second of the two independent surface-wetness estimators in the joint
GridSat plus SSM/I plan (the first is microwave emissivity in ``swi.lsme``). For
each grid cell it builds the day's clear-sky diurnal infrared temperature range
``dT = Tmax - Tmin`` from the eight 3-hourly GridSat-B1 cloud-cleared skin
temperatures. A surface with high thermal inertia, which wet or moist soil has,
damps the diurnal swing, so a SMALL ``dT`` indicates a wetter surface and a large
``dT`` a drier one. This is the apparent-thermal-inertia idea (the Tarpley, Price,
and Carlson lineage), the day-night counterpart to the microwave emissivity.

Clouds are screened with the same ML cloud mask the emissivity chain uses, so the
two estimators share the GridSat-B1 anchor and the same clear-sky sampling. The
catch is that a clean ``dT`` needs clear observations near both the afternoon
maximum and the pre-dawn minimum, so ``lst_max`` and ``lst_min`` are returned for
screening.
"""

import os

import numpy as np

from . import io_gridsat


def diurnal_range_on_grid(root, year, month, day, dst_lat, dst_lon,
                          hours=None, ml_root=None, min_obs=3):
    """Per-cell clear-sky diurnal infrared range from a day of GridSat files.

    root holds the GRIDSAT-CLOUD .nc files (the isccp/ directory in Ken's
    layout); pass ml_root (the ml/ directory) to screen with the ML cloud mask.
    Reads each available 3-hourly slot, regrids its cloud-cleared Ts onto
    (dst_lat, dst_lon), and reduces over the day.

    Returns a dict on the target grid: dT (Tmax - Tmin in K, NaN where fewer than
    min_obs clear observations), tmax, tmin, n_obs (clear-observation count), and
    lst_max / lst_min (the local solar time of the warmest / coolest clear slot,
    to confirm the swing was actually sampled across the diurnal cycle).
    """
    if hours is None:
        hours = io_gridsat.GRIDSAT_HOURS
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    stack, have = [], []
    for h in hours:
        p = io_gridsat.gridsat_cld_path(root, year, month, day, h)
        if not os.path.exists(p):
            continue
        mlp = None
        if ml_root is not None:
            cand = io_gridsat.ml_mask_path(ml_root, year, month, day, h)
            mlp = cand if os.path.exists(cand) else None
        ts, _ = io_gridsat.ts_on_grid(p, dst_lat, dst_lon, ml_path=mlp)
        stack.append(ts)
        have.append(h)
    if not have:
        raise FileNotFoundError(
            f"no GridSat cloud files for {year}-{month:02d}-{day:02d} under {root}")

    stack = np.asarray(stack)                       # (T, nlat, nlon), NaN if not clear
    have = np.asarray(have, dtype=np.float64)
    return reduce_diurnal(stack, have, dst_lon, min_obs=min_obs)


def reduce_diurnal(stack, hours, dst_lon, min_obs=3, require_bracket=True,
                   day_window=(9.0, 18.0)):
    """Reduce a (T, nlat, nlon) clear-sky Ts stack to the diurnal-range fields.

    Pure array logic (no I/O), so it is unit-testable. hours is the length-T UTC
    hour of each slice; dst_lon the target longitudes (for local solar time).

    Clear sky is temporally autocorrelated, so a cell's surviving clear slots are
    often clustered at similar local times and then dT underestimates the true
    diurnal swing. With require_bracket (default), dT is kept only where the
    warmest clear slot falls in the daytime window and the coolest falls at night,
    so dT reflects a genuine day-minus-night difference. day_window is the LST
    range counted as daytime.
    """
    stack = np.asarray(stack, dtype=np.float64)
    hours = np.asarray(hours, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)
    finite = np.isfinite(stack)
    n_obs = finite.sum(axis=0)
    # Fill missing slots with -inf / +inf so plain max/min and argmax/argmin are
    # NaN-safe and warning-free; mask the empty cells back to NaN afterward.
    big = np.where(finite, stack, -np.inf)
    small = np.where(finite, stack, np.inf)
    idx_max = np.argmax(big, axis=0)
    idx_min = np.argmin(small, axis=0)
    tmax = big.max(axis=0)
    tmin = small.min(axis=0)
    lst = lambda hr: (hours[hr] + dst_lon[None, :] / 15.0) % 24.0
    no = n_obs == 0
    tmax = np.where(no, np.nan, tmax)
    tmin = np.where(no, np.nan, tmin)
    lst_max = np.where(no, np.nan, lst(idx_max))
    lst_min = np.where(no, np.nan, lst(idx_min))
    dlo, dhi = day_window
    valid = n_obs >= min_obs
    if require_bracket:
        warm_ok = (lst_max >= dlo) & (lst_max <= dhi)        # warm slot in daytime
        cool_ok = (lst_min < dlo) | (lst_min > dhi)          # cool slot at night (wraps midnight)
        valid = valid & warm_ok & cool_ok
    dT = np.where(valid, tmax - tmin, np.nan)
    return {"dT": dT, "tmax": tmax, "tmin": tmin, "n_obs": n_obs,
            "lst_max": lst_max, "lst_min": lst_min}


def split_root(gridsat_root):
    """Resolve a GridSat top directory into (isccp_dir, ml_dir or None).

    Mirrors run_lsme: <root>/isccp holds the GRIDSAT-CLOUD .nc files and
    <root>/ml the ML masks; fall back to the root itself if there is no isccp/.
    """
    isccp = os.path.join(gridsat_root, "isccp")
    ml = os.path.join(gridsat_root, "ml")
    root = isccp if os.path.isdir(isccp) else gridsat_root
    ml = ml if os.path.isdir(ml) else None
    return root, ml
