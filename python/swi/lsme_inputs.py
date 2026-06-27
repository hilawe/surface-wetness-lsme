"""Reusable LSME-input helpers (library form, no argv).

The LSME-driver scripts used to call into `scripts/run_lsme.py` for the shared
helpers (`gridsat_ts`, `era5_field`, `build_inputs`), so the scripts imported
each other and faked `sys.argv` to drive `build_inputs`. The Codex review
flagged that pattern as a drift source. This module is the library form, with
explicit keyword arguments rather than CLI-flag parsing. `scripts/run_lsme.py`
still owns the CLI front end and now delegates to these helpers.

Functions
---------
placeholder_skin_temperature(lat, lon)
    Smooth latitude band, not encoding microwave Tb.
era5_field(csu_path, var, lat, lon)
    Read one ERA5 variable for the CSU file's date onto the grid, or None.
gridsat_ts(gridsat_arg, lat, lon, *, csu_path, pass_, sat_id)
    Cloud-cleared GridSat-B1 Ts and clear mask on the grid, or None.
build_inputs(csu_path, lat, lon, *, pass_, ts_source, gridsat_arg, atmos_mode)
    Centralized choice of Ts / atmospheric temperature / water vapor / domain
    for the LSME inversion. Returns (ts, ts_label, t_atm, t_atm_label, tcwv,
    tcwv_label, clear, clear_label).
"""

import os
import re
import warnings

import numpy as np

from . import io_era5_atmos, io_gridsat, telsem

# Codex F3 fix: per-pass local-solar-time anchor for the GridSat overpass slot.
# Descending = morning for the morning chain (F-08/F-11/F-13/F-17), 3-4 hours
# later for the late-morning chain (F-10/F-14/F-15/F-16/F-18). When the
# satellite is unknown the morning-chain LSTs are used with a warning.
_OVERPASS_LST_BY_CHAIN = {
    "morning":      {"dsc":  6.0, "asc": 18.0},
    "late_morning": {"dsc":  9.5, "asc": 21.5},
}
_SAT_CHAIN = {
    "f08": "morning", "f11": "morning", "f13": "morning", "f17": "morning",
    "f10": "late_morning", "f14": "late_morning", "f15": "late_morning",
    "f16": "late_morning", "f18": "late_morning",
}


def sat_id_from_csu_path(path):
    """Extract the satellite id (e.g. 'f13') from a CSU FCDR-GRID filename, or None."""
    m = re.search(r"_(F\d{2})_D\d{8}", os.path.basename(path or ""))
    return m.group(1).lower() if m else None


def overpass_lst(pass_, sat_id=None):
    """Local-solar-time anchor for the GridSat overpass, given pass and satellite."""
    if pass_ not in ("asc", "dsc"):
        raise ValueError(f"pass_ must be 'asc' or 'dsc', got {pass_!r}")
    chain = _SAT_CHAIN.get((sat_id or "").lower())
    if chain is None:
        warnings.warn(
            f"Satellite {sat_id!r} not in the overpass-LST table; assuming "
            f"morning chain (descending ~06h, ascending ~18h).",
            RuntimeWarning, stacklevel=3)
        chain = "morning"
    return _OVERPASS_LST_BY_CHAIN[chain][pass_]


def placeholder_skin_temperature(lat, lon):
    """Smooth latitude-only field for a stand-in skin temperature."""
    ts_1d = 300.0 - 0.55 * np.abs(lat)
    return np.repeat(ts_1d[:, np.newaxis], lon.size, axis=1)


def era5_field(csu_path, var, lat, lon):
    """Load one ERA5 field for the CSU file's date on the (lat, lon) grid, or None.

    Looks for ../era5_atmos/era5_atmos_<YYYYMM>.nc next to the data tree.
    """
    ymd = io_era5_atmos.date_from_csu_name(csu_path)
    if ymd is None:
        return None
    era5 = os.path.join(os.path.dirname(csu_path), "..", "era5_atmos",
                        f"era5_atmos_{ymd[0]}{ymd[1]:02d}.nc")
    if not os.path.exists(era5):
        return None
    return io_era5_atmos.field_on_grid(era5, var, lat, lon, day=ymd, mode="morning")


def gridsat_ts(gridsat_arg, lat, lon, *, csu_path=None, pass_="dsc", sat_id=None):
    """Cloud-cleared GridSat-B1 Ts and clear mask on the grid, or None.

    `gridsat_arg` is either a single GridSat cld file or the cld root directory.
    For a directory the overpass-day composite for the CSU file's date is used,
    with the per-pass and per-satellite LST anchor.
    """
    if not gridsat_arg:
        return None
    if os.path.isdir(gridsat_arg):
        ymd = io_era5_atmos.date_from_csu_name(csu_path) if csu_path else None
        if ymd is None:
            return None
        isccp = os.path.join(gridsat_arg, "isccp")
        ml_root = os.path.join(gridsat_arg, "ml")
        root = isccp if os.path.isdir(isccp) else gridsat_arg
        ml_root = ml_root if os.path.isdir(ml_root) else None
        lst_target = overpass_lst(pass_, sat_id)
        try:
            return io_gridsat.day_overpass_on_grid(
                root, ymd[0], ymd[1], ymd[2], lat, lon,
                ml_root=ml_root, lst_target=lst_target)
        except FileNotFoundError:
            return None
    if os.path.exists(gridsat_arg):
        return io_gridsat.ts_on_grid(gridsat_arg, lat, lon)
    return None


def build_inputs(csu_path, lat, lon, *,
                 pass_="dsc",
                 ts_source="placeholder",
                 gridsat_arg=None,
                 atmos_mode="era5",
                 domain="land"):
    """Centralized LSME input assembly, library form.

    Parameters
    ----------
    csu_path : str
        Path to the CSU FCDR-GRID daily file.
    lat, lon : 1-D arrays
        Target grid axes.
    pass_ : 'dsc' or 'asc'
        Microwave pass; drives the GridSat overpass LST anchor.
    ts_source : 'placeholder', 'gridsat', or 'era5'
        Skin-temperature source. 'gridsat' requires `gridsat_arg`.
    gridsat_arg : str or None
        File or root directory passed to `gridsat_ts`.
    atmos_mode : 'era5' or 'nominal'
        'era5' loads tcwv and t2m from the matching ERA5 file; 'nominal' uses
        a constant atmosphere.
    domain : 'land', 'clear_and_land', or 'all'
        Clear-sky and land masking. 'clear_and_land' requires GridSat.

    Returns
    -------
    (ts, ts_label, t_atm, t_atm_label, tcwv, tcwv_label, clear, clear_label)
    """
    sat_id = sat_id_from_csu_path(csu_path)
    nominal = atmos_mode == "nominal"
    tcwv = None if nominal else era5_field(csu_path, "tcwv", lat, lon)
    t_atm = None if nominal else era5_field(csu_path, "t2m", lat, lon)

    gridsat_clear = None
    ts = placeholder_skin_temperature(lat, lon)
    ts_label = "PLACEHOLDER smooth-latitude field (awaiting GridSat-B1)"

    if ts_source == "gridsat":
        g = gridsat_ts(gridsat_arg, lat, lon, csu_path=csu_path,
                       pass_=pass_, sat_id=sat_id)
        if g is not None:
            ts, gridsat_clear = g
            ts_label = (f"GridSat-B1 cloud-cleared Ts (clr); mask from cld; "
                        f"pass={pass_}, sat={sat_id or 'unknown'}")
        else:
            ts_label = "PLACEHOLDER (GridSat file not found)"
    elif ts_source == "era5":
        skt = era5_field(csu_path, "skt", lat, lon)
        if skt is not None:
            ts = skt
            ts_label = ("ERA5 skin temperature (realistic DEV stand-in; "
                        "NOT the final independent-IR Ts)")

    if tcwv is None:
        tcwv_label = "nominal dry atmosphere"
    else:
        f = tcwv[np.isfinite(tcwv)]
        tcwv_label = (f"ERA5 morning-overpass field, "
                      f"{f.min():.1f}..{f.max():.1f} mm "
                      f"(median {np.median(f):.1f})")
    t_atm_label = "ts - 10 K lapse" if t_atm is None else "ERA5 2 m temperature"

    if domain == "all":
        clear, clear_label = None, "all valid pixels (land + ocean)"
    elif domain == "clear_and_land" or (gridsat_clear is not None
                                        and domain == "land"):
        if gridsat_clear is not None:
            clear = gridsat_clear & telsem.land_mask(lat, lon)
            clear_label = "GridSat clear-sky AND land"
        else:
            clear = telsem.land_mask(lat, lon)
            clear_label = "land only (GridSat unavailable)"
    else:  # "land"
        clear = telsem.land_mask(lat, lon)
        clear_label = "land only (global_land_mask)"
    return (ts, ts_label, t_atm, t_atm_label, tcwv, tcwv_label, clear, clear_label)


def build_inputs_from_argv(csu_path, lat, lon, argv, pass_="dsc"):
    """Argv-compatible wrapper for the legacy scripts/run_lsme entry point.

    Parses the original CLI flags (`--gridsat`, `--era5-ts`, `--nominal`,
    `--all-pixels`) and delegates to the library `build_inputs`. Kept so the
    existing scripts continue to work without changing their call sites.
    """
    if "--nominal" in argv:
        atmos_mode = "nominal"
    else:
        atmos_mode = "era5"
    domain = "all" if "--all-pixels" in argv else "land"

    if "--gridsat" in argv:
        ts_source = "gridsat"
        gridsat_arg = argv[argv.index("--gridsat") + 1]
    elif "--era5-ts" in argv and atmos_mode != "nominal":
        ts_source = "era5"
        gridsat_arg = None
    else:
        ts_source = "placeholder"
        gridsat_arg = None

    return build_inputs(csu_path, lat, lon,
                        pass_=pass_, ts_source=ts_source,
                        gridsat_arg=gridsat_arg,
                        atmos_mode=atmos_mode, domain=domain)
