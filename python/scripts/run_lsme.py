"""Driver for the LSME first-order emissivity harness (Steps 1-2).

Runs the emissivity estimator on one CSU SSM/I FCDR daily file and prints the
per-channel dry-versus-wet contrast. Until Ken Knapp's GridSat-B1 arrives, the
skin temperature is a PLACEHOLDER (a smooth latitude field that does NOT encode
surface wetness, so the emissivity contrast comes from the microwave Tb alone)
and there is no cloud mask. The two TODO markers below are the exact plug-in
points.

Usage:
    python -m scripts.run_lsme [path-to-fcdr.nc] [asc|dsc] [--era5-ts] [--nominal]

  --gridsat F : use Ken's GridSat-B1 cloud-cleared product as the REAL Ts and
              clear-sky mask (F = a GridSat cld file). This is the actual Step-1
              input; takes precedence over --era5-ts.
  --era5-ts : use ERA5 skin temperature as a realistic DEVELOPMENT stand-in for
              Ts (not the final independent-IR Ts) instead of the smooth-latitude
              placeholder. With ERA5 also supplying the atmospheric temperature
              and water vapour, this is a fully realistic end-to-end dry run.
  --nominal : ignore ERA5 entirely (constant atmosphere, placeholder Ts).

Default: a July 1998 F-13 descending file (the ~06h morning overpass).
"""

import os
import sys

import numpy as np

from swi import io_csu_grid, lsme, io_era5_atmos, telsem, io_gridsat

DEFAULT_FILE = "../data/f13_1998/CSU_SSMI_FCDR-GRID_V02R00_F13_D19980715.nc"
DEFAULT_PASS = "dsc"


def placeholder_skin_temperature(lat, lon):
    """Stand-in T_s: smooth, warm in the tropics, cooler at the poles.

    Deliberately independent of the brightness temperature so it cannot
    manufacture the emissivity signal. Replaced by Ken's GridSat-B1 T_s.
    """
    ts_1d = 300.0 - 0.55 * np.abs(lat)          # ~300 K equator, ~250 K pole
    return np.repeat(ts_1d[:, np.newaxis], lon.size, axis=1)


def era5_field(path, var, lat, lon):
    """Load one ERA5 field for this file's month on the (lat, lon) grid, or None.

    Looks for ../era5_atmos/era5_atmos_<YYYYMM>.nc next to the data tree.
    """
    ymd = io_era5_atmos.date_from_csu_name(path)
    if ymd is None:
        return None
    era5 = os.path.join(os.path.dirname(path), "..", "era5_atmos",
                        f"era5_atmos_{ymd[0]}{ymd[1]:02d}.nc")
    if not os.path.exists(era5):
        return None
    return io_era5_atmos.field_on_grid(era5, var, lat, lon, day=ymd, mode="morning")


def gridsat_ts(arg, lat, lon, csu_path=None):
    """Load GridSat clear-sky Ts + clear mask from --gridsat's value, or None.

    arg is either a single GridSat cld FILE (one timestep) or the cld ROOT DIR
    (with per-year subdirs); for a directory the overpass-day composite for the
    CSU file's date is used. Returns (ts, clear) on the (lat, lon) grid or None.
    """
    if not arg:
        return None
    if os.path.isdir(arg):
        ymd = io_era5_atmos.date_from_csu_name(csu_path) if csu_path else None
        if ymd is None:
            return None
        # Ken's layout: <root>/isccp (GRIDSAT-CLOUD .nc) + <root>/ml (ML masks).
        # Use the ISCCP dir for the .nc and the ML mask when both are present.
        isccp = os.path.join(arg, "isccp")
        ml_root = os.path.join(arg, "ml")
        root = isccp if os.path.isdir(isccp) else arg
        ml_root = ml_root if os.path.isdir(ml_root) else None
        try:
            return io_gridsat.day_overpass_on_grid(root, ymd[0], ymd[1], ymd[2],
                                                   lat, lon, ml_root=ml_root)
        except FileNotFoundError:
            return None
    if os.path.exists(arg):
        return io_gridsat.ts_on_grid(arg, lat, lon)
    return None


def build_inputs(path, lat, lon, argv):
    """Assemble (ts, ts_label, t_atm, t_atm_label, tcwv, tcwv_label, clear, clear_label).

    Centralizes the Ts / atmospheric-temperature / water-vapour / domain choice
    from the CLI flags, the ERA5 file, and (when given) Ken's GridSat-B1, so
    run_lsme and validate_telsem stay in sync. Ts precedence: --gridsat (the real
    cloud-cleared Ts) > --era5-ts (dev stand-in) > placeholder.
    """
    nominal = "--nominal" in argv
    tcwv = None if nominal else era5_field(path, "tcwv", lat, lon)
    t_atm = None if nominal else era5_field(path, "t2m", lat, lon)

    gridsat_clear = None
    ts, ts_label = placeholder_skin_temperature(lat, lon), \
        "PLACEHOLDER smooth-latitude field (awaiting GridSat-B1)"
    if "--gridsat" in argv:
        g = gridsat_ts(argv[argv.index("--gridsat") + 1], lat, lon, csu_path=path)
        if g is not None:
            ts, gridsat_clear = g
            ts_label = "GridSat-B1 cloud-cleared Ts (clr); mask from cld"
        else:
            ts_label = "PLACEHOLDER (GridSat file not found)"
    elif "--era5-ts" in argv and not nominal:
        skt = era5_field(path, "skt", lat, lon)
        if skt is not None:
            ts = skt
            ts_label = ("ERA5 skin temperature (realistic DEV stand-in; "
                        "NOT the final independent-IR Ts)")

    if tcwv is None:
        tcwv_label = "nominal dry atmosphere (--nominal or no ERA5 file)"
    else:
        f = tcwv[np.isfinite(tcwv)]
        tcwv_label = (f"ERA5 morning-overpass field, {f.min():.1f}..{f.max():.1f} mm "
                      f"(median {np.median(f):.1f})")
    t_atm_label = "ts - 10 K lapse" if t_atm is None else "ERA5 2 m temperature"

    # Domain: GridSat clear-sky AND land when GridSat is in use; otherwise land
    # only by default (the TELSEM reference is a land atlas; ocean drags it down).
    # --all-pixels disables masking.
    if "--all-pixels" in argv:
        clear, clear_label = None, "all valid pixels (land + ocean)"
    elif gridsat_clear is not None:
        clear = gridsat_clear & telsem.land_mask(lat, lon)
        clear_label = "GridSat clear-sky AND land"
    else:
        clear = telsem.land_mask(lat, lon)
        clear_label = "land only (global_land_mask)"
    return (ts, ts_label, t_atm, t_atm_label, tcwv, tcwv_label, clear, clear_label)


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT_FILE
    pass_ = argv[2] if len(argv) > 2 else DEFAULT_PASS

    lat, lon, tb, sensor = io_csu_grid.read_channels(path, pass_=pass_)
    print(f"file   : {path}")
    print(f"sensor : {sensor}   pass: {pass_}   grid: {tb.shape}")

    ts, ts_label, t_atm, t_atm_label, tcwv, tcwv_label, clear, clear_label = \
        build_inputs(path, lat, lon, argv)
    print(f"Ts     : {ts_label}")
    print(f"domain : {clear_label}")
    print(f"T_atm  : {t_atm_label}")
    print(f"tcwv   : {tcwv_label}\n")

    r = lsme.derive_emissivity(tb, ts, clear=clear, tcwv_mm=tcwv, t_atm=t_atm)
    print(f"clear-sky pixels solved: {r['n_clear']:,}\n")

    print(f"{'ch':>4} {'n':>9} {'e_p10':>7} {'e_med':>7} {'e_p90':>7}   "
          f"{'apparent_med':>12}")
    summ = lsme.contrast_summary(r)
    app = r["apparent"]
    for i, s in enumerate(summ):
        a = app[..., i]
        a = a[np.isfinite(a)]
        amed = float(np.median(a)) if a.size else float("nan")
        print(f"{s['channel']:>4} {s['n']:>9,} {s['p10']:>7.3f} "
              f"{s['median']:>7.3f} {s['p90']:>7.3f}   {amed:>12.3f}")

    print("\nRead: window channels (19V/37V) should split high emissivity over "
          "dry land\nfrom low emissivity over water/wet surfaces. With a real "
          "land + cloud mask\nthe spread tightens to the TELSEM 0.9-vs-0.5 "
          "land/ocean contrast.")
    # TODO Step 3: validate r['emissivity'] against the TELSEM July climatology.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
