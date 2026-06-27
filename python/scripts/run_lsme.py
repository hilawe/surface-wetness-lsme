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


from swi import lsme_inputs as _lsme_inputs

# These helpers now live in swi.lsme_inputs (A2 architecture cleanup); keep
# thin re-exports here so the existing scripts that import from run_lsme keep
# working unchanged.
placeholder_skin_temperature = _lsme_inputs.placeholder_skin_temperature
era5_field = _lsme_inputs.era5_field


# LSST anchor and gridsat_ts also live in swi.lsme_inputs now; thin re-exports.
_OVERPASS_LST_BY_CHAIN = _lsme_inputs._OVERPASS_LST_BY_CHAIN
_SAT_CHAIN = _lsme_inputs._SAT_CHAIN
_overpass_lst = _lsme_inputs.overpass_lst
gridsat_ts = _lsme_inputs.gridsat_ts


def _sat_id_from_csu_path(path):
    """Extract the satellite id from a CSU FCDR-GRID filename (re-export)."""
    return _lsme_inputs.sat_id_from_csu_path(path)


def build_inputs(path, lat, lon, argv, pass_="dsc"):
    """CLI-flag wrapper for swi.lsme_inputs.build_inputs.

    Parses --gridsat, --era5-ts, --nominal, --all-pixels and delegates to the
    library form. Kept here for backward compatibility with the existing
    monthly_lsme / make_two_estimator / validate_telsem callers.
    """
    return _lsme_inputs.build_inputs_from_argv(path, lat, lon, argv, pass_=pass_)


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT_FILE
    pass_ = argv[2] if len(argv) > 2 else DEFAULT_PASS

    lat, lon, tb, sensor = io_csu_grid.read_channels(path, pass_=pass_)
    print(f"file   : {path}")
    print(f"sensor : {sensor}   pass: {pass_}   grid: {tb.shape}")

    ts, ts_label, t_atm, t_atm_label, tcwv, tcwv_label, clear, clear_label = \
        build_inputs(path, lat, lon, argv, pass_=pass_)
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
