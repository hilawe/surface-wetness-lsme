"""Step 3 driver: validate the derived LSME emissivity against TELSEM.

Runs the emissivity harness on one CSU SSM/I FCDR file, loads the TELSEM July
climatology, and prints the per-channel skill (Spearman, Pearson, bias, RMSE)
over land. Until the real NWP-SAF TELSEM2 atlas is wired into
swi.telsem.load_atlas, this falls back to a SYNTHETIC atlas and says so loudly:
the numbers are then a machinery check, not a validation result. The skin
temperature is still the placeholder until Ken Knapp's GridSat-B1 T_s lands.

Usage:
    python -m scripts.validate_telsem [path-to-fcdr.nc] [asc|dsc] [--telsem DIR]
"""

import sys

import numpy as np

from swi import io_csu_grid, lsme, telsem
from scripts.run_lsme import build_inputs, DEFAULT_FILE, DEFAULT_PASS


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = args[0] if len(args) > 0 else DEFAULT_FILE
    pass_ = args[1] if len(args) > 1 else DEFAULT_PASS
    telsem_dir = None
    if "--telsem" in argv:
        telsem_dir = argv[argv.index("--telsem") + 1]
    month = 7

    lat, lon, tb, sensor = io_csu_grid.read_channels(path, pass_=pass_)
    ts, ts_label, t_atm, _, tcwv, _, clear, _ = build_inputs(path, lat, lon, argv,
                                                              pass_=pass_)
    r = lsme.derive_emissivity(tb, ts, tcwv_mm=tcwv, t_atm=t_atm, clear=clear)
    print(f"file   : {path}")
    print(f"sensor : {sensor}   pass: {pass_}   solved: {r['n_clear']:,} px")
    print(f"Ts     : {ts_label}")

    # TELSEM reference: real atlas if available, else synthetic with a notice.
    try:
        atlas_lat, atlas_lon, atlas = telsem.load_atlas(telsem_dir, month,
                                                        lat=lat, lon=lon)
        ref_kind = "REAL NWP-SAF TELSEM2"
    except (FileNotFoundError, NotImplementedError):
        atlas_lat, atlas_lon, atlas = lat, lon, telsem.synthetic_atlas(lat, lon, month)
        ref_kind = "SYNTHETIC (machinery check only, NOT a validation result)"
    print(f"TELSEM : {ref_kind}\n")

    rows = telsem.compare(r["emissivity"], lat, lon, atlas, atlas_lat, atlas_lon)
    print(f"{'ch':>4} {'telsem':>7} {'n':>9} {'spearman':>9} {'pearson':>8} "
          f"{'bias':>7} {'rmse':>6}")
    for s in rows:
        print(f"{s['channel']:>4} {s['telsem']:>7} {s['n']:>9,} "
              f"{s['spearman_r']:>9.3f} {s['pearson_r']:>8.3f} "
              f"{s['bias']:>7.3f} {s['rmse']:>6.3f}")

    print("\nWhen Ken's GridSat-B1 Ts replaces the placeholder and the real "
          "TELSEM2\natlas is loaded, these rows become the Step 3 validation: "
          "rank and\npattern agreement and bias of the derived emissivity vs "
          "the climatology.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
