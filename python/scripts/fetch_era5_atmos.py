"""Fetch ERA5 single-level atmospheric fields for the LSME clear-sky correction.

Pulls the inputs the first-order atmospheric correction needs (Step 2 of
docs/the project documentation):

  total_column_water_vapour   -> per-channel transmissivity (swi.lsme)
  2m_temperature              -> effective atmospheric temperature
  skin_temperature            -> reanalysis cross-check on Ken's GridSat-B1 T_s

Default target is July 1998 (the proof-of-concept month), 0.25 degree, four
synoptic times (00/06/12/18 UTC) so the ~06h local morning overpass can be
interpolated per longitude. Requires a configured ~/.cdsapirc.

Usage:
    python -m scripts.fetch_era5_atmos [YYYY] [MM] [--out DIR]
"""

import os
import sys

VARIABLES = [
    "total_column_water_vapour",
    "2m_temperature",
    "skin_temperature",
]
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def fetch(year, month, out_dir):
    import cdsapi

    os.makedirs(out_dir, exist_ok=True)
    days = [f"{d:02d}" for d in range(1, 32)]
    target = os.path.join(out_dir, f"era5_atmos_{year}{int(month):02d}.nc")
    if os.path.exists(target):
        print(f"already present: {target}")
        return target

    c = cdsapi.Client()
    print(f"requesting ERA5 single-levels {year}-{int(month):02d} -> {target}")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year": str(year),
            "month": f"{int(month):02d}",
            "day": days,
            "time": TIMES,
            "grid": [0.25, 0.25],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        target,
    )
    print(f"done: {target}")
    return target


def main(argv):
    year = int(argv[1]) if len(argv) > 1 and not argv[1].startswith("-") else 1998
    month = int(argv[2]) if len(argv) > 2 and not argv[2].startswith("-") else 7
    out_dir = "../data/era5_atmos"
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
    fetch(year, month, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
