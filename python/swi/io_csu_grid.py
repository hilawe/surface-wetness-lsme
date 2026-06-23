"""Reader for the CSU ICDR-GRID daily 0.25-degree brightness-temperature files.

These NOAA CDR files are already gridded (lat 720 x lon 1440, 0.25 degree,
ascending and descending separated, Tb in Kelvin), so no swath binning is
needed: we read the seven Basist-equivalent channels directly and hand them to
the engine.

Channel mapping to the Basist order [19V 19H 22V 37V 37H 85V 85H]:

- SSM/I  : exact (19.35 / 22.235 / 37.0 / 85.5 GHz).
- SSMIS  : 91.655 GHz substitutes for 85.5 (see the 85-vs-91 note in
           docs/the project documentation).
- AMSR2  : APPROXIMATE only (18.7->19, 23.8->22, 36.5->37, 89.0->85). AMSR2 is a
           different instrument; feeding it into the SSM/I-tuned algorithm is a
           data-path proof of concept, not a validated product.

Variable names are `fcdr_<base>_<asc|dsc>`; fill is -9999.9.
"""

import numpy as np

from .channels import N_CHANNELS
from . import core_numpy

# Base channel names (without the fcdr_ prefix or _asc/_dsc suffix), in Basist
# order P1..P7 = 19V 19H 22V 37V 37H 85V 85H.
SENSOR_CHANNELS = {
    "ssmi":  ["tb19v", "tb19h", "tb22v", "tb37v", "tb37h", "tb85v", "tb85h"],
    "ssmis": ["tb19v", "tb19h", "tb22v", "tb37v", "tb37h", "tb91v", "tb91h"],
    "amsr2": ["tb18v", "tb18h", "tb23v", "tb36v", "tb36h", "tb89va", "tb89ha"],
}

FILL = -9999.9


def detect_sensor(ds):
    """Pick the sensor whose channel set is present in the dataset."""
    for sensor, bases in SENSOR_CHANNELS.items():
        if all(f"fcdr_{b}_asc" in ds.variables for b in bases):
            return sensor
    raise ValueError("no known sensor channel set found in file")


def read_channels(path, pass_="asc", sensor=None):
    """Read the seven Basist-order channels from a CSU ICDR-GRID file.

    Returns (lat, lon, tb, sensor) where tb is (nlat, nlon, 7) float32 Kelvin
    with NaN where any channel is missing/fill.
    """
    import netCDF4 as nc

    if pass_ not in ("asc", "dsc"):
        raise ValueError("pass_ must be 'asc' or 'dsc'")
    ds = nc.Dataset(path)
    try:
        sensor = sensor or detect_sensor(ds)
        bases = SENSOR_CHANNELS[sensor]
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["lon"][:], dtype=np.float64)
        nlat, nlon = lat.size, lon.size
        tb = np.empty((nlat, nlon, N_CHANNELS), dtype=np.float32)
        for i, base in enumerate(bases):
            v = ds[f"fcdr_{base}_{pass_}"][:]
            a = np.ma.filled(v, np.nan).astype(np.float32)
            a[a <= FILL + 1.0] = np.nan          # guard against fill leakage
            tb[:, :, i] = a
    finally:
        ds.close()
    return lat, lon, tb, sensor


def evaluate_file(path, pass_="asc", sensor=None, engine=core_numpy):
    """Read a CSU ICDR-GRID file and run the engine for one pass.

    Returns a dict with lat, lon, sensor, pass, and the output grids temp/wet/
    snow/ret (nlat x nlon), filled with NaN (temp/wet) or -128 (snow/ret) where
    no valid input existed.
    """
    lat, lon, tb, sensor = read_channels(path, pass_=pass_, sensor=sensor)
    nlat, nlon = lat.size, lon.size

    # Flag any required channel that is entirely missing (e.g. the F-17 37V
    # channel, inoperable since 2016). The algorithm needs all seven.
    finite_per_chan = np.isfinite(tb).reshape(-1, N_CHANNELS).sum(axis=0)
    empty_channels = [SENSOR_CHANNELS[sensor][i]
                      for i in range(N_CHANNELS) if finite_per_chan[i] == 0]

    valid = np.isfinite(tb).all(axis=2)          # need all 7 channels

    temp = np.full((nlat, nlon), np.nan, dtype=np.float32)
    wet = np.full((nlat, nlon), np.nan, dtype=np.float32)
    snow = np.full((nlat, nlon), -128, dtype=np.int32)
    ret = np.full((nlat, nlon), -128, dtype=np.int32)

    if valid.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            r = engine.evaluate_kelvin(tb[valid])
        temp[valid] = r.temp
        wet[valid] = r.wet
        snow[valid] = r.snow
        ret[valid] = r.ret

    return {
        "lat": lat, "lon": lon, "sensor": sensor, "pass": pass_,
        "valid": valid, "temp": temp, "wet": wet, "snow": snow, "ret": ret,
        "empty_channels": empty_channels,
    }
