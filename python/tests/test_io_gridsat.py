"""Tests for the GridSat-B1 cloud-cleared reader (guarded on a sample file)."""

import os

import numpy as np
import pytest

from swi import io_gridsat as ig

_SAMP = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gridsat_sample")
SAMPLE = os.path.join(_SAMP, "GRIDSAT-B1.1985.01.01.06.v02r01.nc")        # old probe
SAMPLE98 = os.path.join(_SAMP, "GRIDSAT-CLOUD.1998.07.01.06.nc")          # Ken 1998
SAMPLE_ML = os.path.join(_SAMP, "ML-CLOUD.1998.07.01.06.npz")            # Ken ML mask

# F-13 style 0.25 degree target grid (south-first, lon 0..360)
DST_LAT = np.linspace(-89.875, 89.875, 720)
DST_LON = np.linspace(0.125, 359.875, 1440)


def test_path_builder():
    # default: Ken's flat GRIDSAT-CLOUD layout
    p = ig.gridsat_cld_path("/path/to/swi_data/data/gridsat_cld/isccp", 1998, 7, 15, 6)
    assert p.endswith("/GRIDSAT-CLOUD.1998.07.15.06.nc")
    # old probe layout still available
    q = ig.gridsat_cld_path("/store/isccp/gridsat/data", 1985, 1, 1, 6,
                            prefix="GRIDSAT-B1", suffix=".v02r01.nc", year_subdir=True)
    assert q.endswith("1985/GRIDSAT-B1.1985.01.01.06.v02r01.nc")
    # ML mask path
    m = ig.ml_mask_path("/path/to/swi_data/data/gridsat_cld/ml", 1998, 7, 15, 6)
    assert m.endswith("/ML-CLOUD.1998.07.15.06.npz")


def test_overpass_hour_per_lon_picks_local_06h():
    hours = ig.GRIDSAT_HOURS                       # (0,3,...,21)
    lon = np.array([0.0, 180.0])
    idx = ig.overpass_hour_per_lon(hours, lon)
    # lon 0: local 06h is UTC 06 -> index of 6; lon 180: local 06h is UTC 18
    assert hours[idx[0]] == 6
    assert hours[idx[1]] == 18


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="GridSat sample not staged")
def test_read_clr_cld_shapes_and_conventions():
    lat, lon, clr, clear = ig.read_clr_cld(SAMPLE)
    assert lat.size == 2000 and lon.size == 5143
    assert lon.min() >= 0.0 and lon.max() < 360.0
    assert np.all(np.diff(lon) >= 0)                 # ascending
    assert clr.shape == (2000, 5143)
    assert clear.dtype == bool and clear.any()
    fin = clr[np.isfinite(clr)]
    assert fin.size > 0
    assert 150.0 < fin.min() and fin.max() < 350.0   # Kelvin, physical
    # Ts is only defined where the pixel is clear
    assert np.isfinite(clr[clear]).all()


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="GridSat sample not staged")
def test_ts_on_grid_downsamples_to_quarter_degree():
    ts, clear = ig.ts_on_grid(SAMPLE, DST_LAT, DST_LON)
    assert ts.shape == (720, 1440)
    assert clear.shape == (720, 1440)
    assert clear.sum() > 100                         # some clear cells landed
    fin = ts[np.isfinite(ts)]
    assert 150.0 < fin.min() and fin.max() < 350.0
    assert np.array_equal(clear, np.isfinite(ts))


@pytest.mark.skipif(not (os.path.exists(SAMPLE98) and os.path.exists(SAMPLE_ML)),
                    reason="Ken 1998 GRIDSAT-CLOUD + ML samples not staged")
def test_real_1998_isccp_and_ml_mask():
    # the GRIDSAT-CLOUD .nc reads like the probe file (same clr/cld/grid)
    lat, lon, clr, clear = ig.read_clr_cld(SAMPLE98)
    assert lat.size == 2000 and lon.size == 5143 and clear.any()

    # ML mask is a 2000x5143 uint8 with 0=clear, 1=cloud, 255=fill
    ml = ig.read_ml_mask(SAMPLE_ML)
    assert ml.shape == (2000, 5143)
    assert set(np.unique(ml)).issubset({0, 1, 255})

    # screening with the ML mask gives a clear set close to the ISCCP cld set
    _, _, _, clear_ml = ig.read_clr_cld(SAMPLE98, ml_path=SAMPLE_ML)
    frac_cld = clear.mean()
    frac_ml = clear_ml.mean()
    assert abs(frac_cld - frac_ml) < 0.10            # the two masks roughly agree

    # ts_on_grid with the ML mask produces a physical Ts field on the F-13 grid
    ts, gclear = ig.ts_on_grid(SAMPLE98, DST_LAT, DST_LON, ml_path=SAMPLE_ML)
    assert ts.shape == (720, 1440) and gclear_ok(ts, gclear)


def gclear_ok(ts, clear):
    fin = ts[np.isfinite(ts)]
    return (clear.sum() > 100 and fin.size > 0
            and 150.0 < fin.min() and fin.max() < 350.0
            and np.array_equal(clear, np.isfinite(ts)))
