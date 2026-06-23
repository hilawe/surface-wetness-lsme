"""Tests for the ERA5 atmospheric loader (pure logic + guarded real-file check)."""

import datetime as dt
import os

import numpy as np
import pytest

from swi import io_era5_atmos as io


def test_date_from_csu_name():
    p = "../data/f13_1998/CSU_SSMI_FCDR-GRID_V02R00_F13_D19980715.nc"
    assert io.date_from_csu_name(p) == (1998, 7, 15)
    assert io.date_from_csu_name("no_date_here.nc") is None


def test_flip_to_ascending_flips_lat_and_field():
    lat = np.array([90.0, 0.0, -90.0])             # descending
    field = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lat2, f2 = io._flip_to_ascending(lat, field)
    assert lat2[0] < lat2[-1]
    assert np.array_equal(f2, field[::-1, :])
    # already ascending is untouched
    lat3, f3 = io._flip_to_ascending(lat2, f2)
    assert np.array_equal(f3, f2)


def test_day_indices_selects_the_right_day():
    times = [dt.datetime(1998, 7, 1, h) for h in (0, 6, 12, 18)] + \
            [dt.datetime(1998, 7, 2, h) for h in (0, 6, 12, 18)]
    idx = io._day_indices(times, (1998, 7, 2))
    assert list(idx) == [4, 5, 6, 7]
    assert list(io._day_indices(times, None)) == list(range(8))


def test_morning_field_picks_local_06h_per_longitude():
    hours = [0.0, 6.0, 12.0, 18.0]
    lon = np.array([0.0, 180.0])
    # each time slice is filled with its own time index, so the output value
    # at a longitude equals the chosen time index there
    stack = np.stack([np.full((1, 2), t) for t in range(4)], axis=0)[:, 0, :]
    stack = stack.reshape(4, 1, 2)
    out = io._morning_field(stack, hours, lon)
    # lon 0: local 06h is UTC 06 -> index 1; lon 180: local 06h is UTC 18 -> index 3
    assert out.shape == (1, 2)
    assert out[0, 0] == 1
    assert out[0, 1] == 3


def test_reduce_field_daymean():
    times = [dt.datetime(1998, 7, 1, h) for h in (0, 6, 12, 18)]
    stack = np.stack([np.full((2, 2), v) for v in (10.0, 20.0, 30.0, 40.0)])
    out = io.reduce_field(stack, times, np.array([0.0, 90.0]),
                          day=(1998, 7, 1), mode="daymean")
    assert np.allclose(out, 25.0)


REAL = os.path.join(os.path.dirname(__file__), "..", "..",
                    "data", "era5_atmos", "era5_atmos_199807.nc")


@pytest.mark.skipif(not os.path.exists(REAL), reason="ERA5 July 1998 not staged")
def test_tcwv_on_grid_real_file_physical():
    dst_lat = np.linspace(-89.0, 89.0, 30)
    dst_lon = np.linspace(0.0, 357.0, 60)
    tcwv = io.tcwv_on_grid(REAL, dst_lat, dst_lon, day=(1998, 7, 15))
    assert tcwv.shape == (30, 60)
    finite = tcwv[np.isfinite(tcwv)]
    assert finite.size > 0
    assert finite.min() >= 0.0 and finite.max() < 100.0    # mm, physical


@pytest.mark.skipif(not os.path.exists(REAL), reason="ERA5 July 1998 not staged")
def test_field_on_grid_temperatures_physical():
    dst_lat = np.linspace(-89.0, 89.0, 30)
    dst_lon = np.linspace(0.0, 357.0, 60)
    for var in ("t2m", "skt"):
        f = io.field_on_grid(REAL, var, dst_lat, dst_lon, day=(1998, 7, 15))
        fin = f[np.isfinite(f)]
        assert fin.size > 0
        # Kelvin, physical Earth surface (Antarctic July plateau ~198 K to hot desert)
        assert 180.0 < fin.min() and fin.max() < 340.0
