"""Unit tests for the thermal-inertia diurnal-range reduction (swi.thermal)."""

import numpy as np

from swi import thermal

HOURS8 = np.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=float)


def test_diurnal_range_basic():
    lon = np.array([0.0, 0.0])                       # LST == UTC hour
    series = np.array([280, 278, 282, 290, 295, 292, 286, 282], float)  # max h12, min h3
    stack = np.full((8, 1, 2), np.nan)
    stack[:, 0, 0] = series
    stack[[0, 4], 0, 1] = [285.0, 290.0]             # only two clear obs
    r = thermal.reduce_diurnal(stack, HOURS8, lon, min_obs=3)
    assert np.isclose(r["dT"][0, 0], 17.0)
    assert np.isclose(r["tmax"][0, 0], 295.0)
    assert np.isclose(r["tmin"][0, 0], 278.0)
    assert r["n_obs"][0, 0] == 8
    assert np.isclose(r["lst_max"][0, 0], 12.0)
    assert np.isclose(r["lst_min"][0, 0], 3.0)
    # too few clear obs -> dT masked, but n_obs still reported
    assert r["n_obs"][0, 1] == 2
    assert np.isnan(r["dT"][0, 1])


def test_all_nan_cell_is_nan():
    lon = np.array([0.0])
    stack = np.full((4, 1, 1), np.nan)
    r = thermal.reduce_diurnal(stack, np.array([0, 6, 12, 18.]), lon, min_obs=1)
    assert r["n_obs"][0, 0] == 0
    assert np.isnan(r["dT"][0, 0])
    assert np.isnan(r["tmax"][0, 0])
    assert np.isnan(r["lst_max"][0, 0])


def test_local_solar_time_longitude_offset():
    # warmest slot at UTC h=0; at lon 180 the local solar time is (0 + 180/15) = 12
    lon = np.array([180.0])
    stack = np.full((2, 1, 1), np.nan)
    stack[0, 0, 0] = 300.0
    stack[1, 0, 0] = 290.0
    r = thermal.reduce_diurnal(stack, np.array([0.0, 12.0]), lon, min_obs=1)
    assert np.isclose(r["lst_max"][0, 0], 12.0)
    assert np.isclose(r["dT"][0, 0], 10.0)
