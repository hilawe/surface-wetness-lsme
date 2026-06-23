"""Unit tests for the approximate geostationary VZA geometry (swi.geostationary)."""

import numpy as np

from swi import geostationary as geo


def test_vza_zero_at_subpoint():
    assert np.isclose(geo.vza_from_sublon(0.0, 0.0, 0.0), 0.0, atol=1e-6)


def test_vza_known_central_angle():
    # 60 degrees from the sub-satellite point gives about 68 degrees VZA
    v = geo.vza_from_sublon(0.0, 60.0, 0.0)
    assert 67.0 < v < 69.5


def test_vza_increases_with_distance():
    v10 = geo.vza_from_sublon(0.0, 10.0, 0.0)
    v40 = geo.vza_from_sublon(0.0, 40.0, 0.0)
    assert 0.0 < v10 < v40 < 90.0


def test_vza_nan_beyond_horizon():
    assert np.isnan(geo.vza_from_sublon(0.0, 85.0, 0.0))


def test_min_geo_vza_low_at_subpoint_high_at_seam():
    lat = np.array([0.0])
    lon = np.array([-135.0, -105.0, -75.0])
    vza = geo.min_geo_vza(lat, lon, sublons=(-135.0, -75.0))
    assert vza[0, 0] < 1.0                 # at the GOES-W sub-point
    assert vza[0, 2] < 1.0                 # at the GOES-E sub-point
    assert vza[0, 1] > vza[0, 0]           # the seam between them is higher VZA
    assert 30.0 < vza[0, 1] < 40.0         # ~35 degrees at 30 deg central angle
