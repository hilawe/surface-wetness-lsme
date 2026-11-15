"""The shared land mask.

The function reads nothing but the two coordinate axes, but it is shared by
published subsets of this repository that ship different sets of modules, so
where it lives is a boundary question rather than a matter of taste. While it
lived among the grid utilities, a subset that excludes those could not import
it and failed for anyone who cloned it, although every test here passed.

THIS FILE SHIPS IN EVERY SUBSET, so it imports nothing but the shared module.
Each re-export entry point is tested in the file that travels with it, and an
import added here would fail whichever subset does not carry that module.
"""

import numpy as np

from swi import landmask


LAT = np.array([-10.0, 0.0, 10.0, 40.0])
LON = np.array([0.0, 90.0, 190.0, 350.0])


def test_the_mask_has_the_grid_shape_and_is_boolean():
    out = landmask.land_mask(LAT, LON)
    assert out.shape == (LAT.size, LON.size)
    assert out.dtype == bool


def test_longitude_is_accepted_in_either_convention():
    """0..360 and -180..180 name the same cells and must mask the same."""
    lon_360 = np.array([0.0, 90.0, 190.0, 350.0])
    lon_180 = np.array([0.0, 90.0, -170.0, -10.0])
    assert np.array_equal(landmask.land_mask(LAT, lon_360),
                          landmask.land_mask(LAT, lon_180))


def test_it_separates_land_from_ocean():
    """A mask that is all land is the documented no-package fallback.

    The fallback exists so a caller without global_land_mask still runs, and
    the docstring tells callers who need a real mask to check for an all-True
    result. That makes an all-True return indistinguishable from a working
    mask over a land-only box, so this samples one point well inside a
    continent and one well out in an ocean.
    """
    out = landmask.land_mask(np.array([0.0]), np.array([20.0, 200.0]))
    assert out[0, 0], "equatorial Africa should read as land"
    assert not out[0, 1], "the equatorial Pacific should read as ocean"
