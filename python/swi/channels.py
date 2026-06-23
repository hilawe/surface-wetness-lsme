"""Channel conventions and sentinels for the Surface Wetness Index engine.

The Basist decision tree consumes the seven SSMI imager channels per grid cell.
Values are integers in the NATIVE PACKED domain:

    packed = brightness_temperature_Kelvin - 70

which is exactly the byte stored in the daily 1/3-degree grids (0..255).
sig_recog re-adds the +70 K offset internally.
"""

import numpy as np

# Channel indices into the last axis of a (..., 7) array.
CH19V, CH19H, CH22V, CH37V, CH37H, CH85V, CH85H = range(7)
CHANNEL_NAMES = ["19V", "19H", "22V", "37V", "37H", "85V", "85H"]
N_CHANNELS = 7

# Packing offset (sig_recog's NV) and the daily-grid fill byte.
NV = 70
FILL_BYTE = 32

# Output sentinels (see docs/the project documentation).
RTEMP_BAD = -99.0     # undefined / unusable land skin temperature
WET_BAD = -99.0       # condition prevents a wetness retrieval
WET_DRY = 0.0
SNOW_NONE = 0
SNOW_ICE = -1         # ice / glacial
SNOW_BAD = -99        # bad data
SNOW_GAP = -100       # orbital gap


def kelvin_to_packed(tb):
    """Kelvin brightness temperatures -> native packed integers (round, -70)."""
    return np.rint(np.asarray(tb, dtype=np.float64)).astype(np.int32) - NV


def packed_to_kelvin(packed):
    """Native packed integers -> Kelvin (add 70)."""
    return np.asarray(packed, dtype=np.int32) + NV
