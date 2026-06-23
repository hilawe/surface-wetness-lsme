"""pytest bootstrap: make the swi package importable.

The LSME work does not use the Surface Wetness Index decision-tree engine, so
no C library is built here. The companion `surface-wetness-index` repository
hosts the Basist revival and its C oracle.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
