"""The channel reader loads in a tree that carries no engine.

`swi.io_csu_grid` ships in published subsets that carry different sets of
modules. One of them holds the channel reader and no decision-tree engine.
The reader used to import the engine when the module loaded, only to fill in
a default argument of one function, so in that subset nothing that touched
the reader could be imported, and no script there could start. Every test in
the source repository passed throughout, because the engine is always
importable here.

So the condition is made deliberately, in a fresh interpreter that refuses
the engine the way a tree without it would. A fresh interpreter is used
because a module that is already loaded in this one would answer the import
from memory and the refusal would never be consulted.

This file imports nothing but the reader, so it runs in every subset.
"""

import os
import subprocess
import sys

PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROBE = r'''
import sys


class RefuseEngine:
    def find_spec(self, name, path=None, target=None):
        if name == "swi.core_numpy":
            raise ModuleNotFoundError("No module named 'swi.core_numpy'")
        return None


sys.meta_path.insert(0, RefuseEngine())

from swi import io_csu_grid

assert "swi.core_numpy" not in sys.modules, "the engine was loaded"
assert callable(io_csu_grid.read_channels)
print("LOADED")

try:
    io_csu_grid.evaluate_file("no_such_file.nc")
except ImportError as exc:
    print("REFUSED:", exc)
'''


def _probe():
    return subprocess.run([sys.executable, "-c", PROBE], cwd=PYTHON_DIR,
                          capture_output=True, text=True)


def test_the_reader_loads_in_a_tree_without_the_engine():
    run = _probe()
    assert run.returncode == 0, run.stderr[-600:]
    assert "LOADED" in run.stdout


def test_asking_for_the_engine_where_there_is_none_says_so():
    """The one function that needs the engine names what is missing, before
    it opens the file. Left to run on, it would report the missing file and
    say nothing about the engine."""
    run = _probe()
    assert run.returncode == 0, run.stderr[-600:]
    refused = [line for line in run.stdout.splitlines()
               if line.startswith("REFUSED:")]
    assert refused, run.stdout
    assert "engine" in refused[0] and "read_channels" in refused[0]
