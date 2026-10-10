"""Build script for the Cython-accelerated engine (the default AlphaGo.go /
AlphaGo.preprocessing.preprocessing). The frozen pure-Python reference
implementations they superseded now live in deprecated/ and are unaffected by this build.

Run with: python setup_cython.py build_ext --inplace

For profiling, NEURALZ_CYTHON_PROFILE=1 builds the same extensions with Cython's profile
and linetrace hooks, so cProfile and line_profiler see individual Cython functions (the
ladder reader, each feature plane) instead of one opaque call. Those builds run slower:
never deploy or train with one - rebuild without the variable afterwards (switching modes
regenerates the C++ automatically).
"""
import os

import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

PROFILE = os.environ.get("NEURALZ_CYTHON_PROFILE") == "1"
# The generated C++ differs between the two modes but the .pyx sources don't, so Cython's
# own "is it up to date" check can't tell: remember the last mode, regenerate on a change.
_MODE_FILE = os.path.join("build", "cython_mode")
_mode = "profile" if PROFILE else "normal"
try:
    with open(_MODE_FILE) as f:
        _MODE_CHANGED = f.read().strip() != _mode
except OSError:
    _MODE_CHANGED = PROFILE  # no record: a default build needs no forced regeneration
os.makedirs("build", exist_ok=True)
with open(_MODE_FILE, "w") as f:
    f.write(_mode)
_NPY_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
if PROFILE:
    _NPY_MACROS += [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]
_INCLUDE_DIRS = [numpy.get_include()]

extensions = [
    Extension("AlphaGo.go.constants", ["AlphaGo/go/constants.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
    Extension("AlphaGo.go.ladders", ["AlphaGo/go/ladders.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
    Extension("AlphaGo.go.game_state", ["AlphaGo/go/game_state.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
    Extension("AlphaGo.go.group_logic", ["AlphaGo/go/group_logic.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
    Extension("AlphaGo.go.coordinates", ["AlphaGo/go/coordinates.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
    Extension("AlphaGo.go.zobrist", ["AlphaGo/go/zobrist.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
    Extension("AlphaGo.preprocessing.preprocessing",
              ["AlphaGo/preprocessing/preprocessing.pyx"],
              include_dirs=_INCLUDE_DIRS, define_macros=_NPY_MACROS, language="c++"),
]

setup(
    name="RocAlphaGo-cython-extensions",
    # We only want `build_ext --inplace` to compile the extensions below, not a full
    # package distribution - the repo root has other top-level dirs (Data/, SampleGames/,
    # tmpCythonVersion/) that confuse setuptools' automatic package discovery.
    packages=[],
    ext_modules=cythonize(
        extensions,
        compiler_directives=dict({"language_level": "3"},
                                 **({"profile": True, "linetrace": True} if PROFILE else {})),
        force=_MODE_CHANGED,
    ),
)
