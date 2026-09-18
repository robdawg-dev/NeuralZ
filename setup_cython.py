"""Build script for the Cython-accelerated engine (the default AlphaGo.go /
AlphaGo.preprocessing.preprocessing). The frozen pure-Python reference implementations
(AlphaGo/go_python.py, AlphaGo/preprocessing/preprocessing_python.py,
AlphaGo/preprocessing/game_converter_python.py) are unaffected by this build.

Run with: python setup_cython.py build_ext --inplace
"""
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

_NPY_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
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
        compiler_directives={"language_level": "3"},
    ),
)
