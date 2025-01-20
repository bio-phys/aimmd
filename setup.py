"""
This file is part of AIMMD.

AIMMD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AIMMD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AIMMD. If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import numpy
# always prefer setuptools over distutils!
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


CY_EXTS = []
if sys.platform.startswith("darwin"):
    # we are on MacOS, so (probably) without openMP support,
    # i.e. probably we are using ApplClang in gcc compatibility mode
    # which is not really compatible to gcc as it is missing openMP (which is part of a gcc standard install)
    CY_EXTS += [Extension('aimmd.coords._symmetry',
                          ['aimmd/coords/_symmetry.pyx'],
                          include_dirs=["m", numpy.get_include()],
                          extra_compile_args=["-O3", "-march=native"],# "-fopenmp"],
                          # for now just try sans openMP
                          #extra_link_args=['-fopenmp'],
                          )
                ]
else:
    # lets just try with openmp and worst case is we have to deal with every other OS that comes up
    # (for linux this works)
    CY_EXTS += [Extension('aimmd.coords._symmetry',
                          ['aimmd/coords/_symmetry.pyx'],
                          include_dirs=["m", numpy.get_include()],
                          extra_compile_args=["-O3", "-march=native", "-fopenmp"],
                          extra_link_args=['-fopenmp'],
                          )
                ]


# run the setup
setup(
    ext_modules=cythonize(CY_EXTS),
)
