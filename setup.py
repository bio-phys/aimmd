# -*- coding: utf-8 -*-
"""
Created on Sa Nov 10 16:39:29 2018

@author: Hendrik Jung

Usage:
   cd /the/folder/where/this/setup.py/lies
   pip install -e .
Or (not recommended) with:
   python setup.py

If you want linetraceing for the tests of cython functions
give --install-option='--linetrace' to pip install
or --global-option='--linetrace' to pip install
or --linetrace option to setup.py


This setup.py is a strongly modified version,
originally adopted from https://github.com/pypa/sampleproject

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
import os
import sys
# always prefer setuptools over distutils!
from setuptools import setup, find_packages
from setuptools.extension import Extension

# sort out if we'll use linetracing for cython
if '--linetrace' in sys.argv:
    LINETRACE = True
    sys.argv.remove('--linetrace')
else:
    LINETRACE = False

# test for and setup cython
try:
    import Cython
    from Cython.Build import cythonize
    Cython.Compiler.Options.get_directive_defaults()['language_level'] = 3
except (ImportError, ModuleNotFoundError):
    # need cython to build symmetry functions
    raise ModuleNotFoundError("Cython not found. Cython is needed to build the"
                              + " symmetry functions. Please install it and "
                              + "then rerun this setup.")

if LINETRACE:
    Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
    # need this to get coverage of the function definitions
    Cython.Compiler.Options.get_directive_defaults()['binding'] = True

# prepare Cython modules
try:
    import numpy
except (ImportError, ModuleNotFoundError):
    # need numpy to build symmetry functions
    raise ModuleNotFoundError("Numpy not found. Numpy is needed to build the"
                              + " symmetry functions. Please install it and "
                              + "then rerun this setup.")
else:
    # include_dirs must contains the '.' for setup to search
    # all the subfolders of the project root
    include_dirs = ['.', numpy.get_include(), 'm']

CY_EXTS = [Extension('aimmd.coords._symmetry',
                     ['aimmd/coords/_symmetry.pyx'],
                     include_dirs=include_dirs,
                     extra_compile_args=["-O3", "-march=native", "-fopenmp"],
                     extra_link_args=['-fopenmp'])
           ]

# set linetrace macro if wanted
if LINETRACE:
    for ext in CY_EXTS:
        ext.define_macros.append(('CYTHON_TRACE_NOGIL', 1))

# cythonize the extensions
# always recompile
EXT_MODULES = cythonize(CY_EXTS, force=True)

# Get the long description from the README file
HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# Get version and other stuff from __about__.py
about_dct = {'__file__': __file__}
with open(os.path.join(HERE, "aimmd/__about__.py"), 'r') as fp:
    exec(fp.read(), about_dct)

# Define the remaining arguments for setup
NAME = about_dct["__title__"]
PACKAGES = find_packages()
VERSION = about_dct['__version__']
DESCRIPTION = about_dct["__description__"]
URL = about_dct["__url__"]
AUTHOR = about_dct['__author__']
AUTOR_EMAIL = about_dct['__author_email__']
LICENSE = about_dct['__license__']
CLASSIFIERS = [  # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Scientists',
        'Topic :: Science :: Molecular Dynamics',
        'Topic :: Science :: Enhanced Sampling'
        # License
        'License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE',
        # Supported python versions
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        # NOTE: we use asyncio features and type annotations that need >= 3.9
        # NOTE: we use f-strings and therfore require python >= 3.6
        # NOTE: even without f-strings python 2 will not work as intended:
        # 1. we did not take care of integer division vs float division
        # 2. we use binary pickle formats for storing the trainers
        # 3. we use cython with language_level=3
               ]
KEYWORDS = ["science", "MD", "Molecular Dynamics", "Path Sampling",
            "Transition Path Sampling", "TPS", "Machine Learning", "ML",
            "committor", "commitment probability", "reaction coordinate", "RC",
            ]
# List setup dependencies here.
# If you need them at run-time you have to add them to install_requires too
# Be aware of: https://github.com/pypa/setuptools/issues/209
SETUP_REQUIRES = [
        'numpy>=1.17.0',
        'cython',
                  ]
# List run-time dependencies here. These will be installed by pip when
# your project is installed. For an analysis of "install_requires" vs pip's
# requirements files see:
# https://packaging.python.org/en/latest/requirements.html
INSTALL_REQUIRES = [
        'numpy>=1.17.0',  # v>=1.17.0 because we use 'new-style' RNGs
        'cython',
        'scipy',
        'openpathsampling',
        'mdtraj',
        'networkx',
        #'dcgpy',  # dont install dcgpy automatically as it should best be
                   # installed from conda-forge
        'sympy',  # only used with dcgpy atm, but not a dcgpy dependency, so
                  # to take care of installing ourselfs
        # for aimmd.Storage (and for old loading/saving of keras models)
        'h5py>=3',  # need >=3 for the 'new' string handling
        'asyncmd',  # needed for distributed
        'mdanalysis',  # needed for distributed examples, but is an asyncmd
                       # dependency anyway
                    ]
# List additional groups of dependencies here (e.g. development
# dependencies). You can install these using the following syntax,
# for example:
# $ pip install -e .[test]
EXTRAS_REQUIRE = {
        "test": ['pytest', 'pytest-asyncio']
                  }
EXTRAS_REQUIRE["dev"] = (EXTRAS_REQUIRE["test"]
                         + ['coverage', 'pytest-cov',
                            'flake8', 'flake8-alfred', 'flake8-comprehensions',
                            'flake8-docstrings', 'flake8-if-statements',
                            'flake8-logging-format', 'flake8-todo',
                            ]
                         )

# run the setup
setup(
    name=NAME,
    packages=PACKAGES,
    ext_modules=EXT_MODULES,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTOR_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': EXTRAS_REQUIRE["test"],
        'dev': EXTRAS_REQUIRE["dev"],
                    }
)
