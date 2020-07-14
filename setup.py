# -*- coding: utf-8 -*-
"""
Created on Sa Nov 10 16:39:29 2018

@author: Hendrik Jung

Usage:
   cd /the/folder/where/this/setup.py/lies
   pip install -e .
Or (not recommended) with:
   python setup.py

If you want linetrace for the tests
give --install-option='--linetrace' to pip install
or --global-option='--linetrace' to pip install
or --linetrace option to setup.py


This setup.py is a strongly modified version,
originally adopted from https://github.com/pypa/sampleproject

This file is part of ARCD.

ARCD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ARCD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ARCD. If not, see <https://www.gnu.org/licenses/>.
"""
import os
import sys
# always prefer setuptools over distutils!
from setuptools import setup, find_packages
from setuptools.extension import Extension

# sort out if we'll use linetracing
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

CY_EXTS = [Extension('arcd.coords._symmetry',
                     ['arcd/coords/_symmetry.pyx'],
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
with open(os.path.join(HERE, "arcd/__about__.py"), 'r') as fp:
    exec(fp.read(), about_dct)


setup(
    name="arcd",
    packages=find_packages(),
    ext_modules=EXT_MODULES,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=about_dct['__version__'],

    description='''
                Automatic Reaction Coordinate Discovery:
                Machine learning the reaction coordinate from shooting results.
                ''',

    long_description=LONG_DESCRIPTION,

    # The project's main homepage.
    url='https://gitea.kotspeicher.de/hejung/arcd',

    # Author details
    author=about_dct['__author__'],
    author_email=about_dct['__author_email__'],

    # Choose your license
    license=about_dct['__license__'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Scientists',
        'Topic :: Science :: Molecular Dynamics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',  # CI tested
        'Programming Language :: Python :: 3.7',  # CI tested
        'Programming Language :: Python :: 3.8',  # CI tested
        # NOTE: we use f-strings and therfore require python >= 3.6
        # NOTE: even without f-strings python 2 will not work as intended:
        # 1. we did not take care of integer division vs float division
        # 2. we use binary pickle formats for storing the trainers
        # 3. we use cython with language_level=3
    ],

    # What does your project relate to?
    keywords='science md integrators path sampling development',

    # List setup dependencies here.
    # If you need them at run-time you have to add them to install_requires too
    # Be aware of: https://github.com/pypa/setuptools/issues/209
    setup_requires=[
        'numpy',
        'cython',
    ],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
        'cython',
        'openpathsampling',
        'mdtraj',
        'networkx',
        #'dcgpy',
        'sympy',  # only used for dcgpy atm
        'h5py',  # for asrcd.Storage and for old loading/saving of keras models
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[test]
    extras_require={
        'test': ['pytest'],
        'dev': ['coverage', 'pytest', 'pytest-cov',
                'flake8', 'flake8-alfred', 'flake8-comprehensions',
                'flake8-docstrings', 'flake8-if-statements',
                'flake8-logging-format', 'flake8-todo'
                ],
    }
)
