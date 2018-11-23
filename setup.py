#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sa Nov 10 16:39:29 2018

@author: Hendrik Jung

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


Usage:
   cd /the/folder/where/this/setup.py/lies
   pip install -e .
Or (not recommended) with:
   python setup.py

If you want linetrace for the tests
give --install-option='--linetrace' to pip install
or --global-option='--linetrace' to pip install
or --linetrace option to setup.py
"""


import os
import sys
import subprocess
# always prefer setuptools over distutils!
from setuptools import setup, find_packages
from setuptools.extension import Extension


# +-----------------------------------------------------------------------------
# | GET GIT VERSION
# +-----------------------------------------------------------------------------

def get_git_version():
    # Return the git revision as a string
    # copied from numpy setup.py
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v

        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        output = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return output

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = 'Unknown'

    return git_revision


def scandir(directory, files=[], endswith='.pyx'):
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path) and path.endswith(endswith):
            files.append(path.replace(os.path.sep, ".")[:-len(endswith)])
        elif os.path.isdir(path):
            scandir(path, files, endswith=endswith)
    return files


def make_extension(ext_name, endswith='.pyx', **kwargs):
    # should never happen, we have setup_require=['numpy']
    try:
        import numpy
    except ImportError:
        include_dirs = ['.', 'm']
    else:
        #include_dirs = ['.', numpy.get_include(), 'm']
        include_dirs = ['.', numpy.get_include(), 'm',
                        '/opt/conda/envs/arcd_devel/include',
                        '/opt/conda/envs/arcd_devel/include/eigen3',
                        '/home/think/oprogs/OPS/arcd_deps/d-CGP/include']
    ext_path = ext_name.replace(".", os.path.sep)+endswith
    return Extension(
        ext_name,
        [ext_path],
        include_dirs=include_dirs,
        #extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        #extra_link_args=['-fopenmp'],
        extra_link_args=['-lquadmath', '-ltbb', '-lgmp', '-lmpfr',
                         '-L /opt/cona/envs/lib/',
                         '-lboost_python36',
                         ],
        **kwargs,
        # your include_dirs must contains the '.' for setup to search
        # ...all the subfolder of the codeRootFolder
        )


def make_ext_modules(use_cython=True, linetrace=False):
    if use_cython:
        ext_names = scandir('arcd', endswith='.pyx')
        if linetrace:
            extensions = [make_extension(name, endswith='.pyx',
                                         define_macros=[('CYTHON_TRACE_NOGIL', 1)])
                          for name in ext_names]
        else:
            extensions = [make_extension(name, endswith='.pyx')
                          for name in ext_names]
        # always recompile
        return cythonize(extensions, force=True)
    else:
        ext_names = scandir('arcd', endswith='.cpp')
        #ext_names = ['arcd.symreg.sources.optimize',
        #             'arcd.symreg.sources.core']
        extensions = [make_extension(name, endswith='.cpp')
                      for name in ext_names]
        return extensions


# sort out if we'll use cython, linetracing, etc
if '--linetrace' in sys.argv:
    LINETRACE = True
    sys.argv.remove('--linetrace')
else:
    LINETRACE = False

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

if USE_CYTHON and LINETRACE:
    import Cython
    Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
    # need this to get coverage of the function definitions
    Cython.Compiler.Options.get_directive_defaults()['binding'] = True


HERE = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

#import numpy
#include_dirs = ['.', numpy.get_include(), 'm',
#                        '/opt/conda/envs/arcd_devel/include',
#                        '/opt/conda/envs/arcd_devel/include/eigen3',
#                        '/home/think/oprogs/OPS/arcd_deps/d-CGP/include']
#exts = [Extension(
#        'arcd.symreg.sources.optimize',
#        ['arcd/symreg/sources/optimize.cpp'],
#        include_dirs=include_dirs,
#        #extra_compile_args=["-O3", "-march=native", "-fopenmp"],
#        #extra_link_args=['-fopenmp'],
#        extra_link_args=['-lquadmath', '-ltbb', '-lgmp', '-lmpfr',
#                         '-L /opt/cona/envs/lib/',
#                         '-lboost_python36',
#                         '-lboost_system',
#                         '-lboost_timer',
#                         '-lboost_chrono',
#                         '-lboost_serialization',
#                         '-lboost_unit_test_framework',
#                         ],
#        # your include_dirs must contains the '.' for setup to search
#        # ...all the subfolder of the codeRootFolder
#        ),
#        Extension(
#        'arcd.symreg.sources.core',
#        ['arcd/symreg/sources/core.cpp', 'arcd/symreg/sources/docstrings.cpp'],
#        include_dirs=include_dirs,
#        #extra_compile_args=["-O3", "-march=native", "-fopenmp"],
#        #extra_link_args=['-fopenmp'],
#        extra_link_args=['-lquadmath', '-ltbb', '-lgmp', '-lmpfr',
#                         '-L /opt/cona/envs/lib/',
#                         '-lboost_python36',
#                         '-lboost_system',
#                         '-lboost_timer',
#                         '-lboost_chrono',
#                         '-lboost_serialization',
#                         '-lboost_unit_test_framework',
#                         ],
#        # your include_dirs must contains the '.' for setup to search
#        # ...all the subfolder of the codeRootFolder
#        )]
#
setup(
    name="arcd",
    packages=find_packages(),
    #ext_modules=exts,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='''Automatic Reaction Coordinate Discovery: Machine learning the reaction coordinate from shooting results.''',

    long_description=LONG_DESCRIPTION,

    # The project's main homepage.
    url='https://gogs.kotspeicher.de/hejung/arcd',

    # Author details
    author='hejung',
    author_email='hendrik.andre.jung@gmail.com',

    # Choose your license
    license='LGPLv2.1',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Scientists',
        'Topic :: Science :: Molecular Dynamics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.x',  # should work, not tested
        'Programming Language :: Python :: 3.6',  # works, tested
        # NOTE: python 2 will most likely not work as intended:
        # 1. we did not take care of integer division vs float division
        # 2. we use binary pickle formats for storing the trainers
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
        'sympy',
    #    'openpathsampling',
    #    'mdtraj',
    #    'networkx',
    #    'h5py',  # for loading and saving of keras models
    #    'keras',
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[test]
    extras_require={
        'test': ['coverage', 'pytest', 'pytest-cov'],
    }
)
