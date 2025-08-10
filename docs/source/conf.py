# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup ---------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../../')
                                   )
                )

# -- Project information ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata

project = 'aimmd'
copyright = '2018-now, Hendrik Jung'
author = 'Hendrik Jung'
# The full version, including alpha/beta/rc tags
version = release = importlib.metadata.version("aimmd")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",
    'myst_nb',
        ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output --------------------------------------------------
# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/bio-phys/aimmd.git",
    "repository_branch": "main",

    "use_edit_page_button": True,
    "use_source_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    #"use_sidenotes": True,

    "show_toc_level": 2,  # show up to one sub-heading on the right sidebar

    # links with icons in left sidebar
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bio-phys/aimmd.git",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/aimmd/",
            "icon": "https://img.shields.io/pypi/dm/aimmd",
            "type": "url",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for autosummary extension-----------------------------------------
autosummary_imported_members = False  # default = False
autosummary_ignore_module_all = True  # default = True

# -- Options for autodoc extension --------------------------------------------
autodoc_default_options = {
                    # document members (default = False)
                    "members": False,
                           }

# -- Options for intersphinx extension ----------------------------------------
intersphinx_mapping = {
            "asyncmd": ("https://asyncmd.readthedocs.io/en/latest/", None),
            #"h5py": ("https://docs.h5py.org/en/stable", None),
            }

# -- Options for MyST (Parser) ------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath",  # enable math rendering
                          ]

# -- Options for MyST-NB ------------------------------------------------------
nb_execution_mode = "off"  # render the notebooks as they are in the repository
