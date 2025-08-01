# Installation

Independently of how you install aimmd you will need a working installation of one of the machine learning backends ([pytorch], [tensorflow], etc.).
Please refer to their respective documentations for installation instructions.

aimmd runs its TPS simulations either through [asyncmd] (if managing and learning from many simulations simultaneously, possibly on a HPC cluster) or through [openpathsampling] (in the sequential case). [openpathsampling] and [asyncmd] can both easily be installed via pip and are automatically installed when you install aimmd with pip. Note, that for [asyncmd] and/or [openpathsampling] to work you need to install a molecular dynamics engine to perform the trajectory integration. In [asyncmd] the only currently supported engine is [gromacs], while [openpathsampling] can use both [gromacs] and [openMM] (but [openMM] is highly recommended).

## pip install from PyPi

aimmd is published on [PyPi] (since v0.9.1), installing is as easy as:

```bash
pip install aimmd
```

## pip install directly from the repository

In case you you intend to run the tests or {doc}`example notebooks </examples_link/README>` yourself or if want to install the latest and greatest development version of aimmd (see the {doc}`changelog </include_changelog>` for whats new) you will need to install aimmd from the git repository.

This will clone the repository to the current working directory and install aimmd into the current python environment:

```bash
git clone https://github.com/bio-phys/aimmd.git
cd aimmd
pip install .
```

(tests-installation)=
### Tests

Tests use [pytest]. To run them you can install aimmd with the tests requirements. All tests should either pass or be skipped.

This will clone the repository to the current working directory and install aimmd with the tests requirement into the current python environment:

```bash
git clone https://github.com/bio-phys/aimmd.git
cd aimmd
pip install .\[tests\]
# or use
pip install .\[tests-all\]
# to also install optional dependencies needed to run all tests
```

you can then run the tests (against the installed version) as

```bash
pytest
# or use
pytest -v
# to get a more detailed report
```

```{note}
The ``tests-all`` target will also install [coverage] and [pytest-cov], see the [developer installation below](developer-installation) for more.
```

(documentation-installation)=
### Documentation

The documentation can be build with [sphinx], use e.g. the following to build it in html format:

```bash
cd aimmd  # Need to be at the top folder of the repository for the next line to work
sphinx-build -b html docs/source docs/build/html
```

The documentation is located in the `docs/source/` folder and (mostly) written in [MyST] markdown.
[MyST-NB] and the [sphinx-book-theme] are needed to build the documentation and include the example notebooks into it.

```{note}
Use ```pip install .\[docs\]``` to install the requirements needed to build the documentation.
```

(developer-installation)=
## Developer installation

If you intend to contribute to aimmd, it is recommended to use the ``dev`` extra and use an editable install to enable you to directly test your changes:

```bash
git clone https://github.com/bio-phys/aimmd.git
cd aimmd
pip install -e .\[dev\]
```

This will, in addition to the requirements to run the tests and to build the documentation, install [jupyterlab] such that you can easily contribute to the example notebooks.
It will also install [coverage] and its [pytest-cov] plugin such that you have an idea of the test coverage for your newly added code.
To get a nice html coverage report you can run the tests as

```bash
pytest --cov=aimmd --cov-report=html
```

[asyncmd]: https://pypi.org/project/asyncmd/
[coverage]: https://pypi.org/project/coverage/
[GROMACS]: https://www.gromacs.org/
[jupyterlab]: https://jupyterlab.readthedocs.io/en/stable/
[MyST]: https://mystmd.org/
[MyST-NB]: https://myst-nb.readthedocs.io/en/latest/
[openMM]: http://openmm.org/
[openpathsampling]: http://openpathsampling.org/latest/
[PyPi]: https://pypi.org/project/aimmd/
[pytest]: https://docs.pytest.org/en/latest/
[pytest-cov]: https://pypi.org/project/pytest-cov/
[pytorch]: https://pytorch.org
[tensorflow]: https://www.tensorflow.org
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[sphinx-book-theme]: https://sphinx-book-theme.readthedocs.io/en/stable/
