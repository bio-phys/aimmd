# aimmd

## Synopsis

aimmd - AI for Molecular Mechanism Discovery: Machine learning the reaction coordinate from shooting results.

## Code Example

Please see the Ipython notebooks in the example folder for an introduction.

## Motivation

This project exists because finding reaction coordinates of molecular systems is a great way of understanding how they work.

## Installation

aimmd interacts with openpathsampling through hooks which are called at predefined points during the TPS simulation, e.g. after every MC step. To install [openpathsampling]:

```bash
pip install openpathsampling
```

You should also install any molecular dynamics engines you want to use with openpathsampling for TPS, i.e. [openMM] and/or [GROMACS].

Now cd whereever you want to keep your local copy of aimmd, clone the repository and install aimmd using pip, e.g.

```bash
git clone https://github.com/bio-phys/aimmd.git
pip install -e aimmd/
```

For aimmd to be useful you need to install at least one machine learning backend. aimmd supports multiple different backends and can easily be extended to more. The backend is used to define the underlying machine learning models architecture and is used to fit the model. It naturally also defines the type of the model, i.e. neural network, symbolic regresssion, etc.
Currently supported backends are (model types in brackets):

- [pytorch] (neural network)
- [tensorflow]/keras (neural network)
- [dcgpy] (symbolic regression expressions) [Currently no iterative/on-the-fly training possible]

You should be able to install all of them using pip and/or conda. Please refer to their respective documentations for detailed installation instructions.

### TLDR:

```bash
pip install openpathsampling
git clone https://github.com/bio-phys/aimmd.git
pip install -e aimmd/
```

Additionally you will need to install at least one of the machine learning backends, i.e. [pytorch], [tensorflow] and/or [dcgpy].
Note also that you might want to install additional engines for openpathsampling, i.e. [openMM] and/or [GROMACS].

## API Reference

There is none (yet). Please read the example notebooks, the docstrings and the code.

## Tests

Tests use pytest. Use `pytest .` while in the toplevel directory of the repository to run them.

## License

GPL v3

---
<sub>This README.md is printed from 100% recycled electrons.</sub>

[pytorch]: https://pytorch.org
[tensorflow]: https://www.tensorflow.org
[dcgpy]: http://darioizzo.github.io/dcgp/
[openMM]: http://openmm.org/
[openpathsampling]: http://openpathsampling.org/latest/
[GROMACS]: http://www.gromacs.org/
