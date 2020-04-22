# arcd

## Synopsis

arcd - Automatic Reaction Coordinate Discovery: Machine learning the reaction coordinate from shooting results.

## Code Example

Please see the Ipython notebooks in the example folder for an introduction.

## Motivation

This project exists because finding reaction coordinates of molecular systems is a great way of understanding how they work.

## Installation

### Quick and dirty (use a venv or conda environment!):
```bash
pip install git+https://github.com/hejung/openpathsampling.git@PathSampling_Hooks
git clone https://gitea.kotspeicher.de/hejung/arcd.git
pip install -e arcd/
```
Additionally you will need to install at least one of the machine learning backends, i.e. pytorch, tensorflow and/or dcgpy.
Note also that you might want to install additional engines for openpathsampling, i.e. openMM and/or gromacs.

### Detailed and customizable (still best to use a venv or conda environment):
arcd interacts with openpathsampling through hooks which are called at predefined points during the TPS simulation, e.g. after every MC step. Until merged into ops master this feature is only available on my ops fork on github.
To make it available you can either directly install the PathSampling_Hooks branch using pip
```bash
pip install git+https://github.com/hejung/openpathsampling.git@PathSampling_Hooks
```
or clone the repo/add it as additional remote to your git local and then checkout the PathSampling_Hooks branch.
You should also install any molecular dynamics engines you want to use with openpathsampling for TPS, i.e. openMM and/or gromacs.

Now cd whereever you want to keep your local copy of arcd, clone the repository and install arcd using pip, e.g.
```bash
git clone https://gitea.kotspeicher.de/hejung/arcd.git
pip install -e arcd/
```

For arcd to be useful you need to install at least one machine learning backend. arcd supports multiple different backends and can easily be extended to more. The backend is used to define the underlying machine learning models architecture and is used to fit the model. It naturally also defines the type of the model, i.e. neural network, symbolic regresssion, etc.
Currently supported backends are (model types in brackets):
- [pytorch] (neural network)
- [tensorflow]/keras (neural network)
- [dcgpy] (symbolic regression expressions) [Currently no iterative/on-the-fly training possible]
You can install all of them using pip or conda. Please refer to their respective documentations for detailed installation instructions.

## API Reference

There is none yet. Please read the example notebooks, the docstrings and the code.

## Tests

Tests use pytest. Use `pytest .` while in the toplevel directory of the repository to run them.

## Developers

Let people know how they can dive into the project, include important links to things like wiki, issue trackers, coding style guide, irc, twitter accounts if applicable.

## Contributors

You could (and should) give props to all the people who contributed to the code.

## License

GPL v3

---
<sub>This README.md is printed from 100% recycled electrons.</sub>

[pytorch]: https://pytorch.org
[tensorflow]: https://www.tensorflow.org
[dcgpy]: http://darioizzo.github.io/dcgp/
