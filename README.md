# aimmd

## Synopsis

aimmd - AI for Molecular Mechanism Discovery: Machine learning the reaction coordinate from shooting results.

## Code Example

Please see the jupyter notebooks in the `examples` folder.

## Motivation

This project exists because finding reaction coordinates of molecular systems is a great way of understanding how they work.

## Installation

aimmd runs its TPS simulations either through [asyncmd] (if managing and learning from many simulations simultaneously on a HPC cluster) or through [openpathsampling] (in the sequential case). [openpathsampling] can easily be installed via pip and is automatically installed when you install aimmd with pip. To install [asyncmd] please follow the installation instructions there (clone the repository and install localy using pip) as it is not yet on PyPi.

Note, that for [asyncmd] and/or [openpathsampling] to work you need to install a molecular dynamics engine to perform the trajectory integration. In [asyncmd] the only currently supported engine is [gromacs], while [openpathsampling] can use both [gromacs] and [openMM] (but [openMM] is highly recommended).

In addition to [asyncmd] and/or [openpathsampling] to run the TPS simulations you need to install at least one machine learning backend (to actually learn the committor/ the reaction coordinate). aimmd supports multiple different backends and can easily be extended to more. The backend is used to define the underlying machine learning models architecture and is used to fit the model. It naturally also defines the type of the model, i.e. neural network, symbolic regresssion, etc.
Currently supported backends are (model types in brackets):

- [pytorch] (neural network) : [Recommended for steering and learning from simulations iteratively]
- [tensorflow]/keras (neural network) : [Mostly included for legacy reasons]
- [dcgpy] (symbolic regression expressions) [Currently no steering and learning from simulations on-the-fly possible; recommended to build low dimensional interpretable models of the committor]

You should be able to install all of them using pip and/or conda. Please refer to their respective documentations for detailed installation instructions.

To finaly install aimmd, cd whereever you want to keep your local copy of aimmd, clone the repository and install aimmd using pip, e.g.

```bash
git clone https://github.com/bio-phys/aimmd.git
pip install -e aimmd/
```

### TLDR

- For using the `aimmd.distributed` module and steering many simulations on a HPC cluster simultaneously you need to install [asyncmd]. You also need a working installation of [gromacs].
- You will need to install at least one of the machine learning backends, i.e. [pytorch] (recommended for steering and learning from simulations iteratively), [tensorflow] and/or [dcgpy] (recommended for building low dimensional interpretable models).
- You might want to install additional engines for use with the sequential aimmd code building on [openpathsampling], e.g. [openMM].

```bash
git clone https://github.com/bio-phys/aimmd.git
pip install -e aimmd/
```

## API Reference

There is none (yet). Please read the example notebooks, the docstrings and the code.

## Tests

Tests use pytest. Use e.g. `pytest .` while in the toplevel directory of the repository to run them.

## Contributions

Contributions are welcome! Please feel free to open an [issue](https://github.com/bio-phys/aimmd/issues) or [pull request](https://github.com/bio-phys/aimmd/pulls) if you discover any bugs or want to propose a missing feature.

## License

GPL v3

## Citation

If you use aimmd in published work please cite:

- H. Jung, R. Covino, A. Arjun, C. Leitold, C. Dellago, P.G. Bolhuis and G. Hummer. Machine-guided path sampling to discover mechanisms of molecular self-organization. Nature Computational Science 3, 334–345 (2023). doi:[10.1038/s43588-023-00428-z](https://doi.org/10.1038/s43588-023-00428-z)

---
<sub>This README.md is printed from 100% recycled electrons.</sub>

[asyncmd]: https://github.com/bio-phys/asyncmd
[pytorch]: https://pytorch.org
[tensorflow]: https://www.tensorflow.org
[dcgpy]: http://darioizzo.github.io/dcgp/
[openMM]: http://openmm.org/
[openpathsampling]: http://openpathsampling.org/latest/
[GROMACS]: http://www.gromacs.org/
