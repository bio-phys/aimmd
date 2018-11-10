# arcd

## Synopsis

arcd - Automatic Reaction Coordinate Discovery: Machine learning the reaction coordinate from shooting results.

## Code Example

Show what the library does as concisely as possible, developers should be able to figure out **how** your project solves their problem by looking at the code example. Make sure the API you are showing off is obvious, and that your code is short and concise.

```
your code goes here
```

## Motivation

A short description of the motivation behind the creation and maintenance of the project. This should explain **why** the project exists.

## Installation
Dependencies:
- openpathsampling developer install to add my 'pathsamp_hooks' github branch
- keras and backend (I used tensorflow)
- pyaudi and dcgpy (installable via pip)
- cython, pytest, sympy (conda installable)

I recommend using conda or virtualenv to not break your system python installation. The following commands will use conda to create a new environment 'arcd' and install everything you need. All commands assume you have conda installed.
1. Add the conda channels 'conda-forge' and 'omnia' with
```
conda config --prepend channels omnia
conda config --prepend channels conda-forge
```
This will add these two channels with a higher priority than the standard channel. Or move them to the top of the channel list if they are already in there.

2. OPS: Create a new environment 'arcd' and install openpathsampling dependencies.
```
conda create --name arcd python=3 numpy scipy pandas nose jupyter netcdf4 matplotlib openmm pyyaml svgwrite mdtraj ujson networkx openmmtools future
```
Clone the openpathsampling github repository, add github/hejung/openpathsampling as an additional remote and checkout the pathsamp_hooks branch.
cd into the repository, activate the environment you just created (`source activate arcd` or `conda activate arcd` depending on your conda version) and install openpathsampling via pip with `pip install -e .`

3. Install keras and backend. Use conda to install keras dependencies, `conda install -n arcd h5py graphviz pydot`. Then install the backend (I used tensorflow), see the corresponding installation instructions. After installing the backend, keras can be installed via pip from pypi with `pip install keras`.

4. arcd & friends: dependencies
```
cython pytest sympy
```
(get pyaudi  + dcgpy ) <- pip should do that when installing arcd


If you already have an openpathsampling developer install you only need to add https://github.com/hejung/openpathsampling as a remote and checkout my branch with the pathsampling hooks (pathsamp_hooks).

5. To install arcd directly from the repository (this should also install all extra dependencies via pip):
```
git clone https://gogs.kotspeicher.de/hejung/arcd.git
cd arcd
pip install -e .
```

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples. Also write test if you have not yet!

## Developers

Let people know how they can dive into the project, include important links to things like wiki, issue trackers, coding style guide, irc, twitter accounts if applicable.

## Contributors

You could (and should) give props to all the people who contributed to the code.

## License

A short snippet describing the license (GPL, MIT, Apache, etc.)
