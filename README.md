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
- keras and (tensorflow or theano)
- pyaudi and dcgpy (installable via pip)

I recommend using conda or virtualenv to not break your system python installation. The following commands will use conda to create a new environment 'arcd' and install everything you need. All commands assume you have conda installed.
1. Add the conda channels 'conda-forge' and 'omnia' with
```
conda config --prepend channels omnia
conda config --prepend channels conda-forge
```
This will add these two channels with a higher priority than the standard channel. Or move them to the top of the channel list if they are already in there. 

2. Create a new environment 'arcd' and install openpathsampling and arcd dependencies. You should replace 'tensorflow' with 'tensorflow-gpu' if you have a Nvidia GPU and want to use it.
```
conda create --name arcd python=3 numpy scipy pandas nose jupyter netcdf4 matplotlib openmm pyyaml svgwrite mdtraj ujson networkx openmmtools cython h5py graphviz pydot pytest sympy tensorflow pyaudi keras future
```
3. Activate the environment you just created with `source activate arcd` or `conda activate arcd` depending on your conda version.

4. get OPS + developer install
(5. get pyaudi  + dcgpy ) <- pip should do that when installing arcd

If you already have an openpathsampling developer install you only need to add https://github.com/hejung/openpathsampling as a remote and checkout my branch with the pathsampling hooks (pathsamp_hooks).

6. To install arcd directly from the repository (this should also install all extra dependencies via pip):
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
