# arcd

## Synopsis

arcd - Automatic Reaction Coordinate Discovery: Machine learning the reaction coordinate from shooting results.

## Code Example

Please see the Ipython notebooks in the example folder for an introduction.

## Motivation

This project exists because finding reaction coordinates of molecular systems is a great way of understanding how they work.

## Installation
Dependencies:
- openpathsampling developer install
- pytorch and/or keras + backend (tested tensorflow); [if you install both tensorflow and pytorch, install pytorch second to avoid conflicts]
- cython, pytest, sympy, h5py (conda installable)
- pyaudi and dcgpy (installable via pip)

I recommend using conda or virtualenv to not break your system python installation. The following step-by-step instruction will use conda to create a new environment called 'arcd' and install everything you need. All commands assume you have conda already installed.
1. Add the conda channels 'conda-forge' and 'omnia' with
```
conda config --prepend channels omnia
conda config --prepend channels conda-forge
```
This will add these two channels with a higher priority than the standard channel or move them to the top of the channel list if they are already in there.

2. Install OPS: Create a new environment 'arcd' and install the conda package openpathsampling to easily satisfy all dependencies.
```
conda create --name arcd python=3.6 openpathsampling
```

3. arcd dependencies I (keras + tensorflow):
Use the conda package from defaults for easy install
```
conda install -n arcd tensorflow keras -c defaults
```
or if you happen to own a compatible GPU (choose a CUDA version compatible to your system driver)
```
conda install -n arcd tensorflow-gpu keras cudatoolkit=8.0 -c defaults
```
To be able to save keras models you will need to install h5py [highly recommended anyway ;)]
```
conda install -n arcd h5py
```
and to be able to draw them you will need graphviz and pydot
```
conda install -n arcd graphviz pydot
```

4. arcd dependecies II (pytorch):
```
conda install -n arcd pytorch torchvision -c pytorch
```
If you happen to own a compatible GPU, choose a CUDA version compatible to your proprietary driver version, e.g.
```
conda install -n arcd pytorch torchvision cudatoolkit=8.0 -c pytorch
```

5. arcd dependencies III:
```
conda install -n arcd cython pytest sympy
```
(get pyaudi  + dcgpy ) <- pip should do that when installing arcd

6. Now remove openpathsampling and (re-)install it manually from the git repository. You need to clone github.com/hejung/openpathsampling and checkout the pathsampling_hooks branch.
```
conda activate arcd
conda remove --force openpathsampling
cd /where/ever/you/cloned/the/repo/to/openpathsampling
# you should now be in the same folder where the setup.py is located
git checkout pathsampling_hooks
pip install -e .
```

7. Finally install arcd. (This should also install any missing dependencies if installable via pip):
```
git clone https://gitea.kotspeicher.de/hejung/arcd.git
cd arcd
conda activate arcd
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

GPL v3
