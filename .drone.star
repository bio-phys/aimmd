"""
This file is part of AIMMD.

AIMMD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

AIMMD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with AIMMD. If not, see <https://www.gnu.org/licenses/>.
"""


def main(ctx):
  ret_list = []
  if ctx.build.branch == "master":
    # PR against or push to master
    # --runall runs slow tests ('--runslow') and tests for deprecated code (if existant)
    ret_list += [
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.6", pytest_args="--runall"),
      # this one fails with h5py >= 3 (which we want for the distributed storage)
      #make_conda_pipeline(os="linux", arch="amd64", py_version="3.6", pytest_args="--runall"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.7", pytest_args="--runall"),
      make_conda_pipeline(os="linux", arch="amd64", py_version="3.7", pytest_args="--runall"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.8", pytest_args="--runall"),
      make_conda_pipeline(os="linux", arch="amd64", py_version="3.8", pytest_args="--runall"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.9", pytest_args="--runall"),
      make_conda_pipeline(os="linux", arch="amd64", py_version="3.9", pytest_args="--runall"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.10", pytest_args="--runall"),
      make_conda_pipeline(os="linux", arch="amd64", py_version="3.10", pytest_args="--runall"),
    ]
  else:
    # all other branches
    # Note that the test themselves should be fast because we dont run slow tests
    # however the installation takes quite some time, so we do the 4 pip builds
    # (3 is the runner limit on kotspeicher) such that we are done in one go
    ret_list += [
      #make_pip_pipeline(os="linux", arch="amd64", py_version="3.6"),
      # this one fails with h5py >= 3 (which we want for the distributed storage)
      #make_conda_pipeline(os="linux", arch="amd64", py_version="3.6"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.7"),
      #make_conda_pipeline(os="linux", arch="amd64", py_version="3.7"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.8"),
      #make_conda_pipeline(os="linux", arch="amd64", py_version="3.8"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.9"),
      #make_conda_pipeline(os="linux", arch="amd64", py_version="3.9"),
      make_pip_pipeline(os="linux", arch="amd64", py_version="3.10"),
      #make_conda_pipeline(os="linux", arch="amd64", py_version="3.10"),
    ]

  return ret_list

def make_pip_pipeline(os, arch, py_version, pytest_args=""):
  return {
    "kind": "pipeline",
    "name": ("{0}-{1}-py{2}:{3}".format(os, arch, py_version, pytest_args)
             if pytest_args
             else "{0}-{1}-py{2}".format(os, arch, py_version)),
    "platform": {
      "os": os,
      "arch": arch,
    },
    "steps": [
      {
        "name": "test",
        "image": "python:{0}".format(py_version),
        "commands": [
          "pip install --upgrade pip",
          "python --version",
          "pip --version",
          "pip list",
          # install ops pathsampling hooks branch
          #"pip install git+https://github.com/hejung/openpathsampling.git@PathSampling_Hooks",
          "pip install openpathsampling",  # install ops from pypi
          # install deep learning packages
          # TODO: this is hardcoded and not nice for maintenance
          #"pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
          # this tries to install the most recent pytorch with cuda support
          "pip install torch",
          "pip install tensorflow",
          "pip install numpy cython",  # install setup dependecies
          "pip install .[test]",
          "pip list",
          "pytest -v -rs " + pytest_args + " .",
        ]
      },
    ]
  }

# NOTE: we use our own docker repo where we have a conda container
#         where /bin/sh is a symlink to /bin/bash, this gets conda to work
# NOTE 2: Do **not** use conda activate, it does not propagate the exit code
#       ...so we could have sliently failing tests!
def make_conda_pipeline(os, arch, py_version, pytest_args=""):
  return {
    "kind": "pipeline",
    "name": ("{0}-{1}-conda-py{2}:{3}".format(os, arch, py_version, pytest_args)
             if pytest_args
             else "{0}-{1}-conda-py{2}".format(os, arch, py_version)),
    "platform": {
      "os": os,
      "arch": arch,
    },
    "steps": [
      {
        "name": "test",
        "image": "hejung/conda3-drone",
        "commands": [
          "conda config --prepend channels conda-forge",
          "conda config --append channels omnia",
          "conda update -n base conda -q -y -c defaults",
          "conda --version",
          # install openpathsampling directly with the env
          "conda create -n test_env -q -y python={0} compilers openpathsampling".format(py_version),
          "source activate test_env",
          "conda info -e",
          # install deep learning packages
          # TODO: this is CPUonly hardcoded...
          "conda install -q torchvision cpuonly -c pytorch -y",
          "conda install -q numpy cython -y",  # install setup dependecies
          # tensorflow from conda-forge seems to break quite often?!
          # the one from defaults can be some versions behind but usually works
          # (since it is behind we run into the issue with h5py>3 and tf <=2.4 for python3.7)
          #"conda install -c defaults 'tensorflow>=2' -y -q",
          "pip install tensorflow",  # ...so lets use the one from pypi
          "conda list",
          "python --version",
          "pip install .[test]",
          "pytest -v -rs " + pytest_args + " .",
        ]
      },
    ]
  }
