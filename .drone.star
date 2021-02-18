"""
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
"""

def main(ctx):
  return [
    make_pip_pipeline(os="linux", arch="amd64", py_version="3.6"),
    make_pip_pipeline(os="linux", arch="amd64", py_version="3.7"),
    make_pip_pipeline(os="linux", arch="amd64", py_version="3.7", runall=True),
    make_pip_pipeline(os="linux", arch="amd64", py_version="3.8"),
    make_conda_pipeline(os="linux", arch="amd64", py_version="3.6"),
    make_conda_pipeline(os="linux", arch="amd64", py_version="3.7"),
    make_conda_pipeline(os="linux", arch="amd64", py_version="3.7", runall=True),
    # no tensorflow conda package for py3.8 yet
    #make_conda_pipeline(os="linux", arch="amd64", py_version="3.8"),
  ]

def make_pip_pipeline(os, arch, py_version, runall=False):
  return {
    "kind": "pipeline",
    "name": ("{0}-{1}-py{2}-full".format(os, arch, py_version) if runall
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
          "pip install git+https://github.com/hejung/openpathsampling.git@PathSampling_Hooks",
          # install deep learning packages
          # TODO: this is hardcoded and not nice for maintenance
          #"pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html",
          # this tries to install the most recent pytorch with cuda support
          "pip install torch",
          "pip install tensorflow",
          "pip install numpy cython",  # install setup dependecies
          "pip install .[test]",
          "pip list",
          # runall runs slow tests and tests for deprecated code
          ("pytest -v -rs --runall ." if runall
           else "pytest -v -rs --runslow ."),
        ]
      },
    ]
  }

# NOTE: we use our own docker repo where we have a conda container
#         where /bin/sh is a symlink to /bin/bash, this gets conda to work
# NOTE 2: Do **not** use conda activate, it does not propagate the exit code
#       ...so we could have sliently failing tests!
def make_conda_pipeline(os, arch, py_version, runall=False):
  return {
    "kind": "pipeline",
    "name": ("{0}-{1}-conda-py{2}-full".format(os, arch, py_version) if runall
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
          "conda create -n test_env -q -y python={0} compilers".format(py_version),
          "source activate test_env",
          "conda info -e",
          # install deep learning packages
          "conda install 'tensorflow>=2' -y -q",
          # TODO: this is CPUonly hardcoded...
          "conda install -q torchvision cpuonly -c pytorch -y",
          "conda install -q numpy cython -y",  # install setup dependecies
          # install ops pathsampling hooks branch
          "pip install git+https://github.com/hejung/openpathsampling.git@PathSampling_Hooks",
          "conda list",
          "python --version",
          "pip install .[test]",
          # runall runs slow tests and tests for deprecated code
          ("pytest -v -rs --runall ." if runall
           else "pytest -v -rs --runslow ."),
        ]
      },
    ]
  }
