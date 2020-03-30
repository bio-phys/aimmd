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
    make_pipeline(os="linux", arch="amd64", py_version="3.6"),
    make_pipeline(os="linux", arch="amd64", py_version="3.7"),
  ]

def make_pipeline(os, arch, py_version):
  return {
    "kind": "pipeline",
    "name": "{0}-{1}-py{2}".format(os, arch, py_version),
    "platform": {
        "os": os,
        "arch": arch,
    },
    "steps": [
      # NOTE: we can use pip directly to achieve that!
      #{
      #  "name": "clone external dependencies",
      #  "image": "alpine/git",
      #  "commands": [
            # TODO: this only works for linux,
            # but there must be a drone-plugin to do this!?
            # make directory for dependencies and change there
      #      "mkdir external_git_deps",
      #      "cd external_git_deps",
      #      "git clone https://github.com/hejung/openpathsampling.git",
      #      "cd openpathsampling",
      #      "git checkout PathSampling_Hooks",
      #  ]
      #},
      {
          "name": "test",
          "image": "python:{0}".format(py_version),
          "commands": [
              # install ops pathsampling hooks branch
              "pip install git+https://github.com/hejung/openpathsampling.git@PathSampling_Hooks",
              # TODO: this is hardcoded and not nice for maintenance
              # install deep learning packages
              "pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
              "pip install tensorflow"
              "pip install numpy cython",  # install setup dependecies
              "pip install .[test]",
              "pytest .",
          ]
      }
    ]
  }
