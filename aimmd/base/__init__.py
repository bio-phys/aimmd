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
class Properties:
    """Keys to access shooting point properties in trainset iteration."""
    descriptors = "descriptors"
    shot_results = "shot_results"
    weights = "weights"
    q = "log_probs"
    phi = "committors"


# setup dictionary mapping descriptive strings to 'paths' in HDF5 file
_H5PY_PATH_DICT = {"level0": "/aimmd_data"}  # toplevel aimmd group
_H5PY_PATH_DICT["cache"] = _H5PY_PATH_DICT["level0"] + "/cache"  # cache
_H5PY_PATH_DICT["distributed"] = _H5PY_PATH_DICT["level0"] + "/distributed"
_H5PY_PATH_DICT.update({  # these depend on cache, distributed and level0 to be defined
        "rcmodel_store": _H5PY_PATH_DICT["level0"] + "/RCModels",
        "trainset_store": _H5PY_PATH_DICT["level0"] + "/TrainSet",
        "tra_dc_cache": _H5PY_PATH_DICT["cache"] + "/TrajectoryDensityCollectors",
        "distributed_brainstore": _H5PY_PATH_DICT["distributed"] + "/BrainStore",
        "distributed_mcstepcollections": _H5PY_PATH_DICT["distributed"] + "/MCStepCollections",
        "distributed_traj_val_cache": _H5PY_PATH_DICT["cache"] + "/distributed",
                       })
