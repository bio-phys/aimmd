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
import os
import numpy as np
from .trajectory import Trajectory


def nstout_from_mdp(mdp, traj_type="TRR"):
    """Get lowest output frequency for gromacs trajectories from MDP class."""
    if traj_type.upper() == "TRR":
        keys = ["nstxout", "nstvout", "nstfout"]
    elif traj_type.upper() == "XTC":
        keys = ["nstxout-compressed", "nstxtcout"]
    else:
        raise ValueError("traj_type must be one of 'TRR' or 'XTC'.")

    vals = []
    for k in keys:
        try:
            vals += [mdp[k]]
        except KeyError:
            # not set, defaults to "0" (== no output!)
            pass
    nstout = min(vals, default=None)
    if nstout is None:
        raise ValueError("The MDP you passed results in no trajectory output.")
    return nstout


def get_all_traj_parts(folder, deffnm, traj_type="TRR"):
    """Find and return a list of trajectory parts produced by a GmxEngine."""
    # NOTE: this assumes all files/parts are ther, i.e. nothing was deleted
    #       we just check for the highest number and also assume the tpr exists
    def partnum_suffix(num):
        # construct gromacs num part suffix from simulation_part
        num_suffix = str(num)
        while len(num_suffix) < 4:
            num_suffix = "0" + num_suffix
        num_suffix = ".part" + num_suffix
        return num_suffix

    ending = "." + traj_type.lower()
    content = os.listdir(folder)
    filtered = [f for f in content
                if (f.endswith(ending) and f.startswith(deffnm))
                ]
    partnums = [int(f.lstrip(f"{deffnm}.part").rstrip(ending)) for f in filtered]
    max_num = np.max(partnums)
    trajs = [Trajectory(trajectory_file=os.path.join(folder,
                                                     (f"{deffnm}"
                                                      + f"{partnum_suffix(num)}"
                                                      + f"{ending}")
                                                     ),
                        structure_file=os.path.join(folder, f"{deffnm}.tpr")
                        )
             for num in range(1, max_num+1)]
    return trajs


def get_all_file_parts(folder, deffnm, file_ending):
    """Find and return all files with given ending produced by GmxEngine."""
    # NOTE: this assumes all files/parts are ther, i.e. nothing was deleted
    #       we just check for the highest number and also assume they exist
    def partnum_suffix(num):
        # construct gromacs num part suffix from simulation_part
        num_suffix = str(num)
        while len(num_suffix) < 4:
            num_suffix = "0" + num_suffix
        num_suffix = ".part" + num_suffix
        return num_suffix

    content = os.listdir(folder)
    filtered = [f for f in content
                if (f.endswith(file_ending) and f.startswith(deffnm))
                ]
    partnums = [int(f.lstrip(f"{deffnm}.part").rstrip(file_ending))
                for f in filtered]
    max_num = np.max(partnums)
    parts = [os.path.join(folder, (f"{deffnm}{partnum_suffix(num)}{file_ending}"))
             for num in range(1, max_num+1)]
    return parts
