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
