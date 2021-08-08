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
import shutil


def ensure_executable_available(executable):
    """
    Helper function to ensure the given executable is available and executable.

    Takes a relative or absolute path to an executable or the name of an
    executable available in $PATH.
    Returns the full path to the executable.
    """
    if os.path.isfile(os.path.abspath(executable)):
        # see if it is a relative path starting from cwd
        # (or a full path starting with /)
        executable = os.path.abspath(executable)
        if not os.access(executable, os.X_OK):
            raise ValueError(f"{executable} must be executable.")
    elif shutil.which(executable) is not None:
        # see if we find it in $PATH
        executable = shutil.which(executable)
    else:
        raise ValueError(f"{executable} must be an existing path or accesible "
                         + "via the $PATH environment variable.")
    return executable


def accepted_trajs_from_aimmd_storage(storage, per_chain=True, starts=None):
    """
    Find all accepted trial trajectories in an aimmd.distributed storage.

    Parameters:
    -----------
    storage - :class:`aimmd.Storage`
    per_chain - bool (default=True), whether to return the reults seperated
                by chains
    starts - None or list of ints (len=n_chains), the starting step for the
             collection for every chain, if None we will start with the
             first step

    Returns:
    --------
    tras - list of trajectories of the accepted trials
    counts - list of counts, i.e. if a trial was accepted multiple times

    Note, if per_chain is True a list of tras, counts is returned.
    Each entry in the list corresponds to the chain with the same index.
    """
    def accepted_trajs_from_chainstore(chainstore, start):
        # find the last accepted TP to be able to add it again
        # instead of the rejects we could find
        last_accept = start
        found = False
        while not found:
            if chainstore[last_accept].accepted:
                found = True  # not necessary since we use break
                break
            last_accept -= 1
        # now iterate over the storage
        tras = []
        counts = []
        for i, step in enumerate(chainstore[start:]):
            if step.accepted:
                last_accept = i + start
                tras.append(step.path)
                counts.append(1)
            else:
                try:
                    counts[-1] += 1
                except IndexError:
                    # no accepts yet
                    tras.append(chainstore[last_accept].path)
                    counts.append(1)
        return tras, counts

    if starts is None:
        starts = [0 for _ in storage.central_memory]
    if per_chain:
        return [accepted_trajs_from_chainstore(cs, starts[i])
                for i, cs in enumerate(storage.central_memory)
                ]
    else:
        tras = []
        counts = []
        for i, cs in enumerate(storage.central_memory):
            t, c = accepted_trajs_from_chainstore(cs, starts[i])
            tras += t
            counts += c
        return tras, counts
