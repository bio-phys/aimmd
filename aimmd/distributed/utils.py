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


def accepted_trajs_from_aimmd_storage(storage, per_mcstep_collection=True, starts=None):
    """
    Find all accepted trial trajectories in an aimmd.distributed storage.

    Parameters:
    -----------
    storage - :class:`aimmd.Storage`
    per_mcstep_collection - bool (default=True), whether to return the results seperated
                            by mcstep_collections
    starts - None or list of ints (len=n_mcstep_collection), the starting step for the
             collection for every collection, if None we will start with the
             first step

    Returns:
    --------
    tras - list of trajectories of the accepted trials
    counts - list of counts, i.e. if a trial was accepted multiple times

    Note, if per_mcstep_collection is True a list of tras, counts is returned.
    Each entry in the list corresponds to the mcstep_collection with the same index.
    """
    def accepted_trajs_from_mcstep_collection(mcstep_collection, start):
        if start == len(mcstep_collection):
            # this happens when we run the density collection twice without
            # adding/producing a new MCStep into the mcstep_collection,
            # i.e. always when the Densitycollection runs more often than we
            # have PathChainSamplers (when interval > n_samplers)
            return [], []
        elif start > len(mcstep_collection):
            # this should never happen
            raise ValueError(f"start [{start}] can not be > len(mcstep_collection) "
                             f"[{len(mcstep_collection)}].")
        # find the last accepted TP to be able to add it again
        # instead of the rejects we could find
        last_accept = start
        found = False
        while not found:
            if mcstep_collection[last_accept].accepted:
                found = True  # not necessary since we use break
                break
            last_accept -= 1
        # now iterate over the storage
        tras = []
        counts = []
        for i, step in enumerate(mcstep_collection[start:]):
            if step.accepted:
                last_accept = i + start
                tras.append(step.path)
                counts.append(1)
            else:
                try:
                    counts[-1] += 1
                except IndexError:
                    # no accepts yet
                    tras.append(mcstep_collection[last_accept].path)
                    counts.append(1)
        return tras, counts

    if starts is None:
        starts = [0 for _ in storage.mcstep_collections]
    if per_mcstep_collection:
        return [accepted_trajs_from_mcstep_collection(cs, starts[i])
                for i, cs in enumerate(storage.mcstep_collections)
                ]
    else:
        tras = []
        counts = []
        for i, cs in enumerate(storage.mcstep_collections):
            t, c = accepted_trajs_from_mcstep_collection(cs, starts[i])
            tras += t
            counts += c
        return tras, counts
