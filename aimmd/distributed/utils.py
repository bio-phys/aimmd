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
    storage : :class:`aimmd.Storage`
    per_mcstep_collection : bool
        Whether to return the results seperated by mcstep_collections, True by
        default
    starts : None or list of ints (len=n_mcstep_collection)
        The starting step for the collection for every collection, if None we
        will start with the first step

    Returns:
    --------
    tras : list of trajectories
        The accepted trials
    weights : list of weights
        Trials can have different weights, e.g. if a trial was accepted
        multiple times or has an ensemble weight attached to it already

    Note, if per_mcstep_collection is True a list of tras, weights is returned.
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
            if (mcstep_collection[last_accept].accepted
                and mcstep_collection[last_accept].weight > 0):
                found = True  # not necessary since we use break
                break
            last_accept -= 1
        # now iterate over the storage
        tras = []
        weights = []
        for i, step in enumerate(mcstep_collection[start:]):
            if (step.accepted and step.weight > 0):
                last_accept = i + start
                tras.append(step.path)
                weights.append(step.weight)
                last_weight = step.weight  # remember the weight for next iters
            else:
                try:
                    # can only end up here without triggering IndexError if we
                    # did the loop already and defined last_weight
                    weights[-1] += last_weight
                except IndexError:
                    # no accepts yet
                    tras.append(mcstep_collection[last_accept].path)
                    weights.append(mcstep_collection[last_accept].weight)
                    # remember the weight
                    last_weight = mcstep_collection[last_accept].weight
        return tras, weights

    if starts is None:
        starts = [0 for _ in storage.mcstep_collections]
    if per_mcstep_collection:
        return [accepted_trajs_from_mcstep_collection(cs, starts[i])
                for i, cs in enumerate(storage.mcstep_collections)
                ]
    else:
        tras = []
        weights = []
        for i, cs in enumerate(storage.mcstep_collections):
            t, c = accepted_trajs_from_mcstep_collection(cs, starts[i])
            tras += t
            weights += c
        return tras, weights
