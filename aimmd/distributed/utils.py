# This file is part of aimmd
#
# aimmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# aimmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with aimmd. If not, see <https://www.gnu.org/licenses/>.
"""
This file contains utility functions used at various places in :mod:`aimmd.distributed`,
but which can also be useful for users, e.g. a method to get all accepted trials
from an aimmd.distributed storage.
"""
import typing

if typing.TYPE_CHECKING:  # pragma: no cover
    from asyncmd.trajectory.trajectory import Trajectory
    from ..base.storage import MCStepCollection, Storage


def accepted_trajs_from_mcstep_collection(mcstep_collection: "MCStepCollection",
                                          start: int = 0,
                                          return_multiplicity: bool = False,
                                          ) -> "tuple[list[Trajectory], list[float]]":
    """
    Find all accepted trial trajectories and their weights in a aimmd.distributed
    mcstep_collection.

    Parameters
    ----------
    mcstep_collection : MCStepCollection
        The mcstep collection from which the trials should be retrieved
    start : int, optional
        The first trial step number to inspect, by default 0.
    return_multiplicity : bool, optional
        Whether to return the multiplicity instead of the (sum of) weights for
        the trajectories, i.e., if True each trajectory will have a weight of 1
        and the returned weights simply count if a trajectory was accepted multiple
        times.
        By default False.

    Returns
    -------
    tuple[list[Trajectory], list[float]]
        All accepted trial trajectories and their corresponding weights.

    Raises
    ------
    ValueError
        When the start value is larger than the length of the mcstep_collection.
    """
    if start == len(mcstep_collection):
        # this happens when we run the density collection twice without
        # adding/producing a new MCStep into the mcstep_collection,
        # i.e. always when the Densitycollection runs more often than we
        # have PathChainSamplers (when interval > n_samplers)
        return [], []
    if start > len(mcstep_collection):
        # this should never happen
        raise ValueError(f"start [{start}] can not be > len(mcstep_collection) "
                         f"[{len(mcstep_collection)}].")
    # find the last accepted TP to be able to add it again
    # instead of the rejects we could find
    last_accept = start
    found = False
    while not found and last_accept >= 0:
        if (mcstep_collection[last_accept].accepted
                and mcstep_collection[last_accept].weight > 0):
            found = True  # not necessary since we use break
            break
        last_accept -= 1
    if last_accept < 0:
        # no accepts with weight > 0 in storage (yet)
        return [], []
    # now iterate over the storage
    tras = []
    weights = []
    for i, step in enumerate(mcstep_collection[start:]):
        if (step.accepted and step.weight > 0):
            last_accept = i + start
            tras.append(step.path)
            # add and remember the weight (or multiplicity) for next iters
            if return_multiplicity:
                last_weight = 1.
            else:
                last_weight = step.weight
            weights.append(last_weight)
        else:
            try:
                # can only end up here without triggering IndexError if we
                # did the loop already and defined last_weight
                weights[-1] += last_weight
            except IndexError:
                # no accepts yet
                tras.append(mcstep_collection[last_accept].path)
                # again: add and remember the weight (or multiplicity)
                if return_multiplicity:
                    last_weight = 1.
                else:
                    last_weight = mcstep_collection[last_accept].weight
                weights.append(last_weight)
    return tras, weights


def accepted_trajs_from_aimmd_storage(storage: "Storage",
                                      per_mcstep_collection: bool = True,
                                      starts: list[int] | None = None,
                                      return_multiplicity: bool = False,
                                      ):
    """
    Find all accepted trial trajectories in an aimmd.distributed storage.

    If per_mcstep_collection is True a list of tras, weights is returned.
    Each entry in the list corresponds to the mcstep_collection with the same index.

    Parameters:
    -----------
    storage : :class:`aimmd.Storage`
    per_mcstep_collection : bool
        Whether to return the results seperated by mcstep_collections, True by
        default
    starts : None or list of ints (len=n_mcstep_collection)
        The starting step for the collection for every collection, if None we
        will start with the first step
    return_multiplicity : bool, optional
        Whether to return the multiplicity instead of the (sum of) weights for
        the trajectories, i.e., if True each trajectory will have a weight of 1
        and the returned weights simply count if a trajectory was accepted multiple
        times.
        By default False.

    Returns:
    --------
    tras : list of trajectories
        The accepted trials
    weights : list of weights
        Trials can have different weights, e.g. if a trial was accepted
        multiple times or has an ensemble weight attached to it already
    """
    if starts is None:
        starts = [0 for _ in storage.mcstep_collections]
    if per_mcstep_collection:
        return [accepted_trajs_from_mcstep_collection(cs, starts[i],
                                                      return_multiplicity=return_multiplicity,
                                                      )
                for i, cs in enumerate(storage.mcstep_collections)
                ]
    tras = []
    weights = []
    for i, cs in enumerate(storage.mcstep_collections):
        t, c = accepted_trajs_from_mcstep_collection(cs, starts[i],
                                                     return_multiplicity=return_multiplicity,
                                                     )
        tras += t
        weights += c
    return tras, weights
