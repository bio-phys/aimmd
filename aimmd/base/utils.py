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
import logging
import openpathsampling as paths
from .rcmodel import RCModel
from .trainset import TrainSet


logger = logging.getLogger(__name__)


def emulate_production_from_trainset(model, trainset):
    """
    Emulates a TPS production run from given trainset.

    This function retraces the exact same sequence of shooting points
    and generated trial transitions as the original TPS simulation.
    Very useful to quickly test different models/ model architectures
    on existing shooting data without costly trial trajectory propagation.

    NOTE: This can only be used to (re-)train on the same descriptors.

    Parameters
    ----------
        model - :class:`aimmd.base.RCModel` the model to train
        trainset - :class:`aimmd.TrainSet` trainset with shooting data

    Returns
    -------
        model - :class:`aimmd.base.RCModel` the trained model

    """
    new_ts = TrainSet(n_states=trainset.n_states)
    for i in range(len(trainset)):
        descriptors = trainset.descriptors[i:i+1]
        # register_sp expects 2d arrays (as the model always does)
        model.register_sp(descriptors, use_transform=False)
        # get the result,
        # in reality we would need to propagate two trial trajectories
        shot_result = trainset.shot_results[i]
        # append_point expects 1d arrays
        new_ts.append_point(descriptors[0], shot_result)
        # let the model decide if it wants to train
        model.train_hook(new_ts)

    return model


def emulate_production_from_storage(model, storage, n_states):
    """
    Emulates a TPS production run from given trainset.

    This function retraces the exact same sequence of shooting points
    and generated trial transitions as the original TPS simulation.
    Very useful to quickly test different models/ model architectures
    on existing shooting data without costly trial trajectory propagation.

    NOTE: This should only be used to (re-)train on different descriptors
          as it recalculates the potentially costly descriptor_transform
          for every shooting point.

    Parameters
    ----------
        model - :class:`aimmd.base.RCModel` the model to train
        storage - :class:`openpathsampling.Storage` with simulation data
        n_states - number of (meta-)stable states in the TPS simulation

    Returns
    -------
    model, trainset
    where:
        model - :class:`aimmd.base.RCModel` the trained model
        trainset - :class:`aimmd.TrainSet` the newly created trainset

    """
    new_ts = TrainSet(n_states=n_states)
    for step in storage.steps:
        try:
            # not every step has a shooting snapshot
            # e.g. the initial transition path does not
            # steps without shooting snap can not be trained on
            sp = step.change.canonical.details.shooting_snapshot
        except AttributeError:
            # no shooting snap
            continue
        except IndexError:
            # no trials!
            continue
        # let the model predict what it thinks
        model.register_sp(sp)
        # add to trainset
        new_ts.append_ops_mcstep(step)
        # (possibly) train
        model.train_hook(new_ts)

    return model, new_ts


def get_batch_size_from_model_and_descriptors(model, descriptors, max_size=4096) -> int:
    """
    Get a batch size value either from models expected efficiency params or from descriptors size.

    If None is in there take the min(descriptors.shape[0], max_size)
    """
    try:
        batch_size = model.ee_params["batch_size"]
    except (KeyError, AttributeError):
        # either ee_params not set or no batch_size in there
        # so lets (try to) do it in one batch
        batch_size = None
    except Exception as e:
        # raise everything else
        raise e from None
    if batch_size is None:
        # None means (try) in one batch
        batch_size = descriptors.shape[0]
    elif max_size < batch_size:
        # this makes sure that if batch_size is given (i.e. not None)
        # we will use it even if max_size would be smaller
        max_size = batch_size
    # make sure we can not go tooo large even with a None value
    # the below results in an 4 MB descriptors tensor
    # if we have 64 bit floats and 1D descriptors
    # since we expect descriptors to be dim 100 - 1000
    # it is more like 400 - 4000 MB descriptors
    # (or half of that for 32 bit floats)
    #max_size = 4096  # = 1024 * 4
    if batch_size > max_size:
        logger.warning(f"Using batch size {max_size} instead of {batch_size}"
                        + " to make sure we can fit everything in memory."
                           )
        batch_size = max_size
    return batch_size
