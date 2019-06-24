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
import os
import logging
import openpathsampling as paths
from .rcmodel import RCModel
from .trainset import TrainSet
from ..ops.traininghook import TrainingHook as _TrainingHook


logger = logging.getLogger(__name__)


# This function is tied to the TrainingHook as this is the object saving
# and therfore 'deciding' on the name the data has in ops_storage.tags
# and since we want to keep track of those names only once we rebind here
load_trainset = _TrainingHook.load_trainset


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
        model - :class:`arcd.base.RCModel` the model to train
        trainset - :class:`arcd.TrainSet` trainset with shooting data

    Returns
    -------
        model - :class:`arcd.base.RCModel` the trained model

    """
    new_ts = TrainSet(states=trainset.states,
                      # actually we do not use the descriptor_transform
                      descriptor_transform=trainset.descriptor_transform)
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


def emulate_production_from_storage(model, storage, states):
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
        model - :class:`arcd.base.RCModel` the model to train
        storage - :class:`openpathsampling.Storage` with simulation data
        states - list of :class:`openpathsampling.Volume` the (meta-)stable
                 states for the TPS simulation

    Returns
    -------
    model, trainset
    where:
        model - :class:`arcd.base.RCModel` the trained model
        trainset - :class:`arcd.TrainSet` the newly created trainset

    """
    new_ts = TrainSet(states=states,
                      # here we actually use the transform!
                      descriptor_transform=model.descriptor_transform)
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


def load_model_with_storage(fname, mode='r', storage_endswith='.nc', sname=None):
    """
    Load an arcd.RCModel from file.

    Additionally open and return the corresponding ops storage file if it is
    found in the same folder.

    Parameters
    ----------
    fname - str, the name of the model file
    mode - str, mode to open the ops storage file in,
           should be either 'r' or 'a'
    storage_endswith - str, filename extension of the ops storage,
                       used to scan for potential matching storages
    sname - str or None, name of the ops storage file to use,
            if None, we will try to identify the storage by filename extension,
            if given we will ignore storage_endswith and load $sname

    Returns
    -------
    model, storage

    """
    if mode not in ['a', 'r']:
        raise ValueError("mode must be either 'a' or 'r'.")
    if sname is None:
        topdir = os.path.dirname(os.path.abspath(fname))
        files = [f for f in os.listdir(topdir)
                 if not os.path.isdir(os.path.join(topdir, f))]
        sFiles = [f for f in files if f.endswith('.nc')]
        if len(sFiles) > 1:
            raise ValueError('More than one file matches storage filename '
                             + 'extension. Please pass sname to avoid ambiguity.')
        if len(sFiles) == 1:
            # one storage, take it
            sFile = sFiles[0]
            storage = paths.Storage(os.path.join(topdir, sFile), mode)
        else:
            # no storage found, warn but load anyway
            logger.warning('No matching storage found. Proceeding without. '
                           + 'Consider passing sname or storage_endswith if '
                           + 'this is unexpected.')
            storage = None
    else:
        storage = paths.Storage(sname, mode)

    # no that we sorted if and which storage to use: load the model
    state, cls = RCModel.load_state(fname, storage)
    state = cls.fix_state(state)
    model = cls.set_state(state)

    return model, storage


def load_model(fname, storage=None, descriptor_transform=None):
    """
    Load an arcd.RCModel from file. Do not open or load an ops storage.

    You can pass any open ops storage object to read out the
    descriptor_transform or pass the descriptor_transform directly.

    NOTE:
    This is much faster and consumes less memory than load_model_with_storage
    as this does not open any additional ops storages. It is therefore the
    preferred way of loading additional models for comparisson.

    Parameters
    ----------
    fname - str, model filename
    storage - None or open ops storage object,
              if not None we ignore descriptor_transform
    descriptor_transform - None or some callable that takes snapshots/trajectories,
                           the descriptor_transform to set for the model,
                           only has an effect if storage is None

    """
    state, cls = RCModel.load_state(fname, storage)
    state = cls.fix_state(state)
    model = cls.set_state(state)
    if storage is None:
        model.descriptor_transform = descriptor_transform
    return model
