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
import logging
import numpy as np
from .selector import RCModelSelector


logger = logging.getLogger(__name__)


# TODO: can we write a nice function that works for some selectors only?
#       i.e. that is selector specific and lets the user choose the selector?
def set_rcmodel_in_all_selectors(model, simulation):
    """
    Replace all RCModelSelectors models with the given model.

    Useful for restarting TPS simulations, since the aimmd RCModels can not be
    saved by ops together with the RCModelSelector.
    """
    for move_group in simulation.move_scheme.movers.values():
        for mover in move_group:
            if isinstance(mover.selector, RCModelSelector):
                mover.selector.model = model


def analyze_ops_mcstep(mcstep, descriptor_transform, states):
    """
    Extract the shooting results and descriptors from given ops MCStep.

    Parameters:
    -----------
    mcstep - :class:`openpathsampling.MCStep`, the step from which we
             extract the descriptors and shot_results
    descriptor_transform - :class:`openpathsampling.CollectiveVariable`,
                           function transforming (cartesian) atom coordinates
                           to the space in which the model learns
    states - list of :class:`openpathsampling.Volume`, the metastable states
             that could be reached

    Returns:
    --------
    descriptors - numpy ndarray, descriptors/CV-values for the shooting point
    shot_results - numpy ndarray, counts for the states reached in the same
                   order as the states list

    """
    try:
        details = mcstep.change.canonical.details
        shooting_snap = details.shooting_snapshot
    # TODO: warn or pass? if used togehter with other TP generation schemes
    # than shooting, pass is the right thing to do,
    # otherwise this should never happen anyway...but then it might be good
    # to know if it does... :)
    except AttributeError:
        # wrong kind of move (no shooting_snapshot)
        # this could actually happen if we use aimmd in one simulation
        # together with other TPS/TIS schemes
        logger.warning('Tried to add a MCStep that has no '
                       + 'shooting_snapshot.')
    except IndexError:
        # very wrong kind of move (no trials!)
        # I think this should never happen?
        logger.error('Tried to add a MCStep that contains no trials.')
    else:
        # find out which states we reached
        trial_traj = mcstep.change.canonical.trials[0].trajectory
        init_traj = details.initial_trajectory
        test_points = [s for s in [trial_traj[0], trial_traj[-1]]
                       if s not in [init_traj[0], init_traj[-1]]]
        shot_results = np.array([sum(int(state(pt)) for pt in test_points)
                                 for state in states])
        total_count = sum(shot_results)

        # TODO: for now we assume TwoWayShooting,
        # because otherwise we can not redraw v,
        # which would break our independence assumption!
        # (if we ignore the velocities of the SPs)

        # warn if no states were reached,
        # this point makes no contribution to the loss since terms are 0,
        # this makes the 'harmonic loss' from multi-domain models blow up,
        # also some regularization schemes will overcount/overregularize
        if total_count < 2:
            logger.warning('Total states reached is < 2. This probably '
                           + 'means there are uncommited trajectories. '
                           )
        # get and possibly transform descriptors
        # descriptors is a 1d-array, since we use a snap and no traj in CV
        descriptors = descriptor_transform(shooting_snap)
        if not np.all(np.isfinite(descriptors)):
            logger.warning('There are NaNs or infinities in the training '
                           + 'descriptors. \n We used numpy.nan_to_num() '
                           + 'to proceed. You might still want to have '
                           + '(and should have) a look @ \n'
                           + 'np.where(np.isinf(descriptors): '
                           + str(np.where(np.isinf(descriptors)))
                           + 'and np.where(np.isnan(descriptors): '
                           + str(np.where(np.isnan(descriptors))))
            descriptors = np.nan_to_num(descriptors)

        return descriptors, shot_results


def accepted_trials_from_ops_storage(storage, start=0):
    """
    Find all accepted trial trajectories in an ops storage.

    Parameters:
    -----------
    storage - :class:`openpathsampling.Storage`
    start - int (default=0), step from which to start collection

    Returns:
    --------
    tras - list of trajectories of the accepted trials
    counts - list of counts, i.e. if a trial was accepted multiple times
    """
    # find the last accepted TP to be able to add it again
    # instead of the rejects we could find
    last_accept = start
    found = False
    while not found:
        if storage.steps[last_accept].change.canonical.accepted:
            found = True  # not necessary since we use break
            break
        last_accept -= 1
    # now iterate over the storage
    tras = []
    counts = []
    for i, step in enumerate(storage.steps[start:]):
        if step.change.canonical.accepted:
            last_accept = i + start
            tras.append(step.change.canonical.trials[0].trajectory)
            counts.append(1)
        else:
            try:
                counts[-1] += 1
            except IndexError:
                # no accepts yet
                change = storage.steps[last_accept].change
                tras.append(change.canonical.trials[0].trajectory)
                counts.append(1)
    return tras, counts
