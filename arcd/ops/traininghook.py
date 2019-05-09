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
import numpy as np
from openpathsampling.pathsimulators.hooks import PathSimulatorHook
from openpathsampling.collectivevariable import CollectiveVariable
from .selector import RCModelSelector
from ..base.rcmodel import RCModel
from ..base.trainset import TrainSet


logger = logging.getLogger(__name__)


class TrainingHook(PathSimulatorHook):
    """
    OPS PathSimulatorHook to train arcd.RCModels on shooting data.

    Note that the training itself is left to the model, this simply calls
    the models train_hook() function after every MCStep and keeps the
    trainingset up to date.

    Attributes:
    -----------
        model - :class:`arcd.base.RCModel` that predicts RC values
        trainset - :class:`arcd.base.TrainSet` to store the shooting results
        save_model_interval - int, save the current model every
                              save_model_interval MCStep with a suffix
                              indicating the step,
                              Note: can be inifnity to never save
        density_collection - dict, contains parameters to control collection of
                             density of points on TPs,
                             'enabled' - bool, wheter to collect at all
                             'first_collection' - int, step at which we create
                                                  the initial estimate
                             'recreate_interval' - int, interval in which we
                                                   recreate the estimate, i.e.
                                                   use newly predicted probs
                                                   for all points
        train_states - int, number of (virtual) shots from the endpoints of
                       each trial trajectory that lie inside the states,
                       used to enrich the TrainingSet with commited SPs,
                       use a value of 0 to deactivate, default is 10
        save_model_extension - str, the file extension to use when saving the
                               model, same as arcd.RCModel.save_model_extension
        save_model_suffix - str, suffix to append to OPS storage name when
                            constructing the model savename
        save_model_after_simulation - bool, wheter to save the (final) model
                                      after the last MCStep

    """

    implemented_for = ['before_simulation',
                       'after_step',
                       'after_simulation'
                       ]
    # need to have it here, such that we can get it without instantiating
    save_model_extension = RCModel.save_model_extension
    save_model_suffix = '_RCmodel'
    save_model_after_simulation = True
    # we will save the trainset after the last step
    # into sim.storage.tags as $save_trainset_prefix + '.after_step_{:d}'
    save_trainset_prefix = 'arcd.TrainSet.data'
    save_trainset_suffix = '.after_step_{:d}'
    # whether we add invalid MCSteps to the TrainSet
    # this is passed to TrainSet.add_ops_step() as add_invalid
    add_invalid_mcsteps = False

    def __init__(self, model, trainset, save_model_interval=500,
                 density_collection={'enabled': True,
                                     'first_collection': 500,
                                     'recreate_interval': 1000,
                                     },
                 train_states=10,
                 ):
        """Initialize an arcd.TrainingHook."""
        self.model = model
        self.trainset = trainset
        self.save_model_interval = save_model_interval
        density_collection_defaults = {'enabled': True,
                                       'first_collection': 500,
                                       'recreate_interval': 1000,
                                       }
        density_collection_defaults.update(density_collection)
        self.density_collection = density_collection_defaults
        self.train_states = train_states

    def _get_model_from_sim_storage(self, sim):
        if sim.storage is not None:
            spath = sim.storage.abspath
            sdir = os.path.dirname(spath)
            sname = os.path.basename(spath)
            content = os.listdir(sdir)
            possible_mods = [c for c in content
                             if (os.path.isfile(os.path.join(sdir, c))
                                 and sname in c
                                 and c.endswith(self.save_model_suffix
                                                + self.save_model_extension)
                                 )
                             ]
            if len(possible_mods) == 1:
                # only one possible model, take it
                mod_fname = os.path.join(sdir, possible_mods[0])
                # this gives us the correct subclass and a half-fixed state
                # i.e. we set descriptor_transform to the OPS CV
                state, cls = RCModel.load_state(mod_fname, sim.storage)
                # this corrects the rest of the state,
                # e.g. loads the associated ANN with weights
                state = cls.fix_state(state)
                # this finally instantiates the correct RCModel class
                return cls.set_state(state)
            elif len(possible_mods) == 0:
                logger.error('No matching model file found!')
            else:
                logger.error('Multiple matching model files found.')
        else:
            logger.error('Simulation has no attached storage, '
                         + 'can not find a model file.')

    def _create_trainset_from_sim_storage(self, sim, states,
                                          descriptor_transform):
        if sim.storage is not None:
            descriptors, shot_results = self._find_trainset_data(sim.storage)
            if descriptors is not None:
                # we found a trainset in storage
                logger.info('Found old TrainSet data in storage.tags')
                return TrainSet(states, descriptor_transform,
                                descriptors, shot_results)

            logger.info('Could not find old TrainSet data. '
                        + 'Recreating from storage.steps.')
            trainset = TrainSet(states, descriptor_transform)
            for step in sim.storage.steps:
                trainset.append_ops_mcstep(
                                mcstep=step, add_states=self.train_states,
                                add_invalid=self.add_invalid_mcsteps
                                          )
            return trainset
        else:
            logger.error('Can not recreate TrainSet without storage')

    def _find_trainset_data(self, storage):
        keys = list(storage.tags.keys())
        keys = [k for k in keys if self.save_trainset_prefix in k]
        if len(keys) < 1:
            # did not find anything
            return None, None
        strip = (len(self.save_trainset_prefix)
                 + len(self.save_trainset_suffix)
                 - 4  # we ignore the first characters up until the number
                 )
        # find the trainset data with the highest step number
        numbers = [int(k[strip:]) for k in keys]
        max_idx = np.argmax(numbers)
        if keys[max_idx] == len(storage.steps):
            # make sure this trainset is the one saved at last step!
            # if the previous TPS simulation was killed it can happen that
            # the trainset is not saved, we try to correct as good as possible
            # this should at least warn now and will result in TS recreation
            logger.warning('The TrainSet we found does not match the number of'
                           + ' steps in storage. This could mean the'
                           + ' simulation before did not terminate properly.')
            return None, None
        descriptors, shot_results = storage.tags[keys[max_idx]]
        return descriptors, shot_results

    def before_simulation(self, sim):
        """Will be called by OPS Pathsimulator once before the simulation."""
        # if we have no model we will try to reload it
        if self.model is None:
            model = self._get_model_from_sim_storage(sim)
            if model is None:
                raise RuntimeError('RCmodel not set and could not load any'
                                   + ' model from file.')
            self.model = model
            # TODO: this might not always be what we want!
            # we put the loaded model in all RCmodelSelectors...?
            # save the model possibly a second time, but with every Selector?!
            selector_states = []
            for move_group in sim.move_scheme.movers.values():
                for mover in move_group:
                    if isinstance(mover.selector, RCModelSelector):
                        mover.selector.model = model
                        selector_states.append(mover.selector.states)
            logger.info('Restored saved model into TrainingHook and Selector')
        # if we have no trainset try to repopulate it
        if self.trainset is None:
            if len(selector_states) == 1:
                # only one arcd.RCModelSelector, take its states
                states = selector_states[0]
            else:
                raise ValueError('Could not reconstruct states for trainingset'
                                 + '. Please pass a training set with states.')

            self.trainset = self._create_trainset_from_sim_storage(
                                        sim, states, model.descriptor_transform
                                                                   )
            logger.info('Recreated TrainSet from storage.steps')
        # save the descriptor_transform
        # this should essentially be a no-op if it is already in storage
        # but circumvents unhappy users that forgot to save the transform
        if isinstance(self.model.descriptor_transform, CollectiveVariable):
            if sim.storage is not None:
                sim.storage.save(self.model.descriptor_transform)

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        """Will be called by OPS PathSimulator after every MCStep."""
        def iter_tps(storage, start=0):
            # find the last accepted TP to be able to add it again
            # instead of the rejects we could find
            last_accept = start
            found = False
            while not found:
                if storage.steps[last_accept].change.canonical.accepted:
                    found = True  # since we use break this is not neccessary
                    break
                last_accept -= 1
            # now actually iterate over the storage
            for i, step in enumerate(storage.steps[start:]):
                if step.change.canonical.accepted:
                    last_accept = i + start
                    yield step.change.canonical.trials[0].trajectory
                else:
                    change = storage.steps[last_accept].change
                    yield change.canonical.trials[0].trajectory

        # results is the MCStep
        self.trainset.append_ops_mcstep(mcstep=results,
                                        add_states=self.train_states,
                                        add_invalid=self.add_invalid_mcsteps)
        self.model.train_hook(self.trainset)
        if sim.storage is not None:
            if self.density_collection['enabled']:
                # collect density of points on TPs in probability space
                dc = self.model.density_collector
                recreate = self.density_collection['recreate_interval']
                first = self.density_collection['first_collection']
                if step_number - first >= 0:
                    if step_number % recreate == 0:
                        # recreation time
                        dc.evaluate_density_on_trajectories(
                                            model=self.model,
                                            trajectories=iter_tps(sim.storage),
                                            update=False
                                                            )
                    elif step_number - first == 0:
                        # first collection
                        dc.evaluate_density_on_trajectories(
                                            model=self.model,
                                            trajectories=iter_tps(
                                                            sim.storage,
                                                            start=-first
                                                                  ),
                                            update=True
                                                            )
                    else:
                        # only the last step
                        dc.evaluate_density_on_trajectories(
                                            model=self.model,
                                            trajectories=iter_tps(
                                                            sim.storage,
                                                            start=step_number
                                                                    ),
                                            update=True
                                                            )

            if step_number % self.save_model_interval == 0:
                # save the model every save_model_interval MCSteps
                spath = sim.storage.abspath
                fname = (spath + self.save_model_suffix
                         + '_at_step{:d}'.format(step_number)
                         )
                self.model.save(fname)
                logger.info('Saved intermediate RCModel as ' + fname)

    def after_simulation(self, sim):
        """Will be called by OPS PathSimulator once after the simulation."""
        if sim.storage is not None:
            spath = sim.storage.abspath
            # save without step-suffix to reload at simulation start
            fname = spath + self.save_model_suffix
            # we want to overwrite the last final model,
            # such that we always start with a current model
            self.model.save(fname, overwrite=True)
            logger.info('Saved RCmodel as ' + fname)
            # also save the descriptors and shot-results for faster
            # TrainSet recreation
            save_name = (self.save_trainset_prefix
                         + self.save_trainset_suffix.format(sim.step)
                         )
            sim.storage.tags[save_name] = [self.trainset.descriptors,
                                           self.trainset.shot_results]
        else:
            logger.warn('Could not save model, as there is no storage '
                        + 'associated with the simulation.')
