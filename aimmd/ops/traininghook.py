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
import numpy as np
from openpathsampling.beta.hooks import PathSimulatorHook
from openpathsampling import CollectiveVariable
from openpathsampling import Volume
from .selector import RCModelSelector
from ..base.rcmodel import RCModel
from ..base.trainset import TrainSet


logger = logging.getLogger(__name__)


class TrainingHook(PathSimulatorHook):
    """
    OPS PathSimulatorHook to train aimmd.RCModels on shooting data.

    NOTE on continuing simulations, possibly with a new/different RCmodel:
    If model or trainset are None we will try to load them before the
    simulation, however this can obviously only work if continuing an existing
    simulation.
    If replace_model is True (and a model is passed) we will replace all
    references to the old model with the newly passed RCmodel. This can be
    usefull for continuing a TPS simulation with an optimized model
    architecture.

    NOTE that the training itself is left to the model, TrainingHook only calls
    the models train_hook() function after every MCStep and keeps the
    trainingset up to date.

    Attributes:
    -----------
        model - :class:`aimmd.base.RCModel` that predicts RC values
        trainset - :class:`aimmd.base.TrainSet` to store the shooting results
        save_model_interval - int, save the current model every
                              save_model_interval MCStep with a suffix
                              indicating the step,
                              Note: can be inifnity to never save
        replace_model - bool, wheter we should set/replace the RCModel in all
                        associated OPS-selectors
        density_collection - dict, contains parameters to control collection of
                             density of points on TPs,
                             'enabled' - bool, wheter to collect at all
                             'interval' - interval at which we add TPs to the
                                          estimate
                             'first_collection' - int, step at which we create
                                                  the initial estimate
                             'recreate_interval' - int, interval in which we
                                                   recreate the estimate, i.e.
                                                   use newly predicted probs
                                                   for all points
        save_model_extension - str, the file extension to use when saving the
                               model, same as aimmd.RCModel.save_model_extension
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
    save_trainset_prefix = 'aimmd.TrainSet.data'
    save_trainset_suffix = '.after_step_{:d}'
    # no suffix/prefix, states will stay the same the whole simulation
    save_trainset_states = 'aimmd.TrainSet.states'
    # descriptor_transform does not change during simulation either
    save_trainset_descriptor_transform = 'aimmd.TrainSet.descriptor_transform'
    # whether we add invalid MCSteps to the TrainSet
    # this is passed to TrainSet.add_ops_step() as add_invalid
    add_invalid_mcsteps = False

    def __init__(self, model=None, trainset=None, save_model_interval=500,
                 replace_model=False,
                 density_collection={'enabled': True,
                                     'interval': 20,
                                     'first_collection': 500,
                                     'recreate_interval': 1000,
                                     }
                 ):
        """Initialize an aimmd.TrainingHook."""
        self.model = model
        self.trainset = trainset
        self.save_model_interval = save_model_interval
        self._replace_model = replace_model
        density_collection_defaults = {'enabled': True,
                                       'interval': 20,
                                       'first_collection': 500,
                                       'recreate_interval': 1000,
                                       }
        density_collection_defaults.update(density_collection)
        self.density_collection = density_collection_defaults

    @classmethod
    def load_trainset(cls, storage, descriptor_transform=None, states=None):
        """
        Load the most recent TrainSet data from an ops storage.

        Create the TrainSet from steps if no TrainSet data is found in storage.
        Passing descriptor_transform and/or states explicitly will set those in the
        returned TrainSet (instead of the initial values it was saved with).

        Parameters
        ----------
        storage - an open ops storage object
        descriptor_transform - None, str or callable that takes snapshots/trajectories,
                               if str we will try to retrieve the cv with that name
                               from the ops storage,
                               if None we will get the descriptor_transform from
                               storage.tags
        states - None, list of str or list of callables/ops-volumes that take snapshots,
                 if list of str, we will try to load the volumes from storage,
                 if None we will get the states from storage.tags

        Returns
        -------
        aimmd.TrainSet

        """
        save_trainset_states = cls.save_trainset_states
        save_trainset_descriptor_transform = cls.save_trainset_descriptor_transform
        # first try to load the saved states and descriptor_transform if None given
        if descriptor_transform is None:
            descriptor_transform = storage.tags[save_trainset_descriptor_transform]
        if states is None:
            states = storage.tags[save_trainset_states]
        # try to find the stuff with the same names as what we were passed
        if isinstance(descriptor_transform, str):
            descriptor_transform = storage.cvs.find(descriptor_transform)
        if all(isinstance(s, str) for s in states):
            states = [storage.volumes.find(s) for s in states]
        tsdata = cls._find_latest_trainset_data(storage)
        descriptors, shot_results, weights, delta = tsdata
        if descriptors is not None:
            logger.info('Found a TrainSet in storage.tags')
            # we found a trainset in storage
            trainset = TrainSet(states=states,
                                descriptor_transform=descriptor_transform,
                                descriptors=descriptors,
                                shot_results=shot_results,
                                weights=weights,
                                )
            if delta > 0:
                logger.info('Adding {:d} missing steps to the Trainset.'.format(delta))
                # add missing steps if any
                for step in storage.steps[-delta:]:
                    trainset.append_ops_mcstep(step)
        else:
            logger.info('No TrainSet found. Recreating completely.')
            # recreate the trainset from scratch
            trainset = TrainSet(states=states,
                                descriptor_transform=descriptor_transform)
            for step in storage.steps:
                trainset.append_ops_mcstep(
                                           mcstep=step,
                                           add_invalid=False
                                          )
        return trainset

    @classmethod
    def _find_latest_trainset_data(cls, storage):
        # returns descriptors, shot_results, delta_complete
        # here delta_complete is the number of steps missing at the end of the TS
        save_trainset_prefix = cls.save_trainset_prefix
        save_trainset_suffix = cls.save_trainset_suffix
        keys = list(storage.tags.keys())
        keys = [k for k in keys if save_trainset_prefix in k]
        if len(keys) < 1:
            # did not find anything
            return None, None, None, 0
        strip = (len(save_trainset_prefix)
                 + len(save_trainset_suffix)
                 - 4  # we ignore the first characters up until the number
                 )
        # find the trainset data with the highest step number
        numbers = [int(k[strip:]) for k in keys]
        max_idx = np.argmax(numbers)
        last_mccycle = storage.steps[-1].mccycle
        # make sure this trainset is the one saved at last step!
        # if the previous TPS simulation was killed it can happen that
        # the trainset is not saved, we try to correct as good as possible
        delta_complete = last_mccycle - numbers[max_idx]
        if delta_complete != 0:
            logger.warning('The TrainSet we found does not match the number of'
                           + ' steps in storage. This could mean the'
                           + ' simulation before did not terminate properly.'
                           + ' We will try to add the missing steps to continue.')
        data = storage.tags[keys[max_idx]]
        if len(data) == 2:
            # this is for backwards compability:
            # make it possible to load trainsets that do not have weights
            descriptors, shot_results = data
            weights = np.ones((shot_results.shape[0]), dtype=np.float64)
        elif len(data) == 3:
            descriptors, shot_results, weights = data
        return descriptors, shot_results, weights, delta_complete

    @staticmethod
    def _match_model_to_trainset(model, trainset):
        """Update predicted p and q lists if they are shorter than the trainset."""
        if len(model.expected_p) < len(trainset):
            delta = len(model.expected_p) - len(trainset)
            ps = trainset.descriptors[delta:]
            for p in ps:
                model.register_sp(p, use_transform=False)
        return model

    def _get_model_from_sim(self, sim):
        if sim.storage is not None:
            spath = sim.storage.filename
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

    def before_simulation(self, sim):
        """Will be called by OPS Pathsimulator once before the simulation."""
        # if we have no trainset try to load and/or repopulate it
        if self.trainset is None:
            if sim.storage is None:
                raise RuntimeError('Can only reconstruct TrainSet if simulation'
                                   + ' has a storage attached. Please pass a '
                                   + 'TrainSet or use a simulation with storage.')
            states = sim.storage.tags[self.save_trainset_states]
            if states is None:
                raise RuntimeError('Could not reconstruct states for TrainSet'
                                   + '. Please pass a TrainSet with states.')
            self.trainset = self.load_trainset(storage=sim.storage)
            logger.info('Successfully recreated TrainSet from storage.')

        # if we have no model we will try to reload it
        replace_model = False
        if self.model is None:
            model = self._get_model_from_sim(sim)
            if model is None:
                raise RuntimeError('RCmodel not set and could not load any'
                                   + ' model from file.')
            # this is to potentially fix simulations that got killed prematurely
            # and should not do anything if everything went well
            self.model = self._match_model_to_trainset(model=model,
                                                       trainset=self.trainset)
            replace_model = True  # need to set the loaded model in the selector
            logger.info('Successfully loaded the RCModel.')

        # set model in SP-Selector, either the one we just loaded
        # or a new one that was passed if self._replace_model=True
        if self._replace_model or replace_model:
            # TODO: this might not always be what we want!
            # we put the (loaded) model in all RCmodelSelectors...?
            # save the model possibly a second time, but with every Selector?!
            for move_group in sim.move_scheme.movers.values():
                for mover in move_group:
                    if isinstance(mover.selector, RCModelSelector):
                        mover.selector.model = self.model
            logger.info('Registered model in the SP-Selector')

        # save stuff if not already in storage
        # save the trainset states
        if isinstance(self.trainset.states[0], Volume):
            # we check if it is a volume because only ops volumes are guranteed
            # to save and load as expected
            if sim.storage is not None:
                states = sim.storage.tags[self.save_trainset_states]
                # acessing a non-existing tag will return a None value
                if states is None:
                    # therefore we now know that we did not save the states yet
                    sim.storage.tags[self.save_trainset_states] = self.trainset.states
        # save the descriptor_transform
        # this should essentially be a no-op if it is already in storage
        # but circumvents unhappy users that forgot to save the transform
        if isinstance(self.model.descriptor_transform, CollectiveVariable):
            if sim.storage is not None:
                sim.storage.save(self.model.descriptor_transform)
                # also make it available in storage.tags
                dtransform = sim.storage.tags[self.save_trainset_descriptor_transform]
                if dtransform is None:
                    sim.storage.tags[self.save_trainset_descriptor_transform] = self.model.descriptor_transform

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        """Will be called by OPS PathSimulator after every MCStep."""
        def get_tps(storage, start=0):
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

        # results is the MCStep
        self.trainset.append_ops_mcstep(
                                    mcstep=results,
                                    add_invalid=self.add_invalid_mcsteps
                                        )
        self.model.train_hook(self.trainset)
        if sim.storage is not None:
            if step_number % self.save_model_interval == 0:
                # save the model every save_model_interval MCSteps
                spath = sim.storage.filename
                fname = (spath + self.save_model_suffix
                         + '_at_step{:d}'.format(step_number)
                         )
                self.model.save(fname)
                logger.info('Saved intermediate RCModel as ' + fname)

            if self.density_collection['enabled']:
                # collect density of points on TPs in probability space
                dc = self.model.density_collector
                interval = self.density_collection['interval']
                recreate = self.density_collection['recreate_interval']
                first = self.density_collection['first_collection']
                if step_number - first >= 0:
                    if step_number - first == 0:
                        # first collection
                        tps, counts = get_tps(sim.storage, start=-first)
                        dc.add_density_for_trajectories(
                                            model=self.model,
                                            trajectories=tps,
                                            counts=counts
                                                        )
                    elif step_number % interval == 0:
                        # add only the last interval steps
                        tps, counts = get_tps(sim.storage, start=-interval)
                        dc.add_density_for_trajectories(
                                            model=self.model,
                                            trajectories=tps,
                                            counts=counts
                                                        )
                    # Note that this is an if because reevaluation should be
                    # independent of adding new TPs in the same MCStep
                    if step_number % recreate == 0:
                        # reevaluation time
                        dc.reevaluate_density(model=self.model)

    def after_simulation(self, sim):
        """Will be called by OPS PathSimulator once after the simulation."""
        if sim.storage is not None:
            spath = sim.storage.filename
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
                                           self.trainset.shot_results,
                                           self.trainset.weights]
        else:
            logger.warn('Could not save model, as there is no storage '
                        + 'associated with the simulation.')
