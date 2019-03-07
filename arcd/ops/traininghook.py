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
from openpathsampling.pathsimulators.hooks import PathSimulatorHook
from .selector import RCModelSelector
from ..base.rcmodel import RCModel
from ..base.trainset import TrainSet


logger = logging.getLogger(__name__)


class TrainingHook(PathSimulatorHook):
    """
    TODO
    Parameters:
    -----------
        model - :class:`arcd.base.RCModel` that predicts RC values
        trainset - :class:`arcd.base.TrainSet` to store the shooting results
    """
    implemented_for = ['before_simulation',
                       'after_step',
                       'after_simulation'
                       ]
    # need to have it here, such that we can get it without instantiating
    save_model_extension = RCModel.save_model_extension
    save_model_suffix = '_RCmodel'
    save_model_after_simulation = True

    def __init__(self, model, trainset, save_model_interval=100):
        self.model = model
        self.trainset = trainset
        self.save_model_interval = save_model_interval

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
                # this corrects the rest of the state, e.g. load the ANN with weights
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

    def _create_trainset_from_sim_storage(self, sim, states, descriptor_transform):
        if sim.storage is not None:
            trainset = TrainSet(states, descriptor_transform)
            for step in sim.storage.steps:
                trainset.append_ops_mcstep(step)
            return trainset
        else:
            logger.error('Can not recreate TrainSet without storage')

    def before_simulation(self, sim):
        # if we have no model we will try to reload it
        if self.model is None:
            model = self._get_model_from_sim_storage(sim)
            if model is None:
                raise RuntimeError('RCmodel not set and could not load any'
                                   + ' model from file.')
            self.model = model
            # TODO: this might not always be what we want!
            # TODO: we put the loaded model in all RCmodelSelectors...?
            # TODO: save the model possibly a second time, but with every RCModelSelector?!
            selector_states = []
            for move_group in sim.move_scheme.movers.values():
                for mover in move_group:
                    if isinstance(mover.selector, RCModelSelector):
                        mover.selector.model = model
                        selector_states.append(mover.selector.states)
            logger.info('Restored saved model into TrainingHook and RCModelSelector')

        if self.trainset is None:
            if len(selector_states) == 1:
                states = selector_states[0]
            else:
                raise ValueError('Could not reconstruct states for trainingset'
                                 + '. Please pass a training set with states.')

            self.trainset = self._create_trainset_from_sim_storage(
                                        sim, states, model.descriptor_transform
                                                                   )
            logger.info('Recreated TrainSet from storage.steps')

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        # results is the MCStep
        self.trainset.append_ops_mcstep(results)
        self.model.train_hook(self.trainset)
        # save the model every save_model_interval MCSteps
        if sim.storage is not None:
            if step_number % self.save_model_interval == 0:
                spath = sim.storage.abspath
                fname = (spath + self.save_model_suffix
                         + '_at_step{:d}'.format(step_number)
                         )
                self.model.save(fname)
                logger.info('Saved intermediate RCModel as ' + fname)

    def after_simulation(self, sim):
        if sim.storage is not None:
            spath = sim.storage.abspath
            # save without step-suffix to reload at simulation start
            fname = spath + self.save_model_suffix
            # we want to overwrite the last final model,
            # such that we always start with a current model
            self.model.save(fname, overwrite=True)
            logger.info('Saved RCmodel as ' + fname)
        else:
            logger.warn('Could not save model, as there is no storage '
                        + 'associated with the simulation.')
