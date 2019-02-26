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
import logging
from openpathsampling.pathsimulators.hooks import PathSimulatorHook


logger = logging.getLogger(__name__)


class TrainingHook(PathSimulatorHook):
    """
    TODO
    Parameters:
    -----------
        model - :class:`arcd.base.RCModel` that predicts RC values
        trainset - :class:`arcd.base.TrainSet` to store the shooting results
    """
    implemented_for = [#'before_simulation',
                       #'before_step',
                       'after_step',
                       #'after_simulation'
                       ]
    # TODO: load + save models!?
    # TODO: every model should have a load/save routine...!

    def __init__(self, model, trainset):
        self.model = model
        self.trainset = trainset

    def before_simulation(self, sim):
        # TODO: here we could/should try to load an existing model
        pass

    def before_step(self, sim, step_number, step_info, state):
        pass

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        # results is the MCStep
        self.trainset.append_ops_mcstep(results)
        self.model.train_hook(self.trainset)

    def after_simulation(self, sim):
        # TODO: here we should save the model
        # TODO: derive fname from simulation name
        # self.model.save(fname)
        pass
