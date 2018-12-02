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
    Parameters:
    -----------
        model - a wrapped model
        trainer - a trainer object able to train the model
        history - a History object, saving selection and training history
    """
    implemented_for = [#'before_simulation',
                       #'before_step',
                       'after_step',
                       #'after_simulation'
                       ]

    def __init__(self, model, trainer, trainset, history):
        self.model = model
        self.trainer = trainer
        self.trainset = trainset
        self.history = history

    def before_simulation(self, sim):
        pass

    def before_step(self, sim, step_number, step_info, state):
        pass

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        # results is the MCStep
        self.trainset.add_mcstep(results)
        self.trainer.train(self.model, self.trainset, self.history)

    def after_simulation(self, sim):
        pass
