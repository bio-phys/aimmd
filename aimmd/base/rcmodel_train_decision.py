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

logger = logging.getLogger(__name__)


# train_decision functions, their defaults and their docstrings
# NOTE: they are externalised because they are the same for all RCModels
# (at least for all that use an ANN as prediction model)
# but these models all have different __init__ signatures so we need to rewrite __init__
# since we also want to write the class docstring only once we have it here too
_train_decision_docs = {}
_train_decision_defaults = {}
_train_decision_funcs = {}


def train_decision_EEscale(self, trainset):
    """Scale learning rate by EE-factor and trains only if lr > lr_min."""
    train = False
    lr = self.ee_params['lr_0']
    lr *= self.train_expected_efficiency_factor(trainset,
                                                self.ee_params['window'])
    if self._count_train_hook % self.ee_params['interval'] == 0:
        if lr >= self.ee_params['lr_min']:
            train = True
    epochs = self.ee_params['epochs_per_train']
    batch_size = self.ee_params['batch_size']
    logger.info('Decided train=' + str(train) + ', lr=' + str(lr)
                + ', epochs=' + str(epochs)
                + ', batch_size=' + str(batch_size)
                )
    return train, lr, epochs, batch_size

_train_decision_funcs['EEscale'] = train_decision_EEscale
_train_decision_docs['EEscale'] = """
    Controls training by multiplying lr with expected efficiency factor

    ee_params - dict, 'expected efficiency parameters', containing
        lr_0 - float, base learning rate
        lr_min - float, minimal learning rate we still train with
        epochs_per_train - int, if we train we train for this many epochs
        interval - int, we attempt to train every interval MCStep,
                   measured by self.train_hook() calls
        window - int, size of the smoothing window used for expected efficiency
        batch_size - int or None, size of chunks of trainset for training,
                     NOTE: if None, we will use len(trainset), this is needed
                     for using some optimizers, e.g. LBFGS
                                  """
_train_decision_defaults['EEscale'] = {'lr_0': 1e-3,
                                       'lr_min': 1e-4,
                                       'epochs_per_train': 5,
                                       'interval': 3,
                                       'window': 100,
                                       'batch_size': None
                                       }


def train_decision_EErand(self, trainset):
    """Train with probability given by current EEfact."""
    train = False
    self._decisions_since_last_train += 1
    if self._decisions_since_last_train >= self.ee_params['max_interval']:
        train = True
        self._decisions_since_last_train = 0
    elif self._count_train_hook % self.ee_params['interval'] == 0:
        ee_fact = self.train_expected_efficiency_factor(
                                            trainset=trainset,
                                            window=self.ee_params['window']
                                                        )
        if np.random.ranf() < ee_fact:
            train = True
            self._decisions_since_last_train = 0

    lr = None  # will not change the lr
    epochs = self.ee_params['epochs_per_train']
    batch_size = self.ee_params['batch_size']
    logger.info('Decided train=' + str(train) + ', lr=' + str(lr)
                + ', epochs=' + str(epochs)
                + ', batch_size=' + str(batch_size)
                )
    return train, lr, epochs, batch_size

_train_decision_funcs['EErand'] = train_decision_EErand
_train_decision_docs['EErand'] = """
    Do not change learning rate, instead train with frequency given by expected
    efficiency factor, i.e. we train only if np.randoom.ranf() < EE-factor.

    ee_params - dict, 'expected efficiency parameters'
        epochs_per_train - int, if we train we train for this many epochs
        interval - int, we attempt to train every interval MCStep,
                   measured by self.train_hook() calls
        window - int, size of the smoothing window used for expected efficiency
        batch_size - int or None, size of chunks of trainset for training,
                     NOTE: if None, we will use len(trainset), this is needed
                     for using some optimizers, e.g. LBFGS
                              """
_train_decision_defaults['EErand'] = {'epochs_per_train': 1,
                                      'interval': 2,
                                      'max_interval': 10,
                                      'window': 100,
                                      'batch_size': None
                                      }
