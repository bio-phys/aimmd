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
from abc import abstractmethod
from keras import backend as K
from ..base.rcmodel import RCModel
from .utils import load_keras_model


logger = logging.getLogger(__name__)


class KerasRCModel(RCModel):
    """
    Wraps a Keras model for use with arcd.
    """
    # need to have it here, such that we can get it without instantiating
    save_nnet_suffix = '_keras.h5'

    def __init__(self, nnet, descriptor_transform=None):
        super().__init__(descriptor_transform)
        self.nnet = nnet
        self.log_train_decision = []
        self.log_train_loss = []
        self._count_train_hook = 0

    @property
    def n_out(self):
        return self.nnet.output_shape[1]

    @classmethod
    def set_state(cls, state):
        obj = cls(nnet=state['nnet'])
        obj.__dict__.update(state)
        return obj

    @classmethod
    def fix_state(cls, state):
        nnet = load_keras_model(state['nnet'])
        state['nnet'] = nnet
        return state

    def save(self, fname, overwrite=False):
        self.nnet.save(fname + self.save_nnet_suffix, overwrite=overwrite)
        # keep a ref to the network
        nnet = self.nnet
        # but replace with the path to file in self.__dict__
        self.nnet = fname + self.save_nnet_suffix
        # let super save the state dict
        super().save(fname, overwrite)
        # and restore the nnet such that self stays functional
        self.nnet = nnet

    def _log_prob(self, descriptors):
        return self.nnet.predict(descriptors)

    def train_hook(self, trainset):
        self._count_train_hook += 1
        train, new_lr, epochs = self.train_decision(trainset)
        self.log_train_decision.append([train, new_lr, epochs])
        if new_lr:
            K.set_value(self.nnet.optimizer.lr, new_lr)
        if train:
            self.log_train_loss.append([self._train_epoch(trainset)
                                        for _ in range(epochs)])

    @abstractmethod
    def train_decision(self, trainset):
        # should return train, new_lr, epochs
        # i.e. a bool, a float or None and an int
        pass

    def _train_epoch(self, trainset, batch_size=128, shuffle=True):
        # train for one epoch == one pass over the trainset
        loss = 0.
        for descriptors, shot_results in trainset.iter_batch(batch_size, shuffle):
            # multiply by batch lenght to get proper average loss per point
            loss += (self.nnet.train_on_batch(x=descriptors, y=shot_results)
                     * len(shot_results)
                     )
        # *2 to get loss per shot as for pytorch models, i.e. we assume TwoWayShooting
        loss /= 2.*len(trainset)
        return loss


class EEKerasRCModel(KerasRCModel):
    """
    Expected efficiency Keras RCModel wrapper
    """
    def __init__(self, nnet, descriptor_transform=None,
                 ee_params={'lr_0': 1e-3,
                            'lr_min': 1e-4,
                            'epochs_per_train': 5,
                            'interval': 3,
                            'window': 100}
                 ):
        super().__init__(nnet, descriptor_transform)
        # make it possible to pass only the altered values in dictionary
        ee_params_defaults = {'lr_0': 1e-3,
                              'lr_min': 1e-4,
                              'epochs_per_train': 5,
                              'interval': 3,
                              'window': 100}
        ee_params_defaults.update(ee_params)
        self.ee_params = ee_params_defaults

    def train_decision(self, trainset):
        # TODO: atm this is the same as for Pytorch Expected efficiecny models!
        # TODO: we should deduplicate this somehow...
        train = False
        lr = self.ee_params['lr_0']
        lr *= self.train_expected_efficiency_factor(trainset,
                                                    self.ee_params['window'])
        if self._count_train_hook % self.ee_params['interval'] == 0:
            if lr >= self.ee_params['lr_min']:
                train = True
        epochs = self.ee_params['epochs_per_train']
        logger.info('Decided train={:d}, lr={:.3e}, epochs={:d}'.format(train,
                                                                        lr,
                                                                        epochs)
                    )
        return train, lr, epochs
