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
import os
import copy
import numpy as np
from abc import abstractmethod
from tensorflow.keras import backend as K
from ..base.rcmodel import RCModel
from ..base.rcmodel_train_decision import (_train_decision_funcs,
                                           _train_decision_defaults,
                                           _train_decision_docs)
from .utils import load_keras_model


logger = logging.getLogger(__name__)


class KerasRCModel(RCModel):
    """
    Wraps a Keras model for use with arcd.
    """
    # need to have it here, such that we can get it without instantiating
    save_nnet_suffix = '_keras.h5'

    def __init__(self, nnet, descriptor_transform=None):
        self.nnet = nnet
        self.log_train_decision = []
        self.log_train_loss = []
        self._count_train_hook = 0
        # need to call super __init__ last such that it can make use of
        # the properties and methods we implement here
        super().__init__(descriptor_transform)

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
        if not os.path.exists(state['nnet']):
            # try fixing changed absolute paths by taking
            # ops_storage_dir + base filname
            # this enables us to copy the whole folder containing OPS-storage
            # and model to another location/machine
            fname = os.path.basename(state['nnet'])
            state['nnet'] = os.path.join(state['_pickle_file_dirname'], fname)
        nnet = load_keras_model(state['nnet'])
        state['nnet'] = nnet
        return state

    def save(self, fname, overwrite=False):
        self.nnet.save(fname + self.save_nnet_suffix, overwrite=overwrite)
        # keep a ref to the network
        nnet = self.nnet
        # but replace with the name of the file in self.__dict__
        self.nnet = fname + self.save_nnet_suffix
        # let super save the state dict
        super().save(fname, overwrite)
        # and restore the nnet such that self stays functional
        self.nnet = nnet

    def _log_prob(self, descriptors):
        return self.nnet.predict(descriptors)

    def train_hook(self, trainset):
        self._count_train_hook += 1
        train, new_lr, epochs, batch_size = self.train_decision(trainset)
        self.log_train_decision.append([train, new_lr, epochs, batch_size])
        if new_lr is not None:
            logger.info('Setting learning rate to {:.3e}'.format(new_lr))
            self.set_lr(new_lr)
        if train:
            logger.info('Training for {:d} epochs'.format(epochs))
            self.log_train_loss.append([self.train_epoch(trainset,
                                                         batch_size=batch_size
                                                         )
                                        for _ in range(epochs)])

    def test_loss(self, trainset):
        loss = self.nnet.evaluate(x=trainset.descriptors,
                                  y=trainset.shot_results,
                                  sample_weight=trainset.weights,
                                  verbose=0)
        # loss is the mean loss per training point
        # so multiply by sum of weights of points and divide through weighted
        # number of shots to return loss normalized per shot
        return loss * np.sum(trainset.weights) / np.sum(np.sum(trainset.shot_results, axis=-1)
                                                        * trainset.weights)

    @abstractmethod
    def train_decision(self, trainset):
        # should return train, new_lr, epochs
        # i.e. a bool, a float or None and an int
        raise NotImplementedError

    def set_lr(self, new_lr):
        K.set_value(self.nnet.optimizer.lr, new_lr)

    def train_epoch(self, trainset, batch_size=128, shuffle=True):
        # train for one epoch == one pass over the trainset
        loss = 0.
        for des, shots, weights in trainset.iter_batch(batch_size, shuffle):
            # multiply by batch lenght to get total loss per batch
            # and then at the ernd the correct average loss per shooting point
            loss += (self.nnet.train_on_batch(x=des, y=shots,
                                              sample_weight=weights)
                     * np.sum(weights)
                     )
        # get loss per shot as for pytorch models,
        # the lossFXs are not normalized in any way
        return loss / np.sum(np.sum(trainset.shot_results, axis=-1)
                             * trainset.weights
                             )


class EEScaleKerasRCModel(KerasRCModel):
    """Expected efficiency scale KerasRCModel."""
    __doc__ += _train_decision_docs['EEscale']

    def __init__(self, nnet, descriptor_transform=None,
                 ee_params=_train_decision_defaults["EEscale"]):
        super().__init__(nnet, descriptor_transform)
        # make it possible to pass only the altered values in dictionary
        ee_params_defaults = copy.deepcopy(_train_decision_defaults['EEscale'])
        ee_params_defaults.update(ee_params)
        self.ee_params = ee_params_defaults

    train_decision = _train_decision_funcs['EEscale']


class EERandKerasRCModel(KerasRCModel):
    """Expected efficiency randomized KerasRCModel."""
    __doc__ += _train_decision_docs['EErand']

    def __init__(self, nnet, optimizer,
                 ee_params=_train_decision_defaults['EErand'],
                 descriptor_transform=None, loss=None):
        super().__init__(nnet, optimizer, descriptor_transform, loss)
        # make it possible to pass only the altered values in dictionary
        defaults = copy.deepcopy(_train_decision_defaults['EErand'])
        defaults.update(ee_params)
        self.ee_params = defaults
        self._decisions_since_last_train = 0

    train_decision = _train_decision_funcs['EErand']
