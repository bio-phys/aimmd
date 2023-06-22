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
import copy
import h5py
import numpy as np
from abc import abstractmethod
from tensorflow.keras import backend as K
from ..base import Properties
from ..base.rcmodel import RCModel, RCModelAsyncMixin
from ..base.rcmodel_train_decision import (_train_decision_funcs,
                                           _train_decision_defaults,
                                           _train_decision_docs)
from ..base.utils import get_batch_size_from_model_and_descriptors
from .utils import load_keras_model


logger = logging.getLogger(__name__)


class KerasRCModel(RCModel):
    """Wraps a Keras model for use with aimmd."""

    def __init__(self, nnet, states, descriptor_transform=None, cache_file=None):
        # get n_out from model
        n_out = nnet.output_shape[1]
        super().__init__(states=states,
                         descriptor_transform=descriptor_transform,
                         cache_file=cache_file,
                         n_out=n_out)
        self.nnet = nnet
        self.log_train_decision = []
        self.log_train_loss = []
        self._count_train_hook = 0

    # Implemented in base RCModel
    #@property
    #def n_out(self):
    #    return self.nnet.output_shape[1]

    def object_for_pickle(self, group, overwrite=True,
                          name=None, storage_directory=None, **kwargs):
        def create_extlink_and_empty_subfile(fname, parent_group):
            # fname is expected to be a relative path starting at storage dir
            # also we expect that current directory is the storage directory
            ext_link = h5py.ExternalLink(fname, "/")
            parent_group["KerasRCModel"] = ext_link
            # create the external subfile and close it directly
            f = h5py.File(fname, mode="w")
            f.close()
            # open the link/external subfile and return the group
            return parent_group["KerasRCModel"]

        if name is None:
            raise ValueError("name must be given to deduce the filename.")
        if not name.lower().endswith(".h5"):
            name += ".h5"
        # create the directory if it does not exist
        if not os.path.isdir(
                os.path.join(storage_directory,
                             f"{group.file.filename}_KerasModelsSaveFiles")):
            os.mkdir(os.path.join(storage_directory,
                                  f"{group.file.filename}_KerasModelsSaveFiles"))
        # change to the storage directory to use relative paths for the extlinks
        old_dir = os.path.abspath(os.getcwd())  # remember current directory
        os.chdir(storage_directory)
        subfile = os.path.join(f"{group.file.filename}_KerasModelsSaveFiles", name)
        try:
            model_grp = group['KerasRCModel']  # just to check if it is there
        except KeyError:
            # file/group does not exist yet, create a link for the external file
            model_grp = create_extlink_and_empty_subfile(fname=subfile,
                                                         parent_group=group)
        else:
            if overwrite:
                # remove the old file so we can let keras recreate from scratch
                os.unlink(subfile)
                model_grp = create_extlink_and_empty_subfile(fname=subfile,
                                                             parent_group=group)
            else:
                raise RuntimeError(
                            "KerasRCModel file exists but overwrite=False."
                                   )
        # back to the old current directory
        os.chdir(old_dir)
        # save NN
        # NOTE: this is a dirty hack, tf checks if the file is a h5py.File
        #       but it uses only methods of the h5py.Group, so we set the
        #       class to File and reset to group after saving the model :)
        model_grp.__class__ = h5py.File
        self.nnet.save(model_grp, include_optimizer=True)
        model_grp.__class__ = h5py.Group
        # create return object which can be pickled, i.e. self without NN
        state = self.__dict__.copy()
        state["nnet"] = None
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        # and call supers object_for_pickle in case there is something left
        # in ret_obj.__dict__ that we can not pickle
        return super(__class__, ret_obj).object_for_pickle(group,
                                                           overwrite=overwrite,
                                                           **kwargs)

    def complete_from_h5py_group(self, group):
        model_grp = group["KerasRCModel"]
        # NOTE: same hack as for saving: pretend it is a h5py.File to tensorflow
        model_grp.__class__ = h5py.File
        self.nnet = load_keras_model(model_grp)
        model_grp.__class__ = h5py.Group
        # see if there is something left to do for super
        return super().complete_from_h5py_group(group)

    def _log_prob(self, descriptors, batch_size):
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                    model=self, descriptors=descriptors,
                                                                   )
        #predictions = []
        #n_split = (descriptors.shape[0] // batch_size) + 1
        #for descript_part in np.array_split(descriptors, n_split):
        #    pred = self.nnet.predict(descript_part, batch_size=batch_size)
        #    predictions.append(pred)
        #return np.concatenate(predictions, axis=0)
        return self.nnet.predict(descriptors, batch_size=batch_size)

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

    def test_loss(self, trainset, batch_size=None):
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                    model=self, descriptors=trainset.descriptors,
                                                                   )
        loss = self.nnet.evaluate(x=trainset.descriptors,
                                  y=trainset.shot_results,
                                  batch_size=batch_size,
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

    def train_epoch(self, trainset, batch_size=None, shuffle=True):
        # train for one epoch == one pass over the trainset
        loss = 0.
        for target in trainset.iter_batch(batch_size, shuffle):
            # multiply by batch lenght to get total loss per batch
            # and then at the ernd the correct average loss per shooting point
            loss += (self.nnet.train_on_batch(
                                    x=target[Properties.descriptors],
                                    y=target[Properties.shot_results],
                                    sample_weight=target[Properties.weights],
                                              )
                     * np.sum(target['weights'])
                     )
        # get loss per shot as for pytorch models,
        # the lossFXs are not normalized in any way
        return loss / np.sum(np.sum(trainset.shot_results, axis=-1)
                             * trainset.weights
                             )


# the async version is the same, it just uses the async mixin class
class KerasRCModelAsync(RCModelAsyncMixin, KerasRCModel):
    pass


class EEScaleKerasRCModelMixin:
    """Expected efficiency scale KerasRCModel."""
    __doc__ += _train_decision_docs['EEscale']

    def __init__(self, nnet, states, descriptor_transform=None,
                 ee_params=_train_decision_defaults["EEscale"], cache_file=None):
        super().__init__(nnet=nnet, states=states,
                         descriptor_transform=descriptor_transform,
                         cache_file=cache_file)
        # make it possible to pass only the altered values in dictionary
        ee_params_defaults = copy.deepcopy(_train_decision_defaults['EEscale'])
        ee_params_defaults.update(ee_params)
        self.ee_params = ee_params_defaults

    train_decision = _train_decision_funcs['EEscale']


class EEScaleKerasRCModel(EEScaleKerasRCModelMixin, KerasRCModel):
    pass


class EEScaleKerasRCModelAsync(EEScaleKerasRCModelMixin, KerasRCModelAsync):
    pass


class EERandKerasRCModelMixin:
    """Expected efficiency randomized KerasRCModel."""
    __doc__ += _train_decision_docs['EErand']

    def __init__(self, nnet, states, descriptor_transform=None,
                 ee_params=_train_decision_defaults['EErand'], cache_file=None):
        super().__init__(nnet=nnet, states=states,
                         descriptor_transform=descriptor_transform,
                         cache_file=cache_file)
        # make it possible to pass only the altered values in dictionary
        defaults = copy.deepcopy(_train_decision_defaults['EErand'])
        defaults.update(ee_params)
        self.ee_params = defaults
        self._decisions_since_last_train = 0

    train_decision = _train_decision_funcs['EErand']


class EERandKerasRCModel(EERandKerasRCModelMixin, KerasRCModel):
    pass


class EERandKerasRCModelAsync(EERandKerasRCModelMixin, KerasRCModelAsync):
    pass
