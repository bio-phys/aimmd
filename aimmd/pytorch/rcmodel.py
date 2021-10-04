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
import copy
import torch
import numpy as np
from abc import abstractmethod
from ..base import Properties
from ..base.rcmodel import RCModel, RCModelAsyncMixin
from ..base.rcmodel_train_decision import (_train_decision_funcs,
                                           _train_decision_defaults,
                                           _train_decision_docs)
# TODO: Buffered version or non-buffered version?
from ..base.storage import (BytesStreamtoH5py, BytesStreamtoH5pyBuffered,
                            H5pytoBytesStream)
from ..base.utils import get_batch_size_from_model_and_descriptors
from .utils import get_closest_pytorch_device, optimizer_state_to_device


logger = logging.getLogger(__name__)


# LOSS FUNCTIONS
def binomial_loss(target):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = target[Properties.q]
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]
    t1 = shots[:, 0] * torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)

    return weights.dot(torch.where(shots[:, 0] == 0, zeros, t1)
                       + torch.where(shots[:, 1] == 0, zeros, t2))


def binomial_loss_vect(target):
    """
    Loss for a binomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    Same as binomial_loss, but returns a vector loss values per point.
    Needed for multidomain RCModels to train the classifier.

    NOTE: NOT normalized.
    """
    q = target[Properties.q]
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]
    t1 = shots[:, 0] * torch.log(1. + torch.exp(q[:, 0]))
    t2 = shots[:, 1] * torch.log(1. + torch.exp(-q[:, 0]))
    zeros = torch.zeros_like(t1)

    return weights * (torch.where(shots[:, 0] == 0, zeros, t1)
                      + torch.where(shots[:, 1] == 0, zeros, t2)
                      )


def multinomial_loss(target):
    """
    Loss for multinomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    NOTE: This is NOT normalized.
    """
    q = target[Properties.q]
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]
    # log-sum-exp trick
    maxes = torch.max(q, dim=1, keepdim=True).values  # returns a named tuple (values, indices)
    ln_Z = maxes + torch.log(torch.sum(torch.exp(q - maxes), dim=1, keepdim=True))
    zeros = torch.zeros_like(shots)
    return weights.dot(torch.sum(
                        torch.where(shots == 0, zeros, (ln_Z - q) * shots),
                        dim=1
                                 )
                       )


def multinomial_loss_vect(target):
    """
    Loss for multinomial process.

    target - dictionary containing shooting point properties and NN output,
             keys are as in Properties

    Same as multinomial_loss, but returns a vector of loss values per point.
    Needed for multidomain RCModels to train the classifier.

    NOTE: NOT normalized.
    """
    q = target[Properties.q]
    shots = target[Properties.shot_results]
    weights = target[Properties.weights]
    # log-sum-exp trick
    maxes = torch.max(q, dim=1, keepdim=True).values  # returns a named tuple (values, indices)
    ln_Z = maxes + torch.log(torch.sum(torch.exp(q - maxes), dim=1, keepdim=True))
    zeros = torch.zeros_like(shots)
    return weights * torch.sum(
                            torch.where(shots == 0, zeros, (ln_Z - q) * shots),
                            dim=1
                              )


# RCModels using one ANN
class PytorchRCModel(RCModel):
    """Wrap pytorch neural networks for use with aimmd."""

    def __init__(self, nnet, optimizer, states, descriptor_transform=None,
                 loss=None, cache_file=None, n_out=None):
        # try to get number of outputs, i.e. predicted probabilities
        if n_out is None:
            try:
                # works if the last layer is linear
                n_out = list(nnet.modules())[-1].out_features
            except AttributeError:
                pass
            try:
                # works if the pytorch module has set n_out attribute/property
                # (as e.g. our own pytorch module containers do)
                n_out = nnet.n_out
            except AttributeError:
                pass
        super().__init__(states=states,
                         descriptor_transform=descriptor_transform,
                         cache_file=cache_file,
                         n_out=n_out,
                         )
        self.nnet = nnet  # a pytorch.nn.Module
        # any pytorch.optim optimizer, model parameters need to be registered already
        self.optimizer = optimizer
        self.log_train_decision = []
        self.log_train_loss = []
        self._count_train_hook = 0
        # needed to create the tensors on the correct device
        self._device = next(self.nnet.parameters()).device
        self._dtype = next(self.nnet.parameters()).dtype
        if loss is not None:
            # if custom loss given we take that
            self.loss = loss
        else:
            # otherwise we take the correct one for given n_out
            if self.n_out == 1:
                self.loss = binomial_loss
            else:
                self.loss = multinomial_loss

    # NOTE: the base RCModel now implements this (since it knows about the states)
    #@property
    #def n_out(self):
        # TODO: make us of both versions: i.e. check if nnet has n_out attribute
        # if not we can at least try to get the number of outputs, this way
        # users can use any model with last layer linear...!
        # FIXME:TODO: only works if the last layer is linear!
        #return list(self.nnet.modules())[-1].out_features
        # NOTE also not ideal, this way every pytorch model needs to set self.n_out
    #    return self.nnet.n_out

    # NOTE: NEW LOADING-SAVING API
    def object_for_pickle(self, group, overwrite=True, **kwargs):
        """
        Return pickleable object equivalent to self.

        Write everything we can not pickle to the h5py group.

        Parameters:
        -----------
        group - h5py group to write additional data to
        overwrite - bool, wheter to overwrite existing data in h5pygroup
        """
        state = self.__dict__.copy()
        state['nnet_class'] = self.nnet.__class__
        state['optimizer_class'] = self.optimizer.__class__
        state['nnet_call_kwargs'] = self.nnet.call_kwargs
        nnet_state = self.nnet.state_dict()
        optim_state = self.optimizer.state_dict()
        # we set them to None because we know to load them from h5py
        state['nnet'] = None
        state['optimizer'] = None
        if (not overwrite) and ('nnet' in group):
            # make sure we only overwrite if we want to
            raise RuntimeError("Model already exists but overwrite=False.")
        # save nnet and optimizer to the h5py group
        # we can just require the dsets, if they exist it is ok to overwrite
        nnet_dset = group.require_dataset('nnet', dtype=np.uint8,
                                          maxshape=(None,), shape=(0,),
                                          chunks=True,
                                          )
        optim_dset = group.require_dataset('optim', dtype=np.uint8,
                                           maxshape=(None,), shape=(0,),
                                           chunks=True,
                                           )
        # TODO: unbuffered for now: do we want buffer? which size?
        with BytesStreamtoH5py(nnet_dset) as stream_file:
            torch.save(nnet_state, stream_file)
        with BytesStreamtoH5py(optim_dset) as stream_file:
            torch.save(optim_state, stream_file)
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        # and call supers object_for_pickle in case there is something left
        # in ret_obj.__dict__ that we can not pickle
        return super(__class__, ret_obj).object_for_pickle(group,
                                                            overwrite=overwrite,
                                                            **kwargs)

    def complete_from_h5py_group(self, group, device=None):
        """
        Restore working state.

        Parameters:
        -----------
        group - h5py group with optional additional data
        device - None or torch device, if given will overwrite the torch model
                 restore location, if None will try to restore to a device
                 'close' to where it was saved from
        """
        if device is None:
            device = get_closest_pytorch_device(self._device)
        # instatiate and load the neural network
        nnet = self.nnet_class(**self.nnet_call_kwargs)
        with H5pytoBytesStream(group['nnet']) as stream_file:
            nnet_state = torch.load(stream_file, map_location=device)
        nnet.load_state_dict(nnet_state)
        del self.nnet_class
        del self.nnet_call_kwargs
        self.nnet = nnet.to(device)  # should be a no-op?!
        # now load the optimizer
        # first initialize with defaults
        optimizer = self.optimizer_class(self.nnet.parameters())
        del self.optimizer_class
        with H5pytoBytesStream(group['optim']) as stream_file:
            optim_state = torch.load(stream_file, map_location=device)
        # i think we do not need this?: put optimizer state on correct device
        #opt_sdict = optimizer_state_to_device(self.optimizer, device)
        optimizer.load_state_dict(optim_state)
        self.optimizer = optimizer
        return super().complete_from_h5py_group(group)

    @abstractmethod
    def train_decision(self, trainset):
        # this should decide if we train or not
        # return tuple(train, new_lr, epochs, batch_size)
        # train -> bool
        # new_lr -> float or None; if None: no change
        # epochs -> number of passes over the training set
        # batch_size -> size of the chunks of the trainset to use for training
        pass

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
        self.nnet.eval()  # put model in evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for target in trainset.iter_batch(batch_size, False):
                # create descriptors and results tensors where the model lives
                target = {key: torch.as_tensor(val,
                                               device=self._device,
                                               dtype=self._dtype
                                               )
                          for key, val in target.items()
                          }
                q_pred = self.nnet(target[Properties.descriptors])
                target[Properties.q] = q_pred
                loss = self.loss(target)
                total_loss += float(loss)
        self.nnet.train()  # and back to train mode
        return total_loss / np.sum(np.sum(trainset.shot_results, axis=-1)
                                   * trainset.weights
                                   )

    def set_lr(self, new_lr):
        # TODO: new_lr could be a list of different values if we have more parametersets...
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = new_lr

    def train_epoch(self, trainset, batch_size=None, shuffle=True):
        # one pass over the whole trainset
        # returns loss per shot averaged over whole training set
        total_loss = 0.
        for target in trainset.iter_batch(batch_size, shuffle):
            # define closure func so we can use conjugate gradient or LBFGS
            def closure():
                self.optimizer.zero_grad()
                # create descriptors and results tensors where the model lives
                targ = {key: torch.as_tensor(val,
                                             device=self._device,
                                             dtype=self._dtype
                                             )
                        for key, val in target.items()
                        }
                q_pred = self.nnet(targ[Properties.descriptors])
                targ[Properties.q] = q_pred
                loss = self.loss(targ)
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)
            total_loss += float(loss)
        return total_loss / np.sum(np.sum(trainset.shot_results, axis=-1)
                                   * trainset.weights
                                   )

    def _log_prob(self, descriptors, batch_size):
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                    model=self, descriptors=descriptors,
                                                                   )
        n_split = (descriptors.shape[0] // batch_size) + 1
        predictions = []
        self.nnet.eval()  # put model in evaluation mode
        # no gradient accumulation for predictions!
        with torch.no_grad():
            for descript_part in np.array_split(descriptors, n_split):
                # we do this to create the descriptors array on same
                # devive (GPU/CPU) where the model lives
                descript_part = torch.as_tensor(descript_part, device=self._device,
                                                dtype=self._dtype)
                # move the prediction tensor to cpu (if not there already) than convert to numpy
                pred = self.nnet(descript_part).cpu().numpy()
                predictions.append(pred)
        self.nnet.train()  # make model trainable again
        return np.concatenate(predictions, axis=0)


# the async version is the same, it just uses the async mixin class
class PytorchRCModelAsync(RCModelAsyncMixin, PytorchRCModel):
    pass


class EEScalePytorchRCModelMixin:
    """Expected efficiency scale PytorchRCModel."""
    __doc__ += _train_decision_docs['EEscale']

    def __init__(self, nnet, optimizer, states,
                 ee_params=_train_decision_defaults['EEscale'],
                 descriptor_transform=None, loss=None, cache_file=None,
                 n_out=None):
        super().__init__(
                                nnet=nnet, optimizer=optimizer, states=states,
                                descriptor_transform=descriptor_transform,
                                loss=loss, cache_file=cache_file, n_out=n_out,
                                         )
        # make it possible to pass only the altered values in dictionary
        defaults = copy.deepcopy(_train_decision_defaults['EEscale'])
        defaults.update(ee_params)
        self.ee_params = defaults

    train_decision = _train_decision_funcs['EEscale']


class EEScalePytorchRCModel(EEScalePytorchRCModelMixin, PytorchRCModel):
    pass


class EEScalePytorchRCModelAsync(EEScalePytorchRCModelMixin, PytorchRCModelAsync):
    pass


class EERandPytorchRCModelMixin:
    """Expected efficiency randomized PytorchRCModel."""
    __doc__ += _train_decision_docs['EErand']

    def __init__(self, nnet, optimizer, states,
                 ee_params=_train_decision_defaults['EErand'],
                 descriptor_transform=None, loss=None, cache_file=None,
                 n_out=None):
        super().__init__(
                                nnet=nnet, optimizer=optimizer, states=states,
                                descriptor_transform=descriptor_transform,
                                loss=loss, cache_file=cache_file, n_out=n_out,
                                         )
        # make it possible to pass only the altered values in dictionary
        defaults = copy.deepcopy(_train_decision_defaults['EErand'])
        defaults.update(ee_params)
        self.ee_params = defaults
        self._decisions_since_last_train = 0

    train_decision = _train_decision_funcs['EErand']


class EERandPytorchRCModel(EERandPytorchRCModelMixin, PytorchRCModel):
    pass


class EERandPytorchRCModelAsync(EERandPytorchRCModelMixin, PytorchRCModelAsync):
    pass


# (Bayesian) ensemble RCModel
class EnsemblePytorchRCModel(RCModel):
    """
    Wrapper for an ensemble of N pytorch models.

    Should be trained using a Hamiltonian Monte Carlo optimizer to get draws
    from the posterior distribution and not just the maximum a posteriori estimate.
    We initialize and train every model independently, such that the parameters
    should be decorrelated and stay that way. In fact we are doing N Markov chains
    in NN weight space.

    Training uses a Hamiltonian Monte Carlo algorithm (see e.g. MacKay pp.492).
    TODO: clarify what we actually do when we know it
    Predictions are done by averaging over the ensemble of NNs.
    """

    def __init__(self, nnets, optimizers, states, descriptor_transform=None,
                 loss=None, cache_file=None, n_out=None):
        assert len(nnets) == len(optimizers)  # one optimizer per model!
        # try to get number of outputs, i.e. predicted probabilities
        # we assume that they all have the same number of outputs,
        # otherwise it will/would not work anyway... :)
        if n_out is None:
            try:
                # works if the last layer is linear
                n_out = list(nnets[0].modules())[-1].out_features
            except AttributeError:
                pass
            try:
                # works if the pytorch module has set n_out attribute/property
                # (as e.g. our own pytorch module containers do)
                n_out = nnets[0].n_out
            except AttributeError:
                pass
        super().__init__(states=states,
                         descriptor_transform=descriptor_transform,
                         cache_file=cache_file,
                         n_out=n_out)
        self.nnets = nnets  # list of pytorch.nn.Modules
        # list of pytorch optimizers, one per model
        # any pytorch.optim optimizer, model parameters need to be registered already
        self.optimizers = optimizers
        self.log_train_decision = []
        self.log_train_loss = []
        self._count_train_hook = 0
        self._count_train_epochs = 0
        # needed to create the tensors on the correct device
        self._devices = [next(nnet.parameters()).device for nnet in self.nnets]
        self._dtypes = [next(nnet.parameters()).dtype for nnet in self.nnets]
        self._nnets_same_device = all(self._devices[0] == dev
                                      for dev in self._devices)
        if loss is not None:
            # if custom loss given we take that
            self.loss = loss
        else:
            # otherwise we take the correct one for given n_out
            if self.n_out == 1:
                self.loss = binomial_loss
            else:
                self.loss = multinomial_loss

    # NOTE: implemented by RCModel base
    #@property
    #def n_out(self):
        # FIXME:TODO: only works if the last layer is a linear layer
        # but it can have any activation func, just not an embedding etc
        # FIXME: we assume all nnets have the same number of outputs
        #return list(self.nnets[0].modules())[-1].out_features
        # NOTE also not ideal, this way the pytorch model needs to set self.n_out
    #    return self.nnets[0].n_out

    # NOTE: NEW LOADING-SAVING API
    def object_for_pickle(self, group, overwrite=True, **kwargs):
        """
        Return pickleable object equivalent to self.

        Write everything we can not pickle to the h5py group.

        Parameters:
        -----------
        group - h5py group to write additional data to
        overwrite - bool, wheter to overwrite existing data in h5pygroup
        """
        state = self.__dict__.copy()
        state['nnets_classes'] = [net.__class__ for net in self.nnets]
        state['nnets_call_kwargs'] = [net.call_kwargs for net in self.nnets]
        state['nnets'] = None
        nnets_state = [net.state_dict() for net in self.nnets]
        state['optimizers_classes'] = [o.__class__ for o in self.optimizers]
        state['optimizers'] = None
        optims_state = [o.state_dict() for o in self.optimizers]
        if (not overwrite) and ('nnets' in group):
            # make sure we only overwrite if we want to
            raise RuntimeError("Model already exists but overwrite=False.")
        # save nnets and optimizers to the h5py group
        # we can just require the dsets, if they exist it is ok to overwrite
        nnet_dset = group.require_dataset('nnets', dtype=np.uint8,
                                          maxshape=(None,), shape=(0,),
                                          chunks=True,
                                          )
        optim_dset = group.require_dataset('optims', dtype=np.uint8,
                                           maxshape=(None,), shape=(0,),
                                           chunks=True,
                                           )
        # TODO: unbuffered for now: do we want buffer? which size?
        with BytesStreamtoH5py(nnet_dset) as stream_file:
            torch.save(nnets_state, stream_file)
        with BytesStreamtoH5py(optim_dset) as stream_file:
            torch.save(optims_state, stream_file)
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        # and call supers object_for_pickle in case there is something left
        # in ret_obj.__dict__ that we can not pickle
        return super(EnsemblePytorchRCModel,
                     ret_obj).object_for_pickle(group, overwrite=overwrite, **kwargs)

    def complete_from_h5py_group(self, group, devices=None):
        """
        Restore working state.

        Parameters:
        -----------
        group - h5py group with optional additional data
        devices - None or list of torch devices, if given will overwrite the
                  torch models restore locations, if None will try to restore
                  to a device 'close' to where they were initially saved from
        """
        # instatiate and load the neural networks
        nnets = [nc(**kwargs) for nc, kwargs in zip(self.nnets_classes,
                                                    self.nnets_call_kwargs)]
        del self.nnets_classes
        del self.nnets_call_kwargs
        if devices is None:
            devices = [get_closest_pytorch_device(d) for d in self._devices]
        self._devices = devices
        with H5pytoBytesStream(group['nnets']) as stream_file:
            nnets_state = torch.load(stream_file)
        for nnet, d, s in zip(nnets, devices, nnets_state):
            nnet.load_state_dict(s)
            nnet.to(d)
        self.nnets = nnets
        # now load the optimizers
        optimizers = [clas(net.parameters())
                      for clas, net in zip(self.optimizers_classes, self.nnets)
                      ]
        del self.optimizers_classes
        with H5pytoBytesStream(group['optims']) as stream_file:
            optims_state = torch.load(stream_file)
        for opt, s, d in zip(optimizers, optims_state, self._devices):
            s = optimizer_state_to_device(s, d)
            opt.load_state_dict(s)
        self.optimizers = optimizers
        return super(EnsemblePytorchRCModel, self).complete_from_h5py_group(group)

    @abstractmethod
    def train_decision(self, trainset):
        # this should decide if we train or not
        # return tuple(train, new_lr, epochs)
        # train -> bool
        # new_lr -> float or None; if None: no change
        # epochs -> number of passes over the training set
        raise NotImplementedError

    def train_hook(self, trainset):
        # called by TrainingHook after every TPS MCStep
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

    def log_prob(self, descriptors, use_transform=True, batch_size=None):
        return self._log_prob(descriptors,
                              use_transform=use_transform,
                              batch_size=batch_size,
                              )

    def _log_prob(self, descriptors, use_transform, batch_size):
        p = self(descriptors,
                 use_transform=use_transform,
                 batch_size=batch_size,
                 )
        if p.shape[1] == 1:
            return -np.log(1. / p - 1.)
        return np.log(p)

    # NOTE: prediction happens in here,
    # since we have to do the weighting in probability space
    def __call__(self, descriptors, use_transform=True, batch_size=None,
                 detailed_predictions=False):
        if self.n_out == 1:
            def p_func(q):
                return 1. / (1. + torch.exp(-q))
        else:
            def p_func(q):
                exp_q = torch.exp(q)
                return exp_q / torch.sum(exp_q, dim=1, keepdim=True)

        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)

        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                            model=self, descriptors=descriptors,
                                                                   )

        n_split = (descriptors.shape[0] // batch_size) + 1
        [nnet.eval() for nnet in self.nnets]
        # TODO: do we need no_grad? or is eval and no_grad redundant?
        plists = [[] for _ in self.nnets]
        with torch.no_grad():
            for descript_part in np.array_split(descriptors, n_split):
                if self._nnets_same_device:
                    descript_part = torch.as_tensor(descript_part,
                                                    device=self._devices[0],
                                                    dtype=self._dtypes[0]
                                                    )
                for i, nnet in enumerate(self.nnets):
                    if not self._nnets_same_device:
                        descript_part = torch.as_tensor(descript_part,
                                                        device=self._devices[i],
                                                        dtype=self._dtypes[i]
                                                        )
                    plists[i].append(p_func(nnet(descript_part)).cpu().numpy())
        plist = [np.concatenate(l, axis=0) for l in plists]
        p_mean = sum(plist)
        p_mean /= len(plist)
        [nnet.train() for nnet in self.nnets]
        if detailed_predictions:
            return p_mean, plist
        return p_mean

    def test_loss(self, trainset, batch_size=None):
        # TODO/FIXME: this assumes binomial/multinomial loss!
        # TODO/FIXME: we should rewrite it in terms of self.loss if possible
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                model=self, descriptors=trainset.descriptors,
                                                                   )
        # calculate the test loss for the combined weighted prediction
        # i.e. the loss the model suffers when used as a whole
        # Note that self.__call__() puts the model in evaluation mode
        p = self(trainset.descriptors, use_transform=False, batch_size=batch_size)
        if self.n_out == 1:
            p = p[:, 0]  # make it a 1d array
            # NOTE: the only NaNs we can/should have are generated by multiplying
            # 0 with ln(0), which should be zero anyway
            t1 = trainset.shot_results[:, 0] * np.log(1 - p)
            t2 = trainset.shot_results[:, 1] * np.log(p)
            zeros = np.zeros_like(t1)
            loss = - (np.sum(trainset.weights
                             * (np.where(trainset.shot_results[:, 0] == 0, zeros, t1)
                                + np.where(trainset.shot_results[:, 1] == 0, zeros, t2)
                                )
                             )
                      / np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
                      )
        else:
            log_p = np.log(p)
            zeros = np.zeros_like(log_p[0])
            loss = - (np.sum(trainset.weights
                             * np.sum([np.where(n == 0, zeros, n * lp)
                                       for n, lp in zip(trainset.shot_results, log_p)],
                                      axis=1
                                      )
                             )
                      / np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
                      )
        return loss

    def train_epoch(self, trainset, batch_size=None, shuffle=True):
        # one pass over the whole trainset
        # returns loss per shot averaged over whole training set
        total_loss = np.zeros((len(self.nnets),))
        for i, (nnet, optimizer) in enumerate(zip(self.nnets, self.optimizers)):
            dev = self._devices[i]
            dtype = self._dtypes[i]
            for target in trainset.iter_batch(batch_size, shuffle):
                # define closure func so we can use conjugate gradient or LBFGS
                def closure():
                    optimizer.zero_grad()
                    # create descriptors and results tensors where the model lives
                    targ = {key: torch.as_tensor(val,
                                                 device=dev,
                                                 dtype=dtype
                                                 )
                            for key, val in target.items()
                            }
                    q_pred = nnet(targ[Properties.descriptors])
                    targ[Properties.q] = q_pred
                    loss = self.loss(targ)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                total_loss[i] += float(loss)
        self._count_train_epochs += 1
        Z = np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
        return total_loss / Z

    def set_lr(self, new_lr):
        # TODO: new_lr could be a list of different values if we have more parametersets...
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = new_lr


# TODO: NOTE: Should probably not be used, as changing the learning rate breaks
# detailed balance for HamiltionianMonteCarlo
class EEScaleEnsemblePytorchRCModel(EnsemblePytorchRCModel):
    """Expected efficiency scaling EnsemblePytorchRCModel."""
    __doc__ += _train_decision_docs['EEscale']

    def __init__(self, nnets, optimizers, states,
                 ee_params=_train_decision_defaults['EEscale'],
                 descriptor_transform=None, loss=None, cache_file=None,
                 n_out=None):
        super().__init__(nnets=nnets, optimizers=optimizers, states=states,
                         descriptor_transform=descriptor_transform,
                         loss=loss, cache_file=cache_file, n_out=n_out)
        defaults = copy.deepcopy(_train_decision_defaults['EEscale'])
        defaults.update(ee_params)
        self.ee_params = defaults

    train_decision = _train_decision_funcs['EEscale']


class EERandEnsemblePytorchRCModel(EnsemblePytorchRCModel):
    """Expected efficiency randomized EnsemblePytorchRCModel."""
    __doc__ += _train_decision_docs['EErand']

    def __init__(self, nnets, optimizers, states,
                 ee_params=_train_decision_defaults['EErand'],
                 descriptor_transform=None, loss=None, cache_file=None,
                 n_out=None):
        super().__init__(nnets=nnets, optimizers=optimizers, states=states,
                         descriptor_transform=descriptor_transform,
                         loss=loss, cache_file=cache_file, n_out=n_out)
        # make it possible to pass only the altered values in dictionary
        defaults = copy.deepcopy(_train_decision_defaults['EErand'])
        defaults.update(ee_params)
        self.ee_params = defaults
        self._decisions_since_last_train = 0

    train_decision = _train_decision_funcs['EErand']


# MULTIDOMAIN RCModels
class MultiDomainPytorchRCModel(RCModel):
    """
    Wrapper for multi domain pytorch RCModels.

    Literature: "Towards an AI physicist for unsupervised learning"
                 by Wu + Tegmark (arXiv:1810.10525)
    """
    def __init__(self, pnets, cnet, poptimizer, coptimizer, states,
                 descriptor_transform=None, gamma=-1, loss=None,
                 one_hot_classify=False, cache_file=None, n_out=None):
        # try to get number of outputs, i.e. predicted probabilities
        # again assuming all pnets have the same number of outputs
        if n_out is None:
            try:
                # works if the last layer is linear
                n_out = list(pnets[0].modules())[-1].out_features
            except AttributeError:
                pass
            try:
                # works if the pytorch module has set n_out attribute/property
                # (as e.g. our own pytorch module containers do)
                n_out = pnets[0].n_out
            except AttributeError:
                pass
        super().__init__(states=states,
                         descriptor_transform=descriptor_transform,
                         cache_file=cache_file,
                         n_out=n_out)
        # pnets = list of predicting networks
        # poptimizer = optimizer for prediction networks
        # cnet = classifier deciding which network to take
        # coptimizer optimizer for classification networks
        self.pnets = pnets
        self.cnet = cnet
        # any pytorch.optim optimizer, model parameters need to be registered already
        self.poptimizer = poptimizer
        self.coptimizer = coptimizer
        self.gamma = gamma
        # TODO: make this a property such that we can not (re-) set it during object livetime?
        # TODO: would this prevent users from doing stupid stuff? or does it not matter anyways because
        # TODO: we also change the classification during trainign anyways?
        # TODO: I think it does not matter...(?)
        self.one_hot_classify = one_hot_classify
        self.log_train_decision = []
        self.log_train_loss = []
        self.log_ctrain_decision = []
        self.log_ctrain_loss = []
        self._count_train_hook = 0
        # needed to create the tensors on the correct device
        self._pdevices = [next(pnet.parameters()).device
                          for pnet in self.pnets]
        self._pdtypes = [next(pnet.parameters()).dtype
                         for pnet in self.pnets]
        # we assume same dtype too, if all are on same device
        self._pnets_same_device = all(self._pdevices[0] == dev
                                      for dev in self._pdevices)
        # _device and _dtype are for cnet
        self._cdevice = next(self.cnet.parameters()).device
        self._cdtype = next(self.cnet.parameters()).dtype
        if loss is not None:
            # if custom loss given we take that
            # TODO: do we need to check if output is vectorial or will it fail anyway if it is not?
            self.loss = loss
        else:
            # otherwise we take the correct one for given n_out
            if self.n_out == 1:
                self.loss = binomial_loss_vect
            else:
                self.loss = multinomial_loss_vect

    # NOTE: implemented by base RCModel
    #@property
    #def n_out(self):
        # FIXME:TODO: only works if the last layer is linear!
        # all networks have the same number of out features
        #return list(self.pnets[0].modules())[-1].out_features
        # NOTE also not ideal, this way the pytorch model needs to set self.n_out
    #    return self.pnets[0].n_out

    # NOTE: NEW LOADING-SAVING API
    def object_for_pickle(self, group, overwrite=True, **kwargs):
        """
        Return pickleable object equivalent to self.

        Write everything we can not pickle to the h5py group.

        Parameters:
        -----------
        group - h5py group to write additional data to
        overwrite - bool, wheter to overwrite existing data in h5pygroup
        """
        state = self.__dict__.copy()
        state['pnets_class'] = [pn.__class__ for pn in self.pnets]
        state['pnets_call_kwargs'] = [pn.call_kwargs for pn in self.pnets]
        pnets_state = [pn.state_dict() for pn in self.pnets]
        state['pnets'] = None
        state['cnet_class'] = self.cnet.__class__
        state['cnet_call_kwargs'] = self.cnet.call_kwargs
        cnet_state = self.cnet.state_dict()
        state['cnet'] = None
        # now the optimizers
        state['poptimizer_class'] = self.poptimizer.__class__
        poptimizer_state = self.poptimizer.state_dict()
        state['poptimizer'] = None
        state['coptimizer_class'] = self.coptimizer.__class__
        coptimizer_state = self.coptimizer.state_dict()
        state['coptimizer'] = None
        if (not overwrite) and ('pnets' in group):
            raise RuntimeError("Model already exists but overwrite=False.")
        pnet_dset = group.require_dataset('pnets', dtype=np.uint8,
                                          maxshape=(None,), shape=(0,),
                                          chunks=True,
                                          )
        poptim_dset = group.require_dataset('poptim', dtype=np.uint8,
                                            maxshape=(None,), shape=(0,),
                                            chunks=True,
                                            )
        cnet_dset = group.require_dataset('cnet', dtype=np.uint8,
                                          maxshape=(None,), shape=(0,),
                                          chunks=True,
                                          )
        coptim_dset = group.require_dataset('coptim', dtype=np.uint8,
                                            maxshape=(None,), shape=(0,),
                                            chunks=True,
                                            )
        # TODO: unbuffered for now: do we want buffer? which size?
        with BytesStreamtoH5py(pnet_dset) as stream_file:
            torch.save(pnets_state, stream_file)
        with BytesStreamtoH5py(poptim_dset) as stream_file:
            torch.save(poptimizer_state, stream_file)
        with BytesStreamtoH5py(cnet_dset) as stream_file:
            torch.save(cnet_state, stream_file)
        with BytesStreamtoH5py(coptim_dset) as stream_file:
            torch.save(coptimizer_state, stream_file)

        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        # and call supers object_for_pickle in case there is something left
        # in ret_obj.__dict__ that we can not pickle
        return super(MultiDomainPytorchRCModel,
                     ret_obj).object_for_pickle(group, overwrite=overwrite, **kwargs)

    def complete_from_h5py_group(self, group, pdevices=None, cdevice=None):
        """
        Restore working state.

        Parameters:
        -----------
        group - h5py group with optional additional data
        devices - None or list of torch devices, if given will overwrite the
                  torch models restore locations, if None will try to restore
                  to a device 'close' to where they were initially saved from
        """
        # instatiate and load the neural networks
        pnets = [nc(**kwargs) for nc, kwargs in zip(self.pnets_class,
                                                    self.pnets_call_kwargs)]
        del self.pnets_class
        del self.pnets_call_kwargs
        if pdevices is None:
            pdevices = [get_closest_pytorch_device(d) for d in self._pdevices]
        self._pdevices = pdevices
        with H5pytoBytesStream(group['pnets']) as stream_file:
            pnets_state = torch.load(stream_file)
        for net, d, s in zip(pnets, pdevices, pnets_state):
            net.load_state_dict(s)
            net.to(d)
        self.pnets = pnets
        # now cnet
        cnet = self.cnet_class(**self.cnet_call_kwargs)
        del self.cnet_class
        del self.cnet_call_kwargs
        if cdevice is None:
            cdevice = get_closest_pytorch_device(self._cdevice)
        self._cdevice = cdevice
        with H5pytoBytesStream(group['cnet']) as stream_file:
            cnet_state = torch.load(stream_file, map_location=cdevice)
        cnet.load_state_dict(cnet_state)
        cnet.to(cdevice)  # should be a no-op?!!
        self.cnet = cnet
        # now load the optimizers
        poptimizer = self.poptimizer_class([{'params': pnet.parameters()}
                                            for pnet in self.pnets]
                                           )
        with H5pytoBytesStream(group['poptim']) as stream_file:
            poptim_state = torch.load(stream_file, map_location=pdevices[0])
        poptimizer.load_state_dict(poptim_state)
        self.poptimizer = poptimizer
        coptimizer = self.coptimizer_class(self.cnet.parameters())
        with H5pytoBytesStream(group['coptim']) as stream_file:
            coptim_state = torch.load(stream_file, map_location=cdevice)
        coptimizer.load_state_dict(coptim_state)
        self.coptimizer = coptimizer
        del self.poptimizer_class
        del self.coptimizer_class
        return super(MultiDomainPytorchRCModel, self).complete_from_h5py_group(group)

    @abstractmethod
    def train_decision(self, trainset):
        # this should decide if we train or not
        # TODO: possibly return/set the learning rate?!
        # return tuple(train, new_lr, epochs)
        # train -> bool
        # new_lr -> float or None; if None: no change
        # epochs -> number of passes over the training set
        pass

    @abstractmethod
    def train_decision_classifier(self, trainset, cnet_target):
        # decide if we train the classifier
        # should return (train, lr, epochs) as train_decision()
        pass

    def train_hook(self, trainset):
        # TODO/FIXME: this expects the train decission to not return
        #             batch_size, i.e. it only works with the
        #             hardcoded decissions below and not with the
        #             general external functions
        # TODO: different train decisions for different prediction nets?
        # committor prediction nets
        self._count_train_hook += 1
        train, new_lr, epochs = self.train_decision(trainset)
        self.log_train_decision.append([train, new_lr, epochs])
        if new_lr is not None:
            logger.info('Setting learning rate to {:.3e}'.format(new_lr))
            self.set_lr_popt(new_lr)
        if train:
            logger.info('Training for {:d} epochs'.format(epochs))
            self.log_train_loss.append([self.train_epoch_pnets(trainset)
                                        for _ in range(epochs)])

        # classifier
        cnet_target = self.create_cnet_targets(trainset)
        train_c, new_lr_c, epochs_c = self.train_decision_classifier(trainset, cnet_target)
        self.log_ctrain_decision.append([train_c, new_lr_c, epochs_c])
        if new_lr_c is not None:
            logger.info('Setting classifier learning rate to {:.3e}'.format(new_lr_c))
            self.set_lr_copt(new_lr_c)
        if train_c:
            logger.info('Training classifier for {:d} epochs'.format(epochs_c))
            self.log_ctrain_loss.append([self.train_epoch_cnet(trainset, cnet_target)
                                         for _ in range(epochs_c)])

    def test_loss(self, trainset, loss='L_pred', batch_size=None):
        """
        Calculate the test loss over given TrainSet.

        Parameters:
        -----------
        trainset - `:class:aimmd.TrainSet` for which to calculate the loss
        loss - str, one of:
               'L_pred' - calculates the loss suffered for weighted prediction
               'L_mod{:d}' - calculates suffered by a specific pnet,
                             where {:d} is an int index to a pnet
               'L_gamma' - calculates the generalized mean loss over all models
               'L_class' - calculates the loss suffered by classifier
        batch_size - int, number of training points in a single batch

        Note that batch_size is ignored for loss='L_pred'.

        """
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                model=self, descriptors=trainset.descriptors,
                                                                   )

        if loss == 'L_pred':
            return self._test_loss_pred(trainset, batch_size)
        elif 'L_mod' in loss:
            mod_num = int(loss.lstrip('L_mod'))
            if not (0 <= mod_num < len(self.pnets)):
                raise ValueError('Can only calculate "L_mod" for a model index'
                                 + ' that is smaller than len(self.pnets).')
            return self._test_loss_pnets(trainset, batch_size)[mod_num]
        elif loss == 'L_gamma':
            return self._test_loss_pnets(trainset, batch_size)[-1]
        elif loss == 'L_class':
            return self._test_loss_cnet(trainset, batch_size)
        else:
            raise ValueError("'loss' must be one of 'L_pred', 'L_mod{:d}', "
                             + "'L_gamma' or 'L_class'")

    def _test_loss_pred(self, trainset, batch_size):
        # TODO/FIXME: assumes binomial/multinomial loss!
        # calculate the test loss for the combined weighted prediction
        # p_i = \sum_m p_c(m) * p_i(m)
        # i.e. the loss the model would suffer when used as a whole
        # Note that self.__call__() puts the model in evaluation mode
        p = self(trainset.descriptors, use_transform=False, batch_size=batch_size)
        if self.n_out == 1:
            p = p[:, 0]  # make it a 1d array
            zeros = np.zeros_like(p)
            t1 = trainset.shot_results[:, 0] * np.log(1 - p)
            t2 = trainset.shot_results[:, 1] * np.log(p)
            loss = - (np.sum(trainset.weights
                             * (np.where(trainset.shot_results[:, 0] == 0, zeros, t1)
                                + np.where(trainset.shot_results[:, 1] == 0, zeros, t2)
                                )
                             )
                      / np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
                      )
        else:
            log_p = np.log(p)
            zeros = np.zeros_like(log_p[0])
            loss = - (np.sum(trainset.weights
                             * np.sum([np.where(n == 0, zeros, n * lp)
                                       for n, lp in zip(trainset.shot_results, log_p)],
                                      axis=1
                                      )
                             )
                      / np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
                      )
        return loss

    def _test_loss_pnets(self, trainset, batch_size):
        # returns the test losses for all pnets and the L_gamma combined loss value
        # as np.array([L(mod_0), L(mod_1), ..., L(mod_m), L_gamma])
        # evaluation mode for all prediction networks
        self.pnets = [pn.eval() for pn in self.pnets]
        loss_by_model = [0 for _ in self.pnets]
        total_loss = 0
        # very similiar to _train_epoch_pnets but without gradient collection
        with torch.no_grad():
            for target in trainset.iter_batch(batch_size, False):
                if self._pnets_same_device:
                    # create descriptors and results tensors where the models live
                    target = {key: torch.as_tensor(val,
                                                   device=self._pdevices[0],
                                                   dtype=self._pdtypes[0]
                                                   )
                              for key, val in target.items()
                              }
                # we collect the results on the device of the first pnet
                l_m_sum = torch.zeros((target[Properties.descriptors].shape[0],),
                                      device=self._pdevices[0],
                                      dtype=self._pdtypes[0]
                                      )
                for i, pnet in enumerate(self.pnets):
                    if not self._pnets_same_device:
                        # create descriptors and results tensors where the models live
                        target = {key: torch.as_tensor(val,
                                                       device=self._pdevices[i],
                                                       dtype=self._pdtypes[i]
                                                       )
                                  for key, val in target.items()
                                  }
                    q_pred = pnet(target[Properties.descriptors])
                    target[Properties.q] = q_pred
                    l_m = self.loss(target)
                    loss_by_model[i] += float(torch.sum(l_m))
                    l_m_sum += torch.pow(l_m, self.gamma).to(l_m_sum.device)
                # end models loop
                L_gamma = torch.sum(torch.pow(l_m_sum / len(self.pnets), 1/self.gamma))
                total_loss += float(L_gamma)
            # end trainset loop
        # end torch.no_grad()
        # back to training mode for all pnets
        self.pnets = [pn.train() for pn in self.pnets]
        return (np.concatenate((loss_by_model, [total_loss]))
                / np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
                )

    def _test_loss_cnet(self, trainset, batch_size):
        # TODO/FIXME: the batchsize here is quite useless, we load everything into memory anyways!
        cnet_targets = self.create_cnet_targets(trainset, batch_size)
        self.cnet.eval()  # evaluation mode
        with torch.no_grad():
            total_loss = 0
            descriptors = torch.as_tensor(trainset.descriptors,
                                          device=self._cdevice,
                                          dtype=self._cdtype)
            weights = torch.as_tensor(trainset.weights,
                                      device=self._cdevice,
                                      dtype=self._cdtype)
            if batch_size is None:
                batch_size = len(trainset)
            n_batch = int(len(trainset) / batch_size)
            rest = len(trainset) % batch_size
            for b in range(n_batch):
                des = descriptors[b*batch_size:(b+1)*batch_size]
                tar = cnet_targets[b*batch_size:(b+1)*batch_size]
                ws = weights[b*batch_size:(b+1)*batch_size]
                log_probs = self.cnet(des)
                # pack stuff into our dictionary so we can use the same multinomial loss
                target = {Properties.q: log_probs,
                          Properties.descriptors: des,
                          Properties.shot_results: tar,
                          Properties.weights: ws
                          }
                loss = multinomial_loss(target)
                total_loss += float(loss)
            if rest > 0:
                # the rest
                des = descriptors[n_batch*batch_size:n_batch*batch_size + rest]
                tar = cnet_targets[n_batch*batch_size:n_batch*batch_size + rest]
                ws = weights[n_batch*batch_size:n_batch*batch_size + rest]
                log_probs = self.cnet(des)
                target = {Properties.q: log_probs,
                          Properties.descriptors: des,
                          Properties.shot_results: tar,
                          Properties.weights: ws
                          }
                loss = multinomial_loss(target)
                total_loss += float(loss)
        # end torch.no_grad()
        self.cnet.train()  # back to train mode
        # normalize classifier loss per point in trainset
        # this is the same as the per shot normalization,
        # because we only have one event (one correct model) per point
        return total_loss / np.sum(trainset.weights)

    def train_epoch_pnets(self, trainset, batch_size=None, shuffle=True):
        # one pass over the whole trainset
        # returns loss per shot averaged over whole training set as list,
        # one fore each model by idx and last entry is the combined multidomain loss
        total_loss = 0.
        loss_by_model = np.array([0. for _ in self.pnets])
        for target in trainset.iter_batch(batch_size, shuffle):
            def closure():
                self.poptimizer.zero_grad()
                l_by_mod = np.zeros_like(loss_by_model)
                if self._pnets_same_device:
                    # create descriptors and results tensors where the models live
                    # create descriptors and results tensors where the models live
                    targ = {key: torch.as_tensor(val,
                                                 device=self._pdevices[0],
                                                 dtype=self._pdtypes[0]
                                                 )
                            for key, val in target.items()
                            }
                # we collect the results on the device of the first pnet
                l_m_sum = torch.zeros((targ[Properties.descriptors].shape[0],),
                                      device=self._pdevices[0],
                                      dtype=self._pdtypes[0])
                for i, pnet in enumerate(self.pnets):
                    if not self._pnets_same_device:
                        # create descriptors and results tensors where the models live
                        # create descriptors and results tensors where the models live
                        targ = {key: torch.as_tensor(val,
                                                     device=self._pdevices[i],
                                                     dtype=self._pdtypes[i]
                                                     )
                                for key, val in target.items()
                                }
                    q_pred = pnet(targ[Properties.descriptors])
                    targ[Properties.q] = q_pred
                    l_m = self.loss(targ)
                    l_by_mod[i] = float(torch.sum(l_m))
                    l_m_sum += torch.pow(l_m, self.gamma).to(l_m_sum.device)
                # end models loop
                L_gamma = torch.sum(torch.pow(l_m_sum / len(self.pnets), 1/self.gamma))
                L_gamma.backward()
                return L_gamma, l_by_mod
            L_gamma, l_by_mod = self.poptimizer.step(closure)
            total_loss += float(L_gamma)
            loss_by_model += l_by_mod
        # end trainset loop
        return (np.concatenate((loss_by_model, [total_loss]))
                / np.sum(trainset.weights * np.sum(trainset.shot_results, axis=1))
                )

    def create_cnet_targets(self, trainset, batch_size=128):
        # build the trainset for classifier,
        # i.e. which model has the lowest loss for each point in trainset
        targets_out = torch.zeros((len(trainset), len(self.pnets)),
                                  device=self._cdevice,
                                  dtype=self._cdtype)
        fill = 0
        # put prediction nets in evaluation mode
        self.pnets = [pn.eval() for pn in self.pnets]
        with torch.no_grad():
            for target in trainset.iter_batch(batch_size, shuffle=False):
                if self._pnets_same_device:
                    # create descriptors and results tensors where the models live
                    # create descriptors and results tensors where the models live
                    target = {key: torch.as_tensor(val,
                                                   device=self._pdevices[0],
                                                   dtype=self._pdtypes[0]
                                                   )
                              for key, val in target.items()
                              }
                # we collect the results on the device of the first pnet
                l_m_arr = torch.zeros((target[Properties.descriptors].shape[0], len(self.pnets)),
                                      device=self._pdevices[0],
                                      dtype=self._pdtypes[0])
                for i, pnet in enumerate(self.pnets):
                    if not self._pnets_same_device:
                        # create descriptors and results tensors where the models live
                        # create descriptors and results tensors where the models live
                        target = {key: torch.as_tensor(val,
                                                       device=self._pdevices[i],
                                                       dtype=self._pdtypes[i]
                                                       )
                                  for key, val in target.items()
                                  }
                    q_pred = pnet(target[Properties.descriptors])
                    target[Properties.q] = q_pred
                    l_m = self.loss(target)
                    l_m_arr[:, i] = l_m.to(l_m_arr.device)
                # end models loop
                # find minimum loss value model indexes for each point
                # and fill ones into targets_out at that index
                min_idxs = l_m_arr.argmin(dim=1)
                bs = min_idxs.shape[0]  # not every batch is created equal, i.e. different lengths
                targets_out[fill + torch.arange(bs), min_idxs] = 1
                fill += bs
            # end batch over trainset loop
        # end torch nograd
        # put pnets back to train mode
        self.pnets = [pn.train() for pn in self.pnets]
        return targets_out

    def train_epoch_cnet(self, trainset, cnet_targets, batch_size=128, shuffle=True):
        # TODO/FIXME: batchsize is useless here, we load everything into MEM anyways!
        total_loss = 0
        descriptors = torch.as_tensor(trainset.descriptors,
                                      device=self._cdevice,
                                      dtype=self._cdtype)
        weights = torch.as_tensor(trainset.weights,
                                  device=self._cdevice,
                                  dtype=self._cdtype)
        if shuffle:
            shuffle_idxs = torch.randperm(len(trainset))
            descriptors = descriptors[shuffle_idxs]
            cnet_targets = cnet_targets[shuffle_idxs]
            weights = weights[shuffle_idxs]

        if batch_size is None:
            batch_size = len(trainset)
        n_batch = int(len(trainset) / batch_size)
        rest = len(trainset) % batch_size
        for b in range(n_batch):
            des = descriptors[b*batch_size:(b+1)*batch_size]
            counts = cnet_targets[b*batch_size:(b+1)*batch_size]
            ws = weights[b*batch_size:(b+1)*batch_size]
            def closure():
                self.coptimizer.zero_grad()
                log_probs = self.cnet(des)
                tar = {Properties.descriptors: des,
                       Properties.shot_results: counts,
                       Properties.weights: ws,
                       Properties.q: log_probs
                       }
                loss = multinomial_loss(tar)
                loss.backward()
                return loss
            loss = self.coptimizer.step(closure)
            total_loss += float(loss)

        # the rest
        des = descriptors[n_batch*batch_size:n_batch*batch_size + rest]
        counts = cnet_targets[n_batch*batch_size:n_batch*batch_size + rest]
        ws = weights[n_batch*batch_size:n_batch*batch_size + rest]
        def closure():
            self.coptimizer.zero_grad()
            log_probs = self.cnet(des)
            tar = {Properties.descriptors: des,
                   Properties.shot_results: counts,
                   Properties.weights: ws,
                   Properties.q: log_probs
                   }
            loss = multinomial_loss(tar)
            loss.backward()
            return loss
        loss = self.coptimizer.step(closure)
        total_loss += float(loss)

        # normalize classifier loss per point in trainset
        return total_loss / np.sum(trainset.weights)

    def set_lr_popt(self, new_lr):
        # TODO: new_lr could be a list of different values if we have more parametersets...
        # especially here where we train different pnets with the same optimizer!
        for i, param_group in enumerate(self.poptimizer.param_groups):
            param_group['lr'] = new_lr

    def set_lr_copt(self, new_lr):
        # TODO: new_lr could be a list of different values if we have more parametersets...
        for i, param_group in enumerate(self.coptimizer.param_groups):
            param_group['lr'] = new_lr

    # NOTE ON PREDICTIONS:
    # we have to predict probabilities, weight them with classifier and then
    # go back to q space, therefore here the real prediction is in __call__(),
    # i.e. the function that returns committment probabilities
    # _log_prob() and log_prob() then recalculate log_probs from the commitment probs
    # for binom q = ln(1/p_B - 1)
    # for multinom q_i = ln(p_i) + ln(Z),
    # where we can choose Z freely and set it to 1, such that ln(z) = 0
    # using self.q() will then fix Z such that q 'feels' like an RC
    def _log_prob(self, descriptors, batch_size):
        # TODO/FIXME: this is never called...?
        return self.q(descriptors, use_transform=False, batch_size=batch_size)

    def log_prob(self, descriptors, use_transform=True, batch_size=None):
        p = self(descriptors, use_transform=use_transform, batch_size=batch_size)
        if p.shape[1] == 1:
            return -np.log(1. / p - 1.)
        return np.log(p)

    def __call__(self, descriptors, use_transform=True,
                 batch_size=None, domain_predictions=False):
        # returns the probabilities,
        # we decide here if we transform, as this is our initial step even if we back-calculate q
        # if wanted and self.descriptor_transform is defined we use it before prediction
        # if domain_predictions=True we will return a tuple (p_weighted, [p_m for m in self.pnets])
        if self.n_out == 1:
            def p_func(q):
                return 1. / (1. + torch.exp(-q))
        else:
            def p_func(q):
                exp_q = torch.exp(q)
                return exp_q / torch.sum(exp_q, dim=1, keepdim=True)

        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)

        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                    model=self, descriptors=descriptors,
                                                                   )
        # get the probabilities for each model from classifier
        p_c = self._classify(descriptors=descriptors,
                             batch_size=batch_size)  # returns p_c on self._cdevice
        if self.one_hot_classify:
            # TODO: can we do this smarter?
            max_idxs = p_c.argmax(dim=1)
            p_c = torch.zeros_like(p_c)
            p_c[torch.arange(max_idxs.shape[0]), max_idxs] = 1
        n_split = (descriptors.shape[0] // batch_size) + 1
        p_c = p_c.cpu().numpy()
        self.pnets = [pn.eval() for pn in self.pnets]  # pnets to evaluation
        pred = np.zeros((p_c.shape[0], self.n_out))
        idx_start = 0
        # now committement probabilities
        with torch.no_grad():
            for descript_part in np.array_split(descriptors, n_split):
                idx_end = idx_start + descript_part.shape[0]
                descript_part = torch.as_tensor(descript_part,
                                                device=self._pdevices[0],
                                                dtype=self._pdtypes[0])
                if domain_predictions:
                    p_m_lists = [[] for _ in self.pnets]
                for i, pnet in enumerate(self.pnets):
                    # .to() should be a no-op if they are all on the same device (?)
                    descript_part = descript_part.to(self._pdevices[i])
                    q = pnet(descript_part)
                    p = p_func(q).cpu().numpy()
                    if domain_predictions:
                      p_m_lists[i].append(p)
                    pred[idx_start:idx_end] += p_c[idx_start:idx_end, i:i+1] * p
                # end pnets loop
                idx_start = idx_end
            # end batch/descriptor parts loop
        # end torch.no_grad()
        self.pnets = [pn.train() for pn in self.pnets]  # back to train
        if domain_predictions:
            p_m_list = [np.concatenate(l, axis=0) for l in p_m_lists]
            return (pred, p_m_list)
        return pred

    def classify(self, descriptors, use_transform=True, batch_size=None):
        """
        Returns the probabilities the classifier assigns to each model
        for given descriptors.
        """
        # this is just a wrapper around _classify to convert to numpy
        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)
        if batch_size is None:
            batch_size = get_batch_size_from_model_and_descriptors(
                                    model=self, descriptors=descriptors,
                                                                   )
        # get the probabilities for each model from classifier
        # and move them to cpu to numpy before return
        return self._classify(descriptors=descriptors,
                              batch_size=batch_size).cpu().numpy()

    def _classify(self, descriptors, batch_size):
        # return classifier model probabilities for descriptors
        # descriptors is expected to be numpy or torch tensor
        # returns a torch.tensor on the same device the classifier lives on
        n_split = (descriptors.shape[0] // batch_size) + 1
        self.cnet.eval()  # classify in evaluation mode
        p_cs = []
        with torch.no_grad():
            for descript_part in np.array_split(descriptors, n_split):
                descript_part = torch.as_tensor(descript_part, device=self._cdevice,
                                                dtype=self._cdtype)
                q_c = self.cnet(descript_part)
                exp_q_c = torch.exp(q_c)
                # p_c is the probability the classifier assigns the point to be in models class
                p_cs += [exp_q_c / torch.sum(exp_q_c, dim=1, keepdim=True)]

        self.cnet.train()  # and make the cnet trainable again
        return torch.cat(p_cs)


# TODO: Use the decision function and documentation we already have and use for the other models!
class EEMDPytorchRCModel(MultiDomainPytorchRCModel):
    """
    Expected efficiency MultiDomainPytorchRCModel.
    Controls training by multiplying lr with expected efficiency factor

    ee_params - dict, 'expected efficiency parameters'
        lr_0 - float, base learning rate
        lr_min - float, minimal learning rate we still train with
        epochs_per_train - int, if we train we train for this many epochs
        interval - int, we attempt to train every interval MCStep,
                   measured by self.train_hook() calls
        window - int, size of the smoothing window used for expected efficiency
    """
    def __init__(self, pnets, cnet, poptimizer, coptimizer, states,
                 gamma=-1, ee_params={'lr_0': 1e-3,
                                      'lr_min': 1e-4,
                                      'epochs_per_train': 5,
                                      'interval': 3,
                                      'window': 100},
                 ctrain_params={'lr_0': 1e-3,
                                'lr_min': 1e-4,
                                'epochs_per_train': 5,
                                'interval': 6,
                                'window': 100},
                 #ctrain_params={'rel_tol': 0.01,
                 #               'epochs_per_train': 5,
                 #               'interval': 3,
                 #               'max_interval': 20},
                 descriptor_transform=None, loss=None,
                 one_hot_classify=False, cache_file=None, n_out=None):
        super().__init__(pnets, cnet, poptimizer, coptimizer, states,
                         descriptor_transform, gamma, loss,
                         one_hot_classify, cache_file, n_out)
        # make it possible to pass only the altered values in dictionary
        ee_params_defaults = {'lr_0': 1e-3,
                              'lr_min': 1e-4,
                              'epochs_per_train': 5,
                              'interval': 3,
                              'window': 100}
        ee_params_defaults.update(ee_params)
        self.ee_params = ee_params_defaults
        ctrain_params_defaults = {'lr_0': 1e-3,
                                  'lr_min': 1e-4,
                                  'epochs_per_train': 5,
                                  'interval': 6,
                                  'window': 100}
        #ctrain_params_defaults = {'rel_tol': 0.01,
        #                          'epochs_per_train': 5,
        #                          'interval': 3,
        #                          'max_interval': 20}
        ctrain_params_defaults.update(ctrain_params)
        self.ctrain_params = ctrain_params_defaults

    def train_decision(self, trainset):
        # TODO: atm this is the same as for EERCModel
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

    def train_decision_classifier(self, trainset, cnet_targets):
        # use the same expected efficiency factor as for prediction networks
        # but possibly with different lr_0, lr_min, window and friends
        train = False
        lr = self.ctrain_params['lr_0']
        lr *= self.train_expected_efficiency_factor(trainset,
                                                    self.ctrain_params['window'])
        if self._count_train_hook % self.ctrain_params['interval'] == 0:
            if lr >= self.ctrain_params['lr_min']:
                train = True
        epochs = self.ctrain_params['epochs_per_train']
        logger.info('Decided train={:d}, lr={:.3e}, epochs={:d}'.format(train,
                                                                        lr,
                                                                        epochs)
                    )
        return train, lr, epochs

    # TODO: not used atm
    # TODO: do we even need this?, test if it is usefull compared to expected efficiency
    def train_decision_classifier_const_loss(self, trainset, cnet_targets):
        # if loss increased less than rel_tol since last train we do not train
        # otherwise train with predefined lr of coptimizer for epochs_per_train epochs
        train = False
        if self._count_train_hook % self.ctrain_params['interval'] != 0:
            # not a step @ which we check if we train
            pass
        elif len(self.log_ctrain_loss) <= 0:
            # we never trained yet
            train = True
        else:
            last_loss = self.log_ctrain_loss[-1][-1]
            train_hist = np.asarray(self.log_ctrain_decision[-self.ctrain_params['max_interval']:])[:, 0]
            # get current loss
            with torch.no_grad():
                descriptors = torch.as_tensor(trainset.descriptors,
                                              device=self._cdevice,
                                              dtype=self._cdtype)
                log_probs = self.cnet(descriptors)
                loss = float(multinomial_loss(log_probs, cnet_targets))
            # and decide if we train
            if (loss - last_loss) / last_loss >= self.ctrain_params['rel_tol']:
                train = True
            elif np.sum(train_hist) == 0:
                # we did not train for the last max_interval steps
                train = True

        epochs = self.ctrain_params['epochs_per_train']
        logger.info('Decided for classifier train={:d}, epochs={:d}'.format(train,
                                                                            epochs)
                    )

        return train, None, epochs
