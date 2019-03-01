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
import torch
import numpy as np
from abc import abstractmethod
from ..base.rcmodel import RCModel


logger = logging.getLogger(__name__)


# LOSS FUNCTIONS
def binomial_loss(input, target):
    """
    Loss for a binomial process.
    input is the predicted log likelihood,
    target are the true event counts, i.e. states reached for TPS

    NOTE: This is NOT normalized in any way.
    """
    return torch.sum(target[:, 0] * torch.log(1. + torch.exp(input[:, 0]))
                     + target[:, 1] * torch.log(1. + torch.exp(-input[:, 0]))
                     )


def binomial_loss_vect(input, target):
    """
    Loss for a binomial process.
    input is the predicted log likelihood,
    target are the true event counts, i.e. states reached for TPS

    Same as binomial_loss, but returns a vector loss values per point.
    Needed for multidomain RCModels to train the classifier.
    """
    return (target[:, 0] * torch.log(1. + torch.exp(input[:, 0]))
            + target[:, 1] * torch.log(1. + torch.exp(-input[:, 0]))
            )


def multinomial_loss(input, target):
    """
    Loss for multinomial process.
    input are the predicted unnormalized loglikeliehoods,
    target the corresponding true event counts

    NOTE: This is NOT normalized in any way.
    """
    ln_Z = torch.log(torch.sum(torch.exp(input), dim=1, keepdim=True))
    return torch.sum((ln_Z - input) * target)


def multinomial_loss_vect(input, target):
    """
    Loss for multinomial process.
    input are the predicted unnormalized loglikeliehoods,
    target the corresponding true event counts

    Same as multinomial_loss, but returns a vector of loss values per point.
    Needed for multidomain RCModels to train the classifier.
    """
    ln_Z = torch.log(torch.sum(torch.exp(input), dim=1, keepdim=True))
    return torch.sum((ln_Z - input) * target, dim=1)


def _fix_pytorch_device(location):
    """
    Checks if location is an available pytorch.device, else
    returns the pytorch device that is `closest` to location.
    Adopted from pytorch.serialization.validate_cuda_device
    """
    if isinstance(location, torch.device):
        location = str(location)
    if not isinstance(location, str):
        raise ValueError("location should be a string or torch.device")
    if location[5:] == '':
        device = 0
    else:
        device = max(int(location[5:]), 0)
    if 'cuda' in location:
        if not torch.cuda.is_available():
            # no cuda, go to CPU
            logger.info('Restoring on CPU, since CUDA is not available.')
            return torch.device('cpu')
        if device >= torch.cuda.device_count():
            # other cuda device ID
            logger.info('Restoring on a different CUDA device.')
            return torch.device('cuda')
    return device


# RCModels using one ANN
class PytorchRCModel(RCModel):
    """
    Wraps pytorch neural networks for use with arcd
    """
    def __init__(self, nnet, optimizer, descriptor_transform=None, loss=None):
        super().__init__(descriptor_transform)
        self.nnet = nnet  # a pytorch.nn.Module
        # any pytorch.optim optimizer, model parameters need to be registered already
        self.optimizer = optimizer
        self.log_train_decision = []
        self.log_train_loss = []
        self._count_train_hook = 0
        # needed to create the tensors on the correct device
        self._device = next(self.nnet.parameters()).device
        self._dtype = next(self.nnet.parameters()).dtype
        if loss:
            # if custom loss given we take that
            self.loss = loss
        else:
            # otherwise we take the correct one for given n_out
            if self.n_out == 1:
                self.loss = binomial_loss
            else:
                self.loss = multinomial_loss

    @property
    def n_out(self):
        # FIXME:TODO: only works if the last layer is linear!
        return list(self.nnet.modules())[-1].out_features

    @classmethod
    def set_state(cls, state):
        obj = cls(nnet=state['nnet'], optimizer=state['optimizer'])
        obj.__dict__.update(state)
        return obj

    @classmethod
    def fix_state(cls, state):
        # restore the nnet
        nnet = state['nnet_class'](**state['nnet_call_kwargs'])
        del state['nnet_class']
        del state['nnet_call_kwargs']
        dev = _fix_pytorch_device(state['_device'])
        nnet.to(dev)
        nnet.load_state_dict(state['nnet'])
        state['nnet'] = nnet
        # and the optimizer
        # TODO: the least we can do is write TESTS!
        # TODO: we assume that there is only one param group
        optimizer = state['optimizer_class'](nnet.parameters())
        del state['optimizer_class']
        optimizer.load_state_dict(state['optimizer'])
        state['optimizer'] = optimizer
        return state

    def save(self, fname, overwrite=False):
        # keep a ref to the network
        nnet = self.nnet
        # but replace with state_dict in self.__dict__
        self.nnet_class = nnet.__class__
        self.nnet_call_kwargs = nnet.call_kwargs
        self.nnet = nnet.state_dict()
        # same for optimizer
        optimizer = self.optimizer
        self.optimizer_class = optimizer.__class__
        self.optimizer = optimizer.state_dict()
        # now let super save the state dict
        super().save(fname, overwrite)
        # and restore nnet and optimizer such that self stays functional
        self.nnet = nnet
        self.optimizer = optimizer
        # and remove uneccessary keys to self.__dict__
        del self.nnet_class
        del self.nnet_call_kwargs
        del self.optimizer_class

    @abstractmethod
    def train_decision(self, trainset):
        # this should decide if we train or not
        # TODO: possibly return/set the learning rate?!
        # return tuple(train, new_lr, epochs)
        # train -> bool
        # new_lr -> float or None; if None: no change
        # epochs -> number of passes over the training set
        pass

    def train_hook(self, trainset):
        self._count_train_hook += 1
        train, new_lr, epochs = self.train_decision(trainset)
        self.log_train_decision.append([train, new_lr, epochs])
        if new_lr:
            logger.info('Setting learning rate to {:.3e}'.format(new_lr))
            self._set_lr(new_lr)
        if train:
            logger.info('Training for {:d} epochs'.format(epochs))
            self.log_train_loss.append([self._train_epoch(trainset)
                                        for _ in range(epochs)])

    def test_loss(self, trainset, batch_size=128):
        self.nnet.eval()  # put model in evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for descriptors, shot_results in trainset.iter_batch(batch_size, False):
                # create descriptors and results tensors where the model lives
                descriptors = torch.as_tensor(descriptors, device=self._device,
                                              dtype=self._dtype)
                shot_results = torch.as_tensor(shot_results, device=self._device,
                                               dtype=self._dtype)
                q_pred = self.nnet(descriptors)
                loss = self.loss(q_pred, shot_results)
                total_loss += float(loss)
        self.nnet.train()  # and back to train mode
        return total_loss / np.sum(trainset.shot_results)

    def _set_lr(self, new_lr):
        # TODO: this could (and should) be the same func for all pytorch models using pytorch.optim optimizers
        # TODO: what if the training scheme does not change the LR?
        # adapted from torch.optim.lr_scheduler.ReduceLROnPlateau._reduce_lr()
        # TODO: new_lr could be a list of different values if we have more parametersets...
        for i, param_group in enumerate(self.optimizer.param_groups):
            #old_lr = float(param_group['lr'])
            #new_lr = max(old_lr * self.factor, self.min_lrs[i])
            param_group['lr'] = new_lr

    def _train_epoch(self, trainset, batch_size=128, shuffle=True):
        # one pass over the whole trainset
        # returns loss per shot averaged over whole training set
        total_loss = 0.
        for descriptors, shot_results in trainset.iter_batch(batch_size, shuffle):
            self.optimizer.zero_grad()
            # create descriptors and results tensors where the model lives
            descriptors = torch.as_tensor(descriptors, device=self._device,
                                          dtype=self._dtype)
            shot_results = torch.as_tensor(shot_results, device=self._device,
                                           dtype=self._dtype)
            q_pred = self.nnet(descriptors)
            loss = self.loss(q_pred, shot_results)
            total_loss += float(loss)
            loss.backward()
            self.optimizer.step()
        return total_loss / np.sum(trainset.shot_results)

    def _log_prob(self, descriptors):
        self.nnet.eval()  # put model in evaluation mode
        # no gradient accumulation for predictions!
        with torch.no_grad():
            # we do this to create the descriptors array on same GPU/CPU where the model lives
            descriptors = torch.as_tensor(descriptors, device=self._device,
                                          dtype=self._dtype)
            # move the prediction tensor to cpu (if not there already) than convert to numpy
            pred = self.nnet(descriptors).cpu().numpy()
        self.nnet.train()  # make model trainable again
        return pred


class EEPytorchRCModel(PytorchRCModel):
    """
    Expected efficiency PytorchRCModel.
    Controls training by multiplying lr with expected efficiency factor

    ee_params - dict, 'expected efficiency parameters'
        lr_0 - float, base learning rate
        lr_min - float, minimal learning rate we still train with
        epochs_per_train - int, if we train we train for this many epochs
        interval - int, we attempt to train every interval MCStep,
                   measured by self.train_hook() calls
        window - int, size of the smoothing window used for expected efficiency
    """
    def __init__(self, nnet, optimizer, ee_params={'lr_0': 1e-3,
                                                   'lr_min': 1e-4,
                                                   'epochs_per_train': 5,
                                                   'interval': 3,
                                                   'window': 100},
                 descriptor_transform=None, loss=None):
        super().__init__(nnet, optimizer, descriptor_transform, loss)
        # make it possible to pass only the altered values in dictionary
        ee_params_defaults = {'lr_0': 1e-3,
                              'lr_min': 1e-4,
                              'epochs_per_train': 5,
                              'interval': 3,
                              'window': 100}
        ee_params_defaults.update(ee_params)
        self.ee_params = ee_params_defaults

    def train_decision(self, trainset):
        # TODO: atm this is the same as for EEMDRCModel, but EEMDRCmodel will probably diverge?
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


# MULTIDOMAIN RCModels
class MultiDomainPytorchRCModel(RCModel):
    """
    Wrapper for multi domain pytorch RCModels.
    Inspired by "Towards an AI physicist for unsupervised learning" by Wu + Tegmark (arXiv:1810.10525)
    """
    def __init__(self, pnets, cnet, poptimizer, coptimizer,
                 descriptor_transform=None, gamma=-1, loss=None):
        # pnets = list of predicting newtworks
        # poptimizer = optimizer for prediction networks
        # cnet = classifier deciding which network to take
        # coptimizer optimizer for classification networks
        super().__init__(descriptor_transform)
        self.pnets = pnets
        self.cnet = cnet
        # any pytorch.optim optimizer, model parameters need to be registered already
        self.poptimizer = poptimizer
        self.coptimizer = coptimizer
        self.gamma = gamma
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
        if loss:
            # if custom loss given we take that
            # TODO: do we need to check if output is vectorial or will it fail anyway if it is not?
            self.loss = loss
        else:
            # otherwise we take the correct one for given n_out
            if self.n_out == 1:
                self.loss = binomial_loss_vect
            else:
                self.loss = multinomial_loss_vect

    @property
    def n_out(self):
        # FIXME:TODO: only works if the last layer is linear!
        # all networks have the same number of out features
        return list(self.pnets[0].modules())[-1].out_features

    @classmethod
    def set_state(cls, state):
        obj = cls(pnets=state['pnets'], cnet=state['cnet'],
                  poptimizer=state['poptimizer'],
                  coptimizer=state['coptimizer'])
        obj.__dict__.update(state)
        return obj

    @classmethod
    def fix_state(cls, state):
        # restore the nnet
        pnets = [pn(**ckwargs) for pn, ckwargs
                 in zip(state['pnets_class'], state['pnets_call_kwargs'])]
        del state['pnets_class']
        del state['pnets_call_kwargs']
        for i, pn in enumerate(pnets):
            pn.load_state_dict(state['pnets'][i])
            # try moving the model to the platform it was on
            dev = _fix_pytorch_device(state['_pdevices'][i])
            pn.to(dev)
        state['pnets'] = pnets
        cnet = state['cnet_class'](**state['cnet_call_kwargs'])
        del state['cnet_class']
        del state['cnet_call_kwargs']
        cnet.load_state_dict(state['cnet'])
        dev = _fix_pytorch_device(state['_cdevice'])
        cnet.to(dev)
        state['cnet'] = cnet
        # and the optimizers
        # TODO/FIXME: we assume one param group per prediction net
        poptimizer = state['poptimizer_class']([{'params': pnet.parameters()}
                                                for pnet in pnets])
        del state['poptimizer_class']
        poptimizer.load_state_dict(state['poptimizer'])
        state['poptimizer'] = poptimizer
        coptimizer = state['coptimizer_class'](cnet.parameters())
        del state['coptimizer_class']
        coptimizer.load_state_dict(state['coptimizer'])
        state['coptimizer'] = coptimizer
        return state

    def save(self, fname, overwrite=False):
        # keep a ref to the networks
        pnets = self.pnets
        cnet = self.cnet
        # but replace with state_dict in self.__dict__
        self.pnets_class = [pn.__class__ for pn in pnets]
        self.pnets_call_kwargs = [pn.call_kwargs for pn in pnets]
        self.pnets = [pn.state_dict() for pn in pnets]
        self.cnet_class = cnet.__class__
        self.cnet_call_kwargs = cnet.call_kwargs
        self.cnet = cnet.state_dict()
        # same for optimizers
        poptimizer = self.poptimizer
        self.poptimizer_class = poptimizer.__class__
        self.poptimizer = poptimizer.state_dict()
        coptimizer = self.coptimizer
        self.coptimizer_class = coptimizer.__class__
        self.coptimizer = coptimizer.state_dict()
        # now let super save the state dict
        super().save(fname, overwrite)
        # and restore nnet and optimizer such that self stays functional
        self.pnets = pnets
        self.cnet = cnet
        self.poptimizer = poptimizer
        self.coptimizer = coptimizer
        # and remove unecessary keys
        del self.pnets_class
        del self.pnets_call_kwargs
        del self.cnet_class
        del self.cnet_call_kwargs
        del self.coptimizer_class
        del self.poptimizer_class

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
        # TODO: different train decisions for different prediction nets?
        # committor prediction nets
        self._count_train_hook += 1
        train, new_lr, epochs = self.train_decision(trainset)
        self.log_train_decision.append([train, new_lr, epochs])
        if new_lr:
            logger.info('Setting learning rate to {:.3e}'.format(new_lr))
            self._set_lr_popt(new_lr)
        if train:
            logger.info('Training for {:d} epochs'.format(epochs))
            self.log_train_loss.append([self._train_epoch_pnets(trainset)
                                        for _ in range(epochs)])

        # classifier
        cnet_target = self._create_cnet_targets(trainset)
        train_c, new_lr_c, epochs_c = self.train_decision_classifier(trainset, cnet_target)
        self.log_ctrain_decision.append([train_c, new_lr_c, epochs_c])
        if new_lr_c:
            logger.info('Setting classifier learning rate to {:.3e}'.format(new_lr_c))
            self._set_lr_copt(new_lr_c)
        if train_c:
            logger.info('Training classifier for {:d} epochs'.format(epochs_c))
            self.log_ctrain_loss.append([self._train_epoch_cnet(trainset, cnet_target)
                                         for _ in range(epochs_c)])

    def test_loss(self, trainset, loss='L_pred', batch_size=128):
        """
        Calculates the test loss over given TrainSet,
        Parameters:
        -----------
        trainset - `:class:arcd.TrainSet` for which to calculate the loss
        loss - str, one of:
               'L_pred' - calculates the loss suffered for weighted prediction
               'L_mod{:d}' - calculates suffered by a specific pnet,
                             where {:d} is an int index to a pnet
               'L_gamma' - calculates the generalized mean loss over all models
               'L_class' - calculates the loss suffered by classifier

        Note that batch_size is ignored for loss='L_pred'.
        """
        if loss is 'L_pred':
            return self._test_loss_pred(trainset)
        elif 'L_mod' in loss:
            mod_num = int(loss.lstrip('L_mod'))
            if not (0 <= mod_num < len(self.pnets)):
                raise ValueError('Can only calculate "L_mod" for a model index'
                                 + ' that is smaller than len(self.pnets).')
            return self._test_loss_pnets(trainset, batch_size)[mod_num]
        elif loss is 'L_gamma':
            return self._test_loss_pnets(trainset, batch_size)[-1]
        elif loss is 'L_class':
            return self._test_loss_cnet(trainset, batch_size)
        else:
            raise ValueError("'loss' must be one of 'L_pred', 'L_mod{:d}', "
                             + "'L_gamma' or 'L_class'")

    def _test_loss_pred(self, trainset):
        # calculate the test loss for the combined weighted prediction
        # p_i = \sum_m p_c(m) * p_i(m)
        # i.e. the loss the model would suffer when used as a whole
        # FIXME: we normalize per shot assuming TwoWayShooting as always
        # self.__call__() puts the model in evaluation mode
        p = self(trainset.descriptors, use_transform=False)
        if self.n_out == 1:
            p = p[:, 0]  # make it a 1d array
            loss = - (np.sum(trainset.shot_results[:, 0] * np.log(1 - p)
                             + trainset.shot_results[:, 1] * np.log(p)
                             )
                      / np.sum(trainset.shot_results)
                      )
        else:
            log_p = np.log(p)
            loss = - (np.sum([n * lp
                              for n, lp in zip(trainset.shot_results, log_p)]
                             )
                      / np.sum(trainset.shot_results)
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
            for descriptors, shot_results in trainset.iter_batch(batch_size, False):
                if self._pnets_same_device:
                    # create descriptors and results tensors where the models live
                    descriptors = torch.as_tensor(descriptors,
                                                  device=self._pdevices[0],
                                                  dtype=self._pdtypes[0])
                    shot_results = torch.as_tensor(shot_results,
                                                   device=self._pdevices[0],
                                                   dtype=self._pdtypes[0])
                # we collect the results on the device of the first pnet
                l_m_sum = torch.zeros((descriptors.shape[0],), device=self._pdevices[0],
                                      dtype=self._pdtypes[0])
                for i, pnet in enumerate(self.pnets):
                    if not self._pnets_same_device:
                        # create descriptors and results tensors where the models live
                        descriptors = torch.as_tensor(descriptors,
                                                      device=self._pdevices[i],
                                                      dtype=self._pdtypes[i])
                        shot_results = torch.as_tensor(shot_results,
                                                       device=self._pdevices[i],
                                                       dtype=self._pdtypes[i])
                    q_pred = pnet(descriptors)
                    l_m = self.loss(q_pred, shot_results)
                    loss_by_model[i] += float(torch.sum(l_m))
                    l_m_sum += torch.pow(l_m, self.gamma).to(l_m_sum.device)
                # end models loop
                L_gamma = torch.sum(torch.pow(l_m_sum / len(self.pnets), 1/self.gamma))
                total_loss += float(L_gamma)
            # end trainset loop
        # end torch.no_grad()
        # back to training mode for all pnets
        self.pnets = [pn.train() for pn in self.pnets]
        return (np.asarray(loss_by_model + [total_loss])
                / np.sum(trainset.shot_results)
                )

    def _test_loss_cnet(self, trainset, batch_size):
        cnet_targets = self._create_cnet_targets(trainset, batch_size)
        self.cnet.eval()  # evaluation mode
        with torch.no_grad():
            total_loss = 0
            descriptors = torch.as_tensor(trainset.descriptors,
                                          device=self._cdevice,
                                          dtype=self._cdtype)
            n_batch = int(len(trainset) / batch_size)
            rest = len(trainset) % batch_size
            for b in range(n_batch):
                des = descriptors[b*batch_size:(b+1)*batch_size]
                tar = cnet_targets[b*batch_size:(b+1)*batch_size]
                log_probs = self.cnet(des)
                loss = multinomial_loss(log_probs, tar)
                total_loss += float(loss)
            # the rest
            des = descriptors[n_batch*batch_size:n_batch*batch_size + rest]
            tar = cnet_targets[n_batch*batch_size:n_batch*batch_size + rest]
            log_probs = self.cnet(des)
            loss = multinomial_loss(log_probs, tar)
            total_loss += float(loss)
        # end torch.no_grad()
        self.cnet.train()  # back to train mode
        # normalize classifier loss per point in trainset
        # this is the same as the per shot normalization,
        # because we only have one event (one correct model) per point
        return total_loss / len(trainset)

    def _train_epoch_pnets(self, trainset, batch_size=128, shuffle=True):
        # one pass over the whole trainset
        # returns loss per shot averaged over whole training set as list,
        # one fore each model by idx and last entry is the combined multidomain loss
        total_loss = 0.
        loss_by_model = [0. for _ in self.pnets]
        for descriptors, shot_results in trainset.iter_batch(batch_size, shuffle):
            self.poptimizer.zero_grad()
            if self._pnets_same_device:
                # create descriptors and results tensors where the models live
                descriptors = torch.as_tensor(descriptors,
                                              device=self._pdevices[0],
                                              dtype=self._pdtypes[0])
                shot_results = torch.as_tensor(shot_results,
                                               device=self._pdevices[0],
                                               dtype=self._pdtypes[0])
            # we collect the results on the device of the first pnet
            l_m_sum = torch.zeros((descriptors.shape[0],), device=self._pdevices[0],
                                  dtype=self._pdtypes[0])
            for i, pnet in enumerate(self.pnets):
                if not self._pnets_same_device:
                    # create descriptors and results tensors where the models live
                    descriptors = torch.as_tensor(descriptors,
                                                  device=self._pdevices[i],
                                                  dtype=self._pdtypes[i])
                    shot_results = torch.as_tensor(shot_results,
                                                   device=self._pdevices[i],
                                                   dtype=self._pdtypes[i])
                q_pred = pnet(descriptors)
                l_m = self.loss(q_pred, shot_results)
                loss_by_model[i] += float(torch.sum(l_m))
                l_m_sum += torch.pow(l_m, self.gamma).to(l_m_sum.device)
            # end models loop
            L_gamma = torch.sum(torch.pow(l_m_sum / len(self.pnets), 1/self.gamma))
            total_loss += float(L_gamma)
            L_gamma.backward()
            self.poptimizer.step()
        # end trainset loop
        return (np.asarray(loss_by_model + [total_loss])
                / np.sum(trainset.shot_results)
                )

    def _create_cnet_targets(self, trainset, batch_size=128):
        # build the trainset for classifier,
        # i.e. which model has the lowest loss for each point in trainset
        targets = torch.zeros((len(trainset), len(self.pnets)),
                              device=self._cdevice,
                              dtype=self._cdtype)
        fill = 0
        # put prediction nets in evaluation mode
        self.pnets = [pn.eval() for pn in self.pnets]
        with torch.no_grad():
            for descriptors, shot_results in trainset.iter_batch(batch_size,
                                                                 shuffle=False):
                if self._pnets_same_device:
                    # create descriptors and results tensors where the models live
                    descriptors = torch.as_tensor(descriptors,
                                                  device=self._pdevices[0],
                                                  dtype=self._pdtypes[0])
                    shot_results = torch.as_tensor(shot_results,
                                                   device=self._pdevices[0],
                                                   dtype=self._pdtypes[0])
                # we collect the results on the device of the first pnet
                l_m_arr = torch.zeros((descriptors.shape[0], len(self.pnets)),
                                      device=self._pdevices[0],
                                      dtype=self._pdtypes[0])
                for i, pnet in enumerate(self.pnets):
                    if not self._pnets_same_device:
                        # create descriptors and results tensors where the models live
                        descriptors = torch.as_tensor(descriptors,
                                                      device=self._pdevices[i],
                                                      dtype=self._pdtypes[i])
                        shot_results = torch.as_tensor(shot_results,
                                                       device=self._pdevices[i],
                                                       dtype=self._pdtypes[i])
                    q_pred = pnet(descriptors)
                    # normalize per shot
                    l_m = (self.loss(q_pred, shot_results)
                           / torch.sum(shot_results, dim=1)
                           )
                    l_m_arr[:, i] = l_m.to(l_m_arr.device)
                # end models loop
                # find minimum loss value model indexes for each point
                # and fill ones into targets at that index
                min_idxs = l_m_arr.argmin(dim=1)
                bs = min_idxs.shape[0]  # not every batch is created equal, i.e. different lengths
                targets[fill + torch.arange(bs), min_idxs] = 1
                fill += bs
            # end batch over trainset loop
        # end torch nograd
        # put pnets back to train mode
        self.pnets = [pn.train() for pn in self.pnets]
        return targets

    def _train_epoch_cnet(self, trainset, cnet_targets, batch_size=128, shuffle=True):
        total_loss = 0
        descriptors = torch.as_tensor(trainset.descriptors,
                                      device=self._cdevice,
                                      dtype=self._cdtype)
        if shuffle:
            shuffle_idxs = torch.randperm(len(trainset))
            descriptors = descriptors[shuffle_idxs]
            cnet_targets = cnet_targets[shuffle_idxs]

        n_batch = int(len(trainset) / batch_size)
        rest = len(trainset) % batch_size
        for b in range(n_batch):
            self.coptimizer.zero_grad()
            des = descriptors[b*batch_size:(b+1)*batch_size]
            tar = cnet_targets[b*batch_size:(b+1)*batch_size]
            log_probs = self.cnet(des)
            loss = multinomial_loss(log_probs, tar)
            total_loss += float(loss)
            loss.backward()
            self.coptimizer.step()
        # the rest
        self.coptimizer.zero_grad()
        des = descriptors[n_batch*batch_size:n_batch*batch_size + rest]
        tar = cnet_targets[n_batch*batch_size:n_batch*batch_size + rest]
        log_probs = self.cnet(des)
        loss = multinomial_loss(log_probs, tar)
        total_loss += float(loss)
        loss.backward()
        self.coptimizer.step()

        # normalize classifier loss per point in trainset
        return total_loss / len(trainset)

    def _set_lr_popt(self, new_lr):
        # TODO: this could (and should) be the same func for all pytorch models using pytorch.optim optimizers
        # TODO: what if the training scheme does not change the LR?
        # adapted from torch.optim.lr_scheduler.ReduceLROnPlateau._reduce_lr()
        # TODO: new_lr could be a list of different values if we have more parametersets...
        for i, param_group in enumerate(self.poptimizer.param_groups):
            #old_lr = float(param_group['lr'])
            #new_lr = max(old_lr * self.factor, self.min_lrs[i])
            param_group['lr'] = new_lr

    def _set_lr_copt(self, new_lr):
        # TODO: this could (and should) be the same func for all pytorch models using pytorch.optim optimizers
        # TODO: what if the training scheme does not change the LR?
        # adapted from torch.optim.lr_scheduler.ReduceLROnPlateau._reduce_lr()
        # TODO: new_lr could be a list of different values if we have more parametersets...
        for i, param_group in enumerate(self.coptimizer.param_groups):
            #old_lr = float(param_group['lr'])
            #new_lr = max(old_lr * self.factor, self.min_lrs[i])
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
    def _log_prob(self, descriptors):
        return self.q(descriptors, use_transform=False)

    def log_prob(self, descriptors, use_transform=True):
        p = self(descriptors, use_transform)
        if p.shape[1] == 1:
            return np.log(1. / p - 1.)
        return np.log(p)

    def __call__(self, descriptors, use_transform=True, domain_predictions=False):
        # returns the probabilities,
        # we decide here if we transform, as this is our initial step even if we back-calculate q
        # if wanted and self.descriptor_transform is defined we use it before prediction
        # if domain_predictions=True we will return a tuple (p_weighted, [p_m for m in self.pnets])
        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)
        # get the probabilities for each model from classifier
        p_c = self._classify(descriptors)  # returns p_c on self._cdevice
        self.pnets = [pn.eval() for pn in self.pnets]  # pnets to evaluation
        # now committement probabilities
        with torch.no_grad():
            descriptors = torch.as_tensor(descriptors,
                                          device=self._pdevices[0],
                                          dtype=self._pdtypes[0])
            pred = torch.zeros((p_c.shape[0], self.n_out),
                               device=self._pdevices[0],
                               dtype=self._pdtypes[0])
            if domain_predictions:
                p_m_list = []
            p_c = p_c.to(self._pdevices[0])
            for i, pnet in enumerate(self.pnets):
                # .to() should be a no-op if they are all on the same device (?)
                descriptors = descriptors.to(self._pdevices[i])
                q = pnet(descriptors)
                if q.shape[1] == 1:
                    p = 1. / (1. + torch.exp(-q))
                else:
                    exp_q = torch.exp(q)
                    p = exp_q / torch.sum(exp_q, dim=1, keepdim=True)
                if domain_predictions:
                    p_m_list.append(p.cpu().numpy())
                p = p.to(self._pdevices[0])
                pred += p_c[:, i:i+1] * p
            # end pnets loop
        # end torch.no_grad()
        self.pnets = [pn.train() for pn in self.pnets]  # back to train
        if domain_predictions:
            return (pred.cpu().numpy(), p_m_list)
        return pred.cpu().numpy()

    def classify(self, descriptors, use_transform=True):
        """
        Returns the probabilities the classifier assigns to each model
        for given descriptors.
        """
        # this is just a wrapper around _classify to convert to numpy
        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)
        # get the probabilities for each model from classifier
        # and move them to cpu to numpy before return
        return self._classify(descriptors).cpu().numpy()

    def _classify(self, descriptors):
        # return classifier model probabilities for descriptors
        # descriptors is expected to be numpy or torch tensor
        # returns a torch.tensor on the same device the classifier lives on
        self.cnet.eval()  # classify in evaluation mode
        with torch.no_grad():
            descriptors = torch.as_tensor(descriptors, device=self._cdevice,
                                          dtype=self._cdtype)
            q_c = self.cnet(descriptors)
            exp_q_c = torch.exp(q_c)
            # p_c is the probability the classifier assigns the point to be in models class
            p_c = exp_q_c / torch.sum(exp_q_c, dim=1, keepdim=True)

        self.cnet.train()  # and make the cnet trainable again
        return p_c


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
    def __init__(self, pnets, cnet, poptimizer, coptimizer,
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
                 descriptor_transform=None, loss=None):
        super().__init__(pnets, cnet, poptimizer, coptimizer,
                         descriptor_transform, gamma, loss)
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
