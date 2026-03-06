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
import asyncio
import logging
import numpy as np
from openpathsampling.engines.snapshot import BaseSnapshot as OPSBaseSnapshot
from openpathsampling.engines.trajectory import Trajectory as OPSTrajectory
from openpathsampling.collectivevariable import CollectiveVariable
from openpathsampling import Volume
from abc import ABC, abstractmethod

from . import _H5PY_PATH_DICT


logger = logging.getLogger(__name__)


class RCModel(ABC):
    """
    All RCModels should subclass this.

    Subclasses (or users) can define a self.descriptor_transform function,
    which will be applied to descriptors before prediction by the model.
    This could e.g. be a OPS-CV to transform OPS trajectories to numpy arrays,
    such that you can call a model on trajectories directly.
    For OPS CollectivVariables we have special support for saving and loading
    if used in conjunction with OPS storages.

    Attributes:
    -----------
        states - list of states, for use in conjunction with ops these should
                 be :class:`openpathsampling.Volume`s or other functions
                 taking a snapshot and returning if it is inside of the state
        descriptor_transform - any function transforming (Cartesian) snapshot
                               coordinates to the descriptor representation in
                               which the model learns, e.g. an
                               :class:`openpathsampling.CollectiveVariable`,
                               see `.coordinates` for examples of functions
                               that can be turned to a MDtrajFunctionCV
        cache_file - the aimmd.Storage used for caching
        n_out - None or int, used to overwrite the deduced number of outputs,
                i.e. to use a multinomial loss and model for a 2 state system,
                normaly we check len(states) and set n_out accordingly:
                    n_out = 1 if len(states) == 2
                    n_out= len(states) if len(states) > 2
        z_sel_scale - float, scale z_sel to [0., z_sel_scale] for multinomial
                      training and predictions
        min_points_ee_factor - minimum number of SPs in TrainSet to calculate
                               expected efficiency factor over
        density_collection_n_bins - number of bins in each probability
                                    direction to collect the density of points
                                    on TPs
                                    NOTE: this has to be set before creating
                                    the model, i.e.
                                    `RCModel.density_collection_n_bins = n_bins`
                                    and then
                                    `model = RCModel(**init_parms)`

    """

    # have it here, such that we can get it without instantiating
    # scale z_sel to [0., z_sel_scale] for multinomial predictions
    z_sel_scale = 25
    # minimum number of SPs in training set for EE factor calculation
    min_points_ee_factor = 10

    def __init__(self, states, descriptor_transform=None, n_out=None):
        """I am an `abc.ABC` and can not be initialized."""
        self.states = states
        self._n_out = n_out
        self.descriptor_transform = descriptor_transform
        # list of expected committment probabilities,
        # should be in the same order as the trainset
        # will be added when selecting a shooting point
        self.expected_p = []

    @property
    def n_out(self):
        """Return the number of model outputs, i.e. states."""
        # need to know if we use binomial or multinomial
        if self._n_out is not None:
            return self._n_out

        n_states = len(self.states)
        self._n_out = n_states if n_states > 2 else 1
        return self._n_out

    # NOTE on saving and loading models:
    #   if your generic RCModel subclass contains only pickleable objects in
    #   its self.__dict__ you do not have to do anything.
    #   otherwise you might need to implement object_for_pickle and
    #   complete_from_h5py_group, which will be called on saving and loading
    #   respectively, both will be called with a h5py group as argument which
    #   you can use use for saving and loading.
    #   Have a look at the pytorchRCModels code.
    def object_for_pickle(self, group, overwrite=True, **kwargs):
        state = self.__dict__.copy()  # shallow copy -> take care of lists etc!
        if isinstance(state['descriptor_transform'], CollectiveVariable):
            # replace OPS CVs by their name to reload from OPS storage
            state['descriptor_transform'] = state['descriptor_transform'].name
        # create a new list
        # replace ops volumes by their name, keep everything else
        state["states"] = [s.name if isinstance(s, Volume) else s
                           for s in self.states
                           ]
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        return ret_obj

    def complete_from_h5py_group(self, group):
        # only add what we did not pickle
        # (except the ops CollectiveVariable but that needs special care
        #  since it comes from ops storage)
        return self

    def complete_from_ops_storage(self, ops_storage):
        if isinstance(self.descriptor_transform, str):
            self.descriptor_transform = ops_storage.cvs.find(self.descriptor_transform)
        else:
            raise ValueError("self.descriptor_transform does not seem to be a "
                             + "string indicating the name of the ops CV.")
        for i, s in enumerate(self.states):
            if isinstance(s, str):
                try:
                    self.states[i] = ops_storage.volumes.find(s)
                except KeyError:
                    logger.warn(f"There seems to be no state with name {s} in"
                                + " the ops storage.")
        return self

    @abstractmethod
    def train_hook(self, trainset):
        """Will be called by aimmd.TraininHook after every MCStep."""
        raise NotImplementedError

    @abstractmethod
    def _log_prob(self, descriptors, batch_size):
        # returns the unnormalized log probabilities for given descriptors
        # descriptors is a numpy array with shape (n_points, n_descriptors)
        # the output is expected to be an array of shape (n_points, n_out)
        raise NotImplementedError

    @abstractmethod
    def test_loss(self, trainset):
        """Return test loss per shot in trainset."""
        raise NotImplementedError

    def train_expected_efficiency_factor(self, trainset, window):
        """
        Calculate (1 - {n_TP}_true / {n_TP}_expected)**2

        {n_TP}_true - summed over the last 'window' entries of the given
                      trainsets transitions
        {n_TP}_expected - calculated from self.expected_p assuming
                          2 independent shots per point

        Note that we will return min(1, EE-Factor).
        Note also that we will always return 1 if there are less than
        self.min_points_ee_factor points in the trainset.

        """
        # make sure there are enough points, otherwise take less
        # NOTE this also enables starting with non-empty trainingsets
        #      as we will always take at maximum as many points as we
        #      have expected_p values for
        n_points = min(len(trainset), len(self.expected_p), window)
        if n_points < self.min_points_ee_factor:
            # we can not reasonably estimate EE factor due to too less points
            return 1
        n_tp_true = sum(trainset.transitions[-n_points:])
        p_ex = np.asarray(self.expected_p[-n_points:])
        if self.n_out == 1:
            n_tp_ex = np.sum(2 * (1 - p_ex[:, 0]) * p_ex[:, 0])
        else:
            n_tp_ex = 2 * np.sum([p_ex[:, i] * p_ex[:, j]
                                  for i in range(self.n_out)
                                  for j in range(i + 1, self.n_out)
                                  ])
        factor = (1 - n_tp_true / n_tp_ex)**2
        logger.info('Calculated expected efficiency factor '
                    + '{:.3e} over {:d} points.'.format(factor, n_points))
        return min(1, factor)

    def register_sp(self, shoot_snap, use_transform=True):
        """Will be called by aimmd.RCModelSelector after selecting a SP."""
        self.expected_p.append(self(shoot_snap, use_transform)[0])

    def _apply_descriptor_transform(self, descriptors):
        # apply descriptor_transform if wanted and defined
        # returns either unaltered descriptors or transformed descriptors
        if self.descriptor_transform is not None:
            # transform OPS snapshots to trajectories to get 2d arrays
            if isinstance(descriptors, OPSBaseSnapshot):
                descriptors = OPSTrajectory([descriptors])
            descriptors = self.descriptor_transform(descriptors)
        return descriptors

    def log_prob(self, descriptors, use_transform=True, batch_size=None):
        """
        Return the unnormalized log probabilities for given descriptors.

        For n_out=1 only the log probability to reach state B is returned.
        If batch_size is None we will try to get a default value from the
        models training parameters.
        """
        # if self.descriptor_transform is defined we use it before prediction
        # otherwise we just apply the model to descriptors
        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)
        return self._log_prob(descriptors, batch_size=batch_size)

    def q(self, descriptors, use_transform=True, batch_size=None):
        """
        Return the reaction coordinate value(s) for given descriptors.

        For n_out=1 the RC towards state B is returned,
        otherwise the RC towards the state is returned for each state.
        If batch_size is None we will try to get a default value from the
        models training parameters.
        """
        if self.n_out == 1:
            return self.log_prob(descriptors,
                                 use_transform=use_transform,
                                 batch_size=batch_size,
                                 )
        else:
            log_prob = self.log_prob(descriptors,
                                     use_transform=use_transform,
                                     batch_size=batch_size)
            rc = [(log_prob[..., i:i+1]
                   - np.log(np.sum(np.exp(np.delete(log_prob, [i], axis=-1)),
                                   axis=-1, keepdims=True
                                   )
                            )
                   ) for i in range(log_prob.shape[-1])]
            return np.concatenate(rc, axis=-1)

    def __call__(self, descriptors, use_transform=True, batch_size=None):
        """
        Return the commitment probability/probabilities.

        Returns p_B if n_out=1,
        otherwise the committment probabilities towards the states.
        If batch_size is None we will try to get a default value from the
        models training parameters.
        """
        log_prob = self.log_prob(descriptors,
                                 use_transform=use_transform,
                                 batch_size=batch_size)
        if self.n_out == 1:
            return self._p_binom(log_prob)
        return self._p_multinom(log_prob)

    def _p_binom(self, log_prob):
        return 1/(1 + np.exp(-log_prob))

    def _p_multinom(self, log_probs):
        exp_log_p = np.exp(log_probs)
        return exp_log_p / np.sum(exp_log_p, axis=1, keepdims=True)

    def z_sel(self, descriptors, use_transform=True, batch_size=None):
        """
        Return the value of the selection coordinate z_sel.

        It is zero at the most optimal point conceivable.
        For n_out = 1 this is simply the unnormalized log probability.
        For n_out > 1 see method `self._z_sel_multinom`.
        If batch_size is None we will try to get a default value from the
        models training parameters.
        """
        if self.n_out == 1:
            return self.q(descriptors,
                          use_transform=use_transform,
                          batch_size=batch_size,
                          )[:, 0]  # make z_sel 1d!
        return self._z_sel_multinom(descriptors,
                                    use_transform=use_transform,
                                    batch_size=batch_size,
                                    )

    def _z_sel_multinom(self, descriptors, use_transform, batch_size):
        """
        Multinomial selection coordinate.

        This expression is zero if (and only if) x is at the point of
        maximaly conceivable p(TP|x), i.e. all p_i are equal.
        z_{sel}(x) always lies in [0, self.z_sel_scale], where
        z_{sel}(x)=self.z_sel_scale implies p(TP|x)=0.
        We can therefore select the point for which this
        expression is closest to zero as the optimal SP.
        """
        p = self(descriptors,
                 use_transform=use_transform,
                 batch_size=batch_size)
        # the prob to be on any TP is 1 - the prob to be on no TP
        # to be on no TP means beeing on a "self transition" (A->A, etc.)
        reactive_prob = 1 - np.sum(p * p, axis=1)
        return (
                (self.z_sel_scale / (1 - 1 / self.n_out))
                * (1 - 1 / self.n_out - reactive_prob)
                )


class RCModelAsyncMixin:
    """
    RCModelAsync for coroutine descriptor transforms.

    Has awaitable __call__, q, z_sel, ..., i.e. everything that involves the
    descriptor_transform. Parent class docstring:
    """

    __doc__ += RCModel.__doc__  # and add the rest of the docstring
    # NOTE: we "steal" the method docstrings from the RCModel as they are
    #       the same (except for async/non-async)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overwrite the register_sp func for the async model to raise an error and
    # make sure people are warned
    def register_sp(self, shoot_snap, use_transform=True):
        """
        Will be called by aimmd.RCModelSelector after selecting a SP.

        NOTE: The expected efficiency for the SPs in the async case is done by
              the usual functions, except that the SPs and their expected probs
              are added to the model by the TrainingTask, which also adds them
              to the training set in the same order (which allows us to easily
              make sure the orders match even if we have many samplers running
              in parallel).
        """
        raise NotImplementedError("Registering shooting points for the "
                                  "expected efficiency calculation is done by "
                                  "the `TrainingTask`. See the docstring of "
                                  "this method for more.")

    async def _apply_descriptor_transform(self, descriptors):
        if self.descriptor_transform is not None:
            # Note: I think we dont need to check if it is an ops snapshot,
            #       those should only be a thing for non-async models
            descriptors = await self.descriptor_transform(descriptors)
        return descriptors

    async def log_prob(self, descriptors, use_transform=True, batch_size=None):
        if use_transform:
            descriptors = await self._apply_descriptor_transform(descriptors)
        return self._log_prob(descriptors, batch_size=batch_size)

    log_prob.__doc__ = RCModel.log_prob.__doc__

    async def q(self, descriptors, use_transform=True, batch_size=None):
        if self.n_out == 1:
            return await self.log_prob(descriptors,
                                       use_transform=use_transform,
                                       batch_size=batch_size)
        else:
            log_prob = await self.log_prob(descriptors,
                                           use_transform=use_transform,
                                           batch_size=batch_size)
            rc = [(log_prob[..., i:i+1]
                   - np.log(np.sum(np.exp(np.delete(log_prob, [i], axis=-1)),
                                   axis=-1, keepdims=True
                                   )
                            )
                   ) for i in range(log_prob.shape[-1])]
            return np.concatenate(rc, axis=-1)

    q.__doc__ = RCModel.q.__doc__

    async def __call__(self, descriptors, use_transform=True, batch_size=None):
        log_prob = await self.log_prob(descriptors,
                                       use_transform=use_transform,
                                       batch_size=batch_size)
        if self.n_out == 1:
            return self._p_binom(log_prob)
        return self._p_multinom(log_prob)

    __call__.__doc__ = RCModel.__call__.__doc__

    async def z_sel(self, descriptors, use_transform=True, batch_size=None):
        if self.n_out == 1:
            return (await self.q(descriptors,
                                 use_transform=use_transform,
                                 batch_size=batch_size))[:, 0]  # make z_sel 1d
        return await self._z_sel_multinom(descriptors,
                                          use_transform=use_transform,
                                          batch_size=batch_size)

    z_sel.__doc__ = RCModel.z_sel.__doc__

    async def _z_sel_multinom(self, descriptors, use_transform, batch_size):
        p = await self(descriptors,
                       use_transform=use_transform,
                       batch_size=batch_size)
        # the prob to be on any TP is 1 - the prob to be on no TP
        # to be on no TP means beeing on a "self transition" (A->A, etc.)
        reactive_prob = 1 - np.sum(p * p, axis=1)
        return (
                (self.z_sel_scale / (1 - 1 / self.n_out))
                * (1 - 1 / self.n_out - reactive_prob)
                )

    _z_sel_multinom.__doc__ = RCModel._z_sel_multinom.__doc__
