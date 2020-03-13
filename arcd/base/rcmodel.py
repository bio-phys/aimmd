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
import pickle
import numpy as np
from openpathsampling.engines.snapshot import BaseSnapshot as OPSBaseSnapshot
from openpathsampling.engines.trajectory import Trajectory as OPSTrajectory
from openpathsampling.collectivevariable import CollectiveVariable
from abc import ABC, abstractmethod


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
        descriptor_transform - any function transforming (Cartesian) snapshot
                               coordinates to the descriptor representation in
                               which the model learns, e.g. an
                               :class:`openpathsampling.CollectiveVariable`,
                               see `.coordinates` for examples of functions
                               that can be turned to a MDtrajFunctionCV
        save_model_extension - str, the file extension to use when saving
        z_sel_scale - float, scale z_sel to [0., z_sel_scale] for multinomial
                      training and predictions
        density_collection_n_bins - number of bins in each probability
                                    direction to collect the density of points
                                    on TPs
                                    NOTE: this has to be set before creating
                                    the model, i.e.
                                    `RCModel.density_collection_n_bins = n_bins`
                                    and then
                                    `model = RCModel(**init_parms)`

    """

    # need to have it here, such that we can get it without instantiating
    save_model_extension = '.pckl'
    # scale z_sel to [0., z_sel_scale] for multinomial predictions
    z_sel_scale = 25
    # minimum number of SPs in training set for EE factor calculation
    min_points_ee_factor = 10
    # (TP) density collection params
    # NOTE: Everything here only makes it possible to collect TP densities
    # by attaching a TrajectoryDensityCollector, however for this feature
    # to have any effect we need to regularly update the density estimate!
    # Updating the density estimate is done by the corresponding TrainingHook,
    # and unless using the density is enabled there we will not waste
    # computing power with unecessary updates of the density
    density_collection_n_bins = 10

    def __init__(self, descriptor_transform=None):
        """I am an `abc.ABC` and can not be initialized."""
        self.descriptor_transform = descriptor_transform
        self.expected_p = []
        self.expected_q = []
        self.density_collector = TrajectoryDensityCollector(
                                            n_dim=self.n_out,
                                            bins=self.density_collection_n_bins
                                                                )

    @property
    @abstractmethod
    def n_out(self):
        """Return the number of model outputs, i.e. states."""
        # need to know if we use binomial or multinomial
        raise NotImplementedError

    # NOTE ON SAVING AND LOADING MODELS
    # you do not have to do anything if your generic RCModel subclass contains
    # only picklable python objects in its obj.__dict__
    # (except for obj.descriptor_transform which we handle here)
    # Otherwise you will need to implement obj.set_state() and obj.fix_state()
    # and obj.save(), while obj.load_state() should stay untouched
    # Have a look at the KerasRCModels and PytorchRCModels to see how
    # In general state will be a dict, obj.fix_state() should reset all values
    # to the correct python objects
    # and cls.set_state(state) should return the object with given state
    # obj.save() should make all obj.__dict__ values picklable
    # and then call super().save(fname) to save the model
    @classmethod
    def set_state(cls, state):
        """Return an object of the same class with given internal state."""
        obj = cls()
        obj.__dict__.update(state)
        return obj

    @classmethod
    def fix_state(self, state):
        """Corrects a given loaded state to an operational state."""
        return state

    @classmethod
    def load_state(cls, fname, ops_storage=None):
        """Return internal state from given file, possibly (re)set OPS CVs."""
        with open(fname, 'rb') as pfile:
            state = pickle.load(pfile)
        # set pickle_file_dirname variable so we can load models saved besides
        # as e.g. the keras hdf5 files
        state['_pickle_file_dirname'] = os.path.abspath(os.path.dirname(fname))
        if (ops_storage is not None):
            transform = state['descriptor_transform']
            if isinstance(transform, str):
                # we just assume it is the name of the OPS CV
                state['descriptor_transform'] = ops_storage.cvs.find(transform)
            else:
                raise ValueError('Could not load descriptor_transform from '
                                 + 'ops_storage.')
        sub_class = state['__class__']
        del state['__class__']
        # to make this a generally applicable function,
        # we return state and the correct subclass to call
        # such that we can do state=sub_class.fix_state(state)
        # to first get the final operational state dict
        # and then call sub_class.set_state(state)
        # which returns the correctly initialized obj
        return state, sub_class

    def save(self, fname, overwrite=False):
        """Save internal state to file."""
        state = self.__dict__.copy()
        state['__class__'] = self.__class__
        if isinstance(state['descriptor_transform'], CollectiveVariable):
            # replace OPS CVs by their name to reload from OPS storage
            state['descriptor_transform'] = state['descriptor_transform'].name
        # now save
        if not fname.endswith(self.save_model_extension):
            # make sure we have the correct extension
            fname += self.save_model_extension
        if os.path.exists(fname) and not overwrite:
            raise IOError('File {:s} exists.'.format(fname))
        with open(fname, 'wb') as pfile:
            # NOTE: we need python >= 3.4 for protocol=4
            pickle.dump(state, pfile, protocol=4)

    @abstractmethod
    def train_hook(self, trainset):
        """Will be called by arcd.TraininHook after every MCStep."""
        raise NotImplementedError

    @abstractmethod
    def _log_prob(self, descriptors):
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
        logger.info('Calculcated expected efficiency factor '
                    + '{:.3e} over {:d} points.'.format(factor, n_points))
        return min(1, factor)

    def register_sp(self, shoot_snap, use_transform=True):
        """Will be called by arcd.RCModelSelector after selecting a SP."""
        self.expected_q.append(self.q(shoot_snap, use_transform)[0])
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

    def log_prob(self, descriptors, use_transform=True):
        """
        Return the unnormalized log probabilities for given descriptors.

        For n_out=1 only the log probability to reach state B is returned.
        """
        # if self.descriptor_transform is defined we use it before prediction
        # otherwise we just apply the model to descriptors
        if use_transform:
            descriptors = self._apply_descriptor_transform(descriptors)
        return self._log_prob(descriptors)

    def q(self, descriptors, use_transform=True):
        """
        Return the reaction coordinate value(s) for given descriptors.

        For n_out=1 the RC towards state B is returned,
        otherwise the RC towards the state is returned for each state.
        """
        if self.n_out == 1:
            return self.log_prob(descriptors, use_transform)
        else:
            log_prob = self.log_prob(descriptors, use_transform)
            rc = [(log_prob[..., i:i+1]
                   - np.log(np.sum(np.exp(np.delete(log_prob, [i], axis=-1)),
                                   axis=-1, keepdims=True
                                   )
                            )
                   ) for i in range(log_prob.shape[-1])]
            return np.concatenate(rc, axis=-1)

    def __call__(self, descriptors, use_transform=True):
        """
        Return the commitment probability/probabilities.

        Returns p_B if n_out=1,
        otherwise the committment probabilities towards the states.
        """
        if self.n_out == 1:
            return self._p_binom(descriptors, use_transform)
        return self._p_multinom(descriptors, use_transform)

    def _p_binom(self, descriptors, use_transform):
        q = self.q(descriptors, use_transform)
        return 1/(1 + np.exp(-q))

    def _p_multinom(self, descriptors, use_transform):
        exp_log_p = np.exp(self.log_prob(descriptors, use_transform))
        return exp_log_p / np.sum(exp_log_p, axis=1, keepdims=True)

    def z_sel(self, descriptors, use_transform=True):
        """
        Return the value of the selection coordinate z_sel.

        It is zero at the most optimal point conceivable.
        For n_out=1 this is simply the unnormalized log probability.
        """
        if self.n_out == 1:
            return self.q(descriptors, use_transform)
        return self._z_sel_multinom(descriptors, use_transform)

    def _z_sel_multinom(self, descriptors, use_transform):
        """
        Multinomial selection coordinate.

        This expression is zero if (and only if) x is at the point of
        maximaly conceivable p(TP|x), i.e. all p_i are equal.
        z_{sel}(x) always lies in [0, self.z_sel_scale], where
        z_{sel}(x)=self.z_sel_scale implies p(TP|x)=0.
        We can therefore select the point for which this
        expression is closest to zero as the optimal SP.
        """
        p = self._p_multinom(descriptors, use_transform)
        # the prob to be on any TP is 1 - the prob to be on no TP
        # to be on no TP means beeing on a "self transition" (A->A, etc.)
        reactive_prob = 1 - np.sum(p * p, axis=1)
        return (
                (self.z_sel_scale / (1 - 1 / self.n_out))
                * (1 - 1 / self.n_out - reactive_prob)
                )


class TrajectoryDensityCollector:
    """
    Keep track of density of points on trajectories projected to probabilities.

    Usually trajectories will be transition paths and the space will be the
    space spanned by the commitment probabilities predicted by a RCModel.
    The inverse of the estimate of the density of points on TPs can be used as
    an additional factor in the shooting point selection to flatten the
    distribution of proposed shooting points in commitment probability space.

    """

    def __init__(self, n_dim, bins=10):
        """
        Initialize a TPDensityCollector.

        Parameters:
        -----------
        n_dim - int, dimensionality of the histogram space
        bins - number of bins in each direction

        """
        self.n_dim = n_dim
        self.bins = bins
        self.density_histogram = np.zeros(tuple(bins for _ in range(n_dim)))
        self._counts = np.empty((0, 0))
        self._descriptors = np.empty((0, 0))
        self._fill_pointer = 0
        # need to know the number of forbidden bins, i.e. where sum_prob > 1
        bounds = np.arange(0., 1., 1./bins)
        # this magic line creates all possible combinations of probs
        # as an array of shape (bins**n_probs, n_probs)
        combs = np.array(np.meshgrid(*tuple(bounds for _ in range(self.n_dim)))
                         ).T.reshape(-1, self.n_dim)
        sums = np.sum(combs, axis=1)
        n_forbidden_bins = len(np.where(sums > 1)[0])
        self._n_allowed_bins = bins**self.n_dim - n_forbidden_bins

    def _extend_if_needed(self, tp_len, descriptor_dim, add_entries=5000):
        """Extend internal storage arrays if next TP would not fit."""
        shadow_len = self._descriptors.shape[0]
        # make sure we always make space for the whole TP
        add = max((add_entries, tp_len))
        if shadow_len == 0:
            # first creation of the arrays
            self._counts = np.zeros((add,), dtype=np.float64)
            self._descriptors = np.zeros((add, descriptor_dim),
                                         dtype=np.float64)
        elif shadow_len <= self._fill_pointer + tp_len:
            # extend by at least the tp_len
            self._counts = np.concatenate((self._counts,
                                           np.zeros((add,), dtype=np.float64))
                                          )
            self._descriptors = np.concatenate(
                    (self._descriptors,
                     np.zeros((add, descriptor_dim), dtype=np.float64))
                                               )

    def _append(self, tp_descriptors, counts_arr):
        # we expect tp_descriptors to be of shape (n_points, descriptor_dim)
        # and counts_arr to be of shape (n_points,)
        tp_len, descriptor_dim = tp_descriptors.shape
        self._extend_if_needed(tp_len=tp_len, descriptor_dim=descriptor_dim)
        self._descriptors[self._fill_pointer:self._fill_pointer+tp_len] = tp_descriptors
        self._counts[self._fill_pointer:self._fill_pointer+tp_len] = counts_arr
        self._fill_pointer += tp_len

    def evaluate_density_on_trajectories(self, model, trajectories, counts=None):
        """
        Evaluate the density on the given trajectories, also **recreates** the
        complete density estimate using the current model predictions.
        Additionally store trajectories/descriptors for later reevaluation.

        See self.reevaluate_density() to only recreate the density estimate
        without adding new trajectories.

        Parameters:
        -----------
        model - arcd.base.RCModel predicting commitment probabilities
        trajectories - iterator/iterable of trajectories to evaluate
        counts - None or list of weights for the trajectories,
                 i.e. we will add every trajectory count times to the histo,
                 if None, every trajectory is added once

        """
        # add descriptors to self
        if counts is None:
            counts = len(trajectories) * [1.]
        for tra, c in zip(trajectories, counts):
            descriptors = model.descriptor_transform(tra)
            self._append(tp_descriptors=descriptors,
                         counts_arr=np.full((descriptors.shape[0]), c)
                         )
        # get current density estimate for all stored descriptors
        self.reevaluate_density(model=model)

    def reevaluate_density(self, model):
        """
        Reevaluate the density for all stored trajectories.

        Will replace the density histogram with a new density estimate for all
        trajectories from current models prediction.

        Parameters:
        -----------
        model - arcd.base.RCModel predicting commitment probabilities

        """
        pred = model(self._descriptors[:self._fill_pointer],
                     use_transform=False)
        histo, edges = np.histogramdd(
                            sample=pred,
                            bins=self.bins,
                            range=[[0., 1.]
                                   for _ in range(self.n_dim)],
                            weights=self._counts[:self._fill_pointer]
                                      )
        self.density_histogram = histo

    def get_density(self, probabilities):
        """
        Return the current density estimate for a given probability vector.

        Parameters:
        -----------
        probabilities - numpy.ndarray, shape=(n_points, self.n_dim)

        Returns:
        --------
        densities - numpy.ndarray, shape=(n_points,), values of the density
                    at the given points in probability-space

        """
        # we take the min to make sure we are always in the
        # histogram range, even if p = 1
        idxs = tuple(np.intp(
                        np.minimum(np.floor(probabilities[:, i] * self.bins),
                                   self.bins - 1)
                             )
                     for i in range(self.n_dim)
                     )
        return self.density_histogram[idxs]

    def get_correction(self, probabilities):
        """
        Return the 'flattening factor' for the observed density of points.

        The factor is calculated as
         (total_count + n_allowed_bins) / (counts[probabilities] + 1),
        i.e. the factor is 1 / rho(probabilities),
        but we use the Laplace-m-estimator / Add-one-smoothing
        to make sure we do not have zero density anywhere.

        """
        dens = self.get_density(probabilities)
        norm = np.sum(self.density_histogram)
        return (norm + self._n_allowed_bins) / (dens + 1)
