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
import copy
import numpy as np
from openpathsampling.engines.snapshot import BaseSnapshot as OPSBaseSnapshot
from openpathsampling.engines.trajectory import Trajectory as OPSTrajectory
from openpathsampling.collectivevariable import CollectiveVariable
from openpathsampling import Volume
from abc import ABC, abstractmethod
from .storage import Storage


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
        cache_file - the arcd.Storage used for caching
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
    # (TP) density collection params
    # NOTE: Everything here only makes it possible to collect TP densities
    # by attaching a TrajectoryDensityCollector, however for this feature
    # to have any effect we need to regularly update the density estimate!
    # Updating the density estimate is done by the corresponding Hook,
    # and unless using the density is enabled by attaching the hook, we will
    # not waste computing power with unecessary updates of the density
    density_collection_n_bins = 10

    def __init__(self, states, descriptor_transform=None, cache_file=None, n_out=None):
        """I am an `abc.ABC` and can not be initialized."""
        self.states = states
        self._n_out = n_out
        self.descriptor_transform = descriptor_transform
        self.expected_p = []
        self.expected_q = []
        self.density_collector = TrajectoryDensityCollector(
                                        n_dim=self.n_out,
                                        bins=self.density_collection_n_bins,
                                        cache_file=cache_file
                                                            )

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
    def object_for_pickle(self, group, overwrite=True):
        state = self.__dict__.copy()  # shallow copy -> take care of lists etc!
        if isinstance(state['descriptor_transform'], CollectiveVariable):
            # replace OPS CVs by their name to reload from OPS storage
            state['descriptor_transform'] = state['descriptor_transform'].name
        # create a new list
        # replace ops volumes by their name, keep everything else
        state["states"] = [s.name if isinstance(s, Volume) else s
                           for s in self.states
                           ]
        # take care of density collector
        state["density_collector"] = state["density_collector"].object_for_pickle(group,
                                                                                  overwrite=overwrite,
                                                                                  )
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        return ret_obj

    def complete_from_h5py_group(self, group):
        # only add what we did not pickle
        # for now this is only the density collector
        # (except the ops CollectiveVariable but that needs special care
        #  since it comes from ops storage)
        self.density_collector = self.density_collector.complete_from_h5py_group(group)
        return self

    def complete_from_ops_storage(self, ops_storage):
        if isinstance(self.descriptor_transform, str):
            self.descriptor_transform = ops_storage.cvs.find(self.descriptor_transform)
            return self
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

    def __init__(self, n_dim, bins=10, cache_file=None):
        """
        Initialize a TrajectoryDensityCollector.

        Parameters:
        -----------
        n_dim - int, dimensionality of the histogram space
        bins - number of bins in each direction

        """
        self.n_dim = n_dim
        self.bins = bins
        self.density_histogram = np.zeros(tuple(bins for _ in range(n_dim)))
        self._cache_file = cache_file
        self._fill_pointer = 0
        if cache_file is None:
            self._counts = np.empty((0, 0))
            self._descriptors = np.empty((0, 0))
            self._cache = None
        else:
            self._create_h5py_cache(cache_file=cache_file)
            self._counts = None
            self._descriptors = None

        # need to know the number of forbidden bins, i.e. where sum_prob > 1
        bounds = np.arange(0., 1., 1./bins)
        # this magic line creates all possible combinations of probs
        # as an array of shape (bins**n_probs, n_probs)
        combs = np.array(np.meshgrid(*tuple(bounds for _ in range(self.n_dim)))
                         ).T.reshape(-1, self.n_dim)
        sums = np.sum(combs, axis=1)
        n_forbidden_bins = len(np.where(sums > 1)[0])
        self._n_allowed_bins = bins**self.n_dim - n_forbidden_bins

    @property
    def cached(self):
        """Return True if we cache the descriptors on file."""
        return (self._cache_file is not None)

    @property
    def cache_file(self):
        return self._cache_file

    @cache_file.setter
    def cache_file(self, val):
        if self.cached:
            # we already have a cache, so copy it
            self._create_h5py_cache(val, copy_from=self._cache)
            self._cache_file = val
        else:
            # need to copy the in memory numpy cache to h5py
            # get a ref to descriptors and counts
            descriptors = self._descriptors[:self._fill_pointer]
            counts = self._counts[:self._fill_pointer]
            # create cache, also replaces self._descriptors
            self._create_h5py_cache(val)
            self._cache_file = val
            self._fill_pointer = 0  # reset fill pointer so we can use append
            self.append(tra_descriptors=descriptors, multiplicity=counts)

    def _create_h5py_cache(self, cache_file, copy_from=None):
        # the next line looks weird, but ensures the user can pass either
        # arcd storages or h5py files directly
        cache_file = cache_file.file
        if cache_file.mode == 'r':
            # file open in read only mode
            # TODO(?): for now we just make them available, any appending will fail
            #          we can change the cachefile as self.cache_file = new_file
            logger.warn("arcd storage passed as density collector cache file "
                        + "is open in read-only mode. "
                        + "No appending will be possible.")
            # copy_from can be None, but we don't care since we check
            # self._fill_pointer every time before trying to copy
            # (if nothing is there we don't try to copy)
            self._cache = copy_from
        else:
            # file open in write/append modes: copy from storage to cache area
            id_str = str(id(self))
            # we keep the path to the cache at a central location
            # and ONLY ONCE, so it is probably best to have them defined in
            # storage.py (?) as constants and import them from there
            traDC_cache_grp = cache_file.require_group(
                                        Storage.h5py_path_dict["tra_dc_cache"]
                                                       )
            if copy_from is None:
                # nothing to copy, create empty group
                self._cache = traDC_cache_grp.create_group(id_str)
            else:
                # copy saved stuff to cache while creating
                cache_file.copy(copy_from, traDC_cache_grp, name=id_str)
                self._cache = traDC_cache_grp[id_str]
        # same for all file modes
        if self._fill_pointer > 0:
            # we can only access if we already have something in the datasets
            # (since we create them on first append)
            self._descriptors = self._cache["descriptors"]
            self._counts = self._cache["counts"]
        else:
            # we set to None to (try to) create the datsets the first time we access
            # creation will fail for read-only arcd storages
            self._descriptors = None
            self._counts = None

    def object_for_pickle(self, group, overwrite=True):
        if self._cache_file is None:
            # for now just pickle the internal cache numpy arrays
            # (we could also write them to hdf5 as we do below)
            return self
        else:
            self._cache.copy(".", group, name="TrajectoryDensityCollector")
            state = self.__dict__.copy()
            state["_cache_file"] = "enabled"
            state["_cache"] = None
            state["_descriptors"] = None
            state["_counts"] = None
            ret_obj = self.__class__.__new__(self.__class__)
            ret_obj.__dict__.update(state)
            return ret_obj

    def complete_from_h5py_group(self, group):
        if self._cache_file == "enabled":
            # restore cache
            self._cache_file = group.file
            self._create_h5py_cache(self._cache_file,
                                    copy_from=group["TrajectoryDensityCollector"],
                                    )
        return self

    def __del__(self):
        if self._cache is not None:
            # delete cache h5py group
            del self._cache.parent[self._cache.name]

    def _extend_if_needed_cached(self, tra_len, descriptor_dim, add_entries=4000):
        """Extend cache if next trajectory would not fit."""
        # make sure we always make space for the whole tra
        add = max((add_entries, tra_len))
        if self._descriptors is None:
            # create h5py datasets
            self._descriptors = self._cache.create_dataset(
                                            name="descriptors",
                                            shape=(add, descriptor_dim),
                                            maxshape=(None, descriptor_dim),
                                            dtype="f",
                                                           )
            self._counts = self._cache.create_dataset(
                                            name="counts",
                                            shape=(add,),
                                            maxshape=(None,),
                                            dtype="i8",
                                                      )
        else:
            # possibly extend existing datasets
            shadow_len = self._counts.shape[0]
            if shadow_len <= self._fill_pointer + tra_len:
                # extend
                self._counts.resize(shadow_len + add, axis=0)
                self._descriptors.resize(shadow_len + add, axis=0)

    def _extend_if_needed(self, tra_len, descriptor_dim, add_entries=2000):
        """Extend internal storage arrays if next TP would not fit."""
        shadow_len = self._descriptors.shape[0]
        # make sure we always make space for the whole tra
        add = max((add_entries, tra_len))
        if shadow_len == 0:
            # first creation of the arrays
            self._counts = np.zeros((add,), dtype=np.float64)
            self._descriptors = np.zeros((add, descriptor_dim),
                                         dtype=np.float64)
        elif shadow_len <= self._fill_pointer + tra_len:
            # extend by at least the tra_len
            self._counts = np.concatenate((self._counts,
                                           np.zeros((add,), dtype=np.float64))
                                          )
            self._descriptors = np.concatenate(
                    (self._descriptors,
                     np.zeros((add, descriptor_dim), dtype=np.float64))
                                               )

    def _append(self, tra_descriptors, counts_arr):
        # we expect tra_descriptors to be of shape (n_points, descriptor_dim)
        # and counts_arr to be of shape (n_points,)
        tra_len, descriptor_dim = tra_descriptors.shape
        if self._cache_file is None:
            self._extend_if_needed(tra_len=tra_len,
                                   descriptor_dim=descriptor_dim,
                                   )
        else:
            self._extend_if_needed_cached(tra_len=tra_len,
                                          descriptor_dim=descriptor_dim,
                                          )
        self._descriptors[self._fill_pointer:self._fill_pointer+tra_len] = tra_descriptors
        self._counts[self._fill_pointer:self._fill_pointer+tra_len] = counts_arr
        self._fill_pointer += tra_len

    def append(self, tra_descriptors, multiplicity=1):
        """
        Append trajectory descriptors to internal cache.

        Parameters:
        -----------
        tra_descriptors - numpy.array
        multiplicity - int (default=1), weight for trajectory in ensemble,
                       can also be 1d numpy.array with len=len(tra_descriptors)
        """
        if isinstance(multiplicity, (int, np.int)):
            multiplicity = np.full((tra_descriptors.shape[0]), multiplicity)
        self._append(tra_descriptors, multiplicity)

    def add_density_for_trajectories(self, model, trajectories, counts=None):
        """
        Evaluate the density on the given trajectories.

        Only **add** the counts for the added trajectories according to the
        current models predictions to the existing histogram in probability
        space.
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
        len_before = self._fill_pointer
        # add descriptors to self
        if counts is None:
            counts = len(trajectories) * [1.]
        for tra, c in zip(trajectories, counts):
            descriptors = model.descriptor_transform(tra)
            self.append(tra_descriptors=descriptors, multiplicity=c)
        # now predict for the newly added
        pred = model(self._descriptors[len_before:self._fill_pointer],
                     use_transform=False)
        histo, edges = np.histogramdd(sample=pred,
                                      bins=self.bins,
                                      range=[[0., 1.]
                                             for _ in range(self.n_dim)],
                                      weights=self._counts[len_before:self._fill_pointer]
                                      )
        # and add to self.histogram
        self.density_histogram += histo

    def reevaluate_density_add_trajectories(self, model, trajectories, counts=None):
        """
        Revaluate the density for all stored trajectories using the current
        models predictions **after** adding the given trajectories to store
        for later reevaluation.

        See self.reevaluate_density() to only recreate the density estimate
        without adding new trajectories.
        See self.add_density_for_trajectories() to only update the estimate for
        the added trajectories without recreating the complete estimate.

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
            self.append(tra_descriptors=descriptors, multiplicity=c)
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

    def get_counts(self, probabilities):
        """
        Return the current counts in bin for a given probability vector.

        Parameters:
        -----------
        probabilities - numpy.ndarray, shape=(n_points, self.n_dim)

        Returns:
        --------
        counts - numpy.ndarray, shape=(n_points,), values of the density
                 counter at the given points in probability-space

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
        dens = self.get_counts(probabilities)
        norm = np.sum(self.density_histogram)
        return (norm + self._n_allowed_bins) / (dens + 1)
