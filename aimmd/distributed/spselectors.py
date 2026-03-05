# This file is part of aimmd
#
# aimmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# aimmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with aimmd. If not, see <https://www.gnu.org/licenses/>.
"""
This file contains various shooting point selector classes.

They are used to select a shooting configuration either from an input trajectory
or a predefined reservoir of configurations (also trajectories, but with associated
weights for each frame).
"""
import abc
import asyncio
import logging
import collections.abc
import typing
import numpy as np
import numpy.typing as npt

from asyncmd import Trajectory
from asyncmd.trajectory.convert import FrameExtractor

from ..base.density_collection import DensityCollector
from .dataclasses import DensityAdaptionParameters
from .dataclasses_internal import PathSamplingSimStateInfo
from .utils import accepted_trajs_from_mcstep_collection

if typing.TYPE_CHECKING:  # pragma: no cover
    from ..base.rcmodel import RCModelAsyncMixin


logger = logging.getLogger(__name__)


class SPSelector(abc.ABC):
    """Abstract base class for shooting point selectors."""

    # NOTE: if this is True the value returned by :meth:`probability` is the
    #       (unnormalized) ensemble weight of the trajectory in the target
    #       ensemble instead of the picking probability for the snapshot from
    #       trajectory
    #       Returning the ensemble weight of the trajectory enables to accept
    #       every trial with a weight (as in [1]) and therefore avoid the MCMC
    #       chain needed in standard TPS. It allows shooting from an ensemble
    #       of SPs (instead of SPs picked from the last accepted TP), but it
    #       necessitates knowing the Boltzmann weight of every potential SP
    #       (or the factor by which the frequency of configurations in the SP
    #        ensemble differs from an Boltzmann ensemble)
    #       [1] Falkner et al, Mach. Learn. Sci. Technol., 2023,
    #           doi:10.1088/2632-2153/acf55c
    probability_is_ensemble_weight = False

    def __init__(self) -> None:
        """Initialize a `SPSelector`."""
        self._rng = np.random.default_rng()

    @abc.abstractmethod
    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          simstate_info: PathSamplingSimStateInfo, *,
                          model: "RCModelAsyncMixin",
                          ) -> float:
        """Return the proposal probability for the given snapshot."""
        # NOTE:
        # This function always needs to be called with (snapshot, trajectory),
        # for SPSelectors that depend on a model this method also needs a
        # RCModel (in its async form) to calculate the picking probability
        # (or ensemble weight)
        # For SPSelectors that draw a SP from an in-trajectory this method
        # should be called with (snapshot, trajectory) and returns the
        # probability to draw the given snapshot from trajectory
        # For SPSelectors that draw from an ensemble of SPs this function needs
        # the newly generated trajectory (and possibly the snapshot?),
        # otherwise we can not calculate the ensemble weight for the newly
        # generated trajectory
        raise NotImplementedError

    @abc.abstractmethod
    async def pick(self, outfile: str, frame_extractor: FrameExtractor,
                   simstate_info: PathSamplingSimStateInfo, *,
                   # we can use simstate_info, e.g. step_num for tuning the selection
                   # distribution over time or for knowing when to reset/reestimate density
                   model: "RCModelAsyncMixin",
                   trajectory: Trajectory | None = None,
                   ) -> Trajectory:
        """Pick and return a snapshot to shot from."""
        # NOTE:
        # For SPSelectors that draw from an in-trajectory this method should
        # take a trajectory
        # For SPSelectors that select biased according to a reaction coordinate
        # this method should take a RCModel (in its async form)
        # For SPSelectors that draw from a predefined ensemble this does not
        # necessarily need any arguments except the output path for the drawn
        # snapshot and the FrameExtractor to use
        # NOTE: In aimmd.distributed pick does not register the SP with model!
        #       I.e. we do stuff different than in the ops selector class.
        #       For the distributed case we need to save the predicted
        #       commitment probabilities at the shooting point with the MCStep,
        #       such that we can make sure that they are added to the central
        #       RCmodel in the same order as the shooting results are added to
        #       the trainset. They are therefore calculated in the mover and
        #       attached to the finished MCStep to be later added to the model
        #       and the trainset by the training BrainTask.
        raise NotImplementedError


class UniformSPSelector(SPSelector):
    """Select shooting points uniformly from the given in-trajectory."""

    def __init__(self, exclude_frames: int = 0) -> None:
        """
        Initialize a `UniformSPSelector`.

        Parameters
        ----------
        exclude_frames : int, optional
            How many frames to exclude from the selection at the start and end
            of the trajectory, e.g. if exclude_frames=2 we exclude the first
            and last 2 frames, by default 0
        """
        super().__init__()
        self.exclude_frames = exclude_frames

    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          simstate_info: PathSamplingSimStateInfo, *,
                          model: "RCModelAsyncMixin",
                          ) -> float:
        """
        Return proposal probability to pick (any) `snapshot` from `trajectory`.

        For the `UniformSPSelector` this just returns
        1 / (len(trajectory) - 2 * exclude_frames), because every snapshot is
        equally likely.

        Parameters
        ----------
        snapshot : Trajectory
            The snapshot in question.
        trajectory : Trajectory
            The trajectory to pick snapshot from.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            Completely ignored in uniform selection.

        Returns
        -------
        float
            The probability to pick any `snapshot` from `trajectory`.
        """
        # FIXME? we currently dont check if the snapshot is in Trajectory, i.e.
        #       if we could even pick this snapshot from the Trajectory
        if (traj_len := len(trajectory)) <= 2 * self.exclude_frames:
            # can not pick a frame if the trajectory is too short!
            return 0.
        return 1 / (traj_len - 2 * self.exclude_frames)

    async def pick(self, outfile: str, frame_extractor: FrameExtractor,
                   simstate_info: PathSamplingSimStateInfo, *,
                   model: "RCModelAsyncMixin",
                   trajectory: Trajectory | None = None,
                   ) -> Trajectory:
        """
        Pick and return a random snapshot from `trajectory`.

        Write out the picked snapshot using `frame_extractor` as `outfile`.

        Parameters
        ----------
        outfile : str
            Absolute or relative path to write the picked snapshot to.
        frame_extractor : FrameExtractor
            `asyncmd.FrameExtractor` subclass used to extract the chosen frame
            from `trajectory`.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            Completely ignored in uniform selection.
        trajectory : Trajectory
            The trajectory to pick the snapshot from, required although by
            default None.

        Returns
        -------
        Trajectory
            The chosen snapshot.

        Raises
        ------
        ValueError
            If no input `trajectory` is given to pick a snapshot from.
        ValueError
            If the input `trajectory` is to short to pick a snapshot from after
            discarding the first and last `exclude_frames` frames.
        """
        if trajectory is None:
            raise ValueError(
                "The `UniformSPSelector` requires an input trajectory to pick"
                " a snapshot from."
                             )
        if (traj_len := len(trajectory)) <= 2 * self.exclude_frames:
            raise ValueError(
                    f"Can not pick a SP from a trajectory of len {traj_len}"
                    f" while excluding {self.exclude_frames} at each end."
                             )
        chosen_idx = self._rng.integers(traj_len - 2 * self.exclude_frames)
        logger.debug("Selected SP with idx %d on trajectory %s of len %d.",
                     chosen_idx, trajectory, len(trajectory))
        # get the frame from the trajectory and write it out
        snapshot = await frame_extractor.extract_async(outfile=outfile,
                                                       traj_in=trajectory,
                                                       idx=chosen_idx)
        return snapshot


# pylint counts both, the "private" attributes (like _scale) and the corresponding
# properties (like scale), so we disable the warning
class RCModelSPSelector(SPSelector):  # pylint: disable=too-many-instance-attributes
    """
    Select SPs biased towards the transition state according to a RCModel.

    Baseclass for both, selection of SPs from a given in-trajectory (i.e. a TP)
    and selection of SPs from a predefined ensemble of configurations (i.e. an
    equilibrium ensemble). This class provides the functions to calculate the
    (unnormalized) selection biases for the configurations in trajectories
    according to the RCModels prediction under a given selection distribution.
    The input to the distribution is the models `z_sel`, which corresponds to
    the log-committor `q` in the two-state case, and is in any case always 0 at
    the transition state ensemble.
    It supports the use of density adaptation, i.e. it can optionally take into
    account a flattening factor correcting for the density of configurations
    along the predicted committor using various schemes. See also the
    :class:`DensityAdaptionParameters` for more.

    Note that this class is an ABC, i.e. it can not be instantiated because it
    is missing two required methods `pick_snapshot` and `probability`.
    """

    def __init__(self, scale: float, distribution: str, *,
                 density_adaptation_params: DensityAdaptionParameters | None = None,
                 f_sel: collections.abc.Callable | None = None,
                 ) -> None:
        """
        Initialize a `RCModelSPSelector`.

        Parameters
        ----------
        scale : float
            Scale of the distribution, smaller values result in a more peaked
            selection of shooting points around the predicted transition state.
            For the Lorentzian distribution scale = gamma,
            while for the Gaussian distribution scale = 2 * sigma**2.
        distribution : str
            A string indicating the distribution to use when selecting SPs
            around the transition state, can be "lorentzian", "gaussian",
            "uniform_phi", "uniform", or "custom" (see also `f_sel`).
        density_adaptation_params : DensityAdaptionParameters or None
            Parameters for density adaption, which includes an additional
            correction factor to flatten the density of potential SP configurations
            along the predicted committor. See also the :class:`DensityAdaptionParameters`
            for more.
            If set to `None`, no density adaption will be performed.
        f_sel : Callable or None, optional
            If given, sets the shooting point selection distribution function
            to any Callable taking a single np.array of shape (n_frames,) and
            returning an array of the same shape containing bias weights.
            Note that, in this case `distribution` will be ignored and set to
            "custom".
        """
        super().__init__()
        if f_sel is not None:
            self.f_sel = f_sel
        else:
            # the self.f_sel setter already sets self._distribution to "custom"
            self.distribution = distribution
        self.scale = scale
        self.density_adaptation_params = density_adaptation_params
        self.density_collector: DensityCollector | None = None
        self._last_density_collection = 0
        self._last_density_reevaluation = 0

    @property
    def distribution(self) -> str:
        """
        Return/set the shooting point selection distribution f_sel(z).

        Can be either "lorentzian", "gaussian", "uniform_phi", or "uniform".
        For "lorentzian" f_sel(z) = scale / (scale**2 + z**2).
        For "gaussian" f_sel(z) = exp(-z**2 / scale).
        For "uniform_phi" f_sel(z) = exp(-z) / (1 + exp(-z))**2 , resulting in
        a uniform selection weight along the committor phi = 1 / (1 + exp(-z)).
        For "uniform" f_sel(z) = 1, resulting in a selection that favours
        points close to the states, f_sel(phi) = 1 / (phi - phi**2).
        """
        return self._distribution

    @distribution.setter
    def distribution(self, val: str) -> None:
        if val.lower() == 'gaussian':
            self._f_sel = self._gaussian
            self._distribution = val
        elif val.lower() == 'lorentzian':
            self._f_sel = self._lorentzian
            self._distribution = val
        elif val.lower() == 'uniform_phi':
            self._f_sel = self._uniform_phi
            self._distribution = val
        elif val.lower() == 'uniform':
            self._f_sel = self._uniform
            self._distribution = val
        else:
            raise ValueError(
                'Distribution must be one of: "lorentzian", "gaussian", '
                '"uniform_phi", or "uniform".')

    def _lorentzian(self, z: npt.NDArray) -> npt.NDArray:
        return self.scale / (self.scale**2 + z**2)

    def _gaussian(self, z: npt.NDArray) -> npt.NDArray:
        return np.exp(-z**2 / self.scale)

    def _uniform_phi(self, z: npt.NDArray) -> npt.NDArray:
        # just the jacobian d phi / dz = d/dz 1 / (1 + exp(-z))
        # for 1d RCModels
        with np.errstate(over="ignore"):
            # avoid overflow errors from square, if we get it f_sel is zero
            # anyway
            ret = np.exp(-z) / (1 + np.exp(-z))**2
        # however we do not want the selection probability to be zero anywhere
        # so replace it with a small value
        return np.nan_to_num(ret, copy=False, nan=1e-50)

    def _uniform(self, z: npt.NDArray) -> npt.NDArray:
        # this is uniform in z, so f_sel(phi) = d z / dphi = 1 / (phi - phi**2)
        # for 1d RCModels
        return np.ones_like(z, dtype=z.dtype)

    @property
    def scale(self) -> float:
        """Return/set scale of the shooting point selection distribution."""
        return self._scale

    @scale.setter
    def scale(self, val: float) -> None:
        self._scale = float(val)

    @property
    def f_sel(self) -> collections.abc.Callable:
        """
        Return/set the shooting point selection distribution function directly.

        The function must take a single argument z (a np.array with
        shape=(n_frames,)) and return a np.array of the same shape as z
        containing the (unnormalized) bias weights.
        """
        return self._f_sel

    @f_sel.setter
    def f_sel(self, val: collections.abc.Callable) -> None:
        self._f_sel = val
        self._distribution = "custom"

    async def sum_bias(self, trajectory: Trajectory, model: "RCModelAsyncMixin",
                       exclude_frames: int = 0,
                       ) -> np.floating:
        """
        Return the sum of all selection biases for `trajectory` under `model`.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory over which the bias values are summed.
        model : RCModelAsyncMixin
            The reaction coordinate model used to calculate the biases.
        exclude_frames : int
            How many frames to exclude at the start and end of the trajectory,
            e.g., if exclude_frames=2 we exclude the first and last 2 frames,
            by default 0.

        Returns
        -------
        float
            The sum of all bias values on `trajectory` under `model`.
        """
        biases = await self.biases(trajectory, model,
                                   exclude_frames=exclude_frames,
                                   )
        return np.sum(biases)

    async def biases(self, trajectory: Trajectory, model: "RCModelAsyncMixin",
                     exclude_frames: int = 0,
                     ) -> npt.NDArray[np.floating]:
        """
        Return array with bias values for each configuration in `trajectory`.

        Note that the returned array is shorter than `trajectory` if
        `exclude_frames` > 0, because then the first and last `exclude_frames`
        frames will be discarded.

        Parameters
        ----------
        trajectory : Trajectory
            The trajectory to calculate bias values for.
        model : RCModelAsyncMixin
            The reaction coordinate model from which the biases are calculated.
        exclude_frames : int
            How many frames to exclude at the start and end of the trajectory,
            e.g., if exclude_frames=2 we exclude the first and last 2 frames,
            by default 0.

        Returns
        -------
        npt.NDArray[np.floating]
            One dimensional array (of the same length as `trajectory` - 2 * `exclude_frames`)
            with the bias values.
        """
        z_sels = await model.z_sel(trajectory)
        if (any_nan := np.any(np.isnan(z_sels))):
            logger.error("The model predicts NaNs. We used np.nan_to_num to proceed")
            z_sels = np.nan_to_num(z_sels)
        ret = self._f_sel(z_sels)
        if self.density_adaptation_params is not None:
            committor_probs = await model(trajectory)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = self.density_collector.get_correction(committor_probs)
            ret *= density_fact
        if exclude_frames > 0:
            ret = ret[exclude_frames:-exclude_frames]
        return ret

    async def pick(self, outfile: str, frame_extractor: FrameExtractor,
                   simstate_info: PathSamplingSimStateInfo, *,
                   model: "RCModelAsyncMixin",
                   trajectory: Trajectory | None = None,
                   ) -> Trajectory:
        # here we take care of adding trajectories to density adaption and
        # optionally update the density estimate from stored trajectories
        # using the current model
        await self._pre_pick(simstate_info=simstate_info, model=model)
        snap = await self.pick_snapshot(outfile=outfile, frame_extractor=frame_extractor,
                                        simstate_info=simstate_info, trajectory=trajectory,
                                        model=model,
                                        )
        return snap

    @abc.abstractmethod
    async def pick_snapshot(self, outfile: str, frame_extractor: FrameExtractor,
                            simstate_info: PathSamplingSimStateInfo, *,
                            model: "RCModelAsyncMixin",
                            trajectory: Trajectory | None = None,
                            ) -> Trajectory:
        """
        Subclasses must implement this to pick/select a shooting snapshot.

        Parameters
        ----------
        outfile : str
            The filename of the selected snapshot to write out.
        frame_extractor : FrameExtractor
            The FrameExtractor class that will be used to extract the selected
            snapshot.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            The committor model to use to select the snapshot.
        trajectory : Trajectory | None, optional
            An optional input trajectory to select the snapshot from, e.g. an
            input transition path. Can be None for example when selecting from
            a pregenerated reservoir of configurations, by default None.

        Returns
        -------
        Trajectory
            The selected shooting snapshot.
        """
        raise NotImplementedError

    async def _pre_pick(self, simstate_info: PathSamplingSimStateInfo,
                        model: "RCModelAsyncMixin",
                        ) -> None:
        """
        Perform needed operations for density adaption pre-pick.

        This method will reset or reevaluate the density estimate and add the
        given trajectory to the estimate, all depending on the settings in
        `density_adaption_params`.
        In case the density collector has not been initialized, this method will
        initialize it.

        Parameters
        ----------
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            The committor model used for this step.
        trajectory : Trajectory | None, optional
            The trajectory to add to the density estimate, usually this will be
            an input transition path. Can be None for example when selecting from
            a reservoir of configurations, by default None.
        """
        if self.density_adaptation_params is None:  # we do not do density adaption
            return
        if self.density_collector is None:  # setup DensityCollector (if we need one)
            self.density_collector = DensityCollector(
                                        n_dim=model.n_out,
                                        bins=self.density_adaptation_params.n_bins,
                                        )
            if len(self.density_adaptation_params.trajectories_to_flatten):
                await self.density_collector.add_density_for_trajectories_async(
                        model=model,
                        trajectories=self.density_adaptation_params.trajectories_to_flatten,
                        weights=self.density_adaptation_params.trajectories_to_flatten_weights,
                        )
        # NOTE:
        # Different density correction schemes are possible depending on the
        # combination of density_adaption_params
        # Possible options include:
        # 1.) By resetting the density (i.e. clearing it) and then adding only
        #     the trajectory we pick from, the density correction scheme is similar to
        #     the one from Lazzeri et al, https://pubs.acs.org/doi/10.1021/acs.jctc.3c00821
        #     Here the density of only the configurations we pick from is flattened and
        #     this will therefore be potentially more efficient then the next option
        # 2.) Not resetting and always adding the trajectories from pick we arrive at
        #     a density correction scheme similar to the one from the original aimmd
        #     publication, where the density of configurations on transition paths, p(x|TP),
        #     is flattened. In this case regular reevaluations using the current models
        #     predictions make sense.
        # 3.) Instantiating the density adaption with trajectories and (equilibrium) weights,
        #     e.g., the trajectories used as a shooting point "reservoir" for shooting from
        #     equilibrium configurations, this will flatten the density of configurations in
        #     the reservoir. Here also regular reevaluations of the stored density estimate
        #     using the current models predictions make sense (at least if the model changes
        #     over time)

        # first do any reset and recalculation of committor predictions for previously
        # stored trajectories (if we do so)
        if self.density_adaptation_params.reset_before_pick:
            self.density_collector.reset()
        if self.density_adaptation_params.reevaluate_density_interval is not None:
            # check if we do reevaluation at all (not None)
            # and if so check if it is a step_num (which will be the total number
            # of steps done so far in the sampler) at which we reevaluate
            if (
                simstate_info.step_num - self._last_density_reevaluation
                >= self.density_adaptation_params.reevaluate_density_interval
            ):
                await self.density_collector.reevaluate_density_async(model=model)
                self._last_density_reevaluation = simstate_info.step_num
        # now add trajectories (if we do so)
        if self.density_adaptation_params.add_trajectories_from_sampler:
            # get the accepted trajectories produced since the SP-selector last ran
            step_collection = simstate_info.brain.samplers[
                simstate_info.sampler_idx
                ].mcstep_collection
            trajectories, weights = accepted_trajs_from_mcstep_collection(
                                        mcstep_collection=step_collection,
                                        start=self._last_density_collection,
                                        )
            # and remember until where we have already added
            self._last_density_collection = len(step_collection)
            await self.density_collector.add_density_for_trajectories_async(
                                                model=model,
                                                trajectories=trajectories,
                                                weights=weights,
                                                )


class RCModelSPSelectorFromTraj(RCModelSPSelector):
    """
    Select SPs from a given in-trajectory (usually a TP) using a RCModel.

    SP selection is biased towards the current best guess of the transition
    state of the RCModel according to the models `z_sel` method. The SPs are
    selected according to bias weights according to `distribution` f_sel(z_sel)
    centered around z_sel = 0.
    """

    def __init__(
        self,
        scale: float = 1.0,
        distribution: str = "lorentzian",
        *,
        density_adaptation_params: (
            DensityAdaptionParameters | None
        ) = DensityAdaptionParameters(scheme="p_x_TP",
                                      reevaluate_density_interval=10),
        exclude_frames: int = 1,
        f_sel: collections.abc.Callable | None = None,
    ) -> None:
        """
        Initialize a `RCModelSPSelectorFromTraj`.

        Parameters
        ----------
        scale : float, optional
            Scale of the SP selection distribution, by default 1.
        distribution : str
            A string indicating the distribution to use when selecting SPs
            around the transition state, can be "lorentzian", "gaussian",
            "uniform_phi", "uniform", or "custom" (see also `f_sel`).
        density_adaptation_params : DensityAdaptionParameters or None
            Parameters for density adaption, which includes an additional
            correction factor to flatten the density of potential SP configurations
            along the predicted committor. See also the :class:`DensityAdaptionParameters`
            for more. If set to `None`, no density adaption will be performed.
            By default density adaption according to the density of points on
            transition paths, p(x|TP), will be performed.
        exclude_frames : int, optional
            How many frames to exclude from the selection at the start and end
            of the trajectory, e.g. if exclude_frames=2 we exclude the first
            and last 2 frames, by default 1.
        f_sel : Callable or None, optional
            If given, sets the shooting point selection distribution function
            to any Callable taking a single np.array of shape (n_frames,) and
            returning an array of the same shape containing bias weights.
            Note that, in this case `distribution` will be ignored and set to
            "custom".
        """
        super().__init__(scale=scale,
                         distribution=distribution,
                         density_adaptation_params=density_adaptation_params,
                         f_sel=f_sel,
                         )
        # whether we allow to choose first and last frame
        # if False they will also not contribute to sum_bias and accept/reject
        self.exclude_frames = exclude_frames

    @property
    def exclude_frames(self) -> int:
        """Return/set the number of excluded frames at the beginning and end."""
        return self._exclude_frames

    @exclude_frames.setter
    def exclude_frames(self, val) -> None:
        self._exclude_frames = int(val)

    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          simstate_info: PathSamplingSimStateInfo, *,
                          model: "RCModelAsyncMixin",
                          ) -> float:
        """
        Return proposal probability to pick `snapshot` from `trajectory`.

        For the `RCModelSPSelectorFromTraj` the proposal probability is equal
        to f(snap) / sum_bias(traj), where
        f(snap) = f_sel(z_sel_snap) * density_adapt
        and
        sum_bias(traj) is the sum over f(snap) for all snapshots in the
        trajectory, i.e. the normalizing sum.

        Parameters
        ----------
        snapshot : Trajectory
            The snapshot in question.
        trajectory : Trajectory
            The trajectory to pick `snapshot` from.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            The reaction coordinate model used to pick the snapshot.

        Returns
        -------
        float
            Proposal probability to pick `snapshot` from `trajectory`.
        """
        if len(trajectory) <= 2 * self.exclude_frames:
            # Trajectory is too short to select a snapshot from if we discard
            # exclude_frames at the start and end, so f = p = 0.
            return 0.
        sum_bias, f_val = await asyncio.gather(
                                        self.sum_bias(trajectory, model,
                                                      exclude_frames=self.exclude_frames),
                                        # dont exclude any frames for the bias of the snapshot
                                        self.biases(snapshot, model, exclude_frames=0),
                                        )
        # only scalar arrays can be cast to float, so this (also) ensures that
        # snapshot was indeed a 1-frame trajectory (i.e. had length=1)
        f_val = float(f_val)
        if not sum_bias:
            logger.error(
                    "All shooting point weights are 0."
                    " Using equal probability for each configuration."
                    )
            return 1. / (len(trajectory) - 2 * self.exclude_frames)
        return f_val / sum_bias

    async def pick_snapshot(self, outfile: str, frame_extractor: FrameExtractor,
                            simstate_info: PathSamplingSimStateInfo, *,
                            model: "RCModelAsyncMixin",
                            trajectory: Trajectory | None = None,
                            ) -> Trajectory:
        """
        Pick and return a shooting snapshot from `trajectory`.

        The selected snapshot is biased towards model.z_sel = 0 using the
        selection distribution. Write out the picked snapshot using
        `frame_extractor` as `outfile`.

        Parameters
        ----------
        outfile : str
            Path to write out the selected snapshot.
        frame_extractor : FrameExtractor
            The frame extractor to use when getting the snapshot from the
            trajectory, e.g. RandomVelocities for TwoWayShooting.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            The RCModel that is used to bias the selection towards the current
            best guess of the transition state.
        trajectory : Trajectory
            The input trajectory from which we select the snapshot.

        Returns
        -------
        Trajectory
            The selected shooting snapshot.

        Raises
        ------
        ValueError
            If the input `trajectory` is to short to pick a snapshot from after
            discarding the first and last `exclude_frames` frames.
        """
        if trajectory is None:
            raise ValueError(
                "The `RCModelSPSelectorFromTraj` needs a trajectory to pick a"
                " snapshot from.")
        if len(trajectory) <= 2 * self.exclude_frames:
            # Trajectory is too short to select a snapshot from if we discard
            # exclude_frames at the start and end
            raise ValueError(
                f"Can not pick a SP from a trajectory of len {len(trajectory)}"
                f" while excluding {self.exclude_frames} at each end."
                             )
        biases = await self.biases(trajectory, model,
                                   exclude_frames=self.exclude_frames,
                                   )
        if not (sum_bias := np.sum(biases)):
            logger.error("Model not able to give educated guess. Choosing at random.")
            # we can not give any meaningful advice and choose at random
            # choose in [self.exclude_frames, len(traj) - self.exclude_frames )
            idx = self._rng.integers(self.exclude_frames,
                                     len(trajectory) - self.exclude_frames,
                                     dtype=int,
                                     )
        else:
            # if self.exclude_frames > 0, biases will have the length of
            # traj - 2 * self.exclude_frames, i.e. biases already excludes the
            # excluded frames. That in turn means the idx we choose here is
            # shifted by self.exclude_frames in that case, e.g. idx=0 means
            # frame_idx=self.exclude_frames in the trajectory (and we can not
            # choose the last self.exclude_frames frames because biases ends
            # before)
            idx = self._rng.choice(len(biases), p=biases/sum_bias)
            idx += self.exclude_frames  # add in the exclude_frames shift

        logger.debug("Selected SP with idx %d on trajectory %s of len %d.",
                     idx, trajectory, len(trajectory))
        # get the frame from the trajectory and write it out
        snapshot = await frame_extractor.extract_async(outfile=outfile,
                                                       traj_in=trajectory,
                                                       idx=idx)
        return snapshot


class RCModelSPSelectorFromEQ(RCModelSPSelector):
    """
    Select SPs from an equilibrium ensemble biased using a RCModel.

    The equilibrium ensemble of shooting points is supplied as a list of
    trajectories with associated equilibrium weights.

    SP selection is biased towards the current best guess of the transition
    state of the RCModel according to the models `z_sel` method. The SPs are
    selected according to their equilibrium weight and biased according to
    `distribution` f_sel(z_sel) centered around z_sel = 0.

    Note that :meth:`probability` does not return the probability to pick a
    snapshot on a trajectory, but instead the (unnormalized) ensemble weight of
    the given trajectory. This fact is also indicated by the attribute
    self.probability_is_ensemble_weight being True.
    """

    probability_is_ensemble_weight = True

    def __init__(
        self,
        scale: float = 1.0,
        distribution: str = "lorentzian",
        *,
        trajectories: list[Trajectory],
        equilibrium_weights: list[npt.NDArray[np.floating]],
        density_adaption_params: (
            DensityAdaptionParameters | None
        ) = DensityAdaptionParameters(reevaluate_density_interval=10,
                                      reset_before_pick=False,
                                      add_trajectories_from_sampler=False,
                                      ),
        f_sel: collections.abc.Callable | None = None,
    ) -> None:
        """
        Initialize a `RCModelSPSelectorFromEQ`.

        Parameters
        ----------
        trajectories : list[Trajectory]
            A list with trajectories containing the potential shooting points,
            i.e. the ensemble of configurations. See also the notes below.
        equilibrium_weights : list[npt.NDArray[np.floating]]
            A list of np.arrays with the equilibrium weights associated with
            each configuration in `trajectories`. See also the notes below.
        scale : float, optional
            Scale of the SP selection distribution, by default 1.
        distribution : str
            A string indicating the distribution to use when selecting SPs
            around the transition state, can be "lorentzian", "gaussian",
            "uniform_phi", "uniform", or "custom" (see also `f_sel`).
        density_adaptation_params : DensityAdaptionParameters or None
            Parameters for density adaption, which includes an additional
            correction factor to flatten the density of potential SP configurations
            along the predicted committor. See also the :class:`DensityAdaptionParameters`
            for more. If set to `None`, no density adaption will be performed.
            Note that, `trajectories` and `equilibrium_weights` will both be set
            in the density_adaption_parameters as trajectories and weights.
            Additionally it is ensured that no new trajectories are added during
            pick. I.e., density adaption will always flatten the committor distribution
            observed in the potential shooting points.
        f_sel : Callable or None, optional
            If given, sets the shooting point selection distribution function
            to any Callable taking a single np.array of shape (n_frames,) and
            returning an array of the same shape containing bias weights.
            Note that, in this case `distribution` will be ignored and set to
            "custom".

        Raises
        ------
        ValueError
            If the number of configurations in `trajectories` does not match
            the number of equilibrium weights supplied.

        Notes
        -----
        `trajectories` can either be a list of Trajectory objects or a single
        Trajectory object, similarly `equilibrium_weights` can be either a list
        of numpy arrays or a single numpy array.
        `equilibrium_weights` is expected to contain the multiplicative factors
        by which the frequency of observation for each configuration/frame in
        `trajectories` differs from the equilibrium ensemble. The factors do
        not need to be normalized (as long as they share a common partition
        function). Naturally it is also possible to pass the normalized
        equilibrium probability for each configuration in `trajectories` as
        `equilibrium_weights`.
        Depending on the origin of the configurations in `trajectories`, one
        could e.g. use for `equilibrium_weights`:
          - all ones if the `trajectories` are equilibrium trajectories.
          - exp(beta V_{bias}(x)) for each structure from a biased sampling run
            if all configurations where generated using the same V_{bias}.
          - the equilibrium weights from binless WHAM if the configurations are
            from multiple biased sampling runs using different biasing
            potentials V_{bias}.
        """
        # NOTE: we expect equilibrium_weight to be the multiplicative factor by
        #       which the ensemble weight for the structure differs from
        #       equilibrium, e.g.:
        #       -> all ones if the structures come from an equilibrium traj
        #       -> exp(\beta V_{bias}(x)) for each structure from a biased
        #          sampling run
        #       -> the equilibrium weights from binless WHAM if from multiple
        #          biased sampling runs
        if density_adaption_params is not None:
            # Set trajectories and weights for density adaption
            density_adaption_params.trajectories_to_flatten = trajectories
            density_adaption_params.trajectories_to_flatten_weights = equilibrium_weights
            # ensure we dont reset so we dont loose the trajectories
            density_adaption_params.reset_before_pick = False
            # and ensure that we do not add the trajectories from previous steps in the sampler
            density_adaption_params.add_trajectories_from_sampler = False
        super().__init__(scale=scale,
                         distribution=distribution,
                         density_adaptation_params=density_adaption_params,
                         f_sel=f_sel,
                         )
        # concatenate the equilibrium weights into one array
        equilibrium_weights = np.concatenate(equilibrium_weights, axis=0)
        # make sure trajectories and equilibrium_weights match in length
        if equilibrium_weights.shape[0] != sum(len(t) for t in trajectories):
            raise ValueError(
                "Mismatch between trajectory and equilibrium_weights lengths!"
                             )
        self.trajectories = trajectories
        self.equilibrium_weights = equilibrium_weights

    # NOTE: We normalize by using the sum of biases from the reservoir ensemble
    #       (i.e. from self.trajectories)
    #       I think we must normalize to get [\sum_i p_bias(x_i)]^-1 correct,
    #       if we do not normalize we run into trouble when the network
    #       predictions change (and with it the selection biases and Z_bias)
    #       (I [hejung] think this is the best we can do because it assumes
    #        the new points (on the trajectory) come from the reservoir
    #        ensemble and are already included in the normalizing sum [Z_bias])
    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          simstate_info: PathSamplingSimStateInfo, *,
                          model: "RCModelAsyncMixin",
                          ) -> float:
        """
        Return the equilibrium ensemble weight for `trajectory`.

        Note that, this method does not return the probability to pick
        `snapshot` from `trajectory` (as for SPSelectors that pick the shooting
        snapshot from an input trajectory). It still needs to be called with a
        value for `snapshot` due to API consistency reasons.

        Parameters
        ----------
        snapshot : Trajectory
            Ignored, not needed to calculate ensemble weight of `trajectory`.
        trajectory : Trajectory
            The trajectory to calculate the ensemble weight for.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            The reaction coordinate model used to pick the snapshot.

        Returns
        -------
        float
            Ensemble weight for `trajectory`.
        """
        # get bias according to current model estimate of TSE
        all_biases = await asyncio.gather(
                                *(self.biases(trajectory=t,
                                              model=model)
                                  for t in self.trajectories + [trajectory]
                                  )
                                          )
        biases_for_sp_ensemble = np.concatenate(all_biases[:-1], axis=0)
        biases_for_new_traj = all_biases[-1]
        z_bias = np.sum(biases_for_sp_ensemble)
        return z_bias / np.sum(biases_for_new_traj)

    async def pick_snapshot(self, outfile: str, frame_extractor: FrameExtractor,
                            simstate_info: PathSamplingSimStateInfo, *,
                            model: "RCModelAsyncMixin",
                            trajectory: Trajectory | None = None,
                            ) -> Trajectory:
        """
        Pick and return a snapshot from the ensemble of potential snapshots.

        The selected snapshot is biased towards model.z_sel = 0 using the
        selection distribution. Write out the picked snapshot using
        `frame_extractor` as `outfile`.

        Parameters
        ----------
        outfile : str
            Path to write out the selected snapshot.
        frame_extractor : FrameExtractor
            The frame extractor to use when getting the snapshot from the
            trajectory, e.g. RandomVelocities for TwoWayShooting.
        simstate_info: PathSamplingSimStateInfo
            Dataclass carrying the current state of the pathsampling simulation
            and information about the current step, e.g. the step_num, workdir.
        model : RCModelAsyncMixin
            The RCModel that is used to bias the selection towards the current
            best guess of the transition state.
        trajectory : typing.Optional[Trajectory], optional
            Ignored, by default None. No trajectory is needed to pick a
            snapshot from. The argument is only retained to keep API
            compatibility with the other SPSelectors.

        Returns
        -------
        Trajectory
            The selected shooting snapshot.
        """
        # get bias according to current model estimate of TSE
        all_biases = await asyncio.gather(*(self.biases(trajectory=t,
                                                        model=model)
                                            for t in self.trajectories)
                                          )
        all_biases = np.concatenate(all_biases, axis=0)
        # get effective weight (which corrects for possible non-EQ frequency of
        # configurations in self.trajectories)
        effective_weights = all_biases * self.equilibrium_weights
        # pick an index (to the full ensemble) according to effective weight
        index = self._rng.choice(effective_weights.shape[0], size=None,
                                 p=effective_weights/np.sum(effective_weights),
                                 )
        # find the trajectory in self.trajectories that corresponds to index
        traj_index = 0
        while index >= len(self.trajectories[traj_index]):
            # and we directly find the correct frame index in it too...
            index -= len(self.trajectories[traj_index])
            traj_index += 1
        logger.debug("Selected SP with idx %d on trajectory %s.",
                     index, self.trajectories[traj_index])
        # write out the chosen snapshot using frame_extractor
        snapshot = await frame_extractor.extract_async(
                                        outfile=outfile,
                                        traj_in=self.trajectories[traj_index],
                                        idx=index,
                                                       )
        return snapshot
