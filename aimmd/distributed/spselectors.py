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
import abc
import asyncio
import logging
import numpy as np

from asyncmd import Trajectory
from asyncmd.trajectory.convert import FrameExtractor


logger = logging.getLogger(__name__)


class SPSelector(abc.ABC):
    """Abstract base class for shooting point selectors."""
    probability_is_ensemble_weight = False

    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    @abc.abstractmethod
    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          **kwargs) -> float:
        """
        Return the proposal probability for the given snapshot.
        """
        # For SPSelectors that draw a SP from an in-trajectory this method
        # should be called with (snapshot, trajectory)
        # For SPSelectors that draw from an ensemble of SPs this function also
        # needs the trajectory this snapshot now belongs to, otherwise we can
        # not calculate the ensemble weight for the newly generated trajectory
        raise NotImplementedError

    @abc.abstractmethod
    async def pick(self, outfile: str, frame_extractor: FrameExtractor,
                   **kwargs) -> Trajectory:
        """Pick and return a snapshot to shot from."""
        # For SPSelectors that draw from an in-trajectory this method should
        # take a trajectory
        # For SPSelectors that draw from a predefined ensemble this does not
        # necessarily need any arguments except the output path for the drawn
        # snapshot
        raise NotImplementedError


# TODO: DOCUMENT
class UniformSPSelector(SPSelector):
    """Select shooting points uniformly from the given in-trajectory."""

    def __init__(self, exclude_frames: int = 0) -> None:
        """
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
                          **kwargs) -> float:
        # TODO? we currently dont check if the snapshot is in Trajectory, i.e.
        #       if we could even pick this snapshot from the Trajectory
        traj_len = len(trajectory)
        if traj_len <= 2 * self.exclude_frames:
            # can not pick a frame if the trajectory is too short!
            return 0.
        return 1 / (traj_len - 2 * self.exclude_frames)

    async def pick(self, outfile: str, frame_extractor: FrameExtractor,
                   trajectory: Trajectory, **kwargs) -> Trajectory:
        traj_len = len(trajectory)
        if traj_len <= 2 * self.exclude_frames:
            raise ValueError("Can not pick a SP from a trajectory of len "
                             f"{traj_len} while excluding {self.exclude_frames}"
                             " at each end.")
        chosen_idx = self._rng.integers(traj_len - 2 * self.exclude_frames)
        logger.debug("Selected SP with idx %d on trajectory %s of len %d.",
                     chosen_idx, trajectory, len(trajectory))
        # get the frame from the trajectory and write it out
        snapshot = await frame_extractor.extract_async(outfile=outfile,
                                                       traj_in=trajectory,
                                                       idx=chosen_idx)
        return snapshot


# TODO: Document!
# TODO: (finish) make a superclass for both/all RCModelSelectors
#       we should try to move all possible methods to the superclass
class RCModelSPSelector(SPSelector):
    """
    Select SPs biased towards the current best guess of the transition state of
    the RCmodel.
    """
    def __init__(self, scale: float, distribution: str,
                 density_adaptation: bool = True) -> None:
        super().__init__()
        self.distribution = distribution
        self.scale = scale
        self.density_adaptation = density_adaptation

    @property
    def distribution(self):
        """Return the name of the shooting point selection distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, val):
        if val.lower() == 'gaussian':
            self._f_sel = self._gaussian
            self._distribution = val
        elif val.lower() == 'lorentzian':
            self._f_sel = self._lorentzian
            self._distribution = val
        else:
            raise ValueError('Distribution must be one of: '
                             + '"gaussian" or "lorentzian"')

    def _lorentzian(self, z):
        return self.scale / (self.scale**2 + z**2)

    def _gaussian(self, z):
        return np.exp(-z**2/self.scale)

    async def f(self, snapshot, trajectory, model):
        """
        Return the unnormalized proposal probability of snapshot from trajectory.
        """
        # we expect that 'snapshot' is a len 1 trajectory!
        z_sel = await model.z_sel(snapshot)
        any_nan = np.any(np.isnan(z_sel))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         'We used np.nan_to_num to proceed')
            z_sel = np.nan_to_num(z_sel)
        # casting to float makes errors when the np-array is not size-1,
        # i.e. we check that snapshot really was a len-1 trajectory
        ret = float(self._f_sel(z_sel))
        if self.density_adaptation:
            committor_probs = await model(snapshot)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = model.density_collector.get_correction(
                                                            committor_probs
                                                                  )
            ret *= float(density_fact)
        if ret == 0.:
            if await self.sum_bias(trajectory) == 0.:
                logger.error("All SP weights are 0. Using equal probabilities.")
                return 1.
        return ret

    async def sum_bias(self, trajectory, model):
        """
        Return the partition function of proposal probabilities for trajectory.
        """
        biases = await self.biases(trajectory, model)
        return np.sum(biases)

    async def biases(self, trajectory, model):
        z_sels = await model.z_sel(trajectory)
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        ret = self._f_sel(z_sels)
        if self.density_adaptation:
            committor_probs = await model(trajectory)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = model.density_collector.get_correction(
                                                            committor_probs
                                                                  )
            ret *= density_fact
        return ret


# TODO: DOCUMENT
class RCModelSPSelectorFromTraj(RCModelSPSelector):
    """
    Select SPs from a given in-trajectory (usualy a TP).

    Selection is biased towards the current best guess of the transition state
    of the RCModel.
    """
    def __init__(self, scale: float = 1., distribution: str = "lorentzian",
                 density_adaptation: bool = True,
                 exclude_first_last_frame: bool = True) -> None:
        super().__init__(scale=scale,
                         distribution=distribution,
                         density_adaptation=density_adaptation,
                         )
        # whether we allow to choose first and last frame
        # if False they will also not contribute to sum_bias and accept/reject
        self.exclude_first_last_frame = exclude_first_last_frame

    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          model, **kwargs):
        """Return proposal probability of the snapshot for this trajectory."""
        # we expect that 'snapshot' is a len 1 trajectory!
        sum_bias, f_val = await asyncio.gather(
                                          self.sum_bias(trajectory, model),
                                          self.f(snapshot, trajectory, model),
                                               )
        if sum_bias == 0.:
            logger.error("All SP weights are 0. Using equal probabilities.")
            if self.exclude_first_last_frame:
                return 1. / (len(trajectory) - 2)
            else:
                return 1. / len(trajectory)
        return f_val / sum_bias

    async def biases(self, trajectory, model):
        ret = await super().biases(trajectory=trajectory, model=model)
        if self.exclude_first_last_frame:
            ret = ret[1:-1]
        return ret

    async def pick(self, outfile: str, frame_extractor: FrameExtractor,
                   trajectory: Trajectory, model) -> Trajectory:
        """
        Pick a shooting snapshot from given trajectory.

        Parameters
        ----------
        outfile : str
            Path to write out the selected snapshot.
        frame_extractor : FrameExtractor
            The framextractor to use when getting the snapshot from the
            trajectory, e.g. RandomVelocities for TwoWayShooting.
        trajectory : Trajectory
            The input trajectory from which we select the snapshot.
        model : aimmd.RCModel
            The RCModel that is used to bias the selection towards the current
            best guess of the transition state.

        Returns
        -------
        Trajectory
            The selected shooting snapshot.
        """
        # NOTE: this does not register the SP with model!
        #       i.e. we do stuff different than in the ops selector
        #       For the distributed case we need to save the predicted
        #       commitment probabilities at the shooting point with the MCStep,
        #       such that we can make sure that they are added to the central
        #       RCmodel in the same order as the shooting results are added to
        #       the trainset
        biases = await self.biases(trajectory, model)
        sum_bias = np.sum(biases)
        if sum_bias == 0.:
            logger.error('Model not able to give educated guess.\
                         Choosing based on luck.')
            # we can not give any meaningfull advice and choose at random
            if self.exclude_first_last_frame:
                # choose from [1, len(traj) - 1 )
                return self._rng.integers(1, len(trajectory) - 1)
            else:
                # choose from [0, len(traj) )
                return self._rng.integers(len(trajectory))

        # if self.exclude_first_last_frame == True
        # biases will be have the length of traj - 2,
        # i.e. biases already excludes the two frames
        # that means the idx we choose here is shifted by one in that case,
        # e.g. idx=0 means frame_idx=1 in the trajectory
        # (and we can not choose the last frame because biases ends before)
        rand = self._rng.random() * sum_bias
        idx = 0
        prob = biases[0]
        while prob <= rand and idx < len(biases) - 1:
            idx += 1
            prob += biases[idx]
        if self.exclude_first_last_frame:
            idx += 1
        logger.debug("Selected SP with idx %d on trajectory %s of len %d.",
                     idx, trajectory, len(trajectory))
        # get the frame from the trajectory and write it out
        snapshot = await frame_extractor.extract_async(outfile=outfile,
                                                       traj_in=trajectory,
                                                       idx=idx)
        return snapshot


# TODO: DOCUMENT!
class RCModelSPSelectorFromEQ(RCModelSPSelector):
    probability_is_ensemble_weight = True
    # select SPs from a biased EQ distribtion so we can shoot in parallel
    # and get a weight for every mcstep (instead of an accept/reject)
    def __init__(self, trajectories: list[Trajectory],
                 equilibrium_weights: list[np.ndarray],
                 scale: float = 1,
                 distribution: str = "lorentzian",
                 density_adaptation: bool = True) -> None:
        # NOTE: we expect equilibrium_weight to be the multiplicative factor by
        #       which the ensemble weight for the structure differs from
        #       equilibrium, e.g.:
        #       -> all ones if the structures come from an equilibrium traj
        #       -> exp(\beta V_{bias}(x)) for each structure from a biased
        #          sampling run
        #       -> the equilibrium weights from binless WHAM if from multiple
        #          biased sampling runs
        super().__init__(scale=scale,
                         distribution=distribution,
                         density_adaptation=density_adaptation,
                         )
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]
        if isinstance(equilibrium_weights, np.ndarray):
            equilibrium_weights = [equilibrium_weights]
        # this looks a bit weired but we always concatenate the equilibrium
        # weights to one large array, if it was just one bevore nothing changes
        equilibrium_weights = np.concatenate(equilibrium_weights, axis=0)
        # make sure trajectories and equilibrium_weights match (at least in length)
        if equilibrium_weights.shape[0] != sum(len(t) for t in trajectories):
            raise ValueError("Mismatch between trajectory and equilibrium_weights lengths!")
        self.trajectories = trajectories
        self.equilibrium_weights = equilibrium_weights

    # TODO: is this correct?!
    #       we normalize by using the sum of biases from the reservoir ensemble
    #       (i.e. from self.trajectories)
    #       I think we must normalize to get [\sum_i p_bias(x_i)]^-1 correct,
    #       if we do not normalize we run into trouble when the network
    #       predictions change (and with it the selection biases and Z_bias)
    #       (I [hejung] think this is the best we can do because it assumes
    #        the new points (on the trajectory) come from the reservoir
    #        ensemble and are already included in the normalizing sum [Z_bias])
    async def probability(self, snapshot: Trajectory, trajectory: Trajectory,
                          model, **kwargs) -> float:
        # get bias according to current model estimate of TSE
        all_biases = await asyncio.gather(*(self.biases(trajectory=t,
                                                        model=model)
                                            for t in self.trajectories + [trajectory])
                                          )
        biases_for_SP_ensemble = np.concatenate(all_biases[:-1], axis=0)
        biases_for_new_traj = all_biases[-1]
        Z_bias = np.sum(biases_for_SP_ensemble)
        return Z_bias / np.sum(biases_for_new_traj)

    async def pick(self, outfile: str, frame_extractor: FrameExtractor, model,
                   **kwargs) -> Trajectory:
        # get bias according to current model estimate of TSE
        all_biases = await asyncio.gather(*(self.biases(trajectory=t,
                                                        model=model)
                                            for t in self.trajectories)
                                          )
        all_biases = np.concatenate(all_biases, axis=0)
        # get effective weight (which corrects for possible non-EQ frequency of
        # configurations in self.trajectories)
        effective_weights = all_biases * self.equilibrium_weights
        # now pick an index (in the full ensemble) according to effective weight
        index = self._rng.choice(effective_weights.shape[0], size=None,
                                 p=effective_weights/np.sum(effective_weights),
                                 )
        # find the right trajectory in self.trajectories
        traj_index = 0
        while index >= len(self.trajectories[traj_index]):
            # and we directly find the correct frame index in it too...
            index -= len(self.trajectories[traj_index])
            traj_index += 1
        # and write out the choosen snapshot using frame_extractor
        logger.debug("Selected SP with idx %d on trajectory %s.",
                     index, self.trajectories[traj_index])
        # get the frame from the trajectory and write it out
        snapshot = await frame_extractor.extract_async(
                                        outfile=outfile,
                                        traj_in=self.trajectories[traj_index],
                                        idx=index,
                                                       )
        return snapshot
