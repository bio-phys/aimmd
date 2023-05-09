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


logger = logging.getLogger(__name__)


class SPSelector(abc.ABC):
    """Abstract base class for shooting point selectors."""
    def __init__(self) -> None:
        self._rng = np.random.default_rng()

    @abc.abstractmethod
    async def probability(self, snapshot, trajectory, model) -> float:
        """Return proposal probability of the snapshot for this trajectory."""
        raise NotImplementedError

    @abc.abstractmethod
    async def pick(self, trajectory, model) -> int:
        """Return the index of the chosen snapshot within trajectory."""
        raise NotImplementedError


# TODO: DOCUMENT
class RCModelSPSelector(SPSelector):
    """Select SPs biased towards current best guess of the transition state."""
    def __init__(self, scale=1., distribution="lorentzian",
                 density_adaptation=True, exclude_first_last_frame=True):
        super().__init__()
        self.distribution = distribution
        self.scale = scale
        self.density_adaptation = density_adaptation
        # whether we allow to choose first and last frame
        # if False they will also not contribute to sum_bias and accept/reject
        self.exclude_first_last_frame = exclude_first_last_frame

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
        """Return the unnormalized proposal probability of a snapshot."""
        # we expect that 'snapshot' is a len 1 trajectory!
        z_sel = await model.z_sel(snapshot)
        any_nan = np.any(np.isnan(z_sel))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         + 'We used np.nan_to_num to proceed')
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

    async def probability(self, snapshot, trajectory, model):
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

    async def sum_bias(self, trajectory, model):
        """
        Return the partition function of proposal probabilities for trajectory.
        """
        biases = await self._biases(trajectory, model)
        return np.sum(biases)

    async def _biases(self, trajectory, model):
        z_sels = await model.z_sel(trajectory)
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         + 'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        ret = self._f_sel(z_sels)
        if self.density_adaptation:
            committor_probs = await model(trajectory)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = model.density_collector.get_correction(
                                                            committor_probs
                                                                  )
            ret *= density_fact.reshape(committor_probs.shape)
        if self.exclude_first_last_frame:
            ret = ret[1:-1]
        return ret

    async def pick(self, trajectory, model):
        """Return the index of the chosen snapshot within trajectory."""
        # NOTE: this does not register the SP with model!
        #       i.e. we do stuff different than in the ops selector
        #       For the distributed case we need to save the predicted
        #       commitment probabilities at the shooting point with the MCStep,
        #       such that we can make sure that they are added to the central
        #       RCmodel in the same order as the shooting results are added to
        #       the trainset
        biases = await self._biases(trajectory, model)
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
        # and return chosen idx
        if self.exclude_first_last_frame:
            idx += 1
        return idx
