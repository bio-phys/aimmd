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
This file contains the helper classes to perform collection of (and correction for)
the density of configurations on trajectories when projected to the selection coordinate.

In the standard case we will attempt to flatten the distribution of (shooting)
configurations along the committor.
"""
import asyncio
import inspect
import logging
import typing

import numpy as np

from ..tools import is_documented_by as _is_documented_by

if typing.TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


logger = logging.getLogger(__name__)


class DensityCollector:
    """
    Keep track of density of configurations on trajectories projected to probabilities.
    """

    def __init__(self, n_dim: int, bins: int = 10) -> None:
        """
        Initialize a :class:`DensityCollector`.

        Parameters
        ----------
        n_dim : int
            The number of dimensions the probability space has in which we are
            histograming.
        bins : int, optional
            The number of bins to use for each probability direction, by default 10.
        """
        self.n_dim = n_dim
        self.bins = bins
        self.density_histogram = np.zeros(tuple(bins for _ in range(n_dim)))
        self._trajectories = []
        self._weights = []
        self._n_samp = 0

        # need to know the number of forbidden bins, i.e. where sum_prob > 1
        bounds = np.arange(0., 1., 1./bins)
        # this magic line creates all possible combinations of probs
        # as an array of shape (bins**n_probs, n_probs)
        combs = np.array(np.meshgrid(*tuple(bounds for _ in range(self.n_dim)))
                         ).T.reshape(-1, self.n_dim)
        sums = np.sum(combs, axis=1)
        n_forbidden_bins = len(np.where(sums > 1)[0])
        self._n_allowed_bins = bins**self.n_dim - n_forbidden_bins

    def reset(self) -> None:
        """
        Reset everything stored in self (density histogram, trajectories and weights).
        """
        self.density_histogram = np.zeros(tuple(self.bins for _ in range(self.n_dim)))
        self._trajectories = []
        self._weights = []
        self._n_samp = 0

    def _check_for_async(self, model) -> bool:
        """
        Check if the given model is async.

        Parameters
        ----------
        model : RCModel
            The model to test

        Returns
        -------
        bool
            Whether the models __call__ method is async.
        """
        return (inspect.iscoroutinefunction(model.__call__)
                # for 'real' models we should not need the second check, but to be sure
                or inspect.iscoroutinefunction(model)
                )

    def add_density_for_trajectories(self, model, trajectories, weights=None):
        """
        Add the density of the given trajectories to density histogram.

        Adds the weights for the added trajectories according to the given models
        predictions to the existing histogram in probability space.
        Additionally stores trajectories/descriptors for later reevaluation.

        Parameters
        ----------
        model : aimmd.base.RCModel
            The model to use for predicting commitment probabilities
        trajectories : Iterable[Trajectory]
            The trajectories to evaluate.
        weights : None | list[float | np.ndarray]
            The weights to use for the trajectories. If None every configuration
            on every trajectory will get a weight of one. If a single float per
            trajectory, each configuration on the corresponding trajectory will
            get a weight corresponding to the value. Can also be a list of np.arrays
            (each of the same length as the corresponding trajectory), i.e. supports
            different weights per configuration.
        """
        if self._check_for_async(model=model):
            raise ValueError(
                "An async model was passed, but this method is not async."
                " See method ``add_density_for_trajectories_async``"
                )
        weights = self._sanitize_weights_for_trajectories(trajectories=trajectories,
                                                          weights=weights)
        # now predict for the newly added
        preds = [model(tra) for tra in trajectories]
        for pred, tra, w in zip(preds, trajectories, weights):
            # add to histogram
            histo, _ = np.histogramdd(sample=pred,
                                      bins=self.bins,
                                      range=[(0., 1.)
                                             for _ in range(self.n_dim)],
                                      weights=w,
                                      )
            self.density_histogram += histo
            self._n_samp += pred.shape[0]
            # store trajectories and weights for reevaluation
            self._trajectories.append(tra)
            self._weights.append(w)

    @_is_documented_by(add_density_for_trajectories)
    # pylint: disable-next=missing-function-docstring
    async def add_density_for_trajectories_async(self, model, trajectories,
                                                 weights=None) -> None:
        if not self._check_for_async(model=model):
            raise ValueError(
                "A non-async model was passed, but this method is async."
                " See method ``add_density_for_trajectories``"
                )
        weights = self._sanitize_weights_for_trajectories(trajectories=trajectories,
                                                          weights=weights)
        # now predict for the newly added
        preds = await asyncio.gather(*(model(tra) for tra in trajectories))
        for pred, tra, w in zip(preds, trajectories, weights):
            # add to histogram
            histo, _ = np.histogramdd(sample=pred,
                                      bins=self.bins,
                                      range=[(0., 1.)
                                             for _ in range(self.n_dim)],
                                      weights=w,
                                      )
            self.density_histogram += histo
            self._n_samp += pred.shape[0]
            # store trajectories and weights for reevaluation
            self._trajectories.append(tra)
            self._weights.append(w)

    def _sanitize_weights_for_trajectories(self, trajectories: list, weights: list | None,
                                           ) -> "list[npt.NDArray]":
        """
        Ensure that weights has the correct/expected shape for each trajectory.

        The correct/expected shape is a np.NDArray of shape=(len(traj),) for each
        trajectory.

        Parameters
        ----------
        trajectories : list
            A list of trajectories for which the weights should match.
        weights : list | None
            The corresponding list of weights to sanitize. Can be None in which
            case every configuration in every Trajectory will get an equal weight.
            Can also be one weight for a whole Trajectory, in which case each
            configuration in that Trajectory will get this value as weight.

        Returns
        -------
        list[npt.NDArray]
            A list with sanitize weights of matching/expected shape.
        """
        if weights is None:
            # give each point an equal weight of 1
            weights = [np.ones(shape=(len(t), )) for t in trajectories]
        else:
            for i, (w, traj) in enumerate(zip(weights, trajectories)):
                # if weights is one weight for the whole trajectory:
                # make it an array of the correct length
                if isinstance(w, (int, np.integer, float, np.floating)):
                    weights[i] = np.full((len(traj), ), w)
        return weights

    def reevaluate_density(self, model) -> None:
        """
        Reevaluate the density for all stored trajectories.

        Will replace the density histogram with a new density estimate for all
        trajectories using the given models prediction.

        Parameters
        ----------
        model : aimmd.base.RCModel
            The model to use fpr predicting commitment probabilities.
        """
        if self._check_for_async(model=model):
            raise ValueError(
                "An async model was passed, but this method is not async."
                " See method ``reevaluate_density_async``"
                )
        # keep a ref to current trajs and weights, then reset self
        trajs, weights = self._trajectories, self._weights
        self.reset()
        # and finally (re)add all trajectories using the current model
        self.add_density_for_trajectories(model=model, trajectories=trajs,
                                          weights=weights)

    @_is_documented_by(reevaluate_density)
    # pylint: disable-next=missing-function-docstring
    async def reevaluate_density_async(self, model):
        if not self._check_for_async(model=model):
            raise ValueError(
                "A non-async model was passed, but this method is async."
                " See method ``reevaluate_density``"
                )
        # keep a ref to current trajs and weights, then reset self
        trajs, weights = self._trajectories, self._weights
        self.reset()
        # and finally (re)add all trajectories using the current model
        await self.add_density_for_trajectories_async(model=model,
                                                      trajectories=trajs,
                                                      weights=weights,
                                                      )

    def get_counts(self, probabilities):
        """
        Return the current counts/density values for a given probability vector.

        Parameters
        ----------
        probabilities : numpy.ndarray
            The commitment probabilities, with shape=(n_points, self.n_dim), for
            which density values will be returned.

        Returns
        -------
        counts : numpy.ndarray, shape=(n_points,)
            Values of the density counter at the given points in probability-space.
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

        The factor is calculated as total_count / counts[probabilities], i.e.
        the factor is proportional to 1 / rho[probabilities].

        Note that we use a variant of add-one-smoothing in case we encounter infinite
        values due to zero density in a bin.
        In case we encounter an infinity we replace the value in that bin by
        (norm + n_bins * weight_1_samp) / weight_1_samp,
        where norm is the sum of the density in all bins, n_bins is the total
        number of (allowed) bins in each probability direction combined, and
        weight_1_samp is the average weight per sample.

        Parameters
        ----------
        probabilities : np.ndarray
            The commitment probabilities, with shape=(n_points, self.n_dim),
            for which correction factor values are to be returned.
        """
        dens = self.get_counts(probabilities)
        if not (norm := np.sum(self.density_histogram)):
            return np.ones_like(dens)  # we dont have any density yet
        with np.errstate(divide="ignore"):
            # ignore errors through division by zero
            factor = norm / dens
        # Now replace all potential infs by a large (but finite) value
        if np.any(np.isinf(factor)):
            # weight_1_samp is the average weight per sample configuration
            weight_1_samp = norm / self._n_samp
            # the bit below is similar to laplace add-one smoothing, just
            # that we use weight_1_samp instead of adding a one, because we
            # do not know if the average weight per sample \approx 1
            # [in laplace add-one smoothing we would use:
            #   (norm + self._n_allowed_bins) / (dens + 1)
            #  here dens = 0 (since we got the inf)]
            #  and self._n_allowed_bins becomes self._n_allowed_bins * weight_1_samp,
            #  i.e., we add one average sample into each allowed bin
            #  (as in laplace add-one smoothing)
            factor[factor == np.inf] = norm/weight_1_samp + self._n_allowed_bins

        return factor
