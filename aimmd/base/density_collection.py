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
import logging

import numpy as np


logger = logging.getLogger(__name__)


class DensityCollector:
    # TODO: expand docstring
    """
    Keep track of density of configurations on trajectories projected to probabilities.
    """

    def __init__(self, n_dim: int, bins: int = 10) -> None:
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
        # TODO: document this!
        self.density_histogram = np.zeros(tuple(self.bins for _ in range(self.n_dim)))
        self._trajectories = []
        self._weights = []
        self._n_samp = 0

    # TODO: cleanup everything below!

    def add_density_for_trajectories(self, model, trajectories, weights=None):
        # TODO: rewrite docstring when we know what this function does!
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
        model - aimmd.base.RCModel predicting commitment probabilities
        trajectories - iterator/iterable of trajectories to evaluate
        counts - None or list of weights for the trajectories,
                 i.e. we will add every trajectory with weight=counts,
                 if None, every trajectory has equal weight,
                 can also be a list of np.arrays (each of the same length as
                 the corresponding trajectory), i.e. supports different weights
                 per configuration.

        """
        if weights is None:
            # give each point an equal weight of 1
            weights = [np.ones(shape=(len(t), )) for t in trajectories]
        # now predict for the newly added
        preds = [model(tra) for tra in trajectories]
        for pred, tra, w in zip(preds, trajectories, weights):
            # and add to histogram
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

    def reevaluate_density(self, model):
        # TODO: rewrite the docstring when we know what this function does!
        """
        Reevaluate the density for all stored trajectories.

        Will replace the density histogram with a new density estimate for all
        trajectories from current models prediction.

        Parameters:
        -----------
        model - aimmd.base.RCModel predicting commitment probabilities

        """
        # keep a ref to current trajs and weights, then reset self
        trajs, weights = self._trajectories, self._weights
        self.reset()
        # and finally (re)add all trajectories using the current model
        self.add_density_for_trajectories(model=model, trajectories=trajs,
                                          weights=weights)

    def get_counts(self, probabilities):
        # TODO: Docstring!
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
        # TODO: Docstring!
        """
        Return the 'flattening factor' for the observed density of points.

        The factor is calculated as total_count / counts[probabilities],
        i.e. the factor is 1 / rho(probabilities).
        Note that we replace the potential infinite values appearing if
        the counts in a bin are zero by a large but finite value derived
        from the idea of add-one-smoothing.
        I.e. instead of our estimated rho = 0 we use 0 < rho << 1 when
        calculating 1 / rho.
        """
        dens = self.get_counts(probabilities)
        if (norm := np.sum(self.density_histogram)) == 0.:
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
            # NOTE: this is what we used before
            #factor[factor == np.inf] = ((norm + self._n_allowed_bins)
            #                            / weight_1_samp
            #                            )

        return factor


# TODO: maybe just merge the two classes into one and have the async versions of
#       the method have an "_async" suffix in their name?! (and a check for async-model?)
#       ...this makes it more readable and we also get rid of the overwrite mismatch for
#        the methods
class DensityCollectorAsync(DensityCollector):
    # NOTE: Take the docstrings of the superclass (i.e. the non-async version)
    __doc__ = ("Async version of the DensityCollector \n"
               + "Parent class docstring: \n"
               + str(DensityCollector.__doc__)
               )
    # NOTE: "steal" docstrings also for the methods we overwrite, they can
    #        (and should) be taken from DensityCollector

    async def add_density_for_trajectories(self, model, trajectories, weights=None):
        if weights is None:
            # give each point an equal weight of 1
            weights = [np.ones(shape=(len(t), )) for t in trajectories]
        # now predict for the newly added
        preds = await asyncio.gather(*(model(tra) for tra in trajectories))
        for pred, tra, w in zip(preds, trajectories, weights):
            # and add to histogram
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

    add_density_for_trajectories.__doc__ = DensityCollector.add_density_for_trajectories.__doc__

    async def reevaluate_density(self, model):
        # keep a ref to current trajs and weights, then reset self
        trajs, weights = self._trajectories, self._weights
        self.reset()
        # and finally (re)add all trajectories using the current model
        await self.add_density_for_trajectories(model=model, trajectories=trajs,
                                                weights=weights)

    reevaluate_density.__doc__ = DensityCollector.reevaluate_density.__doc__
