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
import numpy as np
from ..base.trainset import TrainSet


class HIPRanalysis:
    """
    Relative input importance analysis ('HIPR').

    Parameters:
    -----------
        model - the aimmd.RCModel to perform relative input importance analysis
        trainset - aimmd.TrainSet with unperturbed descriptors and shot_results
        call_kwargs - dict of additional key word arguments to
                      RCModel.test_loss(), e.g. for choosing the test_loss
                      for MultiDomainModels with call_kwargs={'loss':'L_mod0'}
        n_redraw - int, number of times we redraw random descriptors per point
                   in trainset, i.e. if redraw=2 we will average the loss over
                   2*len(trainset) points per model input descriptor


    Literature 'An approach for determining relative input parameter
                importance and significance in artificial neural networks'
                by Stanley J. Kemp, Patricia Zaradic and Frank Hanse
                https://doi.org/10.1016/j.ecolmodel.2007.01.009

    """

    def __init__(self, model, trainset, call_kwargs={}, n_redraw=5):
        """Initialize HIPRanalysis."""
        self.trainset = trainset  # the 'true' trainset
        self.model = model  # any RCModel with a test_loss function
        # fine grained call control, e.g. to select the loss for MultiDomain
        self.call_kwargs = call_kwargs
        # number of times we redraw random descriptors per point/trainset
        # i.e. if redraw=2 we will do 2 HIPR and average the results
        self.n_redraw = n_redraw

    def do_hipr(self, n_redraw=None, return_all=False):
        """
        Perform HIPR analysis and set self.hipr_losses to the result.

        Parameters:
        -----------
        n_redraw - int or None, number of times we redraw random descriptors
                   per point in trainset, i.e. if redraw=2 we will average
                   the loss over 2*len(trainset) points per model input,
                   Note that giving n_redraw here will take precedence over
                   self.n_redraw, we will only use self.n_redraw if n_redraw
                   given here is None
        return_all - bool, if True we will additionally return all calulated
                     losses as 2d array, second axis corresponds to repetitions

        Returns:
        --------
        hipr_losses - a numpy array (shape=(descriptor_dim + 1,)),
                      where hipr_losses[i] corresponds to the loss suffered by
                      replacing the ith input descriptor with random noise,
                      while hipr_losses[-1] is the reference loss over the
                      unmodified TrainSet
        hipr_losses_std - same shape as hipr_losses, array of standard
                          deviations calculated over the n_redraw repetitions

        """
        if n_redraw is None:
            n_redraw = self.n_redraw
        # last entry is for true loss
        hipr_losses = np.zeros((self.trainset.descriptors.shape[1] + 1, n_redraw))
        maxes = np.max(self.trainset.descriptors, axis=0)
        mins = np.min(self.trainset.descriptors, axis=0)
        for i in range(len(maxes)):
            for draw_num in range(n_redraw):
                descriptors = self.trainset.descriptors.copy()
                descriptors[:, i] = ((maxes[i] - mins[i])
                                     * np.random.ranf(size=len(self.trainset))
                                     + mins[i]
                                     )
                ts = TrainSet(
                       self.trainset.n_states,
                       descriptors=descriptors,
                       shot_results=self.trainset.shot_results,
                       weights=self.trainset.weights
                              )
                hipr_losses[i, draw_num] = self.model.test_loss(
                                                            ts,
                                                            **self.call_kwargs,
                                                                )
        # and add reference loss, it is alwats the same, no need for repetitions
        hipr_losses[-1, :] = self.model.test_loss(self.trainset,
                                                  **self.call_kwargs)
        # take the mean
        hipr_losses_mean = np.mean(hipr_losses, axis=1)
        hipr_losses_std = np.std(hipr_losses, axis=1)
        self.hipr_losses = hipr_losses_mean
        self.hipr_losses_std = hipr_losses_std
        if return_all:
            return hipr_losses_mean, hipr_losses_std, hipr_losses
        return hipr_losses_mean, hipr_losses_std

    def do_hipr_plus(self, n_redraw=None, return_all=False):
        """
        Perform HIPR analysis plus and set self.hipr_losses_plus to the result.

        Note that this is not the 'true' HIPR as described in the literature.
        Here we permutate the descriptors randomly instead of drawing random
        values, this conserves the distribution of values over the
        corresponding descriptor dimension and is therefore hopefully more
        sensible if using non-whitened input.

        Parameters:
        -----------
        n_redraw - int or None, number of times we permutate the descriptors
                   in trainset, i.e. if redraw=2 we will average
                   the loss over 2*len(trainset) points per model input,
                   Note that passing n_redraw here will take precedence over
                   self.n_redraw, we will only use self.n_redraw if n_redraw
                   given here is None
        return_all - bool, if True we will additionally return all calulated
                     losses as 2d array, second axis corresponds to repetitions

        Returns:
        --------
        hipr_losses_plus - a numpy array (shape=(descriptor_dim + 1,)),
                           where hipr_losses[i] corresponds to the loss
                           suffered by permutating the ith input descriptor,
                           while hipr_losses[-1] is the reference loss over the
                           unmodified TrainSet
        hipr_losses_plus_std - same shape as hipr_losses, array of standard
                               deviations calculated over the n_redraw repetitions

        """
        if n_redraw is None:
            n_redraw = self.n_redraw
        # last entry is for true loss
        hipr_losses_plus = np.zeros((self.trainset.descriptors.shape[1] + 1, n_redraw))
        n_dim = self.trainset.descriptors.shape[1]
        for i in range(n_dim):
            for draw_num in range(n_redraw):
                descriptors = self.trainset.descriptors.copy()
                permut_idxs = np.random.permutation(len(self.trainset))
                descriptors[:, i] = descriptors[:, i][permut_idxs]
                ts = TrainSet(
                       self.trainset.n_states,
                       descriptors=descriptors,
                       shot_results=self.trainset.shot_results,
                       weights=self.trainset.weights
                              )
                hipr_losses_plus[i, draw_num] = self.model.test_loss(
                                                            ts,
                                                            **self.call_kwargs,
                                                                     )
        # and add reference loss
        hipr_losses_plus[-1, :] = self.model.test_loss(self.trainset,
                                                       **self.call_kwargs
                                                       )
        # take the mean
        hipr_losses_plus_mean = np.mean(hipr_losses_plus, axis=1)
        hipr_losses_plus_std = np.std(hipr_losses_plus, axis=1)
        self.hipr_losses_plus_std = hipr_losses_plus_std
        self.hipr_losses_plus = hipr_losses_plus_mean
        if return_all:
            return hipr_losses_plus_mean, hipr_losses_plus_std, hipr_losses_plus
        return hipr_losses_plus_mean, hipr_losses_plus_std

    def do_hipr_plus_correlations(self, indices, n_redraw=None, return_all=False):
        """
        Perform correlation HIPR analysis plus for given index combinations.

        Permutes the descriptor values for the given indices together, first
        using different permutations for each input and second using the same
        permutation order for all indices in one group.
        This can be used to quantify the correlations between them. Correlated
        descriptors will result in a higher loss when permuted in the same
        permutation order.

        Parameters
        ----------
        indices - list of list or list of 1d arrays, Each entry in the outer
                  list represents a group of indices for which correlations
                  should be calculated,
                  Note that although a group will usually contain two indices
                  any number of indices (>1) is supported
        n_redraw - int or None, number of times we permutate the descriptors
                   in trainset, i.e. if redraw=2 we will average
                   the loss over 2*len(trainset) points per model input,
                   Note that passing n_redraw here will take precedence over
                   self.n_redraw, we will only use self.n_redraw if n_redraw
                   given here is None
        return_all - bool, if True we will additionally return all calulated
                     losses as 3d array, last axis corresponds to repetitions

        Returns
        -------
        correlation_losses - 2d array, first dim corresponds to an index group,
                             the first loss on the second axis is for permuting
                             the descriptors using a different permutation for
                             every descriptor dimension,
                             the second loss is for permuting all chosen
                             descriptor dimensions together in the same order,
                             NOTE: these are mean values over n_redraw
                             realisations
        correlation_losses_std - 2d array, same order as correlation_losses,
                                 std over the n_redraw realisations

        """
        if n_redraw is None:
            n_redraw = self.n_redraw

        losses = np.zeros((len(indices), 2, n_redraw))
        for i, idxs in enumerate(indices):
            for draw_num in range(n_redraw):
                # permute idxs in parallel, but using different permutations
                descriptors = self.trainset.descriptors.copy()
                for idx in idxs:
                    permut_idxs = np.random.permutation(len(self.trainset))
                    descriptors[:, idx] = descriptors[:, idx][permut_idxs]
                ts = TrainSet(
                    self.trainset.n_states,
                    descriptors=descriptors,
                    shot_results=self.trainset.shot_results,
                    weights=self.trainset.weights
                             )
                losses[i, 0, draw_num] += self.model.test_loss(
                                                            ts,
                                                            **self.call_kwargs
                                                               )
                # now permute all idxs together
                descriptors = self.trainset.descriptors.copy()
                permut_idxs = np.random.permutation(len(self.trainset))
                for idx in idxs:
                    descriptors[:, idx] = descriptors[:, idx][permut_idxs]
                ts = TrainSet(
                    self.trainset.n_states,
                    descriptors=descriptors,
                    shot_results=self.trainset.shot_results,
                    weights=self.trainset.weights
                             )
                losses[i, 1, draw_num] += self.model.test_loss(ts, **self.call_kwargs)
        # mean and std over the n_redraw realizations
        losses_mean = np.mean(losses, axis=2)
        losses_std = np.std(losses, axis=2)
        self.correlation_losses = losses_mean
        self.correlation_losses_std = losses_std
        self.correlation_indices = indices
        if return_all:
            return losses_mean, losses_std, losses
        return losses_mean, losses_std
