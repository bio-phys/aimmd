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
import numpy as np
from ..base.trainset import TrainSet


class HIPRanalysis:
    """
    Relative input importance analysis ('HIPR').

    Literature 'An approach for determining relative input parameter
                importance and significance in artificial neural networks'
                by Stanley J. Kemp, Patricia Zaradic and Frank Hanse
                https://doi.org/10.1016/j.ecolmodel.2007.01.009

    """

    def __init__(self, model, trainset, call_kwargs={}, n_redraw=5):
        """
        Relative input importance analysis ('HIPR') as described in literature.

        Parameters:
        -----------
        model - the arcd.RCModel to perform relative input importance analysis
        trainset - arcd.TrainSet with unperturbed descriptors and shot_results
        call_kwargs - dict of additional key word arguments to
                      RCModel.test_loss(), e.g. for choosing the test_loss
                      for MultiDomainModels with call_kwargs={'loss':'L_mod0'}
        n_redraw - int, number of times we redraw random descriptors per point
                   in trainset, i.e. if redraw=2 we will average the loss over
                   2*len(trainset) points per model input descriptor

        """
        self.trainset = trainset  # the 'true' trainset
        self.model = model  # any RCModel with a test_loss function
        # fine grained call control, e.g. to select the loss for MultiDomain
        self.call_kwargs = call_kwargs
        # number of times we redraw random descriptors per point/trainset
        # i.e. if redraw=2 we will do 2 HIPR and average the results
        self.n_redraw = n_redraw

    def do_hipr(self, n_redraw=None):
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

        Returns:
        --------
        hipr_losses - a numpy array (shape=(descriptor_dim + 1,)),
                      where hipr_losses[i] corresponds to the loss suffered by
                      replacing the ith input descriptor with random noise,
                      while hipr_losses[-1] is the reference loss over the
                      unmodified TrainSet

        """
        # last entry is for true loss
        hipr_losses = np.zeros((self.trainset.descriptors.shape[1] + 1,))
        maxes = np.max(self.trainset.descriptors, axis=0)
        mins = np.min(self.trainset.descriptors, axis=0)
        if n_redraw is None:
            n_redraw = self.n_redraw
        for _ in range(n_redraw):
            for i in range(len(maxes)):
                descriptors = self.trainset.descriptors.copy()
                descriptors[:, i] = ((maxes[i] - mins[i])
                                     * np.random.ranf(size=len(self.trainset))
                                     + mins[i]
                                     )
                ts = TrainSet(
                       self.trainset.states,
                       descriptor_transform=self.trainset.descriptor_transform,
                       descriptors=descriptors,
                       shot_results=self.trainset.shot_results
                              )
                hipr_losses[i] += self.model.test_loss(ts, **self.call_kwargs)
        # take the mean
        hipr_losses /= n_redraw
        # and add reference loss
        hipr_losses[-1] = self.model.test_loss(self.trainset,
                                               **self.call_kwargs)
        self.hipr_losses = hipr_losses
        return hipr_losses

    def do_hipr_plus(self, n_redraw=None):
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

        Returns:
        --------
        hipr_losses_plus - a numpy array (shape=(descriptor_dim + 1,)),
                           where hipr_losses[i] corresponds to the loss
                           suffered by permutating the ith input descriptor,
                           while hipr_losses[-1] is the reference loss over the
                           unmodified TrainSet

        """
        # last entry is for true loss
        hipr_losses_plus = np.zeros((self.trainset.descriptors.shape[1] + 1,))
        if n_redraw is None:
            n_redraw = self.n_redraw
        n_dim = self.trainset.descriptors.shape[1]
        for _ in range(n_redraw):
            for i in range(n_dim):
                descriptors = self.trainset.descriptors.copy()
                permut_idxs = np.random.permutation(len(self.trainset))
                descriptors[:, i] = descriptors[:, i][permut_idxs]
                ts = TrainSet(
                       self.trainset.states,
                       descriptor_transform=self.trainset.descriptor_transform,
                       descriptors=descriptors,
                       shot_results=self.trainset.shot_results
                              )
                hipr_losses_plus[i] += self.model.test_loss(ts,
                                                            **self.call_kwargs
                                                            )
        # take the mean
        hipr_losses_plus /= n_redraw
        # and add reference loss
        hipr_losses_plus[-1] = self.model.test_loss(self.trainset,
                                                    **self.call_kwargs
                                                    )
        self.hipr_losses_plus = hipr_losses_plus
        return hipr_losses_plus
