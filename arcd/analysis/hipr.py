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
    # TODO: docstring!
    """
    Relative input importance analysis.

    Literature 'An approach for determining relative input parameter
                importance and significance in artificial neural networks'
                by Stanley J. Kemp, Patricia Zaradic and Frank Hanse
                https://doi.org/10.1016/j.ecolmodel.2007.01.009
    """
    def __init__(self, model, trainset, call_kwargs={}, n_redraw=1):
        self.trainset = trainset  # the 'true' trainset
        self.model = model  # any RCModel with a test_loss function
        # fine grained call control, e.g. to select the loss for MultiDomain
        self.call_kwargs = call_kwargs
        # number of times we redraw random descriptors per point/trainset
        # i.e. if redraw=2 we will do 2 HIPR and average the results
        self.n_redraw = n_redraw

    def do_hipr(self):
        """
        Perform HIPR analysis and set self.hipr_losses to the result.

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
        for _ in range(self.n_redraw):
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
        hipr_losses /= self.n_redraw
        # and add reference loss
        hipr_losses[-1] = self.model.test_loss(self.trainset,
                                               **self.call_kwargs)
        self.hipr_losses = hipr_losses
        return hipr_losses
