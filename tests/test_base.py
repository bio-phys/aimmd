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
from arcd.base.rcmodel import RCModel


class OneOutRCModel(RCModel):
    def __init__(self):
        # work on numpy arrays directly, but test the use of transform
        # using transform should exchange p_A with p_B
        super().__init__(descriptor_transform=lambda x: -x)
        self.n_train = 0

    @property
    def n_out(self):
        return 1

    def train_hook(self, trainset):
        # just to have something to check if
        # and how often we would have trained
        self.n_train += 1

    def _log_prob(self, descriptors):
        # so we have a return value that has the correct shape
        return np.sum(descriptors, axis=1)

    def test_loss(self, trainset):
        # just to be able to instantiate
        return NotImplementedError


class TwoOutRCModel(RCModel):
    def __init__(self):
        # work on numpy arrays directly, but test the use of transform
        super().__init__(descriptor_transform=lambda x: -x)
        self.n_train = 0

    @property
    def n_out(self):
        return 2

    def train_hook(self, trainset):
        # just to have something to check if
        # and how often we would have trained
        self.n_train += 1

    def _log_prob(self, descriptors):
        # so we have a return value that has the correct shape
        q_A = np.sum(descriptors, axis=1, keepdims=True)
        # take q_B = -q_A such that p_A + p_B = 1
        q_B = -np.sum(descriptors, axis=1, keepdims=True)
        return np.concatenate((q_A, q_B), axis=1)

    def test_loss(self, trainset):
        # just to be able to instantiate
        return NotImplementedError


class Test_RCmodel_binomial:
    def test_call(self):
        model = OneOutRCModel()
        n_points = 20
        n_dim = 40
        # draw random descriptors
        descriptors = np.random.normal(loc=-0.1, size=(n_points, n_dim))
        # the transform is simply lambda x: -x, such that using it should
        # exchange the probabilities
        p_B_transform = model(descriptors, use_transform=True)
        p_B_no_transform = model(descriptors, use_transform=False)
        assert np.allclose(p_B_transform, 1 - p_B_no_transform)
