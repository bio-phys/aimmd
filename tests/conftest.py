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
import pytest
import numpy as np
# TODO/BUG: weird stuff: if not importing mdtraj before RCModel
# the tests segfault....!?
import mdtraj
from arcd.base.rcmodel import RCModel
from openpathsampling.engines import Trajectory as OPSTrajectory


@pytest.fixture
def oneout_rcmodel():
    return OneOutRCModel(lambda x: -x)


@pytest.fixture
def oneout_rcmodel_notrans():
    return OneOutRCModel(None)


@pytest.fixture
def oneout_rcmodel_opstrans():
    def transform(x):
        if isinstance(x, OPSTrajectory):
            return np.array([[0.]])
        else:
            return np.array([[-200.]])
    return OneOutRCModel(transform)


@pytest.fixture
def twoout_rcmodel():
    return TwoOutRCModel(lambda x: -x)


@pytest.fixture
def twoout_rcmodel_notrans():
    return TwoOutRCModel(None)


class OneOutRCModel(RCModel):
    def __init__(self, transform=None):
        # work on numpy arrays directly, but test the use of transform
        # using transform should exchange p_A with p_B
        super().__init__(descriptor_transform=transform)
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
        return np.sum(descriptors, axis=1, keepdims=True)

    def test_loss(self, trainset):
        # just to be able to instantiate
        raise NotImplementedError


class TwoOutRCModel(RCModel):
    def __init__(self, transform):
        super().__init__(descriptor_transform=transform)
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
        raise NotImplementedError
