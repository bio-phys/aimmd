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


class Test_RCModel:
    def test_call_binomial(self, oneout_rcmodel):
        model = oneout_rcmodel
        n_points = 40
        n_dim = 5
        descriptors = np.random.normal(size=(n_points, n_dim))
        # the model returns np.sum(input, axis=1) as log_prob
        # the 'transform' is to use -input, such that for binomial,
        # where q_B = -q_A, the probabilities should be interchanged
        p_B_trans = model(descriptors)
        p_B_notrans = model(descriptors, use_transform=False)
        assert np.allclose(p_B_trans, 1 - p_B_notrans)
        assert p_B_trans.shape[0] == n_points
        assert p_B_trans.shape[1] == 1

    def test_call_multinomial(self, twoout_rcmodel):
        model = twoout_rcmodel
        n_points = 40
        n_dim = 5
        descriptors = np.random.normal(size=(n_points, n_dim))
        # the model returns np.sum(input, axis=1) as log_prob for A
        # and -np.sum(input, axis=1) as log_prob for B
        # the 'transform' is to use -input,
        # such that the probabilities should be interchanged
        p_trans = model(descriptors)
        p_notrans = model(descriptors, use_transform=False)
        assert np.allclose(p_trans, 1 - p_notrans)
        assert p_trans.shape[0] == n_points
        assert p_trans.shape[1] == 2
