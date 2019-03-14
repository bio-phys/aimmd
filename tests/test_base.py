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
from openpathsampling.engines import BaseSnapshot


class Test_RCModel:
    def test_binomial(self, oneout_rcmodel):
        model = oneout_rcmodel
        n_points = 40
        n_dim = 5
        descriptors = np.random.normal(size=(n_points, n_dim))
        p_B_trans = model(descriptors)
        p_B_notrans = model(descriptors, use_transform=False)
        q_trans = model.q(descriptors)
        q_notrans = model.q(descriptors, use_transform=False)
        # the model returns np.sum(input, axis=1) as log_prob
        # the 'transform' is to use -input, such that for binomial
        # where q_B = -q_A by construction,
        # the probabilities should be interchanged
        assert np.allclose(q_trans, -q_notrans)
        assert np.allclose(p_B_trans, 1 - p_B_notrans)
        # make sure z_sel is q_B
        assert np.allclose(q_trans, model.z_sel(descriptors))
        # basic shape checks
        assert p_B_trans.shape[0] == n_points
        assert p_B_trans.shape[1] == 1

    def test_multinomial(self, twoout_rcmodel):
        model = twoout_rcmodel
        n_points = 40
        n_dim = 5
        descriptors = np.random.normal(size=(n_points, n_dim))
        p_trans = model(descriptors)
        p_notrans = model(descriptors, use_transform=False)
        q_trans = model.q(descriptors)
        q_notrans = model.q(descriptors, use_transform=False)
        # the model returns np.sum(input, axis=1) as log_prob for A
        # and -np.sum(input, axis=1) as log_prob for B
        # the 'transform' is to use -input,
        # such that the probabilities should be interchanged
        # as we have two states the RC should be same as for binomial
        assert np.allclose(q_trans, -q_notrans)
        assert np.allclose(p_trans, 1 - p_notrans)
        # check multinomial z_sel
        # p_A = 1 - p_B = exp(200) / (exp(200) + exp(-200))
        descriptors = np.array([[100, 100],
                                [0., 0.]])  # p_A = p_B = 0.5
        z_sel = np.array([model.z_sel_scale, 0.])
        assert np.allclose(z_sel, model.z_sel(descriptors))
        # basic shape checks
        assert p_trans.shape[0] == n_points
        assert p_trans.shape[1] == 2

    def test_transforms(self, oneout_rcmodel_notrans, oneout_rcmodel_opstrans):
        mod_notrans = oneout_rcmodel_notrans
        mod_opstrans = oneout_rcmodel_opstrans
        # TODO: these are a bit silly, can we come up with something better?
        assert np.allclose(np.array([0.5]), mod_notrans(np.array([[0.]])))
        assert np.allclose(np.array([0.5]), mod_opstrans(BaseSnapshot()))
