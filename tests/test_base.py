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
from openpathsampling.engines import BaseSnapshot
import arcd


class Test_RCModel:
    # NOTE: we test something very similar in storage tests
    def test_store_model(self, oneout_rcmodel_notrans, tmp_path):
        arcd_store = arcd.Storage(tmp_path / 'Test.h5')
        model = oneout_rcmodel_notrans
        model.expected_p.append('test')
        arcd_store.rcmodels['test'] = model
        loaded_model = arcd_store.rcmodels['test']
        for key, val in model.__dict__.items():
            if key == 'density_collector':
                loaded_dc = loaded_model.__dict__['density_collector']
                for skey, sval in val.__dict__.items():
                    assert np.all(loaded_dc.__dict__[skey] == sval)
            else:
                assert np.all(loaded_model.__dict__[key] == val)

    @pytest.mark.old
    def test_save_fix_load(self, oneout_rcmodel_notrans, tmp_path):
        # NOTE: OLD LOADING/SAVING API
        p = tmp_path / 'Test_RCModel_test_save_fix_load'
        fname = str(p)
        model = oneout_rcmodel_notrans
        model.expected_p.append('test')
        model.save(fname)
        fname += model.save_model_extension
        state, cls = model.__class__.load_state(fname)
        state = cls.fix_state(state)
        loaded_model = cls.set_state(state)
        for key, val in model.__dict__.items():
            if key == 'density_collector':
                loaded_dc = loaded_model.__dict__['density_collector']
                for skey, sval in val.__dict__.items():
                    assert np.all(loaded_dc.__dict__[skey] == sval)
            else:
                assert np.all(loaded_model.__dict__[key] == val)

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

    def test_expected_efficiency_factor(self, oneout_rcmodel_notrans,
                                        twoout_rcmodel_notrans):
        mod_oneout = oneout_rcmodel_notrans
        mod_twoout = twoout_rcmodel_notrans
        # set minimum points for ee-factor calculation to zero, such that we
        # can test easily
        mod_oneout.min_points_ee_factor = 0
        mod_twoout.min_points_ee_factor = 0
        # expect 1 TP from two times p(TP|SP) = 1/2
        # expect 0 TPs from two times p(TP|SP) = 0
        sps = [np.array([[200.]]), np.array([[0.]]),
               np.array([[200.]]), np.array([[0.]])
               ]
        # contains 1 TP
        ts1 = arcd.TrainSet(n_states=2,
                            descriptors=np.concatenate(sps, axis=0),
                            shot_results=np.array([[0., 2.], [1., 1.],
                                                   [0., 2.], [2., 0.]])
                            )
        # contains 2 TPs
        ts2 = arcd.TrainSet(n_states=2,
                            descriptors=np.concatenate(sps, axis=0),
                            shot_results=np.array([[0., 2.], [1., 1.],
                                                   [1., 1.], [2., 0.]])
                            )
        for sp in sps:
            mod_oneout.register_sp(sp)
            mod_twoout.register_sp(sp)
        # EE factor should be zero if n_TP_ex = n_TP_true
        assert np.allclose(0., mod_oneout.train_expected_efficiency_factor(ts1, len(sps)+1))
        assert np.allclose(0., mod_twoout.train_expected_efficiency_factor(ts1, len(sps)+1))
        # window smaller than len(model.expected_p)
        assert np.allclose(0., mod_oneout.train_expected_efficiency_factor(ts1, len(sps)-1))
        assert np.allclose(0., mod_twoout.train_expected_efficiency_factor(ts1, len(sps)-1))
        # EEfactor should be (1 - n_TP_true / n_TP_ex)**2 = (1 - 2/1)**2 = 1
        assert np.allclose(1., mod_oneout.train_expected_efficiency_factor(ts2, len(sps)+1))
        assert np.allclose(1., mod_twoout.train_expected_efficiency_factor(ts2, len(sps)+1))
        # window smaller than len(model.expected_p)
        assert np.allclose(1., mod_oneout.train_expected_efficiency_factor(ts2, len(sps)-1))
        assert np.allclose(1., mod_twoout.train_expected_efficiency_factor(ts2, len(sps)-1))
        # set minimum number of points for ee-factor calculation to a large value
        # to test that we get a 1 back for a 'too short' trainingset
        mod_oneout.min_points_ee_factor = 100
        mod_twoout.min_points_ee_factor = 100
        assert np.allclose(1., mod_oneout.train_expected_efficiency_factor(ts1, len(sps)))
        assert np.allclose(1., mod_twoout.train_expected_efficiency_factor(ts1, len(sps)))
