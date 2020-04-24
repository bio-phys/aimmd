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
import arcd
import numpy as np


class Test_storage:
    def test_trainset(self, tmp_path):
        # TODO: test saving and loading of states + descrptor_transform?!
        fname = str(tmp_path / "test_store.h5")
        storage = arcd.Storage(fname=fname)
        n_p = 2500  # number of points
        n_d = 200  # number of dimensions per point
        add_points = 200  # points to append for the second round
        ts_true = arcd.TrainSet(states=["A", "B"],
                                descriptor_transform=None,
                                descriptors=np.random.random_sample(size=(n_p, n_d)),
                                shot_results=np.random.randint(0, 2, size=(n_p, 2)),
                                )
        storage.save_trainset(ts_true)
        ts_loaded = storage.load_trainset()
        assert np.allclose(ts_true.descriptors, ts_loaded.descriptors)
        assert np.allclose(ts_true.shot_results, ts_loaded.shot_results)
        assert np.allclose(ts_true.weights, ts_loaded.weights)
        # now append something to the trainset and save again
        # this test overwriting of existing trainsets
        for _ in range(add_points):
            ts_true.append_point(descriptors=np.random.random_sample(size=(n_d,)),
                                 shot_results=np.random.randint(0, 2, size=(2,)),
                                 )
        # save and test again
        storage.save_trainset(ts_true)
        ts_loaded2 = storage.load_trainset()
        assert np.allclose(ts_true.descriptors, ts_loaded2.descriptors)
        assert np.allclose(ts_true.shot_results, ts_loaded2.shot_results)
        assert np.allclose(ts_true.weights, ts_loaded2.weights)

    def test_base_model(self, tmp_path,
                        oneout_rcmodel_notrans,
                        twoout_rcmodel_notrans,
                        ):
        def assert_model_equal(true, test):
            for key, val in true.__dict__.items():
                if key == 'density_collector':
                    test_dc = test.__dict__['density_collector']
                    for skey, sval in val.__dict__.items():
                        assert np.all(test_dc.__dict__[skey] == sval)
            else:
                assert np.all(test.__dict__[key] == val)

        fname = str(tmp_path / "test_store.h5")
        storage = arcd.Storage(fname=fname)
        models = [oneout_rcmodel_notrans,
                  twoout_rcmodel_notrans,
                  ]
        # save the models
        for i, mod in enumerate(models):
            storage.rcmodels[str(i)] = mod
        # loaded and test they are equal
        loaded_models = [storage.rcmodels[str(i)] for i in range(len(models))]
        for true_mod, test_mod in zip(models, loaded_models):
            assert_model_equal(true=true_mod, test=test_mod)
        # and again to test overwriting etc
        for i, mod in enumerate(models):
            storage.rcmodels[str(i)] = mod
        loaded_models2 = [storage.rcmodels[str(i)] for i in range(len(models))]
        for true_mod, test_mod in zip(models, loaded_models2):
            assert_model_equal(true=true_mod, test=test_mod)
