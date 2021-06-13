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
import pytest
import aimmd
import numpy as np


class Test_storage:
    @pytest.mark.parametrize("buffsize", [None,  # use unbuffered version
                                          2**17,  # 130 KiB buffer
                                          2**29,  # 530 MiB default
                                          ]
                             )
    def test_MutableObjectShelf(self, tmp_path, buffsize):
        fname = str(tmp_path / "test_store.h5")
        storage = aimmd.Storage(fname=fname)
        grp_name = "/testdata"
        grp = storage.file.require_group(grp_name)
        shelf = aimmd.base.storage.MutableObjectShelf(grp)
        # test that accessing an empty shelf raises the correct error
        with pytest.raises(KeyError):
            _ = shelf.load(buffsize=buffsize)
        objs = [np.random.random_sample(size=(10000, 400))]
        objs += ["test"]
        shelf.save(objs, buffsize=buffsize)
        loaded_objs = shelf.load(buffsize=buffsize)
        for o_true, o_loaded in zip(objs, loaded_objs):
            assert np.all(o_true == o_loaded)
        storage.close()
        storage = aimmd.Storage(fname=fname, mode="a")
        grp = storage.file.require_group(grp_name)
        shelf = aimmd.base.storage.MutableObjectShelf(grp)
        with pytest.raises(RuntimeError):
            # we already stored and set overwrite=False, so we expect an error
            shelf.save(objs, overwrite=False, buffsize=buffsize)
        # and now test if overwriting works
        shelf.save(objs, overwrite=True, buffsize=buffsize)
        loaded_objs2 = shelf.load(buffsize=buffsize)
        for o_true, o_loaded in zip(objs, loaded_objs2):
            assert np.all(o_true == o_loaded)

    def test_trainset(self, tmp_path):
        # TODO: test saving and loading of states + descrptor_transform?!
        fname = str(tmp_path / "test_store.h5")
        storage = aimmd.Storage(fname=fname)
        n_p = 2500  # number of points
        n_d = 200  # number of dimensions per point
        add_points = 200  # points to append for the second round
        ts_true = aimmd.TrainSet(n_states=2,
                                 descriptors=np.random.random_sample(size=(n_p, n_d)),
                                 shot_results=np.random.randint(0, 2, size=(n_p, 2)),
                                 )
        storage.save_trainset(ts_true)
        storage.close()
        storage = aimmd.Storage(fname=fname, mode='a')
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
        storage = aimmd.Storage(fname=fname)
        models = [oneout_rcmodel_notrans,
                  twoout_rcmodel_notrans,
                  ]
        # save the models
        for i, mod in enumerate(models):
            storage.rcmodels[str(i)] = mod
        # load and test that they are equal
        loaded_models = [storage.rcmodels[str(i)] for i in range(len(models))]
        for true_mod, test_mod in zip(models, loaded_models):
            assert_model_equal(true=true_mod, test=test_mod)
        # and again to test overwriting
        for i, mod in enumerate(models):
            storage.rcmodels[str(i)] = mod
        loaded_models2 = [storage.rcmodels[str(i)] for i in range(len(models))]
        for true_mod, test_mod in zip(models, loaded_models2):
            assert_model_equal(true=true_mod, test=test_mod)
        # now yet another way: by iterating over the dictionary
        loaded_models3 = [None for _ in range(len(storage.rcmodels))]
        # need to make sure the order is the same as when saving
        # but we used the indices ans key anyways
        for key, val in storage.rcmodels.items():
            loaded_models3[int(key)] = storage.rcmodels[key]
        # check that they are equal again
        for true_mod, test_mod in zip(models, loaded_models3):
            assert_model_equal(true=true_mod, test=test_mod)
        # test removing model(s)
        del storage.rcmodels["0"]
        assert len(storage.rcmodels) == len(models) - 1
        del storage.rcmodels["1"]
        assert len(storage.rcmodels) == len(models) - 2
