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
        fname = tmp_path / "test_store.h5"
        storage = arcd.Storage(fname=fname)
        n_p = 2500  # number of points
        n_d = 200  # number of dimensions per point
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
