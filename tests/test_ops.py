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
from unittest.mock import MagicMock
import aimmd


class Test_AimmdStorageHook:
    def test_before_simulation(self, tmp_path, oneout_rcmodel):
        # NOTE: this does not test anything really
        # the simulations ops storage is none so we save no ops objects anyways
        aimmd_store = aimmd.Storage(tmp_path / "test.h5")
        trainset = aimmd.TrainSet(n_states=3)
        hook = aimmd.ops.AimmdStorageHook(storage=aimmd_store,
                                          model=oneout_rcmodel,
                                          trainset=trainset
                                          )
        # pretend we have no ops storage attached to pathsimulator
        simulation = MagicMock(storage=None)
        hook.before_simulation(simulation)

    def test_after_step(self, tmp_path, oneout_rcmodel_notrans):
        aimmd_store = aimmd.Storage(tmp_path / "test.h5")
        trainset = aimmd.TrainSet(n_states=3)
        model_prefix = "blub"
        hook = aimmd.ops.AimmdStorageHook(storage=aimmd_store,
                                          model=oneout_rcmodel_notrans,
                                          trainset=trainset,
                                          interval=3,
                                          name_prefix=model_prefix
                                          )
        # call after step with a non divisible step_number
        step_num = 2
        hook.after_step("simulation", step_num, ('step', 'info'), ('state'),
                        "results", "hook_state")
        # make sure nothing got saved, i.e. storage is empty
        with pytest.raises(KeyError):
            ts = aimmd_store.load_trainset()
        assert len(aimmd_store.rcmodels.keys()) == 0
        # now try a divisible step number
        step_num = 3
        hook.after_step("simulation", step_num, ('step', 'info'), ('state'),
                        "results", "hook_state")
        # and make sure all is in there under the names we expect
        # (that what we load is actually what we saved is made sure in storage tests)
        loaded_model = aimmd_store.rcmodels[model_prefix
                                           + f"_after_step_{step_num:d}"]
        loaded_ts = aimmd_store.load_trainset()

    def test_after_simulation(self, tmp_path, oneout_rcmodel_notrans):
        aimmd_store = aimmd.Storage(tmp_path / "test.h5")
        trainset = aimmd.TrainSet(n_states=3)
        model_prefix = "blub"
        hook = aimmd.ops.AimmdStorageHook(storage=aimmd_store,
                                          model=oneout_rcmodel_notrans,
                                          trainset=trainset,
                                          interval=3,
                                          name_prefix=model_prefix
                                          )
        last_step = 420
        simulation = MagicMock(step=last_step)
        hook.after_simulation(simulation, {})
        # make sure the model is in the storage (twice)
        loaded_model = aimmd_store.rcmodels[model_prefix
                                           + f"_after_step_{last_step:d}"]
        loaded_model2 = aimmd_store.rcmodels["most_recent"]
        # and the trainset
        ts = aimmd_store.load_trainset()
        # NOTE: no need to test the correctness of the loaded values
        #       we check that what we load is what we save in storage tests


class Test_DensityCollectionHook:
    @pytest.mark.skip("TODO")
    def test_before_simulation(self):
        pass

    @pytest.mark.skip("TODO")
    def test_after_step(self):
        pass


class Test_TrainingHook:
    @pytest.mark.skip("TODO")
    def test_after_step(self):
        pass
