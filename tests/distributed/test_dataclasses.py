# This file is part of aimmd
#
# aimmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# aimmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with aimmd. If not, see <https://www.gnu.org/licenses/>.
"""
Test suite for aimmd.distributed.dataclasses module.
"""
import pytest
import numpy as np
import os
from asyncmd import Trajectory

from aimmd.distributed.dataclasses import MCstep
from aimmd.distributed.dataclasses import DensityAdaptionParameters


class TestMCstep:
    """Tests for the MCstep dataclass."""

    def test_init_minimal(self, simple_trajectory):
        """Test creating MCstep with only required arguments."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
        )

        assert step.mover is None
        assert step.step_num == 1
        assert step.directory == "/tmp"
        assert step.path is simple_trajectory
        assert step.accepted is False
        assert step.p_acc == 0.0
        assert step.weight == 1.0
        assert step.predicted_committors_sp is None
        assert step.shooting_snap is None
        assert step.states_reached is None
        assert step.trial_trajectories == []
        assert step.default_savename == "mcstep_data.pckl"

    def test_init_full(self, simple_trajectory):
        """Test creating MCstep with all optional arguments."""
        dummy_mover = object()
        dummy_snap = object()
        dummy_committors = np.array([0.5, 0.3, 0.2])
        dummy_states = np.array([1, 1, 0])
        dummy_trajectory = object()

        step = MCstep(
            mover=dummy_mover,
            step_num=5,
            directory="/test/dir",
            path=simple_trajectory,
            accepted=True,
            p_acc=0.75,
            weight=2.5,
            predicted_committors_sp=dummy_committors,
            shooting_snap=dummy_snap,
            states_reached=dummy_states,
            trial_trajectories=[dummy_trajectory],
            default_savename="custom_data.pkl",
        )

        assert step.mover is dummy_mover
        assert step.step_num == 5
        assert step.directory == "/test/dir"
        assert step.path is simple_trajectory
        assert step.accepted is True
        assert step.p_acc == 0.75
        assert step.weight == 2.5
        assert np.array_equal(step.predicted_committors_sp, dummy_committors)
        assert step.shooting_snap is dummy_snap
        assert np.array_equal(step.states_reached, dummy_states)
        assert step.trial_trajectories == [dummy_trajectory]
        assert step.default_savename == "custom_data.pkl"

    def test_default_factory_independence(self, simple_trajectory):
        """Test that trial_trajectories list is independent per instance."""
        step1 = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
        )

        step2 = MCstep(
            mover=None,
            step_num=2,
            directory="/tmp",
            path=simple_trajectory,
        )

        step1.trial_trajectories.append("step1_traj")
        step2.trial_trajectories.append("step2_traj")

        assert len(step1.trial_trajectories) == 1
        assert len(step2.trial_trajectories) == 1
        assert step1.trial_trajectories != step2.trial_trajectories

    def test_transitions_none(self, simple_trajectory):
        """Test transitions property when states_reached is None."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=None,
        )

        assert step.transitions == 0
        assert step.contains_transition is False

    def test_transitions_single_state(self, simple_trajectory):
        """Test transitions property with single state reached."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=np.array([1, 0, 0]),
        )

        assert step.transitions == 0
        assert step.contains_transition is False

    def test_transitions_no_transition(self, simple_trajectory):
        """Test transitions property with states reached but no transition."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=np.array([2, 0, 0]),
        )

        assert step.transitions == 0
        assert step.contains_transition is False

    def test_transitions_one_transition(self, simple_trajectory):
        """Test transitions property with one transition."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=np.array([1, 1]),
        )

        assert step.transitions == 1
        assert step.contains_transition

    def test_transitions_multiple_states_one_transition(self, simple_trajectory):
        """Test transitions property with three states but only one transition."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=np.array([1, 1, 0]),
        )

        assert step.transitions == 1
        assert step.contains_transition

    def test_transitions_multiple_transitions(self, simple_trajectory):
        """Test transitions property with multiple transitions."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=np.array([1, 1, 1]),
        )

        # With 3 states all reached: pairs are (0,1), (0,2), (1,2) = 3 transitions
        assert step.transitions == 3
        assert step.contains_transition

    def test_transitions_max_transitions(self, simple_trajectory):
        """Test transitions with all states reached for larger state space."""
        n_states = 5
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=np.ones(n_states),
        )

        # Number of pairs: C(n, 2) = n*(n-1)/2 = 5*4/2 = 10
        assert step.transitions == 10
        assert step.contains_transition

    def test_contains_transition_none(self, simple_trajectory):
        """Test contains_transition when states_reached is None."""
        step = MCstep(
            mover=None,
            step_num=1,
            directory="/tmp",
            path=simple_trajectory,
            states_reached=None,
        )

        assert step.contains_transition is False

    def test_save_default_name(self, simple_trajectory, tmp_path):
        """Test save method with default filename."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
            accepted=True,
        )

        step.save()

        expected_file = step_dir / "mcstep_data.pckl"
        assert expected_file.exists()

    def test_save_custom_default_name(self, simple_trajectory, tmp_path):
        """Test save method with custom default filename."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
            default_savename="custom.pkl",
        )

        step.save()

        expected_file = step_dir / "custom.pkl"
        assert expected_file.exists()

    def test_save_custom_name(self, simple_trajectory, tmp_path):
        """Test save method with custom filename."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
        )

        step.save(fname=(step_dir / "custom.pkl"))

        expected_file = step_dir / "custom.pkl"
        assert expected_file.exists()

    def test_save_file_exists_no_overwrite(self, simple_trajectory, tmp_path):
        """Test save raises ValueError when file exists and overwrite=False."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
        )

        step.save()
        with pytest.raises(ValueError, match="exists but overwrite=False"):
            step.save()

    def test_save_overwrite(self, simple_trajectory, tmp_path):
        """Test save method with overwrite=True."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
        )

        expected_file = step_dir / "mcstep_data.pckl"

        step.save()
        assert expected_file.exists()
        step.accepted = True
        step.save(overwrite=True)
        assert expected_file.exists()

    def test_load_default(self, simple_trajectory, tmp_path):
        """Test load method with default filename from current directory."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
            accepted=True,
            p_acc=0.8,
        )

        step.save()

        cur_dir = os.getcwd()
        os.chdir(step_dir)
        loaded_step = MCstep.load()
        os.chdir(cur_dir)

        assert loaded_step.step_num == step.step_num
        assert loaded_step.accepted == step.accepted
        assert loaded_step.p_acc == step.p_acc
        assert loaded_step.directory == step.directory

    def test_load_custom_directory(self, simple_trajectory, tmp_path):
        """Test load method with specified directory."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
        )

        step.save()

        loaded_step = MCstep.load(directory=str(step_dir))

        assert loaded_step.step_num == step.step_num
        assert loaded_step.directory == step.directory

    def test_load_custom_filename(self, simple_trajectory, tmp_path):
        """Test load method with custom filename."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        step = MCstep(
            mover=None,
            step_num=1,
            directory=str(step_dir),
            path=simple_trajectory,
            default_savename="mydata.pkl",
        )

        step.save(fname=str(step_dir / "mydata.pkl"))

        loaded_step = MCstep.load(directory=str(step_dir), fname="mydata.pkl")

        assert loaded_step.step_num == step.step_num

    def test_save_load_roundtrip(self, simple_trajectory, tmp_path):
        """Test save and load preserves all data."""
        step_dir = tmp_path / "step1"
        step_dir.mkdir()

        dummy_mover = object()
        dummy_committors = np.array([0.5, 0.3, 0.7])
        dummy_states = np.array([1, 1])

        original_step = MCstep(
            mover=dummy_mover,
            step_num=42,
            directory=str(step_dir),
            path=simple_trajectory,
            accepted=True,
            p_acc=0.95,
            weight=3.14,
            predicted_committors_sp=dummy_committors,
            shooting_snap=simple_trajectory,
            states_reached=dummy_states,
            trial_trajectories=[simple_trajectory, simple_trajectory],
            default_savename="roundtrip.pkl",
        )

        original_step.save()

        loaded_step = MCstep.load(directory=str(step_dir), fname="roundtrip.pkl")

        # Check all attributes
        assert isinstance(loaded_step.mover, object)
        assert loaded_step.step_num == 42
        assert loaded_step.directory == str(step_dir)
        assert loaded_step.accepted is True
        assert loaded_step.p_acc == 0.95
        assert loaded_step.weight == 3.14
        assert np.array_equal(loaded_step.predicted_committors_sp, dummy_committors)
        assert loaded_step.shooting_snap is simple_trajectory
        assert np.array_equal(loaded_step.states_reached, dummy_states)
        assert len(loaded_step.trial_trajectories) == 2
        assert loaded_step.default_savename == "roundtrip.pkl"


class TestDensityAdaptionParameters:
    """Tests for the DensityAdaptionParameters dataclass."""

    def test_init_defaults(self):
        """Test DensityAdaptionParameters with all defaults."""
        params = DensityAdaptionParameters()

        assert params.n_bins == 10
        assert params.scheme is None
        assert params.reevaluate_density_interval is None
        assert params.reset_before_pick is True
        assert params.add_trajectories_from_sampler is True
        assert params.trajectories_to_flatten == []
        assert params.trajectories_to_flatten_weights is None

    def test_init_custom(self, simple_trajectory, trajectory_alphaR):
        """Test DensityAdaptionParameters with custom values and scheme=None."""
        weights1 = np.array([0.5])
        weights2 = np.array([0.3])

        params = DensityAdaptionParameters(
            n_bins=20,
            scheme=None,
            reevaluate_density_interval=5,
            reset_before_pick=False,
            add_trajectories_from_sampler=False,
            trajectories_to_flatten=[simple_trajectory, trajectory_alphaR],
            trajectories_to_flatten_weights=[weights1, weights2],
        )

        assert params.n_bins == 20
        assert params.scheme is None
        assert params.reevaluate_density_interval == 5
        assert params.reset_before_pick is False
        assert params.add_trajectories_from_sampler is False
        assert len(params.trajectories_to_flatten) == 2
        assert np.array_equal(params.trajectories_to_flatten_weights[0], weights1)
        assert np.array_equal(params.trajectories_to_flatten_weights[1], weights2)

    def test_post_init_none(self):
        """Test __post_init__ with scheme=None (no changes)."""
        params = DensityAdaptionParameters(
            n_bins=15,
            scheme=None,
            reset_before_pick=False,
            add_trajectories_from_sampler=False,
        )

        assert params.n_bins == 15
        assert params.reset_before_pick is False
        assert params.add_trajectories_from_sampler is False
        assert params.reevaluate_density_interval is None

    def test_post_init_lazzeri_lowercase(self):
        """Test __post_init__ with lazzeri scheme (lowercase)."""
        params = DensityAdaptionParameters(
            n_bins=10,
            scheme="lazzeri",
            reset_before_pick=False,
            add_trajectories_from_sampler=False,
            reevaluate_density_interval=5,
        )

        assert params.scheme == "lazzeri"
        assert params.reset_before_pick is True
        assert params.add_trajectories_from_sampler is True
        assert params.reevaluate_density_interval is None

    def test_post_init_p_x_tp_lowercase(self):
        """Test __post_init__ with p_x_tp scheme (lowercase)."""
        params = DensityAdaptionParameters(
            n_bins=10,
            scheme="p_x_tp",
            reset_before_pick=True,
            add_trajectories_from_sampler=False,
            reevaluate_density_interval=10,
        )

        assert params.scheme == "p_x_tp"
        assert params.reset_before_pick is False
        assert params.add_trajectories_from_sampler is True
        assert params.reevaluate_density_interval == 10

    def test_post_init_lazzeri_uppercase(self):
        """Test __post_init__ with LAZZERI scheme (uppercase, should be case-insensitive)."""
        params = DensityAdaptionParameters(
            n_bins=10,
            scheme="LAZZERI",
        )

        assert params.reset_before_pick is True
        assert params.add_trajectories_from_sampler is True
        assert params.reevaluate_density_interval is None

    def test_post_init_p_x_tp_uppercase(self):
        """Test __post_init__ with P_X_TP scheme (uppercase, should be case-insensitive)."""
        params = DensityAdaptionParameters(
            n_bins=10,
            scheme="P_X_TP",
        )

        assert params.reset_before_pick is False
        assert params.add_trajectories_from_sampler is True

    def test_attributes_after_lazzeri(self):
        """Test all parameters are correctly set after lazzeri scheme."""
        params = DensityAdaptionParameters(scheme="lazzeri")

        assert params.reset_before_pick is True
        assert params.add_trajectories_from_sampler is True
        assert params.reevaluate_density_interval is None
        assert params.n_bins == 10
        assert params.trajectories_to_flatten == []

    @pytest.mark.parametrize("reevaluate_interval", [None, 10, 20])
    def test_attributes_after_p_x_tp(self, reevaluate_interval):
        """Test all parameters are correctly set after p_x_tp scheme."""
        params = DensityAdaptionParameters(scheme="p_x_tp",
                                           reevaluate_density_interval=reevaluate_interval,
                                           )

        # if reevaluate_interval = None is passed we should end up with the default
        # for scheme="p_x_TP", which is 10
        if reevaluate_interval is None:
            expected_reevaluate_interval = 10
        else:
            expected_reevaluate_interval = reevaluate_interval

        assert params.reset_before_pick is False
        assert params.add_trajectories_from_sampler is True
        assert params.reevaluate_density_interval is not None
        assert params.reevaluate_density_interval == expected_reevaluate_interval
        assert params.n_bins == 10
        assert params.trajectories_to_flatten == []
