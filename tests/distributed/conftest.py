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
Test fixtures for distributed submodule tests.
"""
import pytest
from asyncmd import Trajectory


@pytest.fixture#(scope="session")
def simple_trajectory():
    """Create a simple asyncmd Trajectory for testing."""
    return Trajectory(
        structure_file="examples/distributed/gmx_infiles/ala_300K_amber99sb-ildn.tpr",
        trajectory_files="examples/distributed/gmx_infiles/conf.trr",
    )


@pytest.fixture#(scope="session")
def trajectory_alphaR():
    """Create a Trajectory with snapshot in alpha_R state."""
    return Trajectory(
        structure_file="examples/distributed/gmx_infiles/ala_300K_amber99sb-ildn.tpr",
        trajectory_files="examples/distributed/gmx_infiles/conf_in_alphaR.trr",
    )


@pytest.fixture#(scope="session")
def trajectory_C7eq():
    """Create a Trajectory with snapshot in C7eq state."""
    return Trajectory(
        structure_file="examples/distributed/gmx_infiles/ala_300K_amber99sb-ildn.tpr",
        trajectory_files="examples/distributed/gmx_infiles/conf_in_C7eq.trr",
    )
