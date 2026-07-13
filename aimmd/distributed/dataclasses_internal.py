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
This file contains dataclasses used internally in aimmd distributed, e.g. the
:class:`PathSamplingSimStateInfo`.
"""
import dataclasses
import typing

if typing.TYPE_CHECKING:  # pragma: no cover
    from .pathsampling import Brain


@dataclasses.dataclass
class PathSamplingSimStateInfo:
    """
    Dataclass to describe the (state of) current path sampling simulation step.

    Parameters
    ----------
    brain : Brain
        The :class:`Brain` running the simulation.
    sampler_idx : int
        The index of the sampler in which the step is performed.
    step_num : int
        The step number of the current step in the MC chain. This is the number
        of steps performed in the sampler performing this step.
    step_dir : str
        The (working) directory to use for this :class:`MCstep`.
    continuation : bool, optional
        Whether to (try to) continue an existing step from the files found
        in the ``step_dir``.
    """
    brain: "Brain"
    sampler_idx: int
    step_num: int
    step_dir: str
    continuation: bool
