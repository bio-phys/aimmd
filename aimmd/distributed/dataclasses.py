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
This file contains various dataclasses used to tie together input arguments
commonly used together, such as the MDEngineSpec.
"""
import dataclasses
import typing

from asyncmd.utils import ensure_mdconfig_options

if typing.TYPE_CHECKING:  # pragma: no cover
    from asyncmd.mdengine import MDEngine


@dataclasses.dataclass
class MDEngineSpec:
    """
    Specify MDEngine and other propagation options for trial propagation in
    :class:`CommittorSimulation` or in shooting PathMovers like the
    :class:`TwoWayShootingPathMover`.

    Parameters
    ----------
    engine_cls : type[asyncmd.mdengine.MDEngine]
        The MD engine class to use (uninitialized).
    engine_kwargs : dict[str, Any]
        A dictionary with keyword arguments that can be used to initialize the
        MD engine.
    walltime_per_part : float
        MD propagation will be split in parts of walltime, measured in hours.
    max_steps : int
        The maximum number of integration steps to perform without reaching a
        state, i.e. upper cutoff for uncommitted trials.
    full_precision_traj_type : str
        The trajectory type for the full precision trajectories used by this engine.
        Will be used as file-ending for the shooting points.
        Note: Must be a format that stores velocities and coordinates!
        By default "trr".

    Attributes
    ----------
    output_traj_type : str
        The (potentially lossy) output trajectory type of the engine, will be
        inferred automatically from engine_cls and engine_kwargs.
        Note: Only needs to store coordinates (no velocities).
    """
    engine_cls: type["MDEngine"]
    engine_kwargs: dict[str, typing.Any]
    walltime_per_part: float
    max_steps: int | None = None
    full_precision_traj_type: str = "trr"
    output_traj_type: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """
        Ensure that the mdconfig options are what we expect and infer output_traj_type.
        """
        # make sure the mdconfig options are what we expect
        self.engine_kwargs["mdconfig"] = ensure_mdconfig_options(
                                self.engine_kwargs["mdconfig"],
                                # dont generate velocities, we do that ourself
                                genvel="no",
                                # dont apply constraints at start of simulation
                                continuation="yes",
                                )
        # infer output_traj_type
        try:
            # see if it is set as engine_kwarg
            output_traj_type: str = self.engine_kwargs["output_traj_type"]
        except KeyError:
            # it is not, so we use the engine_class default
            output_traj_type = self.engine_cls.output_traj_type
        self.output_traj_type = output_traj_type.lower()
