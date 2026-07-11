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
commonly used together, such as the MDEngineSpec or the MCstep.
"""
import dataclasses
import os
import pickle
import typing

from asyncmd.utils import ensure_mdconfig_options

if typing.TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import numpy.typing as npt
    from asyncmd.trajectory.trajectory import Trajectory
    from asyncmd.mdengine import MDEngine
    from .pathmovers import PathMover


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


@dataclasses.dataclass
class MCstep:
    mover: "PathMover"
    step_num: int
    directory: str
    path: "Trajectory"
    accepted: bool = False
    p_acc: float = 0.
    weight: float = 1.
    predicted_committors_sp: "npt.NDArray | None" = None
    shooting_snap: "Trajectory | None" = None
    states_reached: "npt.NDArray | None" = None
    trial_trajectories: "list[Trajectory]" = dataclasses.field(
                                                default_factory=lambda: []
                                                )
    default_savename: str = "mcstep_data.pckl"

    @property
    def transitions(self) -> int:
        """
        The number of transitions that could be formed given `states_reached`.

        Usually zero or one. Note that this will be zero if `states_reached` is
        `None`.

        Returns
        -------
        int
            Number of transitions
        """
        if self.states_reached is None:
            # can not determine number of transitions without states_reached
            return 0
        state_pair_idxs = [(i, j) for i in range(self.states_reached.shape[0])
                           for j in range(i + 1, self.states_reached.shape[0])
                           ]
        return sum(self.states_reached[i] * self.states_reached[j]
                   for i,j in state_pair_idxs)

    @property
    def contains_transition(self) -> bool:
        """
        Whether this :class:`MCStep` contains a transition between two different states.

        Note that this will be False if `states_reached` is `None`.

        Returns
        -------
        bool
            Whether this step contains a transition
        """
        return bool(self.transitions)

    def save(self, fname: str | None = None,
             overwrite: bool = False) -> None:
        """
        Save this :class:`MCstep`.

        Parameters
        ----------
        fname : str | None, optional
            The filename to use, by default None. If None the filename is constructed
            from ``self.directory`` and ``self.default_savename`` attributes.
        overwrite : bool, optional
            Whether to overwrite any existing files with the same name, by default False.

        Raises
        ------
        ValueError
            If a file with filename exists but ``overwrite`` is ``False``.
        """
        if fname is None:
            fname = os.path.join(self.directory, self.default_savename)
        if not overwrite and os.path.exists(fname):
            # we check if it exists, because pickle/open will happily overwrite
            raise ValueError(f"{fname} exists but overwrite=False.")
        with open(fname, "wb") as pfile:
            pickle.dump(self, pfile)

    @classmethod
    def load(cls, directory: str | None = None,
             fname: str | None = None) -> "MCstep":
        """
        Load a :class:`MCstep` from file.

        Parameters
        ----------
        directory : str | None, optional
            The directory to load the :class:`MCstep` from, by default None.
            If None the current working directory is used.
        fname : str | None, optional
            The filename to load the :class:`MCstep` from, by default None.
            If None, the class attribute :attr:`MCstep.default_savename` will be
            used.

        Returns
        -------
        MCstep
            The loaded :class:`MCstep`.
        """
        if directory is None:
            directory = os.getcwd()
        if fname is None:
            fname = cls.default_savename
        fname = os.path.join(directory, fname)
        with open(fname, "rb") as pfile:
            obj = pickle.load(pfile)
        return obj


@dataclasses.dataclass
class DensityAdaptionParameters:
    """
    Dataclass to specify parameters for density adaption in :class:`RCModelSPSelector`.

    This dataclass includes predefined schemes that will ensure consistency of
    the arguments used for this scheme. Currently these are "lazzeri" and "p_x_tp",
    the former flattens the committor observed on the input transition path while
    the latter flattens the committor on the ensemble of transition paths observed
    so far.

    Parameters
    ----------
    n_bins : int
        The number of bins to use (in each dimension of the model prediction) in
        the histogram used for density estimation.
    scheme: Literal["lazzeri", "p_x_tp"] | None
        A string indicating one of the aforementioned predefined schemes or None.
        None means no parameter consistency checking will be performed.
    reevaluate_density_interval : int | None
        The interval (in steps done by the sampler) in which to reevaluate the
        density estimate using the current model. None means do not reevaluate
        ever, it should only be used if your model prediction does not change.
        It can be ignored for schemes that reset before every pick, e.g. "lazzeri",
        since the current model is always used to add the trajectory.
    reset_before_pick : bool
        Whether the density estimate should be cleared before every pick and
        before (potentially) adding the trajectory from this pick. Setting this
        to True and adding only the trajectory from each pick one arrives at the
        "lazzeri" scheme. By default True.
    add_trajectories_from_sampler : bool
        Whether to add the trajectories produced in the Markov Chain to the
        density estimate. By default True.
    trajectories_to_flatten : list[Trajectory]
        An (initial) list of trajectories to flatten the density from. When selecting
        configurations from a predefined reservoir, these trajectories should be
        added here. Could also be used to kickstart density adaption with an initial
        guess. By default an empty list is used.
    trajectories_to_flatten_weights : list[np.ndarray] | None
        List of numpy arrays with weights for the `trajectories_to_flatten`, the
        list must contain as many numpy arrays as there are trajectories and the
        lengths of the arrays must match the corresponding trajectories.
        Can also be None, in that case equal weights for each configuration will
        be used (in case `trajectories_to_flatten` is not empty). By default None.
    """
    n_bins: int = 10
    # commonly used predefined schemes that will ensure consistency for (some) arguments
    # TODO: Add a predefined scheme for correcting density from given trajectories
    scheme: typing.Literal["lazzeri", "p_x_tp"] | None = None
    reevaluate_density_interval: int | None = None
    reset_before_pick: bool = True
    add_trajectories_from_sampler: bool = True
    trajectories_to_flatten: "list[Trajectory]" = dataclasses.field(
                                                    default_factory=lambda: []
                                                    )
    trajectories_to_flatten_weights: "list[npt.NDArray[np.floating]] | None" = None

    def __post_init__(self) -> None:
        """Ensure consistency of arguments for predefined schemes."""
        if self.scheme is None:
            return
        if self.scheme.lower() == "lazzeri":
            self.reset_before_pick = True
            self.add_trajectories_from_sampler = True
            # No need to reevaluate as we always only have
            # one trajectory in the density
            self.reevaluate_density_interval = None
        elif self.scheme.lower() == "p_x_tp":
            self.reset_before_pick = False
            self.add_trajectories_from_sampler = True
