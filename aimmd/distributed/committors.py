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
This file contains the implementation of the :class:`CommittorSimulation`.

It also contains the implementation of the two input dataclasses for
:class:`CommittorSimulation`, the :class:`CommittorEngineSpec` and the
:class:`CommittorConfiguration`, as well as various dataclasses used internally
in the :class:`CommittorSimulation`.
"""
import asyncio
import dataclasses
import logging
import os
import shutil
import typing

import numpy as np
from tqdm.asyncio import tqdm_asyncio

from asyncmd import Trajectory
from asyncmd.mdengine import EngineCrashedError
from asyncmd.trajectory.convert import (RandomVelocitiesFrameExtractor,
                                        InvertedVelocitiesFrameExtractor,
                                        )
from asyncmd.trajectory.propagate import (
                                MaxStepsReachedError,
                                ConditionalTrajectoryPropagator,
                                construct_tp_from_plus_and_minus_traj_segments,
                                )
from asyncmd.utils import ensure_mdconfig_options

from ..tools import attach_kwargs_to_object as _attach_kwargs_to_object

if typing.TYPE_CHECKING:  # pragma: no cover
    from asyncmd.mdengine import MDEngine
    from asyncmd.trajectory.functionwrapper import TrajectoryFunctionWrapper
    T = typing.TypeVar("T")  # a generic typevar


logger = logging.getLogger(__name__)


def _is_documented_by(docstring, *format_args):
    """
    Decorator to add the given docstring to the decorated method.
    Optionally perform formatting on the string with ``format_args``.
    """
    def wrapper(target):
        target.__doc__ = docstring.format(*format_args)
        return target
    return wrapper


@dataclasses.dataclass
class CommittorEngineSpec:
    """
    Specify MDEngine and other propagation options for :class:`CommittorSimulation`.

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
class CommittorConfiguration:
    """
    Specify configurations for :class:`CommittorSimulation`

    Parameters
    ----------
    trajectory : asyncmd.Trajectory
        The trajectory on which the configuration lies.
    index : int
        The index of the configuration in the trajectory.
    name : str | None
        An optional name used to identify the configuration.
    """
    trajectory: Trajectory
    index: int
    name: str | None = None


@dataclasses.dataclass
class _CommittorSimulationDirectoryPrefixes:
    """
    Store (default) values for directory (prefixes) used by :class:`CommittorSimulation`.
    """
    configuration_prefix: str = "configuration_"
    trial_prefix: str = "trial_"
    forward: str = "forward_propagation"
    backward: str = "backward_propagation"


@dataclasses.dataclass
class _CommittorSimulationOutFilenames:
    """
    Store (default) values for names of output files generated by :class:`CommittorSimulation`.
    """
    traj_to_state: str = "trajectory_to_state"
    traj_to_state_bw: str = "trajectory_to_state_backward"
    transition: str = "transition_trajectory"
    initial_conf: str = "initial_configuration"
    initial_conf_bw: str = "initial_configuration_backward"
    deffnm_engine: str = "committor"


@dataclasses.dataclass
class _CommittorSimulationMaxRetries:
    """
    Store (default) values for how often the associated :class:`CommittorSimulation`
    retries trials in case of various errors.

    Note: A **retry** value of 0 means try once and dont retry. A retry value
    of 1 means try at most 2 times, i.e. retry once on failure.
    """
    crash: int = 1
    max_steps: int = 0


@dataclasses.dataclass
class _PreparedTrialData:
    """
    Store data needed to run one trial (one direction) of the :class:`CommittorSimulation`.

    Used internally to pass data between methods.
    """
    ns: dict[str, int]
    sp_conf: Trajectory
    workdir: str
    continuation: bool
    tra_out: str | None
    engine_spec: CommittorEngineSpec
    overwrite: bool


class CommittorSimulation:
    # this class is "just" very complex, but most of its methods are properties
    # and pylint also gets thrown off in counting attrs due to the way we set
    # the properties in init (each of them counts twice [one for the property
    # and one for the private/underscored version])...so:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    """
    Run committor simulation for multiple configurations in parallel.

    Given a list of configurations (as :class:`CommittorConfiguration`) and a
    list of states propagate trajectories until any of the states is reached.
    Writes out the concatenated trajectory from the starting configuration to
    the first state reached.
    If `two_way` is True, trials will also be performed with inverted momenta,
    i.e. backward. For these trials concatenated output trajectories will also
    be written, but additionally transition trajectories are written out if the
    forward and backward propagation of the same trial end in different states.
    The transition will be ordered going from the lower index state to the higher
    index state.
    The results/states reached for forward and backward trials are stored and
    returned separately due to the correlation in outcomes between forward and
    backward trials.

    This class will create the following directory path for each trial, with
    ``$CONF_DIR_NAME`` either ``conf_$CONF_NUM`` or the name of the configuration,
    and ``$CONF_NUM`` and ``$TRIAL_NUM`` the indices of the configuration and trial,
    respectively: ``$WORKDIR/$CONF_DIR_NAME/trial_$TRIAL_NUM``.
    Inside of each trial directory will be the shooting points and concatenated
    output trajectories and a directory for each the forward and the backward
    propagation, respectively.
    """
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        workdir: str,
        configurations: list[CommittorConfiguration], *,
        states: list["TrajectoryFunctionWrapper"],
        temperature: float | list[float],
        committor_engine_spec: CommittorEngineSpec | list[CommittorEngineSpec],
        two_way: bool | list[bool] = False,
        **kwargs: dict,
    ) -> None:
        """
        Initialize a :class:`CommittorSimulation`.

        All attributes and properties can be set at initialization by passing
        keyword arguments with their name.

        Note, that the :class:`CommittorSimulation` allows to vary some of the
        simulation parameters on a per configuration basis. These can be either
        a list of values (length must then be equal to the number of `configurations`)
        or a single value.
        This means you can simulate systems differing in the number of
        molecules, at different pressures, or at different temperatures (by using
        different :class:`CommittorEngineSpec` and even perform two way trials
        only for a selected subset of configurations (e.g. the ones you expect
        to be a transition state).

        Parameters
        ----------
        workdir : str
            The working directory of the simulation.
        configurations : list[CommittorConfiguration]
            List of input configurations to perform committor simulation for.
        states : list[TrajectoryFunctionWrapper]
            List of states (stopping conditions) for the trial propagation.
        temperature : float | list[float]
            Temperature used to draw random Maxwell-Boltzmann velocities. Measured
            in degree Kelvin. Can vary on a per-configuration basis or be the
            same for all configurations.
            Note: It is the users responsibility to ensure that the `temperature`
            and other temperatures potentially passed via `committor_engine_spec`
            (e.g. for thermostats in the MD engine) agree!
        committor_engine_spec : CommittorEngineSpec | list[CommittorEngineSpec]
            Description/Specification of the MD engine (including parameters)
            used in the trial propagation. See :class:`CommittorEngineSpec` for
            what is included. Can vary on a per-configuration basis.
        two_way : bool | list[bool], optional
            Whether to perform two way trials, by default False.
            Can vary on a per-configuration basis or be the same for all configurations.

        Raises
        ------
        ValueError
            If the given `name`s for the :class:`CommittorConfiguration`s would
            lead to non-unique directory names.
        """
        # internal counter (needed for the properties [below] to work)
        self._trial_counter = 0  # trials per configuration
        # sort out the arguments we always need/get
        # Note: workdir, configurations, and states are read-only properties
        self._workdir = os.path.relpath(workdir)
        if not os.path.isdir(self.workdir):
            logger.warning("Working directory (workdir=%s) does not exist."
                           "We will create it when we need it.", self.workdir)
            # we will create directories as needed so no need to create it here
            # (as we might error below if there are multiple configurations with
            #  the same name)
        self._configurations = configurations
        self._states = states
        # use the properties to set (and check the values)
        self.temperature = temperature
        self.committor_engine_spec = committor_engine_spec
        self.two_way = two_way
        # lists to store which state which particular trial reached
        self._states_reached: list[list[int | None]] = [
                    [] for _ in range(len(self.configurations))
                    ]
        self._states_reached_bw: list[list[int | None]] = [
                    [] for _ in range(len(self.configurations))
                    ]
        # init the dataclasses before we attach kwargs so we can change the defaults
        # from the dataclasses via kwargs (that have the same names as the properties)
        self._dirs = _CommittorSimulationDirectoryPrefixes()
        self._fnames = _CommittorSimulationOutFilenames()
        self._retries = _CommittorSimulationMaxRetries()
        # variable that stores whether we might miss some backward trials because
        # we changed the two_way setting (we will add them as requested and in run)
        self._potentially_missing_backward_trials = False
        # now attach all additional kwargs to self
        _attach_kwargs_to_object(obj=self, logger=logger, **kwargs)
        # finally, make sure that we will have unique configuration directory names
        conf_dirs = [self._get_conf_dir(conf_num=i)
                     for i in range(len(self.configurations))
                     ]
        if len(conf_dirs) != len(set(conf_dirs)):
            raise ValueError("The directory names for the configurations are "
                             "not unique! Maybe you used the same name for "
                             "multiple configurations?"
                             )

    @property
    def workdir(self) -> str:
        """Toplevel working directory of this simulation."""
        return self._workdir

    @property
    def configurations(self) -> list[CommittorConfiguration]:
        """List of input configurations to perform committor simulation for."""
        return self._configurations

    @property
    def states(self) -> list["TrajectoryFunctionWrapper"]:
        """List of states (stopping conditions) for trial propagation."""
        return self._states

    def _ensure_list_len_or_single(self, val: "T | list[T]", name: str
                                   ) -> "T | list[T]":
        """
        Make sure that a value is either one value or a list of values of the
        same length as self.configurations.
        Used to ensure proper format for values we allow to be set on a
        per-configuration basis, e.g., temperature, engine_spec, two_way.
        """
        if isinstance(val, list):
            if not len(val) == len(self.configurations):
                raise ValueError("Must supply either one or exactly as many"
                                 f"{name} values as configurations.")
        return val

    def _get_value_corresponding_to_configuration(self, value: "T | list[T]",
                                                  conf_num: int,
                                                  ) -> "T":
        """
        Get the corresponding value from value or optionally a list of values
        for configuration identified by conf_num.
        This is used to ensure we always easily get (the correct) single value
        for all the attributes/values we allow to be set on a per-configuration
        basis.
        """
        if isinstance(value, list):
            return value[conf_num]
        return value

    def _ensure_consistent_engine_specs(self) -> None:
        """
        Method called when setting property that influences the trial propagation,
        its sole purpose is to raise an error when we already done any trials, i.e.,
        changing the settings is not possible (easily) anymore.
        """
        if self._trial_counter > 0:
            raise ValueError("Changing the engine specification is not supported after "
                             "trials have been performed or reinitialized from workdir."
                             )

    @property
    def temperature(self) -> float | list[float]:
        """Temperature(s) used to generate Maxwell-Boltzmann velocities, in Kelvin."""
        return self._temperature

    @temperature.setter
    def temperature(self, val: float | list[float]) -> None:
        self._ensure_consistent_engine_specs()
        self._temperature = self._ensure_list_len_or_single(val, "temperature")

    @property
    def committor_engine_spec(self) -> CommittorEngineSpec | list[CommittorEngineSpec]:
        """Specification(s) of the MD engines used for the trial propagation."""
        return self._committor_engine_spec

    @committor_engine_spec.setter
    def committor_engine_spec(self, val: CommittorEngineSpec | list[CommittorEngineSpec],
                              ) -> None:
        self._ensure_consistent_engine_specs()
        self._committor_engine_spec = self._ensure_list_len_or_single(
                                                val, "committor_engine_spec")

    @property
    def two_way(self) -> bool | list[bool]:
        """Whether to perform two way trials."""
        return self._two_way

    @two_way.setter
    def two_way(self, val: bool | list[bool]) -> None:
        self._two_way = self._ensure_list_len_or_single(val, "two_way")
        if self._trial_counter > 0:
            # Note: This is possible, since we just add the backward trials
            #       (and do not change the engine_spec itself)
            logger.warning("Changing two_way after trials have been performed. "
                           "Please call `add_missing_backward_trials` method "
                           "to ensure consistency."
                           )
            self._potentially_missing_backward_trials = True

    def _ensure_consistent_dirnames_and_fnames(self) -> None:
        """
        Method called when setting a directory or filename property, its sole
        purpose is to raise an error when we already have created the directory
        structure, i.e. when changing the names is not possible anymore (easily).
        """
        if self._trial_counter > 0:
            raise ValueError("Changing the directory names is not supported after "
                             "the directory structure has been created. I.e. when "
                             "trials have been performed or reinitialized from workdir."
                             )

    @property
    def directory_configuration_prefix(self) -> str:
        """
        The prefix for the configuration directories, used only for un-named configurations.
        The directory name will be ``$DIRECTORY_CONFIGURATION_PREFIX_$CONF_NUM``.
        """
        return self._dirs.configuration_prefix

    @directory_configuration_prefix.setter
    def directory_configuration_prefix(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._dirs.configuration_prefix = val

    @property
    def directory_trial_prefix(self) -> str:
        """
        The prefix for the trial directories, the directory name will be
        ``$DIRECTORY_TRIAL_PREFIX_$TRIAL_NUM``.
        """
        return self._dirs.trial_prefix

    @directory_trial_prefix.setter
    def directory_trial_prefix(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._dirs.trial_prefix = val

    @property
    def directory_forward(self) -> str:
        """The directory name used for the forward propagation."""
        return self._dirs.forward

    @directory_forward.setter
    def directory_forward(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._dirs.forward = val

    @property
    def directory_backward(self) -> str:
        """The directory name used for the backward propagation."""
        return self._dirs.backward

    @directory_backward.setter
    def directory_backward(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._dirs.backward = val

    @property
    def fileout_trajectory_to_state(self) -> str:
        """
        The filename for the output trajectories to the forward state to be written
        to each trial directory.

        Note: Without fileending, the fileending will be inferred from the respective
        committor engine specification.
        """
        return self._fnames.traj_to_state

    @fileout_trajectory_to_state.setter
    def fileout_trajectory_to_state(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._fnames.traj_to_state = val

    @property
    def fileout_trajectory_to_state_backward(self) -> str:
        """
        The filename for the output trajectories to the backward state to be written
        to each trial directory.

        Note: Without fileending, the fileending will be inferred from the respective
        committor engine specification.
        """
        return self._fnames.traj_to_state_bw

    @fileout_trajectory_to_state_backward.setter
    def fileout_trajectory_to_state_backward(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._fnames.traj_to_state_bw = val

    @property
    def fileout_transition_trajectory(self) -> str:
        """
        The filename for the output transition trajectories to be written to the
        trial directories.

        Note: Without fileending, the fileending will be inferred from the respective
        committor engine specification.
        """
        return self._fnames.transition

    @fileout_transition_trajectory.setter
    def fileout_transition_trajectory(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._fnames.transition = val

    @property
    def fileout_initial_configuration(self) -> str:
        """
        The filename for the initial configuration (including velocities) for the
        forward trial to be written to each trial directory.

        Note: Without fileending, because the file-ending defined in
        :attr:`committor_engine_spec` will be added.
        """
        return self._fnames.initial_conf

    @fileout_initial_configuration.setter
    def fileout_initial_configuration(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._fnames.initial_conf = val

    @property
    def fileout_initial_configuration_backward(self) -> str:
        """
        The filename for the initial configuration (including velocities) for the
        backward trial to be written to each trial directory.

        Note: Without fileending, because the file-ending defined in
        :attr:`committor_engine_spec` will be added.
        """
        return self._fnames.initial_conf_bw

    @fileout_initial_configuration_backward.setter
    def fileout_initial_configuration_backward(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._fnames.initial_conf_bw = val

    @property
    def fileout_deffnm_engine(self) -> str:
        """The deffnm used for the MD engines in trial propagation."""
        return self._fnames.deffnm_engine

    @fileout_deffnm_engine.setter
    def fileout_deffnm_engine(self, val: str) -> None:
        self._ensure_consistent_dirnames_and_fnames()
        self._fnames.deffnm_engine = val

    @property
    def max_retries_on_crash(self) -> int:
        """
        Maximum number of retries (per trial) in case of MD engine crash.

        Note: After max_retries is reached the error will be raised!
        """
        return self._retries.crash

    @max_retries_on_crash.setter
    def max_retries_on_crash(self, val: int) -> None:
        self._retries.crash = val

    @property
    def max_retries_on_max_steps(self) -> int:
        """
        Maximum number of retries (per trial) in case of max_steps reached.

        Note: If no state is reached after max_retries + 1 tries, the trial will
        have state_reached=None and simply not be included in the state_reached
        properties. No error will be raised!
        """
        return self._retries.max_steps

    @max_retries_on_max_steps.setter
    def max_retries_on_max_steps(self, val: int) -> None:
        self._retries.max_steps = val

    @property
    def trial_counter(self) -> int:
        """Return the number of trials done per configuration."""
        return self._trial_counter

    def _process_states_reached(self,
                                internal_states_reached: list[list[int | None]],
                                ) -> np.ndarray:
        """
        Helper function for states_reached and states_reached_backward properties.
        Takes internal states reached representation and turns it into a np.array.
        """
        ret = np.zeros((len(self.configurations), len(self.states)))
        for i, results_for_conf in enumerate(internal_states_reached):
            for state_reached in results_for_conf:
                if state_reached is not None:
                    ret[i][state_reached] += 1
        return ret
    _PROCESS_STATES_REACHED_DOCSTRING = """
        Return states_reached per configuration (i.e. summed over trials).
        These are only the results of the {} propagation.

        Return states_reached as a np.array with shape (n_conf, n_states), where
        the entries give the counts of states reached for a given configuration.
        """

    @property
    @_is_documented_by(_PROCESS_STATES_REACHED_DOCSTRING, "forward")
    # pylint: disable-next=missing-function-docstring
    def states_reached(self) -> np.ndarray:
        return self._process_states_reached(self._states_reached)

    @property
    @_is_documented_by(_PROCESS_STATES_REACHED_DOCSTRING, "backward")
    # pylint: disable-next=missing-function-docstring
    def states_reached_backward(self) -> np.ndarray:
        return self._process_states_reached(self._states_reached_bw)

    def _process_states_reached_per_trial(self,
                                          internal_states_reached: list[list[int | None]],
                                          ) -> np.ndarray:
        """
        Helper method for states_reached_per_trial and states_reached_per_trial_backward.
        Takes internal states reached representation and turns it into a np.array.
        """
        ret = np.zeros(
                (len(self.configurations), self._trial_counter, len(self.states))
                )
        for i, results_for_conf in enumerate(internal_states_reached):
            for j, state_reached in enumerate(results_for_conf):
                if state_reached is not None:
                    ret[i][j][state_reached] += 1
        return ret
    _PROCESS_STATES_REACHED_PER_TRIAL_DOCSTRING = """
        Return states_reached per trial (i.e. single trial results).
        These are only the results of the {} propagation.

        Return a np.array shape (n_conf, n_trials, n_states), where the entries
        give the counts of states reached for every single trial, i.e. summing
        over the last (n_states) axis will always give 1.
        """

    @property
    @_is_documented_by(_PROCESS_STATES_REACHED_DOCSTRING, "forward")
    # pylint: disable-next=missing-function-docstring
    def states_reached_per_trial(self) -> np.ndarray:
        return self._process_states_reached_per_trial(self._states_reached)

    @property
    @_is_documented_by(_PROCESS_STATES_REACHED_DOCSTRING, "backward")
    # pylint: disable-next=missing-function-docstring
    def states_reached_per_trial_backward(self) -> np.ndarray:
        return self._process_states_reached_per_trial(self._states_reached_bw)

    def _find_trajs_to_state(self, traj_fname: str) -> list[list[Trajectory]]:
        """
        Helper function to find trajectories_to_state, also backward, and the
        transitions.
        NOTE: Because we use this for the transitions too, it is important to
        keep the isfile check!
        """
        trajs_to_state = []
        for conf_num, conf in enumerate(self.configurations):
            trajs_per_conf = []
            engine_spec = self._get_value_corresponding_to_configuration(
                                self.committor_engine_spec, conf_num=conf_num,
                                )
            for trial_num in range(self.trial_counter):
                traj_path = os.path.join(
                                self._get_trial_dir(conf_num=conf_num,
                                                    trial_num=trial_num),
                                f"{traj_fname}.{engine_spec.output_traj_type}",
                                )
                # NOTE: taking the input structure_file will always work, but
                #       we will not get the structure file from the engine
                struct_fname = conf.trajectory.structure_file
                if os.path.isfile(traj_path):
                    trajs_per_conf += [Trajectory(trajectory_files=traj_path,
                                                  structure_file=struct_fname)
                                       ]
            trajs_to_state += [trajs_per_conf]
        return trajs_to_state
    _FIND_TRAJS_TO_STATE_DOCSTRING = """
        Return all {} trajectories until a state generated so far.

        The return value is a list of lists, the outer list corresponds to
        configurations, the inner lists to the trials.
        """

    @property
    @_is_documented_by(_FIND_TRAJS_TO_STATE_DOCSTRING, "forward")
    # pylint: disable-next=missing-function-docstring
    def trajectories_to_state(self):
        return self._find_trajs_to_state(
                            traj_fname=self.fileout_trajectory_to_state)

    @property
    @_is_documented_by(_FIND_TRAJS_TO_STATE_DOCSTRING, "backward")
    # pylint: disable-next=missing-function-docstring
    def trajectories_to_state_backward(self):
        return self._find_trajs_to_state(
                            traj_fname=self.fileout_trajectory_to_state_backward,
                            )

    @property
    def transition_trajectories(self) -> list[list[Trajectory]]:
        """
        Return all transition trajectories generated so far.

        A transition is defined as a trajectory that connects two different states.

        The return value is a list of lists, the outer list corresponds to
        configurations, the inner lists to the trials. The inner lists may be
        empty if no transitions exist for a particluar configuration.
        """
        # can not have any transitions if we did no two_way trials
        if (  # this looks a bit strange, but two_way can be a list of bools:
            (isinstance(self.two_way, list) and not any(self.two_way))
            # or a simple bool:
            or not self.two_way
        ):
            return [[] for _ in range(len(self.configurations))]
        # build up a list of transitions,
        # NOTE: we can just use the same logic as the trajs_to_state properties
        #       to build the list of transitions, since the implementation there
        #       checks if the file exists and otherwise does not attempt to add it.
        #       ATM this method here has its own docstring though since it is
        #       sufficiently different.
        return self._find_trajs_to_state(
                                traj_fname=self.fileout_transition_trajectory,
                                )

    def _get_conf_dir(self, conf_num: int) -> str:
        """
        Helper method to get the path to the configuration directory for a given
        ``conf_num``.
        """
        conf = self.configurations[conf_num]
        return os.path.join(
                    self.workdir,
                    (conf.name if conf.name is not None
                     else f"{self.directory_configuration_prefix}{conf_num}"),
                    )

    def _get_trial_dir(self, conf_num: int, trial_num: int) -> str:
        """
        Helper method to get path to the trial directory corresponding to
        ``conf_num`` and ``trial_num``.
        """
        return os.path.join(
                    self._get_conf_dir(conf_num=conf_num),
                    f"{self.directory_trial_prefix}{trial_num}",
                    )

    async def _get_or_generate_sp_fw(self, conf_num: int, trial_num: int,
                                     ) -> Trajectory:
        """
        Get or generate forward shooting point.

        Parameters
        ----------
        conf_num : int
            The number of the configuration to get/generate the SP for.
        trial_num : int
            The number of the trial to get/generate the SP for.

        Returns
        -------
        Trajectory
            The forward shooting point.
        """
        # generate SP forward (or get and return if it already there)
        init_conf = self.configurations[conf_num]
        trial_dir = self._get_trial_dir(conf_num=conf_num, trial_num=trial_num)
        full_prec_out = self._get_value_corresponding_to_configuration(
                                self.committor_engine_spec, conf_num=conf_num,
                            ).full_precision_traj_type
        sp_name = os.path.join(
                    trial_dir,
                    f"{self.fileout_initial_configuration}.{full_prec_out}",
                    )
        if os.path.isfile(sp_name):
            # already there, so just return it
            return Trajectory(
                    trajectory_files=sp_name,
                    # use the structure file of the corresponding configuration
                    # because it will always exist
                    structure_file=init_conf.trajectory.structure_file,
                    )
        # not there so extract (with random MB-vels) and then constrain
        sp_name_unconstrained = os.path.join(
                    trial_dir,
                    (self.fileout_initial_configuration + "_unconstrained"
                     + f".{full_prec_out}"),
                    )
        engine_spec = self._get_value_corresponding_to_configuration(
                            value=self.committor_engine_spec, conf_num=conf_num)
        temperature = self._get_value_corresponding_to_configuration(
                            value=self.temperature, conf_num=conf_num)
        extractor = RandomVelocitiesFrameExtractor(T=temperature)
        # Note: overwrite any existing unconstrained SP, we only care if the
        # constrained SP is there to decide if we (re)generate the SP
        sp_unconstrained = await extractor.extract_async(
                                            outfile=sp_name_unconstrained,
                                            traj_in=init_conf.trajectory,
                                            idx=init_conf.index,
                                            overwrite=True,
                                            )
        # now constrain using the engine (because it knows about the constraints)
        constraints_engine = engine_spec.engine_cls(**engine_spec.engine_kwargs)
        sp = await constraints_engine.apply_constraints(
                                            conf_in=sp_unconstrained,
                                            conf_out_name=sp_name,
                                            workdir=trial_dir
                                            )
        return sp

    async def _get_or_generate_sp_bw(self, conf_num: int, trial_num: int,
                                     sp_fw: Trajectory) -> Trajectory:
        """
        Get or generate backward shooting point from forward shooting point.

        Parameters
        ----------
        conf_num : int
            The number of the configuration to get/generate the SP for.
        trial_num : int
            The number of the trial to get/generate the SP for.
        sp_fw : Trajectory
            The forward shooting point (to invert the velocities).

        Returns
        -------
        Trajectory
            The backward shooting point.
        """
        # generate backward SP from forward SP, i.e. forward must exist!
        # as for _generate_sp_fw, if it is already there we just return it
        trial_dir = self._get_trial_dir(conf_num=conf_num, trial_num=trial_num)
        full_prec_out = self._get_value_corresponding_to_configuration(
                                self.committor_engine_spec, conf_num=conf_num,
                            ).full_precision_traj_type
        sp_name = os.path.join(
                    trial_dir,
                    f"{self.fileout_initial_configuration_backward}.{full_prec_out}",
                    )
        if os.path.isfile(sp_name):
            # already there, so just return it
            return Trajectory(
                    trajectory_files=sp_name,
                    structure_file=sp_fw.structure_file,
                    )
        # generate from forward by inverting velocities
        extractor = InvertedVelocitiesFrameExtractor()
        sp = await extractor.extract_async(outfile=sp_name, traj_in=sp_fw, idx=0)
        return sp

    async def _run_single_trial_propagation(self, trial: _PreparedTrialData,
                                            ) -> tuple[list[Trajectory], int]:
        """
        Run a single trial propagation, no error handling.

        Parameters
        ----------
        trial : _PreparedTrialData
            Dataclass describing prepared trial to propagate.

        Returns
        -------
        tuple[list[Trajectory], int]
            trajectory_segments, state_reached
        """
        os.makedirs(trial.workdir, exist_ok=True)
        propagator = ConditionalTrajectoryPropagator(
                            conditions=self.states,
                            engine_cls=trial.engine_spec.engine_cls,
                            engine_kwargs=trial.engine_spec.engine_kwargs,
                            walltime_per_part=trial.engine_spec.walltime_per_part,
                            max_steps=trial.engine_spec.max_steps,
                            )
        trajs, state_reached = await propagator.propagate(
                                            starting_configuration=trial.sp_conf,
                                            workdir=trial.workdir,
                                            deffnm=self.fileout_deffnm_engine,
                                            continuation=trial.continuation)
        if trial.tra_out is not None:
            await propagator.cut_and_concatenate(trajs=trajs,
                                                 tra_out=trial.tra_out,
                                                 overwrite=trial.overwrite)
        return trajs, state_reached

    async def _prepare_trial_propagation(self, *, conf_num: int, trial_num: int,
                                         overwrite: bool,
                                         add_backward_only: bool,
                                         ) -> list[_PreparedTrialData]:
        """
        Prepare trial propagation for given ``conf_num`` and ``trial_num``.

        Parameters
        ----------
        conf_num : int
            The number of the configuration to get/generate the SP for.
        trial_num : int
            The number of the trial to get/generate the SP for.
        overwrite : bool
            Whether to overwrite potentially existing outfiles (traj_to_state etc).
        add_backward_only : bool
            Whether to only add the potentially missing backward shots for all
            trials performed yet.

        Returns
        -------
        list[_PreparedTrialData]
            List of dataclasses describing the forward and/or backward propagation.
        """
        def get_n_previous_fails(dirname: str, suffix: str) -> int:
            # Helper function to the number of previous fails for a given directory.
            n = 0
            while os.path.isdir(f"{os.path.normpath(dirname)}_{suffix}{n + 1}"):
                n += 1
            return n

        two_way = self._get_value_corresponding_to_configuration(
                                self.two_way, conf_num=conf_num)
        engine_spec = self._get_value_corresponding_to_configuration(
                                self.committor_engine_spec, conf_num=conf_num)
        trial_dir = self._get_trial_dir(conf_num=conf_num, trial_num=trial_num)
        # make sure the directory exists so we can write out the SP(s)
        os.makedirs(trial_dir, exist_ok=True)
        workdir_fw = os.path.join(trial_dir, self.directory_forward)
        # and get/generate the SP
        sp_fw = await self._get_or_generate_sp_fw(conf_num=conf_num,
                                                  trial_num=trial_num)
        # also setup loop variables and put it all in the dataclass
        trial_data = [_PreparedTrialData(
                        ns={"max_steps": get_n_previous_fails(workdir_fw, "max_steps"),
                            "crash": get_n_previous_fails(workdir_fw, "crash")
                            },
                        sp_conf=sp_fw,
                        workdir=workdir_fw,
                        continuation=os.path.isdir(workdir_fw),
                        tra_out=os.path.join(
                                    trial_dir,
                                    (f"{self.fileout_trajectory_to_state}"
                                     + f".{engine_spec.output_traj_type}"),
                                    ) if not add_backward_only else None,
                        engine_spec=engine_spec,
                        overwrite=overwrite,
                        )
                      ]
        if two_way:
            # same as for forward for backward if this is a two_way trial
            workdir_bw = os.path.join(trial_dir, self.directory_backward)
            sp_bw = await self._get_or_generate_sp_bw(conf_num=conf_num,
                                                      trial_num=trial_num,
                                                      sp_fw=sp_fw)
            trial_data += [_PreparedTrialData(
                            ns={"max_steps": get_n_previous_fails(workdir_bw, "max_steps"),
                                "crash": get_n_previous_fails(workdir_bw, "crash")
                                },
                            sp_conf=sp_bw,
                            workdir=workdir_bw,
                            continuation=os.path.isdir(workdir_bw),
                            tra_out=os.path.join(
                                        trial_dir,
                                        (f"{self.fileout_trajectory_to_state_backward}"
                                         + f".{engine_spec.output_traj_type}"),
                                        ),
                            engine_spec=engine_spec,
                            overwrite=overwrite,
                            )
                           ]

        return trial_data

    async def _handle_trial_run_from_prepared_data(
                        self, trial_data: list[_PreparedTrialData],
                        ) -> tuple[
                                tuple[list[Trajectory], int | None],
                                tuple[list[Trajectory], int | None] | typing.Literal["NO TRIAL"]
                                   ]:
        """
        Handle trial propagation (including error handling and retries) from
        prepared trial dataclass.

        Parameters
        ----------
        trial_data : _PreparedTrialData
            Dataclass describing the prepared trial propagation.

        Returns
        -------
        tuple[tuple[list[Trajectory], int | None],
              tuple[list[Trajectory], int | None] | typing.Literal["NO TRIAL"]
              ]
            tuple of (forward_result, backward_result), where each result is a
            tuple of trajectory_segments, state_reached.
            If no backward trial is performed (bc two_way=False), return literal
            "NO TRIAL" for backward.

        Raises
        ------
        EngineCrashedError
            Reraised if the engine crashed more than ``max_retries_on_crash`` times.
        """
        trials = [asyncio.create_task(self._run_single_trial_propagation(trial=trial))
                  for trial in trial_data
                  ]
        trial_results: list[tuple[list[Trajectory], int | None]] = [
                ([], None) for _ in trials
                ]
        pending = set(trials)
        while pending:
            done, pending = await asyncio.wait(pending,
                                               return_when=asyncio.FIRST_EXCEPTION,
                                               )
            for t in done:
                t_idx = trials.index(t)
                if t.exception() is None:
                    # no exception raised, just put the result into results...
                    trial_results[t_idx] = t.result()
                    continue  # and go to the next trial
                # exception handling
                if isinstance(t.exception(), EngineCrashedError):
                    if trial_data[t_idx].ns["crash"] >= self.max_retries_on_crash:
                        # raise it if we are above number of retries
                        raise RuntimeError(
                            "MD propagation for trial in directory "
                            f"{trial_data[t_idx].workdir} "
                            f"failed for the {trial_data[t_idx].ns['crash'] + 1}"
                            "th time."
                            ) from t.exception()
                    # otherwise increase crash counter and handle crash
                    trial_data[t_idx].ns["crash"] += 1
                    log_reason = "the engine crashed"
                    # suffix for moved folder so we have space for retry
                    move_suffix = f"_crash{trial_data[t_idx].ns['crash']}"
                elif isinstance(t.exception(), MaxStepsReachedError):
                    log_reason = "the propagation reached maximum number of steps"
                    if trial_data[t_idx].ns["max_steps"] >= self.max_retries_on_max_steps:
                        # log and get out of here because we will not retry
                        logger.error("MD propagation in folder %s did not reach"
                                     " a state because %s.",
                                     trial_data[t_idx].workdir, log_reason)
                        continue
                    # increase max_steps counter and retry
                    trial_data[t_idx].ns["max_steps"] += 1
                    move_suffix = f"_max_steps{trial_data[t_idx].ns['max_steps']}"
                else:
                    # any other exception (we dont handle): raise it
                    for task in pending:
                        task.cancel()  # cancel potential other tasks
                    raise t.exception() from None
                logger.error("MD propagation in folder %s did not reach a state "
                             "because %s. Will retry with a fresh propagation "
                             "from the same initial configuration now.",
                             trial_data[t_idx].workdir, log_reason)
                # if we got until here we are retrying
                # move the previous directory
                move_dir = f"{os.path.normpath(trial_data[t_idx].workdir)}{move_suffix}"
                shutil.move(trial_data[t_idx].workdir, move_dir)
                logger.info("Moved directory %s to %s. Now retrying a fresh run in %s",
                            trial_data[t_idx].workdir, move_dir,
                            trial_data[t_idx].workdir)
                # create a new trial
                trial_data[t_idx].continuation = False  # set continuation to False
                new_trial = asyncio.create_task(
                    self._run_single_trial_propagation(trial=trial_data[t_idx])
                    )
                # remove the old trial from the list of trials
                _ = trials.pop(t_idx)
                # and insert the new trial run at its position
                trials.insert(t_idx, new_trial)
                # and add it to pending trials so we await it next iteration
                pending.add(new_trial)

        if len(trial_results) > 1:
            return (trial_results[0], trial_results[1])
        return (trial_results[0], "NO TRIAL")

    async def _run_trial(self, *, conf_num: int, trial_num: int,
                         overwrite: bool, add_backward_only: bool,
                         ) -> tuple[int | None,
                                    int | None | typing.Literal["NO TRIAL"]
                                    ]:
        """
        Run a trial specified by ``conf_num`` and ``trial_num``.

        Takes care of propagating the forward and potentially backward trial,
        also generates all output trajectories (trajectory_to_state forward and
        backward, and transition_trajectory).

        Parameters
        ----------
        conf_num : int
            The number of the configuration to get/generate the SP for.
        trial_num : int
            The number of the trial to get/generate the SP for.
        overwrite : bool
            Whether to overwrite potentially existing outfiles (traj_to_state etc).
        add_backward_only : bool
            Whether to only add the potentially missing backward shots for all
            trials performed yet.

        Returns
        -------
        tuple[int | None, int | None | typing.Literal["NO TRIAL"]]
            Tuple of (fw_state_reached, bw_state_reached). state_reached will be
            None if the trial did not reach any state, literal "NO TRIAL" is used
            if no backward trial is performed (because two_way=False).
        """
        prepared_trial = await self._prepare_trial_propagation(
                                        conf_num=conf_num,
                                        trial_num=trial_num,
                                        overwrite=overwrite,
                                        add_backward_only=add_backward_only,
                                        )
        fw_res, bw_res = await self._handle_trial_run_from_prepared_data(
                                                trial_data=prepared_trial,
                                                )
        if bw_res == "NO TRIAL":
            # no backward propagation, so no point in trying to make a transition
            return (fw_res[1], "NO TRIAL")
        # we did a forward and a backward trial, so check if we have a transition
        if (
            # the state can be None for max_steps reached, and then we cant make
            # a transition (obviously)
            fw_res[1] is not None
            and bw_res[1] is not None
            and fw_res[1] != bw_res[1]
        ):
            # and if we ended in two different states, write out the transition
            trial_dir = self._get_trial_dir(conf_num=conf_num, trial_num=trial_num)
            engine_spec = self._get_value_corresponding_to_configuration(
                                self.committor_engine_spec, conf_num=conf_num,
                                )
            logger.info(
                "Forward and backward propagation ended in two different "
                "states for trial in directory %s. "
                "Will write out a transition trajectory now.",
                trial_dir
                )
            transition_traj_name = os.path.join(
                                        trial_dir,
                                        (f"{self.fileout_transition_trajectory}"
                                         + f".{engine_spec.output_traj_type}"),
                                        )
            await construct_tp_from_plus_and_minus_traj_segments(
                        minus_trajs=bw_res[0], minus_state=bw_res[1],
                        plus_trajs=fw_res[0], plus_state=fw_res[1],
                        state_funcs=self.states, tra_out=transition_traj_name,
                        overwrite=overwrite,
                        )

        # and always return the states reached also for no transition
        return (fw_res[1], bw_res[1])

    async def _run(self, *, trials_per_configuration: int, overwrite: bool,
                   n_max_concurrent: int | None, add_backward_only: bool,
                   ) -> None:
        """
        Run ``trial_per_configuration`` (additional) trials for each configuration.

        Parameters
        ----------
        trials_per_configuration : int
            Number of additional trials to perform per configuration.
        overwrite : bool
            Whether to overwrite potentially existing outfiles (traj_to_state etc).
        n_max_concurrent : int | None, optional
            How many trials to run concurrently at maximum. None means unlimited,
            by default None.
        add_backward_only : bool
            Whether to only add the potentially missing backward shots for all
            trials performed yet.
        """
        # construct the tasks all at once,
        # ordering is such that we first finish all trials for configuration 0
        # then configuration 1, i.e. we order by configuration and not by trial_num
        tasks = []
        # if we only add backward shots we need to start with the 0th trial for
        # every configuration
        start_trial = self.trial_counter if not add_backward_only else 0
        end_trial = (self.trial_counter + trials_per_configuration
                     if not add_backward_only else self.trial_counter)
        for conf_num in range(len(self.configurations)):
            tasks += [self._run_trial(conf_num=conf_num,
                                      trial_num=trial_num,
                                      overwrite=overwrite,
                                      add_backward_only=add_backward_only,
                                      )
                      for trial_num in range(start_trial, end_trial)
                      ]
        if n_max_concurrent is not None:
            # wrap all tasks with a semaphore so only n_max_concurrent of them
            # can run simultaneously
            semaphore = asyncio.Semaphore(n_max_concurrent)

            async def sem_task(task):
                async with semaphore:
                    return await task
            tasks = [sem_task(task) for task in tasks]
        # run them all (and add a tqdm status bar)
        results = await tqdm_asyncio.gather(*tasks)
        # results is a list of tuples with the idx to the states reached
        # we unpack it and add it to the internal states_reached counters
        for conf_num in range(len(self.configurations)):
            start = conf_num * trials_per_configuration
            stop = (conf_num + 1) * trials_per_configuration
            # only add the forward results if we actually did forward, i.e.
            # if we did only backward we already know about the fw results
            if not add_backward_only:
                self._states_reached[conf_num] += [r[0] for r in results[start:stop]]
            # only add the backward results where we actually performed a trial
            # due to two_way being True
            self._states_reached_bw[conf_num] += [r[1] for r in results[start:stop]
                                                  if r[1] != "NO TRIAL"
                                                  ]
        # increment internal trial (per struct) counter
        if not add_backward_only:
            # if we only added the missing backward shots, naturally we do not
            # increment
            self._trial_counter += trials_per_configuration

    async def run(self, trials_per_configuration: int, overwrite: bool = False,
                  n_max_concurrent: int | None = None,
                  ) -> None:
        """
        Run ``trials_per_configuration`` additional committor trials for each configuration.

        Parameters
        ----------
        trials_per_configuration : int
            Number of additional trial runs to perform per configuration.
        overwrite : bool, optional
            Whether to overwrite potentially existing output trajectories, e.g.
            trajectory_to_state, by default True.
        n_max_concurrent : int | None, optional
            How many trials to run concurrently at maximum. None means unlimited,
            by default None.
        """
        if self._potentially_missing_backward_trials:
            logger.warning("There might be missing backward trials. "
                           "Will ensure consistency by running "
                           "`add_missing_backward_trials` first.")
            await self.add_missing_backward_trials(overwrite=overwrite,
                                                   n_max_concurrent=n_max_concurrent,
                                                   )
        await self._run(trials_per_configuration=trials_per_configuration,
                        overwrite=overwrite,
                        n_max_concurrent=n_max_concurrent,
                        add_backward_only=False,
                        )

    async def add_missing_backward_trials(self, overwrite: bool = False,
                                          n_max_concurrent: int | None = None,
                                          ) -> None:
        """
        Add potentially missing backward trials, if ``two_way`` has been modified.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite potentially existing output trajectories, e.g.
            trajectory_to_state, by default True.
        n_max_concurrent : int | None, optional
            How many trials to run concurrently at maximum. None means unlimited,
            by default None.
        """
        if self._potentially_missing_backward_trials:
            # only do something if there might be something to do
            await self._run(trials_per_configuration=self._trial_counter,
                            overwrite=overwrite,
                            n_max_concurrent=n_max_concurrent,
                            add_backward_only=True,
                            )
            self._potentially_missing_backward_trials = False

    async def reinitialize_from_workdir(self, overwrite: bool = True,
                                        n_max_concurrent: int | None = None,
                                        ) -> None:
        """
        Reassess all trials in workdir and populate ``states_reached`` counters.

        Possibly extend trials if no state has been reached yet and also add
        potentially missing backward trials.
        Can be used to extend (or shorten) a simulation with modified states or
        to recover after a crash.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite potentially existing output trajectories, e.g.
            trajectory_to_state, by default True.
        n_max_concurrent : int | None, optional
            How many trials to run concurrently at maximum. None means unlimited,
            by default None.
        """
        # make sure we set everything to zero before we start!
        self._trial_counter = 0
        # we will/would add them now if they are missing, so set to False
        self._potentially_missing_backward_trials = False
        self._states_reached = [[] for _ in range(len(self.configurations))]
        self._states_reached_bw = [[] for _ in range(len(self.configurations))]
        # find out how many trials we did per configuration, the first
        # configuration should be the one with the most directories created
        # even if we failed/crashed before everything was created, the
        # run_trial_propagation method will take care of creating the missing
        # directories and initial configurations as appropriate. This way we
        # will end up with as many trials done in each configuration as for the
        # first one
        dir_list = os.listdir(self._get_conf_dir(conf_num=0))
        # build a list of all possible dir names
        # (these will be too many if there are other files in conf dir)
        possible_dirnames = [os.path.split(self._get_trial_dir(conf_num=0,
                                                               trial_num=i)
                                           )[1]  # take only the dirname
                             for i in range(len(dir_list))
                             ]
        # now filter to check that only stuff that is a dir and in possible
        # names will be taken, then count them: this is the number of trials
        # we have done already
        filtered = [d for d in dir_list
                    if (d in possible_dirnames
                        # Note: we can be sure it is there (it just came from
                        # listdir), but we are protecting against there being a
                        # file with the same name as the directory we use ;)
                        and os.path.isdir(os.path.join(
                                            self._get_conf_dir(conf_num=0), d,
                                            )
                                          )
                        )
                    ]
        n_trials = len(filtered)
        # we can just use the run-method as it will start with trial 0
        # and the _prepare_trial_propagation method sorts out if we continue, etc
        await self.run(trials_per_configuration=n_trials, overwrite=overwrite,
                       n_max_concurrent=n_max_concurrent)
