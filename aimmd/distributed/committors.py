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
import os
import logging
import asyncio
import numpy as np

from .trajectory import (Trajectory, RandomVelocitiesFrameExtractor,
                         InvertedVelocitiesFrameExtractor,
                         )
from .mdengine import EngineError, EngineCrashedError
from .logic import (MaxStepsReachedError, TrajectoryPropagatorUntilAnyState,
                    construct_TP_from_plus_and_minus_traj_segments,
                    )
from .gmx_utils import ensure_mdp_options

logger = logging.getLogger(__name__)


# TODO: DOCUMENT! (and clean up)
class CommittorSimulation:
    """
    Run committor simulation for multiple starting configurations in parallel.

    Given a list of starting configurations and a list of states propagate
    trajectories until any of the states is reached. Write out the concatenated
    trajectory from the starting configuration to the first state.
    When twoway shooting is performed additionally write out any potential
    transitions (going from the lower index state to the higher index state).
    Note that the `CommittorSimulation` allows for the simulation of different
    ensembles per starting configuration (see the `__init__` docstring).

    Notable functions:
    ------------------
        - `run(n_per_struct)` performs/adds `n_per_struct` committor trials for
          every starting configuration
        - `reinitialize_from_workdir()` populates the results counters from the
          current workdir, possibly with new/different states and additional
          twoway shooting performed if `self.two_way == True`

    Notable properties:
    -------------------
        - `states_reached`, `states_reached_per_shot` and `shot_counter` keep
          track of the results of the running simulation
        - `trajs_to_state`, `trajs_to_state_bw` and `transitions` give access
          to the resulting concatenated trajectories

    Notable attributes:
    -------------------
        - `fname_traj_to_state`, `fname_traj_to_state`, `fname_transition_traj`
          `deffnm_engine_out` and `deffnm_engine_out_bw` can be used to control
          the names of the output files
        - `max_retries_on_crash`
    """

    # NOTE: the defaults here will results in the layout:
    # $WORKDIR/configuration_$CONF_NUM/shot_$SHOT_NUM,
    # where $WORKDIR is the workdir given at init, and $CONF_NUM, $SHOT_NUM are
    # the index to the input list starting_configurations and a counter for the shots
    # Note that configuration_dir_prefix is only used if no names are given for
    # the configurations
    configuration_dir_prefix = "configuration_"
    shot_dir_prefix = "shot_"
    # together with deffnm this results in "start_conf_trial_bw.trr" and
    # "start_conf_trial_fw.trr"
    start_conf_name_prefix = "start_conf_"
    fname_traj_to_state = "traj_to_state.trr"
    fname_traj_to_state_bw = "traj_to_state_bw.trr"  # only in TwoWay
    fname_transition_traj = "transition_traj.trr"  # only in TwoWay
    deffnm_engine_out = "trial_fw"
    deffnm_engine_out_bw = "trial_bw"  # only in twoway (for runs with inverted v)
    max_retries_on_crash = 2  # maximum number of *retries* on MD engine crash
                              # i.e. setting to 1 means *retry* once on crash

    def __init__(self, workdir, starting_configurations, states, engine_cls,
                 engine_kwargs, engine_run_config, T, walltime_per_part,
                 n_max_concurrent=10, two_way=False, max_steps=None, **kwargs):
        """
        Initialize a `CommittorSimulation`.

        Parameters:
        -----------
        workdir - str, absolute or relative path to an existing working directory
        starting_configurations - list of iterables, each entry in the list is
                                  describing a starting configuration and must
                                  have at least the two entries:
                                   (`aimd.distributed.Trajectory`,
                                    `index_of_conf_in_traj`)
                                  It can optionally have the form:
                                   (`aimd.distributed.Trajectory`,
                                    `index_of_conf_in_traj`,
                                    `name_for_configuration`)
        states - A list of state functions, preferably wrapped using any
                 `aimmd.distributed.TrajectoryFunctionWrapper`
        engine_cls - a subclass of `aimmd.distributed.MDEngine`, the molecular
                     dynamics engine to use
        engine_kwargs - a dictionary with keyword arguments that can be used
                        to instantiate the engine given in `engine_cls`
        engine_run_config - a subclass of `aimmd.distributed.MDConfig`
                            compatible with the given `engine_cls` describing
                            the molecular dynamis parameters to use
        T - float, the temperature to use when generating Maxwell-Boltzmann
            velocities
        walltime_per_part - float, walltime per trajectory segment in hours,
                            note that this does not limit the maximum length of
                            the combined trajectory but only the size/time per
                            single trajectory segment
        n_max_concurrent - int, the maximum number of trials to propagate
                           concurrently, note that for two way simulations you
                           will run 2*`n_max_concurrent` molecular dynamic
                           simulations in parallel
        two_way - bool, wheter to run molecular dynamcis forwards and backwards
                  in time
        max_steps - int or None, the maximum number of integration steps to
                    perform in total per trajectory, note that for two way
                    simulations the combined maximum length of the resulting
                    trajectory will be 2*`max_steps`

        Note that all attributes can be set at intialization by passing keyword
        arguments with their name.

        Note, that the `CommittorSimulation` allows the simulation of different
        physical ensembles for every starting configuration. This is achieved
        by allowing the parameters `engine_cls`, `engine_kwargs`,
        `engine_run_config`, `T` and `twoway` to be either singletons (then
        they aer the same for the whole committor simulation) or a list with of
        same length as `starting_configurations`, i.e. one value per starting
        configuration.
        This means you can simulate systems differing in the number of
        molecules (by changing the topology used in the engine), at different
        pressures (by changing the molecular dynamics parameters passed with
        `engine_run_config`), at different temperatures (by changing `T` and
        the parameters in the `engine_run_config`) and even perform two way
        shots only for a selected subset of starting configurations (e.g. the
        ones you expect to be a transition state).
        """
        def ensure_list(val, length: int, name: str) -> list:
            if isinstance(val, list):
                if not len(val) == length:
                    raise ValueError("Must supply either one or exactly as many"
                                     + f"{name} as starting_configurations.")
            else:
                val = [val] * length
            return val

        # TODO: should some of these be properties?
        self.workdir = os.path.abspath(workdir)
        self.starting_configurations = starting_configurations
        self.states = states
        self.engine_cls = ensure_list(val=engine_cls,
                                      length=len(starting_configurations),
                                      name="engine_cls")
        self.engine_kwargs = ensure_list(val=engine_kwargs,
                                         length=len(starting_configurations),
                                         name="engine_kwargs")
        self.T = ensure_list(val=T, length=len(starting_configurations),
                             name="T")
        self.two_way = ensure_list(val=two_way,
                                   length=len(starting_configurations),
                                   name="two_way")
        # TODO: we assume gmx engines here!
        if isinstance(engine_run_config, list):
            for i in range(len(engine_run_config)):
                engine_run_config[i] = ensure_mdp_options(
                               engine_run_config[i],
                               # dont generate velocities, we do that ourself
                               genvel="no",
                               # dont apply constraints at start of simulation
                               continuation="yes",
                                                          )
        else:
            engine_run_config = ensure_mdp_options(engine_run_config,
                                                   genvel="no",
                                                   continuation="yes",
                                                   )
        self.engine_run_config = ensure_list(val=engine_run_config,
                                             length=len(starting_configurations),
                                             name="engine_run_config")
        self.walltime_per_part = walltime_per_part
        self.n_max_concurrent = n_max_concurrent
        self.max_steps = max_steps
        # make it possible to set all existing attributes via kwargs
        # check the type for attributes with default values
        dval = object()
        for kwarg, value in kwargs.items():
            cval = getattr(self, kwarg, dval)
            if cval is not dval:
                if isinstance(value, type(cval)):
                    # value is of same type as default so set it
                    setattr(self, kwarg, value)
                else:
                    raise TypeError(f"Setting attribute {kwarg} with "
                                    + f"mismatching type ({type(value)}). "
                                    + f" Default type is {type(cval)}."
                                    )
        # set counter etc after to make sure they have the value we expect
        self._shot_counter = 0
        self._states_reached = [[] for _ in range(len(self.starting_configurations))]
        # create directories for the configurations if they dont exist
        # also keep the configuration dirs in a list
        # this way users can choose their favourite name for each configuration
        self._conf_dirs = []
        for i, vals in enumerate(self.starting_configurations):
            if len(vals) >= 3:
                # starting_configurations are tuples/list containing at least
                # the traj(==conf), the index in the traj (==idx)
                # and optionaly a name to use
                conf, idx, name = vals
                conf_dir = os.path.join(self.workdir, f"{name}")
            else:
                conf_dir = os.path.join(self.workdir,
                                        f"{self.configuration_dir_prefix}{str(i)}")
            self._conf_dirs.append(conf_dir)
            if not os.path.isdir(conf_dir):
                # if its not a directory it either exists (then we will err)
                # or we just create it
                os.mkdir(conf_dir)

    @property
    def shot_counter(self):
        """Return the number of shots per configuration."""
        return self._shot_counter

    @property
    def states_reached(self):
        """
        states_reached per configuration (i.e. summed over shots)

        Return states_reached as a np.array with shape (n_conf, n_states),
        where the entries give the counts of states reached, i.e. the format is
        as in an `arcd.TrainSet`.
        """
        ret = np.zeros((len(self.starting_configurations), len(self.states)))
        for i, results_for_conf in enumerate(self._states_reached):
            for state_reached in results_for_conf:
                if state_reached is not None:
                    ret[i][state_reached] += 1
        return ret

    @property
    def states_reached_per_shot(self):
        """
        states_reached per shot (i.e. single trial results)

        Return a np.array shape (n_conf, n_shots, n_states), where the entries
        give the counts of states reached for every single shot, i.e. summing
        over the states axis will always give 1 (or 2 if twoway=True).
        """
        ret = np.zeros((len(self.starting_configurations),
                        self._shot_counter,
                        len(self.states))
                       )
        for i, results_for_conf in enumerate(self._states_reached):
            for j, state_reached in enumerate(results_for_conf):
                if state_reached is not None:
                    ret[i][j][state_reached] += 1
        return ret

    @property
    def trajs_to_state(self):
        """
        Return all forward trajectories until a state generated so far.

        They are sorted as a list of lists. The outer list is configurations,
        the inner list is shots, i.e. the outer list will always have
        len=n_starting_configurations and the inner lists will all have
        len=self.shot_counter.
        """
        trajs_to_state = []
        for cnum, cdir in enumerate(self._conf_dirs):
            trajs_per_conf = []
            for snum in range(self.shot_counter):
                traj_fname = os.path.join(cdir,
                                          f"{self.shot_dir_prefix}{snum}",
                                          f"{self.fname_traj_to_state}")
                struct_fname = os.path.join(cdir,
                                            f"{self.shot_dir_prefix}{snum}",
                                            # TODO/FIXME: only works for gmx!
                                            f"{self.deffnm_engine_out}.tpr")
                if os.path.isfile(traj_fname):
                    trajs_per_conf += [Trajectory(trajectory_file=traj_fname,
                                                  structure_file=struct_fname)
                                       ]
            trajs_to_state += [trajs_per_conf]
        return trajs_to_state

    @property
    def trajs_to_state_bw(self):
        """
        Return all backward trajectories until a state generated so far.

        They are sorted as a list of lists. The outer list is configurations,
        the inner list is shots, i.e. the outer list will always have
        len=n_starting_configurations and the inner lists will all have
        len=self.shot_counter.
        """
        trajs_to_state = []
        for cnum, cdir in enumerate(self._conf_dirs):
            trajs_per_conf = []
            for snum in range(self.shot_counter):
                traj_fname = os.path.join(cdir,
                                          f"{self.shot_dir_prefix}{snum}",
                                          f"{self.fname_traj_to_state_bw}")
                struct_fname = os.path.join(cdir,
                                            f"{self.shot_dir_prefix}{snum}",
                                            # TODO/FIXME: only works for gmx!
                                            f"{self.deffnm_engine_out}.tpr")
                if os.path.isfile(traj_fname):
                    trajs_per_conf += [Trajectory(trajectory_file=traj_fname,
                                                  structure_file=struct_fname)
                                       ]
            trajs_to_state += [trajs_per_conf]
        return trajs_to_state

    @property
    def transitions(self):
        """
        Return all transitions generated so far.

        They are sorted as a list of lists. The outer list is configurations,
        the inner list is shots, i.e. the outer list will always have
        len=n_starting_configurations and the inner lists then just contains
        all transitions for the respective configuration and can also be empty.
        """
        if not self.two_way:
            # can not have transitions
            return [[] for _ in range(len(self._conf_dirs))]
        transitions = []
        for cnum, cdir in enumerate(self._conf_dirs):
            trans_per_conf = []
            for snum in range(self.shot_counter):
                traj_fname = os.path.join(cdir,
                                          f"{self.shot_dir_prefix}{snum}",
                                          f"{self.fname_transition_traj}")
                struct_fname = os.path.join(cdir,
                                            f"{self.shot_dir_prefix}{snum}",
                                            # TODO/FIXME: only works for gmx!
                                            f"{self.deffnm_engine_out}.tpr")
                if os.path.isfile(traj_fname):
                    trans_per_conf += [Trajectory(trajectory_file=traj_fname,
                                                  structure_file=struct_fname)
                                       ]
            transitions += [trans_per_conf]
        return transitions

    async def reinitialize_from_workdir(self, overwrite=False):
        """
        Reassess all trials in workdir and populate states_reached counter.

        Possibly extend trials if no state has been reached yet.
        Add missing backwards shots from scratch if the previous run has been
        with two_way=False and this one has two_way=True.

        If overwrite=True we will allow to overwrite existing concatenated
        output trajectories, i.e. traj_to_state, traj_to_state_bw and
        transition_traj.
        """
        # make sure we set everything to zero before we start!
        self._shot_counter = 0
        self._states_reached = [[] for _ in range(len(self.starting_configurations))]
        # find out how many shots we did per configuration, for now we assume
        # that everything went well and we have an equal number of shots per configuration
        dir_list = os.listdir(os.path.join(self.workdir, self._conf_dirs[0]))
        # build a list of all possible dir names
        # (these will be too many if there are other files in conf dir)
        possible_dirnames = [f"{self.shot_dir_prefix}{i}"
                             for i in range(len(dir_list))
                             ]
        # now filter to check that only stuff that is a dir and in possible names
        # will be taken, then count them: this is the number of shots done already
        filtered = [d for d in dir_list
                    if (d in possible_dirnames
                        and os.path.isdir(os.path.join(self.workdir, self._conf_dirs[0], d))
                        )
                    ]
        n_shots = len(filtered)
        return await self._run(n_per_struct=n_shots, continuation=True,
                               overwrite=overwrite)

    async def run(self, n_per_struct):
        """Run for n_per_struct committor trials for each configuration."""
        return await self._run(n_per_struct=n_per_struct, continuation=False,
                               overwrite=False)

    async def _run_single_trial_ow(self, conf_num, shot_num, step_dir,
                                   continuation, overwrite):
        # construct propagator
        propagator = TrajectoryPropagatorUntilAnyState(
                                    states=self.states,
                                    engine_cls=self.engine_cls[conf_num],
                                    engine_kwargs=self.engine_kwargs[conf_num],
                                    run_config=self.engine_run_config[conf_num],
                                    walltime_per_part=self.walltime_per_part,
                                    max_steps=self.max_steps,
                                                       )
        start_conf_name = os.path.join(step_dir,
                                       (f"{self.start_conf_name_prefix}"
                                        + f"{self.deffnm_engine_out}.trr"),
                                       )
        if not continuation:
            # get starting configuration and write it out with random velocities
            extractor_fw = RandomVelocitiesFrameExtractor(T=self.T[conf_num])
            starting_conf = extractor_fw.extract(
                                outfile=start_conf_name,
                                traj_in=self.starting_configurations[conf_num][0],
                                idx=self.starting_configurations[conf_num][1],
                                                 )
            n = 0
        else:
            starting_conf = Trajectory(
                trajectory_file=start_conf_name,
                structure_file=self.starting_configurations[conf_num][0].structure_file
                                       )
            # get the number of times we crashed/reached max length before
            dir_list = os.listdir(step_dir)
            filtered = [d for d in dir_list
                        if (os.path.isdir(os.path.join(step_dir, d))
                            and d.startswith(f"{self.deffnm_engine_out}")
                            and (d.endswith("max_len") or d.endswith("crash")))
                        ]
            n = len(filtered)
        # and propagate
        round_one = True
        while True:
            try:
                out = await propagator.propagate_and_concatenate(
                                    starting_configuration=starting_conf,
                                    workdir=step_dir,
                                    deffnm=self.deffnm_engine_out,
                                    tra_out=os.path.join(step_dir,
                                                         self.fname_traj_to_state
                                                         ),
                                    continuation=(continuation and round_one),
                                    overwrite=overwrite,
                                                                 )
            except (MaxStepsReachedError, EngineCrashedError) as e:
                log_str = (f"MD engine for configuration {conf_num}, "
                           + f"shot {shot_num}, deffnm {self.deffnm_engine_out}"
                           + f" crashed for the {n + 1}th time.")
                if n < self.max_retries_on_crash:
                    if isinstance(e, EngineCrashedError):
                        subdir = os.path.join(step_dir, (f"{self.deffnm_engine_out}"
                                                         + f"_{n + 1}crash"))
                    elif isinstance(e, MaxStepsReachedError):
                        subdir = os.path.join(step_dir, (f"{self.deffnm_engine_out}"
                                                         + f"_{n + 1}max_len"))
                else:
                    logger.error(log_str + " Not retrying this time.")
                    # TODO: do we want to raise the error?!
                    #       I think this way is better as we can still finish
                    #       the simulation as expected (just with a shot less)
                    #raise e from None
                    return None  # no state reached!
                logger.warning(log_str + " Moving to subfolder and retrying.")
                # we only end up here if there is cleanup/moving to do
                os.mkdir(subdir)
                all_files = os.listdir(step_dir)
                for f in all_files:
                    splits = f.split(".")
                    if splits[0] == f"{self.deffnm_engine_out}":
                        # if it is exactly deffnm_out it can only be
                        # a deffnm.tpr/mdp etc or a deffnm.partXXXX.trr/xtc etc
                        # so move it
                        os.rename(os.path.join(step_dir, f), os.path.join(subdir, f))
                    elif "step" in splits[0] and splits[-1] == "pdb":
                        # the gromacs stepXXXa/b/c/d.pdb files, that are
                        # written on decomposition errors/too high forces etc
                        # move them too!
                        # Note that we assume that only one engine crashes at a time!
                        os.rename(os.path.join(step_dir, f), os.path.join(subdir, f))
            else:
                # no error, return and get out of here
                tra_out, state_reached = out
                return state_reached
            finally:
                n += 1
                round_one = False

    async def _run_single_trial_tw(self, conf_num, shot_num, step_dir,
                                   continuation, overwrite):
        # NOTE: this is a potential misuse of a committor simulation,
        #       see the note further down for more on why it is/should be ok
        # propagators
        propagators = [TrajectoryPropagatorUntilAnyState(
                                    states=self.states,
                                    engine_cls=self.engine_cls[conf_num],
                                    engine_kwargs=self.engine_kwargs[conf_num],
                                    run_config=self.engine_run_config[conf_num],
                                    walltime_per_part=self.walltime_per_part,
                                    max_steps=self.max_steps,
                                                         )
                       for _ in range(2)]
        # forward starting configuration
        start_conf_name_fw = os.path.join(step_dir,
                                          (f"{self.start_conf_name_prefix}"
                                           + f"{self.deffnm_engine_out}.trr"),
                                          )
        continuation_fw = continuation
        if not continuation_fw:
            extractor_fw = RandomVelocitiesFrameExtractor(T=self.T[conf_num])
            starting_conf_fw = extractor_fw.extract(
                                  outfile=start_conf_name_fw,
                                  traj_in=self.starting_configurations[conf_num][0],
                                  idx=self.starting_configurations[conf_num][1],
                                                   )
            n_fw = 0
        else:
            starting_conf_fw = Trajectory(
                trajectory_file=start_conf_name_fw,
                structure_file=self.starting_configurations[conf_num][0].structure_file
                                       )
            # get the number of times we crashed/reached max length before
            dir_list = os.listdir(step_dir)
            filtered = [d for d in dir_list
                        if (os.path.isdir(os.path.join(step_dir, d))
                            and d.startswith(f"{self.deffnm_engine_out}")
                            and (d.endswith("max_len") or d.endswith("crash")))
                        ]
            n_fw = len(filtered)
        # backwards starting configuration (forward with inverted velocities)
        start_conf_name_bw = os.path.join(step_dir,
                                          (f"{self.start_conf_name_prefix}"
                                           + f"{self.deffnm_engine_out_bw}.trr"),
                                          )
        continuation_bw = continuation
        if continuation:
            # check if we ever started the backwards trial
            if os.path.isfile(start_conf_name_bw):
                starting_conf_bw = Trajectory(
                    trajectory_file=start_conf_name_bw,
                    structure_file=self.starting_configurations[conf_num][0].structure_file
                                       )
                continuation_bw = True
            else:
                # if not start backwards trial from scratch
                continuation_bw = False
        if not continuation_bw:
            # write out the starting configuration if it is no continuation
            extractor_bw = InvertedVelocitiesFrameExtractor()
            starting_conf_bw = extractor_bw.extract(
                                  outfile=start_conf_name_bw,
                                  traj_in=starting_conf_fw,
                                  idx=0,
                                                   )
            n_bw = 0
        else:
            # get the number of times we crashed/reached max length before
            dir_list = os.listdir(step_dir)
            filtered = [d for d in dir_list
                        if (os.path.isdir(os.path.join(step_dir, d))
                            and d.startswith(f"{self.deffnm_engine_out_bw}")
                            and (d.endswith("max_len") or d.endswith("crash")))
                        ]
            n_bw = len(filtered)
        # and propagate
        ns = [n_fw, n_bw]
        starting_confs = [starting_conf_fw, starting_conf_bw]
        deffnms_engine_out = [self.deffnm_engine_out, self.deffnm_engine_out_bw]
        continuations = [continuation_fw, continuation_bw]
        trials_pending = [asyncio.create_task(
                            p.propagate(starting_configuration=sconf,
                                        workdir=step_dir,
                                        deffnm=deffnm,
                                        continuation=cont,
                                        )
                            )
                          for p, sconf, deffnm, cont in zip(propagators,
                                                            starting_confs,
                                                            deffnms_engine_out,
                                                            continuations,
                                                            )
                          ]
        trials_done = [None for _ in range(2)]
        while any(t is None for t in trials_done):
            # we leave the loop either when everything is done or via exceptions raised
            done, pending = await asyncio.wait(trials_pending,
                                               return_when=asyncio.FIRST_EXCEPTION,
                                               )
            for t in done:
                t_idx = trials_pending.index(t)
                if isinstance(t.exception(), (EngineCrashedError,
                                              MaxStepsReachedError)):
                    log_str = (f"MD engine for configuration {str(conf_num)}, "
                               + f"shot {str(shot_num)}, "
                               + f"deffm {deffnms_engine_out[t_idx]} crashed "
                               + f"for the {ns[t_idx] + 1}th time.")
                    # catch error raised when gromacs crashes
                    if ns[t_idx] < self.max_retries_on_crash:
                        # move the files to a subdirectory
                        if isinstance(t.exception(), EngineCrashedError):
                            subdir = os.path.join(step_dir,
                                                  (f"{deffnms_engine_out[t_idx]}"
                                                   + f"_{ns[t_idx] + 1}crash")
                                                  )
                        elif isinstance(t.exception(), MaxStepsReachedError):
                            subdir = os.path.join(step_dir,
                                                  (f"{deffnms_engine_out[t_idx]}"
                                                   + f"_{ns[t_idx] + 1}max_len")
                                                  )
                        else:
                            raise RuntimeError("This should never happen!")
                        logger.warning(log_str + " Moving to subdirectory and retrying.")
                        os.mkdir(subdir)
                        all_files = os.listdir(step_dir)
                        for f in all_files:
                            splits = f.split(".")
                            if splits[0] == f"{deffnms_engine_out[t_idx]}":
                                # if it is exactly deffnm_out it can only be
                                # a deffnm.tpr/mdp etc or a deffnm.partXXXX.trr/xtc etc
                                # so move it
                                os.rename(os.path.join(step_dir, f),
                                          os.path.join(subdir, f))
                            elif "step" in splits[0] and splits[-1] == "pdb":
                                # the gromacs stepXXXa/b/c/d.pdb files, that are
                                # written on decomposition errors/too high forces etc
                                # move them too!
                                # Note that we assume that only one engine crashes at a time!
                                os.rename(os.path.join(step_dir, f),
                                          os.path.join(subdir, f))
                        # get the task out of the list
                        _ = trials_pending.pop(t_idx)
                        # resubmit the task
                        trials_pending.insert(
                                        t_idx,
                                        asyncio.create_task(
                                            propagators[t_idx].propagate(
                                               starting_configuration=starting_confs[t_idx],
                                               workdir=step_dir,
                                               deffnm=deffnms_engine_out[t_idx],
                                               # we crashed so there is nothing to continue from anymore
                                               continuation=False,
                                                                         )
                                                            )
                                              )
                        # and increase counter
                        ns[t_idx] += 1
                    else:
                        # check if we already know that this trial crashed
                        # if we do we have set the result to (None, None)
                        if trials_done[t_idx] is None:
                            # reached maximum tries, raise the error and crash the sampling? :)
                            logger.error(log_str + " Not retrying this time.")
                            # TODO: same as for oneway, do we want to raise?!
                            #       I (hejung) think not, since not raising enables
                            #       us to finish the simulation adn get a return
                            #raise t.exception() from None
                            # no trajs, no state reached
                            trials_done[t_idx] = (None, None)
                elif t.exception() is not None:
                    # any other exception
                    # raise directly
                    raise t.exception() from None
                else:
                    # no exception raised
                    # put the result into trials_done at the right idx
                    t_idx = trials_pending.index(t)
                    trials_done[t_idx] = t.result()
        # check where they went: construct TP if possible, else concatenate
        (fw_trajs, fw_state), (bw_trajs, bw_state) = trials_done
        if (fw_state is None) or (bw_state is None):
            # if any of the two trials did not finish we return None, i.e. no state reached
            # TODO: is this what we want? Or should we try to return the state
            #       reached if one of them finishes
            #       I (hejung) think None is best, because a half-crashed trial
            #       should be approached with scrutiny and not just taken as is
            return None
        if fw_state == bw_state:
            logger.info(f"Both trials reached state {fw_state}.")
        else:
            # we can form a TP, so do it (low idx state to high idx state)
            logger.info(f"Forward trajectory reached state {fw_state}, "
                        + f"backward trajectory reached state {bw_state}.")
            if fw_state > bw_state:
                minus_trajs, minus_state = bw_trajs, bw_state
                plus_trajs, plus_state = fw_trajs, fw_state
            else:
                # can only be the other way round
                minus_trajs, minus_state = fw_trajs, fw_state
                plus_trajs, plus_state = bw_trajs, bw_state
            tra_out = os.path.join(step_dir, self.fname_transition_traj)
            # TODO: we currently dont use the return, should call as _ = ... ?
            path_traj = await construct_TP_from_plus_and_minus_traj_segments(
                            minus_trajs=minus_trajs, minus_state=minus_state,
                            plus_trajs=plus_trajs, plus_state=plus_state,
                            state_funcs=self.states, tra_out=tra_out,
                            struct_out=None, overwrite=overwrite,
                                                                             )
            logger.info(f"TP from state {minus_state} to {plus_state} was generated.")
        # TODO: do we want to concatenate the trials to states in any way?
        # i.e. independent of if we can form a TP? or only for no TP cases?
        # NOTE: (answer to todo?!)
        # I think this is best as we can then return the fw trial only
        # and treat all fw trials as truly independent realizations
        # i.e. this makes sure the committor simulation stays a committor
        # simulation, even for users who do not think about velocity
        # correlation times
        out_tra_names = [os.path.join(step_dir, self.fname_traj_to_state),
                         os.path.join(step_dir, self.fname_traj_to_state_bw),
                         ]
        # TODO: we currently dont use the return, should call as _ = ... ?
        concats = await asyncio.gather(*(
                        p.cut_and_concatenate(trajs=trajs, tra_out=tra_out,
                                              overwrite=overwrite)
                        for p, trajs, tra_out in zip(propagators,
                                                     [fw_trajs, bw_trajs],
                                                     out_tra_names
                                                     )
                                         )
                                       )
        # (tra_out_fw, fw_state), (tra_out_bw, bw_state) = concats
        return fw_state

    async def _run_single_trial(self, conf_num, shot_num, two_way,
                                continuation, overwrite):
        step_dir = os.path.join(
                        self.workdir,
                        self._conf_dirs[conf_num],
                        f"{self.shot_dir_prefix}{str(shot_num)}",
                                )
        if not continuation:
            # create directory only for new trials
            os.mkdir(step_dir)
        if two_way:
            state_reached = await self._run_single_trial_tw(
                                                    conf_num=conf_num,
                                                    shot_num=shot_num,
                                                    step_dir=step_dir,
                                                    continuation=continuation,
                                                    overwrite=overwrite,
                                                            )
        else:
            state_reached = await self._run_single_trial_ow(
                                                    conf_num=conf_num,
                                                    shot_num=shot_num,
                                                    step_dir=step_dir,
                                                    continuation=continuation,
                                                    overwrite=overwrite,
                                                            )

        return state_reached

    async def _run(self, n_per_struct, continuation, overwrite):
        # NOTE: make this private so we can use it from reassess with continuation
        #       but avoid unhappy users who dont understand when/how continuation
        #       can/should be used
        # first construct the list of all coroutines
        # Note that calling them will not (yet) schedule them for execution
        # we do this later while respecting self.n_max_concurrent
        # using the little func below
        async def gather_with_concurrency(n, *tasks):
            # https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio/61478547#61478547
            semaphore = asyncio.Semaphore(n)

            async def sem_task(task):
                async with semaphore:
                    return await task
            return await asyncio.gather(*(sem_task(task) for task in tasks))

        # construct the tasks all at once,
        # ordering is such that we first finish all trials for configuration 0
        # then configuration 1, i.e. we order by configuration and not by shotnum
        tasks = []
        for cnum in range(len(self.starting_configurations)):
            tasks += [self._run_single_trial(conf_num=cnum,
                                             shot_num=snum,
                                             two_way=self.two_way[cnum],
                                             continuation=continuation,
                                             overwrite=overwrite,
                                             )
                      for snum in range(self._shot_counter,
                                        self._shot_counter + n_per_struct
                                        )
                      ]
        results = await gather_with_concurrency(self.n_max_concurrent, *tasks)
        # results is a list of idx to the states reached
        # we unpack it and add it to the internal states_reached counter
        for cnum in range(len(self.starting_configurations)):
            self._states_reached[cnum] += results[cnum * n_per_struct:
                                                  (cnum + 1) * n_per_struct]
        # increment internal shot (per struct) counter
        self._shot_counter += n_per_struct
        # TODO: we return the total states reached per shot?!
        #       or should we return only for this run?
        return self.states_reached_per_shot