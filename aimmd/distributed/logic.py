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
import asyncio
import multiprocessing
import inspect
import logging
import functools
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from . import _SEM_MAX_PROCESS
from .trajectory import TrajectoryConcatenator
from .gmx_utils import get_all_traj_parts, nstout_from_mdp


logger = logging.getLogger(__name__)


class MaxStepsReachedError(Exception):
    """
    Error raised when the simulation terminated because the (user-defined)
    maximum number of integration steps/trajectory frames has been reached.
    """
    pass


async def construct_TP_from_plus_and_minus_traj_segments(minus_trajs, minus_state,
                                                         plus_trajs, plus_state,
                                                         state_funcs, tra_out,
                                                         struct_out=None,
                                                         overwrite=False):
    """
    Construct a continous TP from plus and minus segments until states.

    This is used e.g. in TwoWay TPS or if you try to get TPs out of a committor
    simulation. Note, that this inverts all velocities on the minus segments.

    Arguments:
    ----------
    minus_trajs - list of arcd.Trajectories, backward in time,
                  these are going to be inverted
    minus_state - int, idx to the first state reached on the minus trajs
    plus_trajs - list of arcd.Trajectories, forward in time
    plus_state - int, idx to the first state reached on the plus trajs
    state_funcs - list of state functions, the indices to the states must match
                  the minus and plus state indices!
    tra_out - path to the output trajectory file
    struct_out - None or path to a structure file, the structure to associate with
                 the concatenated TP, taken from input trajs if None (the default)
    overwrite - bool (default False), wheter to overwrite an existing output
    """
    # first find the slices to concatenate
    # minus state first
    minus_state_vals = await asyncio.gather(*(state_funcs[minus_state](t)
                                              for t in minus_trajs)
                                            )
    part_lens = [len(v) for v in minus_state_vals]
    # make it into one long array
    minus_state_vals = np.concatenate(minus_state_vals, axis=0)
    # get the first frame in state
    frames_in_minus, = np.where(minus_state_vals)  # where always returns a tuple
    # get the first frame in minus state in minus_trajs, this will become the
    # first frame of the traj since we invert this part
    first_frame_in_minus = np.min(frames_in_minus)
    # I think this is overkill, i.e. we can always expect that
    # first frame in state is in last part?!
    # [this could potentially make this a bit shorter and maybe
    #  even a bit more readable :)]
    # But for now: better be save than sorry :)
    # find the first part in which minus state is reached, i.e. the last one
    # to take when constructing the TP
    last_part_idx = 0
    frame_sum = part_lens[last_part_idx]
    while first_frame_in_minus >= frame_sum:
        last_part_idx += 1
        frame_sum += part_lens[last_part_idx]
    # find the first frame in state (counting from start of last part to take)
    _first_frame_in_minus = (first_frame_in_minus
                             - sum(part_lens[:last_part_idx]))  # >= 0
    # now construct the slices and trajs list (backwards!)
    # the last/first part
    slices = [(_first_frame_in_minus, None, -1)]  # negative stride!
    trajs = [minus_trajs[last_part_idx]]
    # the ones we take fully (if any) [the range looks a bit strange
    # because we dont take last_part_index but include the zero as idx]
    slices += [(-1, None, -1) for _ in range(last_part_idx - 1, -1, -1)]
    trajs += [minus_trajs[i] for i in range(last_part_idx - 1, -1, -1)]

    # now plus trajectories, i.e. the part we put in positive stride
    plus_state_vals = await asyncio.gather(*(state_funcs[plus_state](t)
                                             for t in plus_trajs)
                                           )
    part_lens = [len(v) for v in plus_state_vals]
    # make it into one long array
    plus_state_vals = np.concatenate(plus_state_vals, axis=0)
    # get the first frame in state
    frames_in_plus, = np.where(plus_state_vals)
    first_frame_in_plus = np.min(frames_in_plus)
    # find the part
    last_part_idx = 0
    frame_sum = part_lens[last_part_idx]
    while first_frame_in_plus >= frame_sum:
        last_part_idx += 1
        frame_sum += part_lens[last_part_idx]
    # find the first frame in state (counting from start of last part)
    _first_frame_in_plus = (first_frame_in_plus
                            - sum(part_lens[:last_part_idx]))  # >= 0
    # construct the slices and add trajs to list (forward!)
    # NOTE: here we exclude the starting configuration, i.e. the SP,
    #       such that it is in the concatenated trajectory only once!
    #       (gromacs has the first frame in the trajectory)
    if last_part_idx > 0:
        # these are the trajectory segments we take completely
        # [this excludes last_part_idx so far]
        slices += [(1, None, 1)]
        trajs += [plus_trajs[0]]
        # these will be empty if last_part_idx < 2
        slices += [(0, None, 1) for _ in range(1, last_part_idx)]
        trajs += [plus_trajs[i] for i in range(1, last_part_idx)]
        # add last part (with the last frame as first frame in plus state)
        slices += [(0, _first_frame_in_plus + 1, 1)]
        trajs += [plus_trajs[last_part_idx]]
    else:
        # first and last part is the same, so exclude starting configuration
        # from the same segment that has the last frame as first frame in plus
        slices += [(1, _first_frame_in_plus + 1, 1)]
        trajs += [plus_trajs[last_part_idx]]
    # finally produce the concatenated path
    concat = functools.partial(TrajectoryConcatenator().concatenate,
                               trajs=trajs,
                               slices=slices,
                               tra_out=tra_out,
                               struct_out=struct_out,
                               overwrite=overwrite)
    loop = asyncio.get_running_loop()
    async with _SEM_MAX_PROCESS:
        # NOTE: make sure we do not fork! (not save with multithreading)
        # see e.g. https://stackoverflow.com/questions/46439740/safe-to-call-multiprocessing-from-a-thread-in-python
        ctx = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(1, mp_context=ctx) as pool:
            path_traj = await loop.run_in_executor(pool, concat)
    return path_traj


# TODO: DOCUMENT
class TrajectoryPropagatorUntilAnyState:
    """
    Propagate a trajectory until any of the states is reached.

    This class propagates the trajectory using a given MD engine (class) in
    small chunks (chunksize is determined by walltime_per_part) and checks
    after every chunk is done if any state has been reached.
    It then returns either a list of trajectory parts and the state first
    reached and can also concatenate the parts into one trajectory, which then
    starts with the starting configuration and ends with one frame in the state.

    Notable methods:
    ----------------
    propagate - propagate the trajectory until any state is reached,
                return a list of trajecory segments and the state reached
    cut_and_concatenate - take a list of trajectory segments and form one
                          continous trajectory until the first frame in state
    propagate_and_concatenate - propagate and cut_and_concatenate in sequence
    """
    # NOTE: we assume that every state function returns a list/ a 1d array with
    #       True/False for each frame, i.e. if we are in state at a given frame
    # NOTE: we assume non-overlapping states, i.e. a configuration can not
    #       be inside of two states at the same time, it is the users
    #       responsibility to ensure that their states are sane

    def __init__(self, states, engine_cls, engine_kwargs, walltime_per_part,
                 max_steps=None, max_frames=None):
        """
        Initialize a TrajectoryPropagatorUntilAnyState.

        Parameters:
        -----------
        states - list of state functions, e.g. `aimmd.TrajectoryFunctionWrapper`
                 but can be any callable that takes a trajecory and returns an
                 array of True and False values (one value per frame)
        engine_cls - class of the MD engine to use (uninitialized!)
        engine_kwargs - dictionary of key word arguments needed to initialize
                        the MD engine
        walltime_per_part - float, walltime per trajectory segment in hours
        max_steps - None or int, maximum number of integration steps to try
                    before stopping the simulation because it did not commit
        max_frames - None or int, maximum number of frames to produce before
                     stopping the simulation because it did not commit
        NOTE: max_steps and max_frames are redundant since:
                   max_steps = max_frames * output_frequency
              if both are given max_steps takes precedence
        """
        # states - list of wrapped trajectory funcs
        # engine_cls - mdengine class
        # engine_kwargs - dict of kwargs for instantiation of the engine
        # walltime_per_part - walltime (in h) per mdrun, i.e. traj part/segment
        # NOTE: max_steps takes precedence over max_frames if both are given
        # TODO: do we want max_frames as argument to propagate too? I.e. giving it there to overwrite?
        # max_frames - maximum number of *frames* in all segments combined
        #              note that frames = steps / nstxout
        # max_steps - maximum number of integration steps, i.e. nsteps = frames * nstxout
        self._states = None
        self._state_func_is_coroutine = None
        self.states = states
        self.engine_cls = engine_cls
        self.engine_kwargs = engine_kwargs
        self.walltime_per_part = walltime_per_part
        # find out nstout
        # TODO: we are assuming GMX engines here...at some point we will write
        #       a generic nstout_from_mdconfig method that sorts out which
        #       sorts out which type of mdconfig was passed and then calls the
        #       mdconfig specific helper function, e.g. nstout_from_mdp for mdp
        try:
            traj_type = engine_kwargs["output_traj_type"]
        except KeyError:
            # not in there so it will be the engine default
            traj_type = engine_cls.output_traj_type
        nstout = nstout_from_mdp(engine_kwargs["mdp"], traj_type=traj_type)
        # sort out if we use max-frames or max-steps
        if max_frames is not None and max_steps is not None:
            logger.warning("Both max_steps and max_frames given. Note that "
                           + "max_steps will take precedence.")
        if max_steps is not None:
            self.max_steps = max_steps
        elif max_frames is not None:
            self.max_steps = max_frames * nstout
        else:
            logger.info("Neither max_frames nor max_steps given. "
                        + "Setting max_steps to infinity.")
            # this is a float but can be compared to ints
            self.max_steps = np.inf

    #TODO/FIXME: self._states is a list...that means users can change
    #            single elements without using the setter!
    #            we could use a list subclass as for the MDconfig?!
    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        # I think it is save to assume each state has a .__call__() method? :)
        # so we just check if it is awaitable
        self._state_func_is_coroutine = [inspect.iscoroutinefunction(s.__call__)
                                         for s in states]
        if not all(self._state_func_is_coroutine):
            # and warn if it is not
            logger.warning(
                    "It is recommended to use coroutinefunctions for all "
                    + "states. This can easily be achieved by wrapping any"
                    + " function in a TrajectoryFunctionWrapper. All "
                    + "non-coroutine state functions will be blocking when"
                    + " applied! ([s is coroutine for s in states] = "
                    + f"{self._state_func_is_coroutine})"
                           )
        self._states = states

    async def propagate_and_concatenate(self, starting_configuration, workdir,
                                        deffnm, tra_out, overwrite=False,
                                        continuation=False):
        """
        Chain `propagate` and `concatenate` methods.

        Parameters:
        -----------
        starting_configuration - `aimmd.distributed.Trajectory`
        workdir - absolute or relative path to an existing directory
        deffnm - the name to use for all MD engine output files
        tra_out - the filename of the output trajectory
        overwrite - whether to overwrite any existing output trajectories
        continuation - bool, whether to (try to) continue a previous run
                       with given workdir and deffnm but possibly changed states

        Returns (traj_to_state, idx_of_state_reached)
        """
        # this just chains propagate and cut_and_concatenate
        # usefull for committor simulations, for e.g. TPS one should try to
        # directly concatenate both directions to a full TP if possible
        trajs, first_state_reached = await self.propagate(
                                starting_configuration=starting_configuration,
                                workdir=workdir,
                                deffnm=deffnm,
                                continuation=continuation
                                                          )
        # NOTE: it should not matter too much speedwise that we recalculate
        #       the state functions, they are expected to be wrapped traj-funcs
        #       i.e. the second time we should just get the values from cache
        full_traj, first_state_reached = await self.cut_and_concatenate(
                                                        trajs=trajs,
                                                        tra_out=tra_out,
                                                        overwrite=overwrite,
                                                                        )
        return full_traj, first_state_reached

    async def propagate(self, starting_configuration, workdir, deffnm,
                        continuation=False):
        """
        Propagate trajectory in parts until any of the states is reached.

        Parameters:
        -----------
        starting_configuration - `aimmd.distributed.Trajectory`
        workdir - absolute or relative path to an existing directory
        deffnm - the name to use for all MD engine output files
        continuation - bool, whether to (try to) continue a previous run
                       with given workdir and deffnm but possibly changed states

        Returns (list_of_traj_parts, idx_of_first_state_reached)
        """
        # NOTE: curently this just returns a list of trajs + the state reached
        #       this feels a bit uncomfortable but avoids that we concatenate
        #       everything a quadrillion times when we use the results
        # starting_configuration - Trajectory with starting configuration (or None)
        # workdir - workdir for engine
        # deffnm - trajectory name(s) for engine (+ all other output file names)
        # continuation - bool, if True we will try to continue a previous MD run
        #                from files but possibly with new/differetn states
        # check first if the starting configuration is in any state
        state_vals = await self._state_vals_for_traj(starting_configuration)
        if np.any(state_vals):
            states_reached, frame_nums = np.where(state_vals)
            # gets the frame with the lowest idx where any state is True
            min_idx = np.argmin(frame_nums)
            first_state_reached = states_reached[min_idx]
            logger.error(f"Starting configuration ({starting_configuration}) "
                         + f"is already inside the state with idx {first_state_reached}.")
            # we just return the starting configuration/trajectory
            # state reached is calculated below (is the same for both branches)
            trajs = [starting_configuration]
        else:
            engine = self.engine_cls(**self.engine_kwargs)
            if not continuation:
                await engine.prepare(
                            starting_configuration=starting_configuration,
                            workdir=workdir,
                            deffnm=deffnm,
                                    )
                any_state_reached = False
                trajs = []
                step_counter = 0
            else:
                # NOTE: we assume that the state function could be different
                # so get all traj parts and calculate the state functions on them
                trajs = get_all_traj_parts(workdir, deffnm=deffnm,
                                           traj_type=engine.output_traj_type)
                states_vals = await asyncio.gather(
                                *(self._state_vals_for_traj(t) for t in trajs)
                                                   )
                states_vals = np.concatenate([np.asarray(s) for s in states_vals],
                                             axis=1)
                # see if we already reached a state on the existing traj parts
                any_state_reached = np.any(states_vals)
                if any_state_reached:
                    states_reached, frame_nums = np.where(states_vals)
                    # gets the frame with the lowest idx where any state is True
                    min_idx = np.argmin(frame_nums)
                    first_state_reached = states_reached[min_idx]
                    # already reached a state, get out of here!
                    return trajs, first_state_reached
                # Did not reach a state yet, so prepare the engine to continue
                # the simulation until we reach any of the (new) states
                await engine.prepare_from_files(workdir=workdir, deffnm=deffnm)
                step_counter = engine.steps_done

            while ((not any_state_reached)
                   and (step_counter <= self.max_steps)):
                traj = await engine.run_walltime(self.walltime_per_part)
                state_vals = await self._state_vals_for_traj(traj)
                any_state_reached = np.any(state_vals)
                step_counter = engine.steps_done
                trajs.append(traj)
            if not any_state_reached:
                # left while loop because of max_frames reached
                raise MaxStepsReachedError(
                        f"Engine produced {step_counter} "
                        + f"steps (>= {self.max_steps})."
                                           )
        # state_vals are the ones for the last traj
        # here we get which states are True and at which frame
        states_reached, frame_nums = np.where(state_vals)
        # gets the frame with the lowest idx where any state is True
        min_idx = np.argmin(frame_nums)
        # and now the idx to self.states of the state that was first reached
        # NOTE: if two states are reached simultaneously at min_idx,
        #       this will find the state with the lower idx only
        first_state_reached = states_reached[min_idx]
        return trajs, first_state_reached

    async def cut_and_concatenate(self, trajs, tra_out, overwrite=False):
        """
        Cut out and concatenate the trajectory until the first state is reached.

        The expected input is a list of trajectories, e.g. the output of the
        `propagate` method.

        Parameters:
        -----------
        trajs - list of `aimmd.distributed.Trajectory`, a continous trajectory
                split in seperate parts
        tra_out - the filename of the output trajectory
        overwrite - whether to overwrite any existing output trajectories

        Returns (traj_to_state, idx_of_first_state_reached)
        """
        # trajs is a list of trajectoryes, e.g. the return of propagate
        # tra_out and overwrite are passed directly to the Concatenator
        # NOTE: we assume that frame0 of traj0 is outside of any state
        #       and return only the subtrajectory from frame0 until any state
        #       is first reached (the rest is ignored)
        # get all func values and put them into one big array
        states_vals = await asyncio.gather(
                                *(self._state_vals_for_traj(t) for t in trajs)
                                              )
        # states_vals is a list (trajs) of lists (states)
        # take state 0 (always present) to get the traj part lengths
        part_lens = [len(s[0]) for s in states_vals]  # s[0] is 1d (np)array
        states_vals = np.concatenate([np.asarray(s) for s in states_vals],
                                     axis=1)
        states_reached, frame_nums = np.where(states_vals)
        # gets the frame with the lowest idx where any state is True
        min_idx = np.argmin(frame_nums)
        first_state_reached = states_reached[min_idx]
        first_frame_in_state = frame_nums[min_idx]
        # find out in which part it is
        last_part_idx = 0
        frame_sum = part_lens[last_part_idx]
        while first_frame_in_state >= frame_sum:
            last_part_idx += 1
            frame_sum += part_lens[last_part_idx]
        # find the first frame in state (counting from start of last part)
        _first_frame_in_state = (first_frame_in_state
                                 - sum(part_lens[:last_part_idx]))  # >= 0
        if last_part_idx > 0:
            # trajectory parts which we take fully
            slices = [(0, None, 1) for _ in range(last_part_idx)]
        else:
            # only the first/last part
            slices = []
        # and the last part until including first_frame_in_state
        slices += [(0, _first_frame_in_state + 1, 1)]
        # we fill in all args as kwargs because there are so many
        concat = functools.partial(TrajectoryConcatenator().concatenate,
                                   trajs=trajs[:last_part_idx + 1],
                                   slices=slices,
                                   # take the structure file of the traj, as it
                                   # comes from the engine directly
                                   tra_out=tra_out, struct_out=None,
                                   overwrite=overwrite)
        loop = asyncio.get_running_loop()
        async with _SEM_MAX_PROCESS:
            # NOTE: make sure we do not fork! (not save with multithreading)
            # see e.g. https://stackoverflow.com/questions/46439740/safe-to-call-multiprocessing-from-a-thread-in-python
            ctx = multiprocessing.get_context("forkserver")
            with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                full_traj = await loop.run_in_executor(pool, concat)
        return full_traj, first_state_reached

    async def _state_vals_for_traj(self, traj):
        # return a list of state_func results, one for each state func in states
        if all(self._state_func_is_coroutine):
            # easy, all coroutines
            return await asyncio.gather(*(s(traj) for s in self.states))
        elif not any(self._state_func_is_coroutine):
            # also easy (but blocking), none is coroutine
            return [s(traj) for s in self.states]
        else:
            # need to piece it together
            # first the coroutines concurrently
            coros = [s(traj) for s, s_is_coro
                     in zip(self.states, self._state_func_is_coroutine)
                     if s_is_coro
                     ]
            coro_res = await asyncio.gather(*coros)
            # now either take the result from coro execution or calculate it
            all_results = []
            for s, s_is_coro in zip(self.states, self._state_func_is_coroutine):
                if s_is_coro:
                    all_results.append(coro_res.pop(0))
                else:
                    all_results.append(s(traj))
            return all_results
            # NOTE: this would be elegant, but to_thread() is py v>=3.9
            # we wrap the non-coroutines into tasks to schedule all together
            #all_states_as_coro = [
            #    s(traj) if s_is_cor else asyncio.to_thread(s, traj)
            #    for s, s_is_cor in zip(self.states, self._state_func_is_coroutine)
            #                      ]
            #return await asyncio.gather(*all_states_as_coro)
