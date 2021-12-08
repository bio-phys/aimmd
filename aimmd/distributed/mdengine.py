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
import abc
import copy
import shlex
import random
import string
import shutil
import asyncio
import logging

from . import _SEMAPHORES
from .trajectory import Trajectory
from .slurm import SlurmProcess
from .mdconfig import MDP
from .gmx_utils import nstout_from_mdp, get_all_traj_parts
from .utils import ensure_executable_available


logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Exception raised when something goes wrong with the (MD)-Engine."""
    pass


class EngineCrashedError(EngineError):
    """Exception raised when the (MD)-Engine crashes during a run."""
    pass


class MDEngine(abc.ABC):
    @abc.abstractmethod
    async def apply_constraints(self, conf_in, conf_out_name):
        # apply constraints to given conf_in, write conf_out_name and return it
        raise NotImplementedError

    @abc.abstractmethod
    # TODO: think about the most general interface!
    # NOTE: We assume that we do not change the system for/in one engine,
    #       i.e. .top, .ndx, mdp-object, ...?! should go into __init__
    async def prepare(self, starting_configuration, workdir, deffnm):
        raise NotImplementedError

    @abc.abstractmethod
    # TODO: should this be a classmethod?
    #@classmethod
    async def prepare_from_files(self, workdir, deffnm):
        # this should prepare the engine to continue a previously stopped simulation
        # starting with the last trajectory part in workdir that is compatible with deffnm
        raise NotImplementedError

    @abc.abstractmethod
    async def run_walltime(self, walltime):
        # run for specified walltime
        # NOTE: must be possible to run this multiple times after preparing once!
        raise NotImplementedError

    @abc.abstractmethod
    async def run_steps(self, nsteps, steps_per_part=False):
        # run for specified number of steps
        # NOTE: not sure if we need it, but could be useful
        # NOTE: make sure we can run multiple times after preparing once!
        raise NotImplementedError

    @abc.abstractproperty
    def current_trajectory(self):
        # return current trajectory: Trajectory or None
        # if we retun a Trajectory it is either what we are working on atm
        # or the trajectory we finished last
        raise NotImplementedError

    @abc.abstractproperty
    def running(self):
        raise NotImplementedError

    @abc.abstractproperty
    def steps_done(self):
        raise NotImplementedError


# TODO: DOCUMENT!
# TODO: capture mdrun stdout+ stderr? for local gmx? for slurm it goes to file
# NOTE: with tra we usually mean trr, i.e. a full precision trajectory with velocities
class GmxEngine(MDEngine):
    """
    Steer gromacs molecular dynamics simulation from python.

    An async/await enabled wrapper around gromacs grompp and gromacs mdrun.
    Please use the power of concurrent execution of computationally bound
    subprocesses responsibly...or crash your workstation ;)
    The `SlurmGmxEngine` alleviates this problem somewhat by submitting the
    (computationally expensive) mdruns via SLURM...in that case please have in
    mind that your colleagues might also want to use the cluster :)

    Notable functions:
    ------------------
        - `prepare()` provides grompp functionality
        - `run()`, `run_walltime`, `run_steps()` start/run MD simulations
        - `prepare_from_files()` can be used to continue a previous MD run

    Notable properties:
    -------------------
        - `nstout`, `frames_done`, `steps_done` adn `time_done` provide
          read-only access to the trajectory output frequency and frames/steps/
          time done in total for the current deffnm and workdir

    Notable attributes:
    -------------------
        - `grompp_executable`/`mdrun_executable` can be used to customize the
          name or path to the respective executables
        - `grompp_extra_args`/`mdrun_extra_args` can be used to pass extra
          command line arguments to the respective executables
        - `output_traj_type` sets the trajectory type (ending) this engine
          returns/looks for; Note that we simply ignore all other trajectories,
          i.e. depending on the MDP settings we will still write xtc and trr,
          but return only one of them
    """

    # local prepare and option to run a local gmx (mainly for testing)
    _grompp_executable = "gmx grompp"
    _mdrun_executable = "gmx mdrun"
    # extra_args are expected to be str and will be appended to the end of the
    # respective commands after a separating space,
    # i.e. cmd = base_cmd + " " + extra_args
    grompp_extra_args = ""
    mdrun_extra_args = ""
    output_traj_type = "trr"  # file ending of the returned output trajectories
                              # NOTE: this will be the traj we count frames for
                              #       and check the mdp, etc.
                              #       However this does not mean that no other
                              #       trajs will/can be written, we simply
                              #       ignore them

    def __init__(self, mdp, gro_file, top_file, ndx_file=None, **kwargs):
        """
        Initialize a `GmxEngine`.

        Parameters:
        -----------
        mdp - `aimmd.distributed.MDP`, the molecular dynamics parameters
        gro_file - absolute or relative path to a gromacs structure file
        top_file - absolute or relative path to a gromacs topolgy (.top) file
        ndx_file - (optional) absolute or relative path to a gromacs index file

        Note that all attributes can be set at intialization by passing keyword
        arguments with their name, e.g. mdrun_extra_args="-ntomp 2" to instruct
        gromacs to use 2 openMP threads.
        """
        # make it possible to set any attribute via kwargs
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
        # NOTE: after the kwargs setting to be sure they are what we set/expect
        if not isinstance(mdp, MDP):
            raise TypeError(f"mdp must be of type {MDP}.")
        if mdp["nsteps"] != -1:
            logger.info(f"Changing nsteps from {mdp['nsteps']} to -1 "
                        + "(infinte), run length is controlled via run args.")
            mdp["nsteps"] = -1
        # TODO: ensure that x-out and v-out/f-out are the same (if applicable)?
        self._mdp = mdp
        # TODO: store a hash/the file contents for gro, top, ndx?
        #       to check against when we load from storage/restart?
        #       if we do this do it in the property!
        #       (but still write one hashfunc for all!)
        self.gro_file = gro_file  # sets self._gro_file
        self.top_file = top_file  # sets self._top_file
        self.ndx_file = ndx_file  # sets self._ndx_file
        # dirty hack to make sure we also check for our defaults if they are
        # available + executable
        self.mdrun_executable = self.mdrun_executable
        self.grompp_executable = self.grompp_executable
        self._workdir = None
        self._prepared = False
        # NOTE: frames_done and steps_done do not have an easy relation!
        #       See the steps_done property docstring for more!
        # number of frames produced since last call to prepare
        self._frames_done = 0
        # number of integration steps done since last call to prepare
        self._steps_done = 0
        # integration time since last call to prepare in ps
        self._time_done = 0.
        self._nstout = None  # get this from the mdp only when we need it
        # Popen handle for gmx mdrun, used to check if we are running
        self._proc = None
        # these are set by prepare() and used by run_XX()
        self._simulation_part = None
        self._deffnm = None
        # tpr for trajectory (part), will become the structure/topology file
        self._tpr = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self._proc, asyncio.subprocess.Process):
            # cant pickle the process, + its probably dead when we unpickle :)
            state["_proc"] = None
        return state

    @property
    def grompp_executable(self):
        return self._grompp_executable

    @grompp_executable.setter
    def grompp_executable(self, val):
        # split because mdrun and grompp can be both subcommands of gmx
        test_exe = val.split(" ")[0]
        ensure_executable_available(test_exe)
        self._grompp_executable = val

    @property
    def mdrun_executable(self):
        return self._mdrun_executable

    @mdrun_executable.setter
    def mdrun_executable(self, val):
        test_exe = val.split(" ")[0]
        ensure_executable_available(test_exe)
        self._mdrun_executable = val

    @property
    def current_trajectory(self):
        if self._simulation_part == 0:
            # we could check if self_proc is set (which prepare sets to None)
            # this should make sure that calling current trajectory after
            # calling prepare does not return a traj, as soon as we called
            # run self._proc will be set, i.e. there is still no gurantee that
            # the traj is done, but it will be started always
            # (even when accessing simulataneous to the call to run),
            # i.e. it is most likely done
            # we can also check for simulation part, since it seems
            # gmx ignores that if no checkpoint is passed, i.e. we will
            # **always** start with part0001 anyways!
            # but checking for self._simulation_part == 0 also just makes sure
            # we never started a run (i.e. same as checking self._proc)
            return None
        elif (all(v is not None for v in [self._tpr, self._deffnm])
              and not self.running):
            # self._tpr and self._deffnm are set in prepare, i.e. having them
            # set makes sure that we have at least prepared running the traj
            # but it might not be done yet
            traj = Trajectory(
                    trajectory_file=os.path.join(
                                        self.workdir,
                                        (f"{self._deffnm}"
                                         + f"{self._num_suffix(self._simulation_part)}"
                                         + f".{self.output_traj_type}")
                                                 ),
                    structure_file=os.path.join(self.workdir, self._tpr),
                    nstout=self.nstout,
                              )
            return traj
        else:
            return None

    @property
    def ready_for_run(self):
        return self._prepared and not self.running

    @property
    def running(self):
        if self._proc is None:
            # this happens when we did not call run() yet
            return False
        if self._proc.returncode is None:
            # no return code means it is still running
            return True
        # dont care for the value of the exit code,
        # we are not running anymore if we crashed ;)
        return False

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, value):
        if not os.path.isdir(value):
            raise ValueError(f"Not a directory ({value}).")
        value = os.path.abspath(value)
        self._workdir = value

    @property
    def gro_file(self):
        return self._gro_file

    @gro_file.setter
    def gro_file(self, val):
        if not os.path.isfile(val):
            raise FileNotFoundError(f"gro file not found: {val}")
        val = os.path.abspath(val)
        self._gro_file = val

    @property
    def top_file(self):
        return self._top_file

    @top_file.setter
    def top_file(self, val):
        if not os.path.isfile(val):
            raise FileNotFoundError(f"top file not found: {val}")
        val = os.path.abspath(val)
        self._top_file = val

    @property
    def ndx_file(self):
        return self._ndx_file

    @ndx_file.setter
    def ndx_file(self, val):
        if val is not None:
            # GMX does not require an ndx file, so we accept None
            if not os.path.isfile(val):
                raise FileNotFoundError(f"ndx file not found: {val}")
            val = os.path.abspath(val)
        # set it anyway (even if it is None)
        self._ndx_file = val

    @property
    def dt(self):
        """Integration timestep in ps."""
        return self._mdp["dt"]

    @property
    def time_done(self):
        """
        Integration time since last call to prepare in ps.

        Takes into account 'tinit' from the .mdp file if set.
        """
        try:
            tinit = self._mdp["tinit"]
        except KeyError:
            tinit = 0.
        finally:
            return self._time_done - tinit

    # TODO/FIXME: we assume that all output frequencies are multiples of the
    #             smallest when determing the number of frames etc
    # TODO: check that nstxout == nstvout?!
    @property
    def nstout(self):
        """
        Smallest output frequency for current output_traj_type.
        """
        if self._nstout is None:
            nstout = nstout_from_mdp(self._mdp,
                                     traj_type=self.output_traj_type)
            self._nstout = nstout
        return self._nstout

    @property
    def steps_done(self):
        """
        Number of integration steps done since last call to `self.prepare()`.

        NOTE: steps != frames * nstout
        Some remarks on the relation between frames_done and steps_done:
        Usually (when passing `nsteps` to `run()`) frames_done will be equal to
        steps_done/nstout + 1 because the initial/final configuration will be
        written twice (since then the first/last step is always an output step)
        However as soon as we run for a specific walltime (without specifying
        `nsteps`) stuff gets complicated, then gromacs can potentially stop at
        every neighbor search step (where it also can/will write a checkpoint).
        If that step is not a trajectory output step, no output will be written
        to the traj and then the plus 1 rule for the double written
        initial/final configuration is off (since it will then be a 'normal'
        configuration written just once).
        If however the neighbor search and trajectory output fall togehter on
        the same step the configuration will be written twice (as with `nsteps`
        specified).
        """
        return self._steps_done

    @property
    def frames_done(self):
        """
        Number of frames produced since last call to `self.prepare()`.

        NOTE: frames != steps / nstout
        See the steps_done docstring for more.
        """
        return self._frames_done

    async def apply_constraints(self, conf_in, conf_out_name, wdir="."):
        return await self._0step_md(conf_in=conf_in,
                                    conf_out_name=conf_out_name,
                                    wdir=wdir,
                                    constraints=True,
                                    generate_velocities=False,
                                    local_mdrun=True)

    async def generate_velocities(self, conf_in, conf_out_name, wdir=".",
                                  constraints=True):
        return await self._0step_md(conf_in=conf_in,
                                    conf_out_name=conf_out_name,
                                    wdir=wdir,
                                    constraints=constraints,
                                    generate_velocities=True,
                                    local_mdrun=True)

    async def _0step_md(self, conf_in, conf_out_name, wdir,
                        constraints: bool, generate_velocities: bool,
                        local_mdrun: bool):
        # NOTE: local_mdrun is only relevant for SlurmGmxEngine: this way we
        #       can decide if we do the 0step mdruns locally or via slurm
        if (self.workdir is not None) and (wdir == "."):
            # use own working directory if know/set
            wdir = self.workdir
        if not os.path.isabs(conf_out_name):
            # assume conf_out is to be meant relative to wdir if not an abspath
            conf_out_name = os.path.join(wdir, conf_out_name)
        # work in a subdirectory of wdir to make deleting easy
        # generate its name at random to make sure we can use multiple
        # engines with 0stepMDruns in the same wdir
        run_name = "".join(random.choices((string.ascii_letters
                                           + string.ascii_lowercase
                                           + string.ascii_uppercase),
                                          k=6,
                                          )
                           )
        swdir = os.path.join(wdir, run_name)
        os.mkdir(swdir)
        constraints_mdp = copy.deepcopy(self._mdp)
        constraints_mdp["continuation"] = "no" if constraints else "yes"
        constraints_mdp["gen-vel"] = "yes" if generate_velocities else "no"
        if generate_velocities:
            # make sure we have draw a new/different random number for gen-vel
            constraints_mdp["gen-seed"] = -1
        constraints_mdp["nsteps"] = 0
        await self._run_grompp(workdir=swdir, deffnm=run_name,
                               trr_in=conf_in.trajectory_file,
                               tpr_out=f"{run_name}.tpr",
                               mdp_obj=constraints_mdp)
        # TODO: this is a bit hacky, and should probably not be necessary?
        #       we keep a ref to the 'old' self._proc to reset it after we are
        #       done, because the gmx_mdrun method set self._proc to the running
        #       constraints engine
        #       and it is probably not necessary since no engine should be able
        #       to be runing when/if we are able to call apply_constraints?
        old_proc_val = self._proc
        cmd_str = self._mdrun_cmd(tpr=f"{run_name}.tpr", deffnm=run_name)
        logger.info(f"{cmd_str}")
        await self._acquire_resources_gmx_mdrun(local_mdrun=local_mdrun)
        try:  # this try is just to make sure we always call cleanup/release
            await self._start_gmx_mdrun(cmd_str=cmd_str, workdir=swdir,
                                        local_mdrun=local_mdrun,
                                        deffnm=run_name,
                                        )
            try:
                # self._proc is set by _start_gmx_mdrun!
                exit_code = await self._proc.wait()
            except asyncio.CancelledError:
                self._proc.kill()
                raise  # reraise the error for encompassing coroutines
            else:
                if exit_code != 0:
                    raise EngineCrashedError("Non-zero exit code from mdrun."
                                             + f" Exit code was: {exit_code}.")
                # just get the output trajectory, it is only one configuration
                shutil.move(os.path.join(swdir, (f"{run_name}{self._num_suffix(1)}"
                                                 + f".{self.output_traj_type}")
                                         ),
                            conf_out_name)
                shutil.rmtree(swdir)  # remove the whole directory we used as wdir
                return Trajectory(
                                  trajectory_file=conf_out_name,
                                  # structure file of the conf_in because we
                                  # delete the other one with the folder
                                  structure_file=conf_in.structure_file,
                                  nstout=self.nstout,
                                  )
        finally:
            await self._cleanup_gmx_mdrun(local_mdrun=local_mdrun)
            self._proc = old_proc_val

    async def prepare(self, starting_configuration, workdir, deffnm):
        """
        Prepare a fresh simulation (starting with part0001).

        Parameters:
        -----------
        starting_configuration - `aimmd.distributed.Trajectory` with a trr traj
                                 or None, then the initial configuration is the
                                 gro-file
        workdir - absolute or relative path to an exisiting directory
        deffnm - the name (prefix) to use for all files
        """
        # deffnm is the default name/prefix for all outfiles (as in gmx)
        self.workdir = workdir  # sets to abspath and check if it is a dir
        if starting_configuration is None:
            # enable to start from the initial structure file ('-c' option)
            trr_in = None
        elif isinstance(starting_configuration, Trajectory):
            trr_in = starting_configuration.trajectory_file
        else:
            raise TypeError("starting_configuration must be None or a wrapped "
                            + f"trr ({Trajectory}).")
        self._deffnm = deffnm
        # check 'simulation-part' option in mdp file / MDP options
        # it decides at which .partXXXX the gmx numbering starts,
        # however gromacs ignores it if there is no -cpi [CheckPointIn]
        # so we do the same, i.e. we warn if we detect it is set
        # and check if there is a checkpoint with the right name [deffnm.cpt]
        # if yes we set our internal simulation_part counter to the value from
        # the mdp - 1 (we increase *before* each simulation part)
        try:
            sim_part = self._mdp["simulation-part"]
        except KeyError:
            # the gmx mdp default is 1, it starts at part0001
            # we add one at the start of each run, i.e. the numberings match up
            # and we will have tra=`...part0001.trr` from gmx
            # and confout=`...part0001.gro` from our naming
            self._simulation_part = 0
        else:
            if sim_part > 1:
                cpt_file = os.path.join(self.workdir, f"{deffnm}.cpt")
                if not os.path.isfile(cpt_file):
                    raise ValueError("'simulation-part' > 1 is only possible "
                                     + "if starting from a checkpoint, but "
                                     + f"{cpt_file} does not exists."
                                     )
                logger.warning(f"Starting value for 'simulation-part' > 1 (={sim_part}).")
            self._simulation_part = sim_part - 1

        tpr_out = os.path.join(self.workdir, deffnm + ".tpr")
        self._tpr = tpr_out  # remember the path to use as structure file for out trajs
        await self._run_grompp(workdir=self.workdir, deffnm=self._deffnm,
                               trr_in=trr_in, tpr_out=tpr_out,
                               mdp_obj=self._mdp)
        if not os.path.isfile(self._tpr):
            # better be save than sorry :)
            raise RuntimeError("Something went wrong generating the tpr."
                               + f"{self._tpr} does not seem to be a file.")
        # make sure we can not mistake a previous Popen for current mdrun
        self._proc = None
        self._frames_done = 0  # (re-)set how many frames we did
        self._steps_done = 0
        self._time_done = 0.
        self._prepared = True

    async def _run_grompp(self, workdir, deffnm, trr_in, tpr_out, mdp_obj):
        # NOTE: file paths from workdir and deffnm
        mdp_in = os.path.join(workdir, deffnm + ".mdp")
        # write the mdp file
        mdp_obj.write(mdp_in)
        mdp_out = os.path.join(workdir, deffnm + "_mdout.mdp")
        cmd_str = self._grompp_cmd(mdp_in=mdp_in, tpr_out=tpr_out,
                                   trr_in=trr_in, mdp_out=mdp_out)
        logger.info(f"{cmd_str}")
        # 3 file descriptors: stdin, stdout, stderr
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        try:
            grompp_proc = await asyncio.create_subprocess_exec(
                                                *shlex.split(cmd_str),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=workdir,
                                                               )
            stdout, stderr = await grompp_proc.communicate()
            logger.debug(f"grompp stdout: {stdout.decode()}.")
            # gromacs likes to talk on stderr ;)
            logger.debug(f"grompp stderr: {stderr.decode()}.")
            return_code = grompp_proc.returncode
            logger.info(f"grompp command returned {return_code}.")
        finally:
            # release the semaphore(s)
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
            _SEMAPHORES["MAX_FILES_OPEN"].release()
        if return_code != 0:
            # this assumes POSIX
            raise RuntimeError(f"grompp had non-zero return code ({return_code}).")

    async def prepare_from_files(self, workdir, deffnm):
        """
        Prepare continuation run starting from the last part found in workdir.

        Expects all files to exists, will (probably) fail otherwise.

        Parameters:
        -----------
        workdir - absolute or relative path to an existing directory
        deffnm - the name (prefix) to use for all files, must be the same as
                 for the previous run
        """
        self.workdir = workdir
        previous_trajs = get_all_traj_parts(self.workdir, deffnm=deffnm,
                                            traj_type=self.output_traj_type)
        # load the 'old' mdp_in
        self._mdp = MDP(os.path.join(self.workdir, f"{deffnm}.mdp"))
        self._deffnm = deffnm
        # Note that we dont need to explicitly check for the tpr existing,
        # if it does not exist we will err when getting the traj lengths
        self._tpr = os.path.join(self.workdir, deffnm + ".tpr")
        self._simulation_part = len(previous_trajs)
        # len(t), because for frames we do not care if first frame is in traj
        self._frames_done = sum(len(t) for t in previous_trajs)
        # steps done is the more reliable info if we want to know how many
        # integration steps we did
        self._steps_done = previous_trajs[-1].last_step
        self._time_done = previous_trajs[-1].last_time
        self._proc = None
        self._prepared = True

    # NOTE: this enables us to reuse run and prepare methods in SlurmGmxEngine,
    # i.e. we only need to overwite the next 3 functions to write out the slurm
    # submission script, submit the job and allocate/release different resources
    # NOTE: the kwargs enable us to pass e.g. local_mdrun to the slurm engine
    #  to enable us to run the 0step MDs (velocity generation and constraints)
    #  locally (on the login node or whereever the main code is running)
    async def _start_gmx_mdrun(self, cmd_str, workdir, **kwargs):
        proc = await asyncio.create_subprocess_exec(
                                            *shlex.split(cmd_str),
                                            stdout=asyncio.subprocess.PIPE,
                                            stderr=asyncio.subprocess.PIPE,
                                            cwd=workdir,
                                                    )
        self._proc = proc

    async def _acquire_resources_gmx_mdrun(self, **kwargs):
        # *always* called before any gmx_mdrun, use to get Semaphores for resources
        # for local gmx we need 3 file descriptors: stdin, stdout, stderr
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()
        await _SEMAPHORES["MAX_FILES_OPEN"].acquire()

    async def _cleanup_gmx_mdrun(self, **kwargs):
        # *always* called after any gmx_mdrun, use to release Semaphores and friends
        # read stdout and stderr from gmx mdrun
        stdout, stderr = await self._proc.communicate()
        logger.debug(f"gmx mdrun stdout: {stdout.decode()}")
        logger.debug(f"gmx mdrun stderr: {stderr.decode()}")
        # release the semaphore for the 3 file descriptors
        _SEMAPHORES["MAX_FILES_OPEN"].release()
        _SEMAPHORES["MAX_FILES_OPEN"].release()
        _SEMAPHORES["MAX_FILES_OPEN"].release()

    async def run(self, nsteps=None, walltime=None, steps_per_part=False):
        """
        Run simulation for specified number of steps or/and a given walltime.

        Note that you can pass both nsteps and walltime and the simulation will
        stop on the condition that is reached first.

        Parameters:
        -----------
        nsteps - int or None, integration steps to run for either in total
                 [as measured since the last call to `self.prepare()`]
                 or in the newly generated trajectory part,
                 see also the steps_per_part argument
        walltime - float or None, (maximum) walltime in hours
        steps_per_part - bool (default False), if True nsteps are the steps to
                         do in the new trajectory part
        """
        # generic run method is actually easier to implement for gmx :D
        if not self.ready_for_run:
            raise RuntimeError("Engine not ready for run. Call self.prepare() "
                               + "and/or check if it is still running.")
        if all(kwarg is None for kwarg in [nsteps, walltime]):
            raise ValueError("Neither steps nor walltime given.")
        if nsteps is not None:
            if nsteps % self.nstout != 0:
                raise ValueError(f"nsteps ({nsteps}) must be a multiple of "
                                 + f"nstout ({self.nstout}).")
            if not steps_per_part:
                nsteps = nsteps - self.steps_done
            if nsteps == 0:
                # Return None instead of raising an error, this makes it nicer
                # to use the run method with walltime and total nsteps inside
                # while loops, i.e. we can just call traj = e.run(...) and then
                # while traj is not None: traj = e.run()
                # TODO: this will make it complicated to ever use the GmxEngine
                #       for zero-step simulations to only apply constraints
                return None
            elif nsteps < 0:
                raise ValueError(f"nsteps is too small ({nsteps} steps for "
                                 + "this part). Can not travel backwards in "
                                 + "time...")

        self._simulation_part += 1
        cmd_str = self._mdrun_cmd(tpr=self._tpr, deffnm=self._deffnm,
                                  # TODO: use more/any other kwargs?
                                  maxh=walltime, nsteps=nsteps)
        logger.info(f"{cmd_str}")
        await self._acquire_resources_gmx_mdrun()
        try:  # this try is just to make sure we always call cleanup/release
            await self._start_gmx_mdrun(cmd_str=cmd_str, workdir=self.workdir)
            try:
                # self._proc is set by _start_gmx_mdrun!
                exit_code = await self._proc.wait()
            except asyncio.CancelledError:
                self._proc.kill()
                raise  # reraise the error for encompassing coroutines
            else:
                if exit_code != 0:
                    raise EngineCrashedError("Non-zero exit code from mdrun."
                                             + f" Exit code was: {exit_code}.")
                self._frames_done += len(self.current_trajectory)
                # dont care if we did a little more and only the checkpoint knows
                # we will only find out with the next trajectory part anyways
                self._steps_done = self.current_trajectory.last_step
                self._time_done = self.current_trajectory.last_time
                return self.current_trajectory
        finally:
            await self._cleanup_gmx_mdrun()

    async def run_steps(self, nsteps, steps_per_part=False):
        """
        Run simulation for specified number of steps.

        Parameters:
        -----------
        nsteps - int, integration steps to run for either in total [as measured
                 since the last call to `self.prepare()`]
                 or in the newly generated trajectory part
        steps_per_part - bool (default False), if True nsteps are the steps to
                         do in the new trajectory part
        """
        return await self.run(nsteps=nsteps, steps_per_part=steps_per_part)

    async def run_walltime(self, walltime):
        """
        Run simulation for a given walltime.

        Parameters:
        -----------
        walltime - float or None, (maximum) walltime in hours
        """
        return await self.run(walltime=walltime)

    def _num_suffix(self, sim_part):
        # construct gromacs num part suffix from simulation_part
        num_suffix = str(sim_part)
        while len(num_suffix) < 4:
            num_suffix = "0" + num_suffix
        num_suffix = ".part" + num_suffix
        return num_suffix

    def _grompp_cmd(self, mdp_in, tpr_out, trr_in=None, mdp_out=None):
        # all args are expected to be file paths
        cmd = f"{self.grompp_executable} -f {mdp_in} -c {self.gro_file}"
        cmd += f" -p {self.top_file}"
        if self.ndx_file is not None:
            cmd += f" -n {self.ndx_file}"
        if trr_in is not None:
            # input trr is optional
            # TODO/FIXME?!
            # TODO/NOTE: currently we do not pass '-time', i.e. we just use the
            #            gmx default frame selection: last frame from trr
            cmd += f" -t {trr_in}"
        if mdp_out is None:
            # find out the name and dir of the tpr to put the mdp next to it
            head, tail = os.path.split(tpr_out)
            name = tail.split(".")[0]
            mdp_out = os.path.join(head, name + ".mdout.mdp")
        cmd += f" -o {tpr_out} -po {mdp_out}"
        if self.grompp_extra_args != "":
            # add extra args string if it is not empty
            cmd += f" {self.grompp_extra_args}"
        return cmd

    def _mdrun_cmd(self, tpr, deffnm=None, maxh=None, nsteps=None):
        # use "-noappend" to avoid appending to the trajectories when starting
        # from checkpoints, instead let gmx create new files with .partXXXX suffix
        if deffnm is None:
            # find out the name of the tpr and use that as deffnm
            head, tail = os.path.split(tpr)
            deffnm = tail.split(".")[0]
        #cmd = f"{self.mdrun_executable} -noappend -deffnm {deffnm} -cpi"
        # NOTE: the line above does the same as the four below before the if-clauses
        #       however gromacs -deffnm is deprecated (and buggy),
        #       so we just make our own 'deffnm', i.e. we name all files the same
        #       except for the ending but do so explicitly
        cmd = f"{self.mdrun_executable} -noappend -s {tpr}"
        # always add the -cpi option, this lets gmx figure out if it wants
        # to start from a checkpoint (if there is one with deffnm)
        # cpi (CheckPointIn) is ignored if not present,
        # cpo (CheckPointOut) is the name to use for the (final) checkpoint
        cmd += f" -cpi {deffnm}.cpt -cpo {deffnm}.cpt"
        cmd += f" -o {deffnm}.trr -x {deffnm}.xtc -c {deffnm}.confout.gro"
        cmd += f" -e {deffnm}.edr -g {deffnm}.log"
        if maxh is not None:
            cmd += f" -maxh {maxh}"
        if nsteps is not None:
            cmd += f" -nsteps {nsteps}"
        if self.mdrun_extra_args != "":
            cmd += f" {self.mdrun_extra_args}"
        return cmd


# TODO: DOCUMENT!
class SlurmGmxEngine(GmxEngine):
    # use local prepare (i.e. grompp) of GmxEngine then submit run to slurm
    # we reuse the `GmxEngine._proc` to keep a reference to a `SlurmProcess`
    # which emulates the API of `asyncio.subprocess.Process` and can (for our
    # purposes) be used as a drop-in replacement, therefore we only need to
    # reimplement `_start_gmx_mdrun()`, `_acquire_resources_gmx_mdrun()` and
    # `_cleanup_gmx_mdrun()` to have a working SlurmGmxEngine
    # take submit script as str/file, use pythons .format to insert stuff!
    # TODO: document what we insert! currently: mdrun_cmd, jobname
    # TODO/FIXME: we should insert time if we know it!
    #  (some qeue eligibility can depend on time requested so we should request
    #   the known minimum)

    # TODO: these are possible options, but they result in added dependencies
    #        - jinja2 templates for slurm submission scripts?
    #          (does not look like we gain flexibility but we get more work,
    #           so probably not?!)
    #        - pyslurm for job status checks?!
    #          (it seems submission is frickly/impossible in pyslurm,
    #           so also probably not?!)

    _mdrun_executable = "gmx_mpi mdrun"  # MPI as default for clusters
    # whether we run the 0step MDruns (velocity generation and constraints)
    # without/sans slurm, i.e. locally (on the login node)
    # TODO/FIXME: running sans slurm is not possible using gmx_mpi!
    #             and we will probably also need the option to pass other mdrun_extra_args?!
    constraints_md_sans_slurm = False
    # make slurm executable setable from user-facing code but keep defaults
    # at central location in the `SlurmProcess`
    sacct_executable = SlurmProcess.sacct_executable
    sbatch_executable = SlurmProcess.sbatch_executable
    scancel_executable = SlurmProcess.scancel_executable

    def __init__(self, mdp, gro_file, top_file, sbatch_script, ndx_file=None,
                 **kwargs):
        """
        Initialize a `SlurmGmxEngine`.

        Parameters:
        -----------
        mdp - `aimmd.distributed.MDP`, the molecular dynamics parameters
        gro_file - absolute or relative path to a gromacs structure file
        top_file - absolute or relative path to a gromacs topolgy (.top) file
        sbatch_script - absolute or relative path to a slurm sbatch script
                        or a string with the content of the sbatch script
                        NOTE that the submission script must contain the
                        following placeholders (also see the examples folder):
                            {mdrun_cmd} - will be replaced by the command to
                                          run gmx mdrun
                            {jobname} - will be replaced by the name of the job
                                        containing the deffnm of the mdrun
        ndx_file - (optional) absolute or relative path to a gromacs index file

        Note that all attributes can be set at intialization by passing keyword
        arguments with their name, e.g. mdrun_extra_args="-ntomp 2" to instruct
        gromacs to use 2 openMP threads.
        """
        super().__init__(mdp=mdp, gro_file=gro_file, top_file=top_file,
                         ndx_file=ndx_file, **kwargs)
        # we expect sbatch_script to be a str,
        # but it could be either the path to a submit script or the content of
        # the submission script directly
        # we decide what it is by checking for the shebang
        if not sbatch_script.startswith("#!"):
            # probably path to a file, lets try to read it
            with open(sbatch_script, 'r') as f:
                sbatch_script = f.read()
        self.sbatch_script = sbatch_script

    async def apply_constraints(self, conf_in, conf_out_name, wdir="."):
        return await self._0step_md(conf_in=conf_in,
                                    conf_out_name=conf_out_name,
                                    wdir=wdir,
                                    constraints=True,
                                    generate_velocities=False,
                                    local_mdrun=self.constraints_md_sans_slurm)

    async def generate_velocities(self, conf_in, conf_out_name, wdir=".",
                                  constraints=True):
        return await self._0step_md(conf_in=conf_in,
                                    conf_out_name=conf_out_name,
                                    wdir=wdir,
                                    constraints=constraints,
                                    generate_velocities=True,
                                    local_mdrun=self.constraints_md_sans_slurm)

    async def _start_gmx_mdrun(self, cmd_str, workdir,
                               local_mdrun=False, deffnm=None, **kwargs):
        if local_mdrun:
            # run locally using supers start method
            # (used for 0step MDs: gen-vel and constraints)
            return await super()._start_gmx_mdrun(cmd_str=cmd_str,
                                                  workdir=workdir)
        # create a name from deffnm and partnum
        if deffnm is not None:
            name = deffnm + self._num_suffix(sim_part=1)
        else:
            name = self._deffnm + self._num_suffix(sim_part=self._simulation_part)
        # substitute placeholders in submit script
        script = self.sbatch_script.format(mdrun_cmd=cmd_str, jobname=name)
        # write it out
        fname = os.path.join(workdir, name + ".slurm")
        if os.path.exists(fname):
            # TODO: should we raise an error?
            logger.error(f"Overwriting existing submission file ({fname}).")
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            with open(fname, 'w') as f:
                f.write(script)
        self._proc = SlurmProcess(sbatch_script=fname, workdir=workdir,
                                  sacct_executable=self.sacct_executable,
                                  sbatch_executable=self.sbatch_executable,
                                  scancel_executable=self.scancel_executable,
                                  )
        await self._proc.submit()

    async def _acquire_resources_gmx_mdrun(self, local_mdrun=False, **kwargs):
        if local_mdrun:
            # running locally: use supers acquire method
            # (used for 0step MDs: gen-vel and constraints)
            return await super()._acquire_resources_gmx_mdrun()
        if _SEMAPHORES["SLURM_MAX_JOB"] is not None:
            logger.debug(f"SLURM_MAX_JOB semaphore is {_SEMAPHORES['SLURM_MAX_JOB']}"
                         + " before acquiring.")
            await _SEMAPHORES["SLURM_MAX_JOB"].acquire()
        else:
            logger.debug("SLURM_MAX_JOB semaphore is None")

    async def _cleanup_gmx_mdrun(self, local_mdrun=False, **kwargs):
        if local_mdrun:
            # running locally: use supers cleanup method
            # (used for 0step MDs: gen-vel and constraints)
            return await super()._cleanup_gmx_mdrun()
        if _SEMAPHORES["SLURM_MAX_JOB"] is not None:
            _SEMAPHORES["SLURM_MAX_JOB"].release()

    # TODO: do we even need/want that?
    @property
    def slurm_job_state(self):
        if self._proc is None:
            return None
        return self._proc.slurm_job_state
