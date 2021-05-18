"""
This file is part of ARCD.

ARCD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ARCD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ARCD. If not, see <https://www.gnu.org/licenses/>.
"""
import os
import abc
import shlex
import asyncio
import logging

from .trajectory import Trajectory
from .slurm import SlurmProcess
from .mdconfig import MDP


logger = logging.getLogger(__name__)


class MDEngineError(RuntimeError):
    """Exception raised when something goes wrong with the MDEngine."""
    pass


class EngineCrashedError(MDEngineError):
    """Exception raised when the MDEngine crashes during a run."""
    pass


class MDEngine(abc.ABC):
    @abc.abstractmethod
    # TODO: should we expect (require?) run_config to be a subclass of MDConfig?!
    # TODO: think about the most general interface!
    # NOTE: We assume that we do not change the system for/in one engine,
    #       i.e. .top, .ndx, ...?! should go into __init__
    def prepare(self, starting_configuration, workdir, deffnm, run_config):
        raise NotImplementedError

    @abc.abstractmethod
    def run_walltime(self, walltime):
        # run for specified walltime
        # NOTE: must be possible to run this multiple times after preparing once!
        raise NotImplementedError

    @abc.abstractmethod
    def run_nsteps(self, nsteps):
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

    # NOTE: this is redundant and poll() is not very async/await like
    #@abc.abstractmethod
    #def poll(self):
        # return the return code of last engine run
        # [or None (if the engine is running or never ran)]
    #    raise NotImplementedError

    @abc.abstractproperty
    def returncode(self):
        raise NotImplementedError

    @abc.abstractmethod
    def wait(self):
        raise NotImplementedError

    @abc.abstractproperty
    # TODO do we need this? or is prepare always fast enough that we can block
    #      everything/the main program flow and just wait for it to finish?
    def ready_for_run(self):
        # should be set to True when preparation is done
        raise NotImplementedError


# TODO: DOCUMENT!
# TODO: capture mdrun stdout+ stderr? for local gmx? for slurm it goes to file
# NOTE: with tra we mean trr, i.e. a full precision trajectory with velocities
class GmxEngine(MDEngine):
    # local prepare and option to run a local gmx (mainly for testing)
    grompp_executable = "gmx grompp"
    mdrun_executable = "gmx mdrun"
    # extra_args are expected to be str and will be appended to the end of the
    # respective commands after a separating space,
    # i.e. cmd = base_cmd + " " + extra_args
    grompp_extra_args = ""
    mdrun_extra_args = ""

    # TODO/FIXME: option to pass an index file!
    def __init__(self, gro_file, top_file, **kwargs):
        if not os.path.isfile(gro_file):
            raise FileNotFoundError(f"gro file not found: {gro_file}")
        if not os.path.isfile(top_file):
            raise FileNotFoundError(f"top file not found: {top_file}")
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
        # TODO: store a hash/the file contents for gro and top?
        #       to check against when we load from storage/restart?
        self.gro_file = os.path.abspath(gro_file)
        self.top_file = os.path.abspath(top_file)
        self._workdir = None
        self._prepared = False
        # Popen handle for gmx mdrun, used to check if we are running
        self._proc = None
        # these are set by prepare() and used by run_XX()
        self._simulation_part = None
        self._deffnm = None
        self._run_config = None
        self._tpr = None  # tpr for trajectory (part), will become the topology
        # TODO?!
        # counter for frames already produced, needed to enable run(nsteps)
        # to count steps per part
        # since gromacs takes nsteps since the beginning of the simulation

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self._proc, asyncio.subprocess.Process):
            # cant pickle the process, + its probably dead when we unpickle :)
            state["_proc"] = None
        return state

    @property
    def current_trajectory(self):
        if all(v is not None for v in [self._tpr, self._deffnm]):
            # self._tpr and self._deffnm are set in prepare, i.e. having them
            # set makes sure that we have at least prepared running the traj
            # but it might not be done yet
            # the simulation-part is set to s0 in prepare and incremented in
            # run, if it is 0 we can be sure that there is no traj started yet
            if self._simulation_part == 0:
                return None
            # TODO: check self._run_config if we write trr and/or xtc traj!
            traj = Trajectory(trajectory_file=os.path.join(
                        self.workdir, f"{self._deffnm}{self._num_suffix()}.trr"
                                                           ),
                              topology_file=os.path.join(self.workdir, self._tpr)
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
        if self.returncode is None:
            # no return code means it is still running
            return True
        # dont care for the value of the exit code,
        # we are not running anymore if we crashed ;)
        return False

    @property
    def returncode(self):
        if self._proc is not None:
            return self._proc.returncode
        # Note: we also return None when we never ran, i.e. with no _proc set
        return None

    async def wait(self):
        if self._proc is not None:
            return await self._proc.wait()

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, value):
        value = os.path.abspath(value)
        if not os.path.isdir(value):
            raise ValueError(f"Not a directory ({value}).")
        self._workdir = value

    async def prepare(self, starting_configuration, workdir, deffnm, run_config):
        # we require run_config to be a MDP (class)!
        # deffnm is the default name/prefix for all outfiles (as in gmx)
        self.workdir = workdir  # sets to abspath and check if it is a dir
        if starting_configuration is None:
            # enable to start from the initial structure file ('-c' option)
            trr_in = None
        elif isinstance(starting_configuration, Trajectory):
            trr_in = starting_configuration.trajectory_file
        else:
            raise TypeError(f"starting_configuration must be a wrapped trr ({Trajectory}).")
        if not isinstance(run_config, MDP):
            raise TypeError(f"run_config must be of type {MDP}.")
        if run_config["nsteps"][0] != -1:
            logger.info(f"Changing nsteps from {run_config['nsteps']} to -1 "
                        + "(infinte), run length is controlled via run args.")
            run_config["nsteps"] = -1
        self._run_config = run_config
        self._deffnm = deffnm
        # check 'simulation-part' option in mdp file / MDP options
        # it decides at which .partXXXX the gmx numbering starts
        # TODO: we warn if not zero and also adjust ourself accordingly,
        #       but should we instead set it to zero in the MDP?
        try:
            sim_part = self._run_config["simulation-part"]
        except KeyError:
            # the gmx mdp default is 0, it starts at part0001
            # we add one at the start of each run, i.e. the numberings match up
            # and we will have tra=`...part0001.trr` from gmx
            # and confout=`...part0001.gro` from our naming
            self._simulation_part = 0
        else:
            if sim_part > 0:
                logger.warning("Read non-zero starting value for 'simulation-part'.")
                self._simulation_part = sim_part
        # NOTE: file paths from workdir and deffnm
        mdp_in = os.path.join(self.workdir, deffnm + ".mdp")
        # write the mdp file
        self._run_config.write(mdp_in)
        tpr_out = os.path.join(self.workdir, deffnm + ".tpr")
        self._tpr = tpr_out  # keep a ref to use as topology file for out trajs
        mdp_out = os.path.join(self.workdir, deffnm + "_mdout.mdp")
        cmd_str = self._grompp_cmd(mdp_in=mdp_in, tpr_out=tpr_out,
                                   trr_in=trr_in, mdp_out=mdp_out)
        logger.info(f"{cmd_str}")
        grompp_proc = await asyncio.create_subprocess_exec(
                                                *shlex.split(cmd_str),
                                                stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE,
                                                cwd=self.workdir,
                                                           )
        stdout, stderr = await grompp_proc.communicate()
        return_code = grompp_proc.returncode
        logger.info(f"grompp command returned {return_code}.")
        logger.debug(f"grompp stdout: {stdout.decode()}.")
        # gromacs likes to talk on stderr ;)
        logger.debug(f"grompp stderr: {stderr.decode()}.")
        if return_code != 0:
            # this assumes POSIX
            raise RuntimeError(f"grompp had non-zero return code ({return_code}).")
        if not os.path.isfile(self._tpr):
            # better be save than sorry :)
            raise RuntimeError("Something went wrong generating the tpr."
                               + f"{self._tpr} does not seem to be a file.")
        # make sure we can not mistake a previous Popen for current mdrun
        self._proc = None
        self._prepared = True

    async def _start_gmx_mdrun(self, cmd_str):
        # should enable us to reuse run and prepare methods in SlurmGmxEngine,
        # i.e. we only need to overwite this function to write out the slurm
        # submission script and submit the job
        # TODO: capture sdtout/stderr? This would only work for local gmx...
        proc = await asyncio.create_subprocess_exec(
                                            *shlex.split(cmd_str),
                                            stdout=asyncio.subprocess.PIPE,
                                            stderr=asyncio.subprocess.PIPE,
                                            cwd=self.workdir,
                                                    )
        self._proc = proc

    async def run(self, nsteps=None, walltime=None):
        # generic run method is actually easier to implement for gmx :D
        if not self.ready_for_run:
            raise RuntimeError("Engine not ready for run. Call self.prepare() "
                               + "and/or check if it is still running.")
        if nsteps is None and walltime is None:
            raise ValueError("Neither nsteps nor walltime given.")

        self._simulation_part += 1
        cmd_str = self._mdrun_cmd(tpr=self._tpr, deffnm=self._deffnm,
                                  # TODO: use more/any other kwargs?
                                  maxh=walltime, nsteps=nsteps)
        logger.info(f"{cmd_str}")
        await self._start_gmx_mdrun(cmd_str=cmd_str)
        try:
            exit_code = await self.wait()
        except asyncio.CancelledError:
            self._proc.kill()
        else:
            if exit_code != 0:
                raise EngineCrashedError("Non-zero exit code from mdrun.")
            return self.current_trajectory

    async def run_nsteps(self, nsteps):
        # TODO: should nsteps be the number of steps per part?
        """
        NOTE: nsteps is the total number of steps in all traj-parts combined!
        """
        #        i.e. steps is reset to zero only when calling prepare
        #        if our trajectories would know their len this would be
        #        easy to fix by introducing a separate counter....
        #        we could also parse gmx output to know the last framenum?
        #        so we could do this together with untangeling/catching stdout!
        return await self.run(nsteps=nsteps, walltime=None)

    async def run_walltime(self, walltime):
        return await self.run(nsteps=None, walltime=walltime)

    def _num_suffix(self):
        # construct gromacs num part suffix from simulation_part
        num_suffix = str(self._simulation_part)
        while len(num_suffix) < 4:
            num_suffix = "0" + num_suffix
        num_suffix = ".part" + num_suffix
        return num_suffix

    def _grompp_cmd(self, mdp_in, tpr_out, trr_in=None, mdp_out=None):
        # all args are expected to be file paths
        cmd = f"{self.grompp_executable} -f {mdp_in} -c {self.gro_file}"
        cmd += f" -p {self.top_file}"
        if trr_in is not None:
            # input trr is optional
            # TODO/NOTE: currently we do not pass '-time', i.e. we just use the
            #            gmx default frame selection: last frame from trr
            cmd += f" -t {trr_in}"
        if mdp_out is None:
            # find out the name and dir of the tpr to put the mdp next to it
            head, tail = os.path.split(tpr_out)
            name = tail.split(".")[0]
            mdp_out = os.path.join(head, name + "_mdout.mdp")
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
        # always add the -cpi option, this lets gmx figure out if it wants
        # to start from a checkpoint (if there is one with deffnm)
        cmd = f"{self.mdrun_executable} -noappend -deffnm {deffnm} -cpi"
        if maxh is not None:
            cmd += f" -maxh {maxh}"
        if nsteps is not None:
            cmd += f" -nsteps {nsteps}"
        if self.mdrun_extra_args != "":
            cmd += f" {self.mdrun_extra_args}"
        return cmd


class SlurmGmxEngine(GmxEngine):
    # use local prepare (i.e. grompp) of GmxEngine then submit run to slurm
    # we reuse the `GmxEngine._proc` to keep the jobid
    # therefore we only need to reimplement `poll()` and `_start_gmx_mdrun()`
    # currently take submit script as str/file, use pythons .format to insert stuff!
    # TODO: document what we insert! currently: mdrun_cmd, jobname
    # TODO: we should insert time if we know it?!
    #  (some qeue eligibility can depend on time requested so we should request the known minimum)
    # we get the job state from parsing sacct output

    # TODO: these are improvements for the above options, but they result
    #       in additional dependencies...
    #        - jinja2 templates for slurm submission scripts?!
    #        - pyslurm for job status checks?!
    #          (it seems submission is frickly in pyslurm)

    mdrun_executable = "gmx_mpi mdrun"  # MPI as default for clusters

    def __init__(self, gro_file, top_file, sbatch_script, **kwargs):
        super().__init__(gro_file=gro_file, top_file=top_file, **kwargs)
        # we expect sbatch_script to be a str,
        # but it could be either the path to a submit script or the content of
        # the submission script directly
        # we decide what it is by checking for the shebang
        if not sbatch_script.startswith("#!"):
            # probably path to a file, lets try to read it
            with open(sbatch_script, 'r') as f:
                sbatch_script = f.read()
        self.sbatch_script = sbatch_script

    async def _start_gmx_mdrun(self, cmd_str):
        # create a name from deffnm and partnum
        name = self._deffnm + self._num_suffix()
        # substitute placeholders in submit script
        script = self.sbatch_script.format(mdrun_cmd=cmd_str, jobname=name)
        # write it out
        fname = os.path.join(self.workdir, name + ".slurm")
        if os.path.exists(fname):
            # TODO: should we raise an error?
            logger.error(f"Overwriting exisiting submission file ({fname}).")
        with open(fname, 'w') as f:
            f.write(script)
        self._proc = SlurmProcess(sbatch_script=fname)
        await self._proc.submit()

    # TODO: do we even need/want that?
    @property
    def slurm_job_state(self):
        if self._proc is None:
            return None
        return self._proc.slurm_job_state
