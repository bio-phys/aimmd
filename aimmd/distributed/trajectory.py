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
import collections
import asyncio
import inspect
import logging
import hashlib
import functools
import multiprocessing
import numpy as np
import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor
from scipy import constants


from .slurm import SlurmProcess
from .utils import ensure_executable_available
from . import _SEMAPHORES


logger = logging.getLogger(__name__)


# TODO: DOCUMENT!
# TODO: DaskTrajectoryFunctionWrapper?!
class TrajectoryFunctionWrapper:
    """ABC to define the API and some common methods."""
    def __init__(self, **kwargs) -> None:
        # NOTE: in principal we should set these after the stuff set via kwargs
        #       (otherwise users could overwrite them by passing _id="blub" to
        #        init), but since the subclasses sets call_kwargs again and
        #       have to calculate the id according to their own recipe anyway
        #       we can savely set them here (this enables us to use the id
        #        property at initialization time as e.g. in the slurm_jobname
        #        of the SlurmTrajectoryFunctionWrapper)
        self._id = None
        self._call_kwargs = {}  # init to empty dict such that iteration works
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

    @property
    def id(self) -> str:
        return self._id

    @property
    def call_kwargs(self):
        # return a copy to avoid people modifying entries without us noticing
        # TODO/FIXME: this will make unhappy users if they try to set single
        #             items in the dict!
        return self._call_kwargs.copy()

    @call_kwargs.setter
    def call_kwargs(self, value):
        if not isinstance(value, dict):
            raise ValueError("call_kwargs must be a dictionary.")
        self._call_kwargs = value
        self._id = self._get_id_str()  # get/set ID

    @abc.abstractmethod
    def _get_id_str(self):
        pass

    @abc.abstractmethod
    async def get_values_for_trajectory(self, traj):
        pass

    @abc.abstractmethod
    async def __call__(self, val):
        pass


class PyTrajectoryFunctionWrapper(TrajectoryFunctionWrapper):
    """Wrap python functions for use on `aimmd.distributed.Trajectory."""
    # wrap functions for use on aimmd.distributed.Trajectory
    # makes sure that we check for cached values if we apply the wrapped func
    # to an aimmd.distributed.Trajectory
    def __init__(self, function, call_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self._func = None
        self._func_src = None
        # use the properties to directly calculate/get the id
        self.function = function
        self.call_kwargs = call_kwargs

    def __repr__(self) -> str:
        return (f"PyTrajectoryFunctionWrapper(function={self._func}, "
                + f"call_kwargs={self.call_kwargs})"
                )

    def _get_id_str(self):
        # calculate a hash over function src and call_kwargs dict
        # this should be unique and portable, i.e. it should enable us to make
        # ensure that the cached values will only be used for the same function
        # called with the same arguments
        id = 0
        # NOTE: addition is commutative, i.e. order does not matter here!
        for k, v in self._call_kwargs.items():
            # hash the value
            id += int(hashlib.blake2b(str(v).encode('utf-8')).hexdigest(), 16)
            # hash the key
            id += int(hashlib.blake2b(str(k).encode('utf-8')).hexdigest(), 16)
        # and add the func_src
        id += int(hashlib.blake2b(str(self._func_src).encode('utf-8')).hexdigest(), 16)
        return str(id)  # return a str because we want to use it as dict keys

    @property
    def function(self):
        return self._func

    @function.setter
    def function(self, value):
        try:
            src = inspect.getsource(value)
        except OSError:
            # OSError is raised if source can not be retrieved
            self._func_src = None
            self._id = None
            logger.warning(f"Could not retrieve source for {value}."
                           + " No caching can/will be performed.")
        else:
            self._func_src = src
            self._id = self._get_id_str()  # get/set ID
        finally:
            self._func = value

    async def get_values_for_trajectory(self, traj):
        loop = asyncio.get_running_loop()
        async with _SEMAPHORES["MAX_PROCESS"]:
            # NOTE: make sure we do not fork! (not save with multithreading)
            # see e.g. https://stackoverflow.com/questions/46439740/safe-to-call-multiprocessing-from-a-thread-in-python
            ctx = multiprocessing.get_context("forkserver")
            # fill in additional kwargs (if any)
            if len(self.call_kwargs) > 0:
                func = functools.partial(self.function, **self._call_kwargs)
            else:
                func = self.function
            # use one python subprocess: if func releases the GIL
            # it does not matter anyway, if func is full py 1 is enough
            with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                vals = await loop.run_in_executor(pool, func, traj)
        return vals

    async def __call__(self, value):
        if isinstance(value, Trajectory) and self.id is not None:
            return await value._apply_wrapped_func(self.id, self)
        else:
            # NOTE: i think this should never happen?
            # this will block until func is done, we could use a ProcessPool?!
            # Can we make sure that we are always able to pickle value for that?
            # (probably not since it could be Trajectory and we only have no func_src)
            return self._func(value, **self._call_kwargs)


# TODO: document what we fill/replace in the master sbatch script!
# TODO: document what we expect from the executable!
#       -> accept struct, traj, outfile
#       -> write numpy npy files! (or pass custom load func!)
class SlurmTrajectoryFunctionWrapper(TrajectoryFunctionWrapper):
    """
    Wrap executables to use on `aimmd.distributed.Trajectory` via SLURM.

    The execution of the job is submited to the queueing system with the
    given sbatch script (template).
    The executable will be called with the following positional arguments:
        - full filepath of the structure file associated with the trajectory
        - full filepath of the trajectory to calculate values for
        - full filepath of the file the results should be written to without
          fileending, Note that if no custom loading function is supplied we
          expect that the written file has 'npy' format and the added ending
          '.npy', i.e. we expect the executable to add the ending '.npy' to
          the passed filepath (as e.g. `np.save($FILEPATH, data)` would do)
        - any additional arguments from call_kwargs are added as
          `" {key} {value}" for key, value in call_kwargs.items()`
    See also the examples for a reference (python) implementation of multiple
    different functions/executables for use with this class.

    Notable attributes:
    -------------------
    slurm_jobname - Used as name for the job in slurm and also as part of the
                    filename for the submission script that will be written
                    (and deleted if everything goes well) for every trajectory.
    """
    # make it possible to set values for slurm executables from this class
    # but keep the defaults in one central location (the `SlurmProcess`)
    sacct_executable = SlurmProcess.sacct_executable
    sbatch_executable = SlurmProcess.sbatch_executable
    scancel_executable = SlurmProcess.scancel_executable

    def __init__(self, executable, sbatch_script, call_kwargs={},
                 load_results_func=None, slurm_maxjob_semaphore=None, **kwargs,
                 ):
        """
        Initialize `SlurmTrajectoryFunctionWrapper`.

        Parameters:
        -----------
        executable - absolute or relative path to an executable or name of an
                     executable available via the environment (e.g. via the
                      $PATH variable on LINUX)
        sbatch_script - path to a sbatch submission script file or string with
                        the content of a submission script.
                        NOTE that the submission script must contain the
                        following placeholders (also see the examples folder):
                            {cmd_str} - will be replaced by the command to call
                                        the executable on a given trajectory
                            {jobname} - will be replaced by the name of the job
                                        containing the hash of the function
        call_kwargs - dictionary of additional arguments to pass to the
                      executable, they will be added to the call as pair
                      ' {key} {val}', note that in case you want to pass single
                      command line flags (like '-v') this can be achieved by
                      setting key='-v' and val='', i.e. to the empty string
        load_results_func - None or function to call to customize the loading
                            of the results, if a function it will be called
                            with the full path to the results file (as in the
                            call to the executable) and should return a numpy
                            array containing the loaded values
        slurm_maxjob_semaphore - None or `asyncio.Semaphore`, can be used to
                                 bound the maximum number of submitted jobs
                                 NOTE: The semaphore will be lost upon
                                       unpickling, i.e. also when saving this
                                       to an aimmd.Storage

        Note that all attributes can be set via __init__ by passing them as
        keyword arguments.
        """
        # property defaults before superclass init to be resettable via kwargs
        self._slurm_jobname = None
        super().__init__(**kwargs)
        self._executable = None
        # we expect sbatch_script to be a str,
        # but it could be either the path to a submit script or the content of
        # the submission script directly
        # we decide what it is by checking for the shebang
        if not sbatch_script.startswith("#!"):
            # probably path to a file, lets try to read it
            with open(sbatch_script, 'r') as f:
                sbatch_script = f.read()
        # (possibly) use properties to calc the id directly
        self.sbatch_script = sbatch_script
        self.executable = executable
        self.call_kwargs = call_kwargs
        self.load_results_func = load_results_func
        self.slurm_maxjob_semaphore = slurm_maxjob_semaphore

    @property
    def slurm_jobname(self):
        if self._slurm_jobname is None:
            return f"CVfunc_id_{self.id}"
        return self._slurm_jobname

    @slurm_jobname.setter
    def slurm_jobname(self, val):
        self._slurm_jobname = val

    def __repr__(self) -> str:
        return (f"SlurmTrajectoryFunctionWrapper(executable={self._executable}, "
                + f"call_kwargs={self.call_kwargs})"
                )

    def _get_id_str(self):
        # calculate a hash over executable and call_kwargs dict
        # this should be unique and portable, i.e. it should enable us to make
        # ensure that the cached values will only be used for the same function
        # called with the same arguments
        id = 0
        # NOTE: addition is commutative, i.e. order does not matter here!
        for k, v in self._call_kwargs.items():
            # hash the value
            id += int(hashlib.blake2b(str(v).encode('utf-8')).hexdigest(), 16)
            # hash the key
            id += int(hashlib.blake2b(str(k).encode('utf-8')).hexdigest(), 16)
        # and add the executable hash
        with open(self.executable, "rb") as exe_file:
            # NOTE: we assume that executable is small enough to read at once
            #       if this crashes becasue of OOM we should use chunks...
            data = exe_file.read()
        id += int(hashlib.blake2b(data).hexdigest(), 16)
        return str(id)  # return a str because we want to use it as dict keys

    @property
    def executable(self):
        return self._executable

    @executable.setter
    def executable(self, val):
        exe = ensure_executable_available(val)
        # if we get here it should be save to set, i.e. it exists + has X-bit
        self._executable = exe
        self._id = self._get_id_str()  # get the new hash/id

    async def get_values_for_trajectory(self, traj):
        # first construct the path/name for the numpy npy file in which we expect
        # the results to be written
        tra_dir, tra_name = os.path.split(traj.trajectory_file)
        result_file = os.path.join(tra_dir,
                                   f"{tra_name}_CVfunc_id_{self.id}")
        # we expect executable to take 3 postional args:
        # struct traj outfile
        cmd_str = f"{self.executable} {traj.structure_file}"
        cmd_str += f" {traj.trajectory_file} {result_file}"
        if len(self.call_kwargs) > 0:
            for key, val in self.call_kwargs.items():
                cmd_str += f" {key} {val}"
        # construct jobname
        # TODO: do we want the traj name in the jobname here?!
        #       I think rather not, becasue then we can cancel all jobs for one
        #       trajfunc in one `scancel` (i.e. independant of the traj)
        # now prepare the sbatch script
        script = self.sbatch_script.format(cmd_str=cmd_str,
                                           jobname=self.slurm_jobname)
        # write it out
        sbatch_fname = os.path.join(tra_dir,
                                    tra_name + "_" + self.slurm_jobname + ".slurm")
        if os.path.exists(sbatch_fname):
            # TODO: should we raise an error?
            logger.error(f"Overwriting exisiting submission file ({sbatch_fname}).")
        async with _SEMAPHORES["MAX_FILES_OPEN"]:
            with open(sbatch_fname, 'w') as f:
                f.write(script)
        # and submit it
        slurm_proc = SlurmProcess(sbatch_script=sbatch_fname, workdir=tra_dir,
                                  sacct_executable=self.sacct_executable,
                                  sbatch_executable=self.sbatch_executable,
                                  scancel_executable=self.scancel_executable,
                                  sleep_time=15,  # sleep 15 s between checking
                                  )
        if self.slurm_maxjob_semaphore is not None:
            await self.slurm_maxjob_semaphore.acquire()
        try:  # this try is just to make sure we always release the semaphore
            await slurm_proc.submit()
            # wait for the slurm job to finish
            # also cancel the job when this future is canceled
            try:
                exit_code = await slurm_proc.wait()
            except asyncio.CancelledError:
                slurm_proc.kill()
                raise  # reraise for encompassing coroutines
            else:
                if exit_code != 0:
                    raise RuntimeError(
                                "Non-zero exit code from CV batch job for "
                                + f"executable {self.executable} on "
                                + f"trajectory {traj.trajectory_file} "
                                + f"(slurm jobid {slurm_proc.slurm_jobid})."
                                + f" Exit code was: {exit_code}."
                                       )
                os.remove(sbatch_fname)
                if self.load_results_func is None:
                    # we do not have '.npy' ending in results_file,
                    # numpy.save() adds it if it is not there, so we need it here
                    vals = np.load(result_file + ".npy")
                    os.remove(result_file + ".npy")
                else:
                    # use custom loading function from user
                    vals = self.load_results_func(result_file)
                    os.remove(result_file)
                try:
                    # (try to) remove slurm output files
                    os.remove(
                        os.path.join(tra_dir,
                                     f"{self.slurm_jobname}.out.{slurm_proc.slurm_jobid}",
                                     )
                              )
                    os.remove(
                        os.path.join(tra_dir,
                                     f"{self.slurm_jobname}.err.{slurm_proc.slurm_jobid}",
                                     )
                              )
                except FileNotFoundError:
                    # probably just a naming issue, so lets warn our users
                    logger.warning(
                            "Could not remove SLURM output files. Maybe "
                            + "they were not named as expected? Consider "
                            + "adding '#SBATCH -o ./{jobname}.out.%j'"
                            + " and '#SBATCH -e ./{jobname}.err.%j' to the "
                            + "submission script.")
                return vals
        finally:
            if self.slurm_maxjob_semaphore is not None:
                self.slurm_maxjob_semaphore.release()

    async def __call__(self, value):
        if isinstance(value, Trajectory) and self.id is not None:
            return await value._apply_wrapped_func(self.id, self)
        else:
            raise ValueError("SlurmTrajectoryFunctionWrapper must be called"
                             + " with an `aimmd.distributed.Trajectory` "
                             + f"but was called with {type(value)}.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["slurm_maxjob_semaphore"] = None
        return state


class Trajectory:
    """
    Represent a trajectory.

    Keep track of the paths of the trajectory and the structure file.
    Caches values for (wrapped) functions acting on the trajectory.
    Also makes vailable (and caches) a number of useful attributes, namely:
        - first_step (integration step of first frame in this trajectory [part])
        - last_step (integration step of the last frame in this trajectory [part])
        - dt (the timeintervall between subsequent *frames* [not steps])
        - first_time (the integration time of the first frame)
        - last_time (the integration time of the last frame)
        - length (number of frames in this trajectory)
        - nstout (number of integration steps between subsequent frames)

    NOTE: first_step and last_step is only useful for trajectories that come
          directly from a MDEngine. As soon as the trajecory has been
          concatenated using MDAnalysis (i.e. the `TrajectoryConcatenator`)
          the step information is just the frame number in the trajectory part
          that became first/last frame in the concatenated trajectory.
    """

    def __init__(self, trajectory_file, structure_file, nstout=None, **kwargs):
        # NOTE: we assume tra = trr and struct = tpr
        #       but we also expect that anything which works for mdanalysis as
        #       tra and struct should also work here as tra and struct

        # TODO: currently we do not use kwargs?!
        #dval = object()
        #for kwarg, value in kwargs.items():
        #    cval = getattr(self, kwarg, dval)
        #    if cval is not dval:
        #        if isinstance(value, type(cval)):
        #            # value is of same type as default so set it
        #            setattr(self, kwarg, value)
        #        else:
        #            logger.warn(f"Setting attribute {kwarg} with "
        #                        + f"mismatching type ({type(value)}). "
        #                        + f" Default type is {type(cval)}."
        #                        )
        if os.path.isfile(trajectory_file):
            self.trajectory_file = os.path.abspath(trajectory_file)
        else:
            raise ValueError(f"trajectory_file ({trajectory_file}) must be accessible.")
        if os.path.isfile(structure_file):
            self.structure_file = os.path.abspath(structure_file)
        else:
            raise ValueError(f"structure_file ({structure_file}) must be accessible.")
        # properties
        self.nstout = nstout  # use the setter to make basic sanity checks
        self._len = None
        self._first_step = None
        self._last_step = None
        self._dt = None
        self._first_time = None
        self._last_time = None
        # stuff for caching of functions applied to this traj
        self._func_id_to_idx = {}
        self._func_values = []
        self._h5py_grp = None
        self._h5py_cache = None

    def __len__(self):
        if self._len is not None:
            return self._len
        # create/open a mdanalysis universe to get the number of frames
        u = mda.Universe(self.structure_file, self.trajectory_file,
                         tpr_resid_from_one=True)
        self._len = len(u.trajectory)
        return self._len

    def __repr__(self):
        return (f"Trajectory(trajectory_file={self.trajectory_file},"
                + f" structure_file={self.structure_file})"
                )

    @property
    def nstout(self):
        """Output frequency between subsequent frames in integration steps."""
        return self._nstout

    @nstout.setter
    def nstout(self, val):
        if val is not None:
            # ensure that it is an int
            val = int(val)
        # enable setting to None
        self._nstout = val

    @property
    def first_step(self):
        """The integration step of the first frame."""
        if self._first_step is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[0]
            # NOTE: works only(?) for trr and xtc
            self._first_step = ts.data["step"]
        return self._first_step

    @property
    def last_step(self):
        """The integration step of the last frame."""
        if self._last_step is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[-1]
            # TODO/FIXME:
            # NOTE: works only(?) for trr and xtc
            self._last_step = ts.data["step"]
        return self._last_step

    @property
    def dt(self):
        """The time intervall between subsequent *frames* (not steps) in ps."""
        if self._dt is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            # any frame is fine (assuming they all have the same spacing)
            ts = u.trajectory[0]
            self._dt = ts.data["dt"]
        return self._dt

    @property
    def first_time(self):
        """The integration timestep of the first frame in ps."""
        if self._first_time is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[0]
            self._first_time = ts.data["time"]
        return self._first_time

    @property
    def last_time(self):
        """The integration timestep of the last frame in ps."""
        if self._last_time is None:
            u = mda.Universe(self.structure_file, self.trajectory_file,
                             tpr_resid_from_one=True)
            ts = u.trajectory[-1]
            self._last_time = ts.data["time"]
        return self._last_time

    async def _apply_wrapped_func(self, func_id, wrapped_func):
        if self._h5py_grp is not None:
            # first check if we are loaded and possibly get it from there
            # trajectories are immutable once stored, so no need to check len
            try:
                return self._h5py_cache[func_id]
            except KeyError:
                # not in there
                # send function application to seperate process and wait for it
                vals = await wrapped_func.get_values_for_trajectory(self)
                # we set ignore_existing to not err if another thread was
                # faster than us in calculating the values for the traj
                # this happens when we call the function twice on the same
                # trajectory and the second call is while the first one calculates
                # the CVs, then the cache is still empty when the second thread
                # starts but the values are in there when it finishes
                self._h5py_cache.append(func_id, vals, ignore_existing=True)
                return vals
        # only 'local' cache, i.e. this trajectory has no file associated (yet)
        try:
            # see if it is in cache
            idx = self._func_id_to_idx[func_id]
            return self._func_values[idx]
        except KeyError:
            # if not calculate, store and return
            # send function application to seperate process and wait for it
            vals = await wrapped_func.get_values_for_trajectory(self)
            # check again to make sure it is not been added in the meantime
            # see above why/when this can happen
            try:
                idx = self._func_id_to_idx[func_id]
            except KeyError:
                # not in there so set it
                self._func_id_to_idx[func_id] = len(self._func_id_to_idx)
                self._func_values.append(vals)
            else:
                # someone was faster, do nothing
                logger.debug(f"Local cache values already present for function with id {func_id}."
                             + "Ignoring the newly calculated values.")
                pass
            finally:
                return vals

    def __getstate__(self):
        # enable pickling of Trajecory without call to ready_for_pickle
        # this should make it possible to pass it into a ProcessPoolExecutor
        # and lets us calculate TrajectoryFunction values asyncronously
        # NOTE: this removes everything except the filepaths
        state = self.__dict__.copy()
        state["_h5py_cache"] = None
        state["_h5py_grp"] = None
        #state["_func_values"] = []
        #state["_func_id_to_idx"] = {}
        return state

    def object_for_pickle(self, group, overwrite):
        # TODO/NOTE: we ignore overwrite and assume the group is always empty
        #            (or at least matches this tra and we can add values?)
        # currently overwrite will always be false and we can just ignore it?!
        # and then we can/do also expect group to be empty...?
        state = self.__dict__.copy()
        if self._h5py_grp is not None:
            # we already have a file?
            # lets try to link the two groups?
            group = self._h5py_grp
            # or should we copy? how to make sure we update both caches?
            # do we even want that? I (hejung) think a link is what you would
            # expect, i.e. both stored copies of the traj will have all cached
            # values available
            #group.copy(self._h5py_grp, group)
            state["_h5py_grp"] = None
            state["_h5py_cache"] = None
        else:
            # set h5py group such that we use the cache from now on
            self._h5py_grp = group
            self._h5py_cache = TrajectoryFunctionValueCache(self._h5py_grp)
            for func_id, idx in self._func_id_to_idx.items():
                self._h5py_cache.append(func_id, self._func_values[idx])
            # clear the 'local' cache and empty state, such that we initialize
            # to empty, next time we will get it all from file directly
            self._func_values = state["_func_values"] = []
            self._func_id_to_idx = state["_func_id_to_idx"] = {}
        # make the return object
        ret_obj = self.__class__.__new__(self.__class__)
        ret_obj.__dict__.update(state)
        return ret_obj

    def complete_from_h5py_group(self, group):
        # NOTE: Trajectories are immutable once stored,
        # EXCEPT: adding more cached function values for other funcs
        # so we keep around a ref to the h5py group we load from and
        # can then add the stuff in there
        # (loading is as easy as adding the file-cache because we store
        #  everything that was in 'local' cache in the file when we save)
        self._h5py_grp = group
        self._h5py_cache = TrajectoryFunctionValueCache(group)
        return self


class TrajectoryFunctionValueCache(collections.abc.Mapping):
    """Interface for caching function values on a per trajectory basis."""
    # NOTE: this is written with the assumption that stored trajectories are
    #       immutable (except for adding additional stored function values)
    #       but we assume that the actual underlying trajectory stays the same,
    #       i.e. it is not extended after first storing it

    def __init__(self, root_grp):
        self._root_grp = root_grp
        self._h5py_paths = {"ids": "FunctionIDs",
                            "vals": "FunctionValues"
                            }
        self._ids_grp = self._root_grp.require_group(self._h5py_paths["ids"])
        self._vals_grp = self._root_grp.require_group(self._h5py_paths["vals"])

    def __len__(self):
        return len(self._ids_grp.keys())

    def __iter__(self):
        for idx in range(len(self)):
            yield self._ids_grp[str(idx)].asstr()[()]

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Keys must be of type str.")
        for idx, k_val in enumerate(self):
            if key == k_val:
                return self._vals_grp[str(idx)][:]
        # if we got until here the key is not in there
        raise KeyError("Key not found.")

    def append(self, func_id, vals, ignore_existing=False):
        if not isinstance(func_id, str):
            raise TypeError("Keys (func_id) must be of type str.")
        if (func_id in self) and (not ignore_existing):
            raise ValueError(f"There are already values stored for func_id {func_id}."
                             + " Changing the stored values is not supported.")
        elif (func_id in self) and ignore_existing:
            logger.debug(f"File cached values already present for function with id {func_id}."
                         + "Not adding the new values because ignore_existing=False.")
            return
        # TODO: do we also want to check vals for type?
        name = str(len(self))
        _ = self._ids_grp.create_dataset(name, data=func_id)
        _ = self._vals_grp.create_dataset(name, data=vals)


class TrajectoryConcatenator:
    """
    Create concatenated trajectory from given trajectories and frames.

    The concatenate method takes a list of trajectories and a list of slices,
    returns one trajectory containing only the selected frames in that order.
    Velocities are automatically inverted if the step of a slice is negative,
    this can be controlled via the invert_v_for_negative_step attribute.

    NOTE: We assume that all trajs have the same structure file
          and attach the the structure of the first traj if not told otherwise.
    """

    def __init__(self, invert_v_for_negative_step=True):
        self.invert_v_for_negative_step = invert_v_for_negative_step

    def concatenate(self, trajs, slices, tra_out, struct_out=None,
                    overwrite=False, remove_double_frames=True):
        """
        Create concatenated trajectory from given trajectories and frames.

        trajs - list of `:class:`Trajectory
        slices - list of (start, stop, step)
        tra_out - output trajectory filepath, absolute or relativ to cwd
        struct_out - None or output structure filepath, if None we will take the
                     structure file of the first trajectory in trajs
        overwrite - bool (default=False), if True overwrite existing tra_out,
                    if False and the file exists raise an error
        remove_double_frames - bool (default=True), if True try to remove double
                               frames from the concatenated output
                               NOTE: that we use a simple heuristic, we just
                                     check if the integration time is the same
        """
        tra_out = os.path.abspath(tra_out)
        if os.path.exists(tra_out) and not overwrite:
            raise ValueError(f"overwrite=False and tra_out exists: {tra_out}")
        struct_out = (trajs[0].structure_file if struct_out is None
                      else os.path.abspath(struct_out))
        if not os.path.isfile(struct_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise ValueError(f"Output structure file must exist ({struct_out}).")

        # special treatment for traj0 because we need n_atoms for the writer
        u0 = mda.Universe(trajs[0].structure_file, trajs[0].trajectory_file,
                          tpr_resid_from_one=True)
        start0, stop0, step0 = slices[0]
        # if the file exists MDAnalysis will silently overwrite
        with mda.Writer(tra_out, n_atoms=u0.trajectory.n_atoms) as W:
            for ts in u0.trajectory[start0:stop0:step0]:
                if self.invert_v_for_negative_step and step0 < 0:
                    u0.atoms.velocities *= -1
                W.write(u0.atoms)
                if remove_double_frames:
                    # remember the last timestamp, so we can take it out
                    last_time_seen = ts.data["time"]
            del u0  # should free up memory and does no harm?!
            for traj, sl in zip(trajs[1:], slices[1:]):
                u = mda.Universe(traj.structure_file, traj.trajectory_file,
                                 tpr_resid_from_one=True)
                start, stop, step = sl
                for ts in u.trajectory[start:stop:step]:
                    if remove_double_frames:
                        if last_time_seen == ts.data["time"]:
                            # this is a no-op, as they are they same...
                            #last_time_seen = ts.data["time"]
                            continue  # skip this timestep/go to next iteration
                    if self.invert_v_for_negative_step and step < 0:
                        u.atoms.velocities *= -1
                    W.write(u.atoms)
                    if remove_double_frames:
                        last_time_seen = ts.data["time"]
                del u
        # return (file paths to) the finished trajectory
        return Trajectory(tra_out, struct_out)


class FrameExtractor(abc.ABC):
    # extract a single frame with given idx from a trajectory and write it out
    # simplest case is without modification, but useful modifications are e.g.
    # with inverted velocities, with random Maxwell-Boltzmann velocities, etc.

    @abc.abstractmethod
    def apply_modification(self, universe, ts):
        # this func will is called when the current timestep is at the choosen
        # frame and applies the subclass specific frame modifications to the
        # mdanalysis universe, after this function finishes the frames is
        # written out, i.e. with potential modifications applied
        # no return value is expected or considered,
        # the modifications in the universe are nonlocal anyway
        raise NotImplementedError

    def extract(self, outfile, traj_in, idx, struct_out=None, overwrite=False):
        # TODO: should we check that idx is an idx, i.e. an int?
        # TODO: make it possible to select a subset of atoms to write out
        #       and also for modification?
        # TODO: should we make it possible to extract multiple frames, i.e.
        #       enable the use of slices (and iterables of indices?)
        """
        Extract a single frame from trajectory and write it out.

        outfile - path to output file (relative or absolute)
        traj_in - `:class:Trajectory` from which the original frame is taken
        idx - index of the frame in the input trajectory
        struct_out - None or output structure filepath, if None we will take the
                     structure file of the input trajectory
        overwrite - bool (default=False), if True overwrite existing tra_out,
                    if False and the file exists raise an error
        """
        outfile = os.path.abspath(outfile)
        if os.path.exists(outfile) and not overwrite:
            raise ValueError(f"overwrite=False and outfile exists: {outfile}")
        struct_out = (traj_in.structure_file if struct_out is None
                      else os.path.abspath(struct_out))
        if not os.path.isfile(struct_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise ValueError(f"Output structure file must exist ({struct_out}).")
        u = mda.Universe(traj_in.structure_file, traj_in.trajectory_file,
                         tpr_resid_from_one=True)
        with mda.Writer(outfile, n_atoms=u.trajectory.n_atoms) as W:
            ts = u.trajectory[idx]
            self.apply_modification(u, ts)
            W.write(u.atoms)
        return Trajectory(trajectory_file=outfile, structure_file=struct_out)


class NoModificationFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out without modification."""

    def apply_modification(self, universe, ts):
        pass


class InvertedVelocitiesFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out with inverted velocities."""

    def apply_modification(self, universe, ts):
        ts.velocities *= -1.


class RandomVelocitiesFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out with randomized velocities."""

    def __init__(self, T):
        """Temperature T must be given in degree Kelvin."""
        self.T = T  # in K
        self._rng = np.random.default_rng()

    def apply_modification(self, universe, ts):
        # m is in units of g / mol
        # v should be in units of \AA / ps = 100 m / s
        # which means m [10**-3 kg / mol] v**2 [10000 (m/s)**2]
        # is in units of [ 10 kg m**s / (mol * s**2) ]
        # so we use R = N_A * k_B [J / (mol * K) = kg m**2 / (s**2 * mol * K)]
        # and add in a factor 10 to get 1/σ**2 = m / (k_B * T) in the right units
        scale = np.empty((ts.n_atoms, 3), dtype=np.float64)
        s1d = np.sqrt((self.T * constants.R * 0.1)
                      / universe.atoms.masses
                      )
        # sigma is the same for all 3 cartesian dimensions
        for i in range(3):
            scale[:, i] = s1d
        ts.velocities = self._rng.normal(loc=0, scale=scale)
