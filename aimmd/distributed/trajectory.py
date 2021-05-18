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
import collections
import asyncio
import inspect
import logging
import multiprocessing
import numpy as np
import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor
from scipy import constants

from . import _SEM_MAX_PROCESS


logger = logging.getLogger(__name__)


# TODO: DOCUMENT!
# NOTE: it should be 'easy' to write and use a SlurmTrajectoryFunctionWrapper
#       that submits the function calculation as job to the qeue sys
#       we await the result anyways, so we could just submit the job and then
#       await sleep loop until it is done
class TrajectoryFunctionWrapper:
    # wrap functions for use on arcd.distributed.Trajectory
    # makes sure that we check for cached values if we apply the wrapped func
    # to an arcd.distributed.Trajectory
    def __init__(self, function):
        self.function = function

    def __repr__(self):
        return f"TrajectoryFunctionWrapper(function={self._func})"

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
            logger.warning(f"Could not retrieve source for {value}."
                           + " No caching can/will be performed.")
        else:
            self._func_src = src
        self._func = value

    async def __call__(self, value):
        if isinstance(value, Trajectory) and self._func_src is not None:
            return await value._apply_cached_func(self._func_src, self._func)
        else:
            # this will block until func is done, we could use a ProcessPool?!
            # Can we make sure that we are always able to pickle value for that?
            # (probably not since it could be Trajectory and we only have no func_src)
            return self._func(value)


# TODO: DOCUMENT
class Trajectory:
    """
    Represent a trajectory.

    Keep track of the paths of the trajectory and the topolgy files.
    Caching of values for (wrapped) functions acting on the trajectory.
    """

    def __init__(self, trajectory_file, topology_file):
        # NOTE: we assume tra = trr and top = tpr
        #       but we also expect that anything which works for mdanalysis as
        #       tra and top should also work here as tra and top
        self.trajectory_file = os.path.abspath(trajectory_file)
        self.topology_file = os.path.abspath(topology_file)
        self._len = None
        self._func_src_to_idx = {}
        self._func_values = []
        self._h5py_grp = None
        self._h5py_cache = None

    def __len__(self):
        if self._len is not None:
            return self._len
        # create/open a mdanalysis universe to get the number of frames
        u = mda.Universe(self.topology_file, self.trajectory_file)
        self._len = len(u.trajectory)
        return self._len

    def __repr__(self):
        return (f"Trajectory(trajectory_file={self.trajectory_file},"
                + f" topology_file={self.topology_file})"
                )

    async def _apply_cached_func(self, src, func):
        # TODO: for now we assume it is the right value if it is found in cache
        #       can we do anything to ensure/check that?
        #       store a hash of the file when we apply the func and check that?
        if self._h5py_grp is not None:
            # first check if we are loaded and possibly get it from there
            # trajectories are immutable once stored, so no need to check len
            try:
                return self._h5py_cache[src]
            except KeyError:
                # not in there
                # send function application to seperate process and wait for it
                loop = asyncio.get_running_loop()
                async with _SEM_MAX_PROCESS:
                    # NOTE: make sure we do not fork! (not save with multithreading)
                    # see e.g. https://stackoverflow.com/questions/46439740/safe-to-call-multiprocessing-from-a-thread-in-python
                    ctx = multiprocessing.get_context("forkserver")
                    # use one python subprocess: if func releases the GIL
                    # it does not matter anyway, if func is full py 1 is enough
                    with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                        vals = await loop.run_in_executor(pool, func, self)
                self._h5py_cache.append(src, vals)
                return vals
        # only 'local' cache, i.e. this trajectory has no file associated (yet)
        try:
            # see if it is in cache
            idx = self._func_src_to_idx[src]
            return self._func_values[idx]
        except KeyError:
            # if not calculate, store and return
            # send function application to seperate process and wait for it
            loop = asyncio.get_running_loop()
            async with _SEM_MAX_PROCESS:
                # NOTE: make sure we do not fork! (not save with multithreading)
                # see e.g. https://stackoverflow.com/questions/46439740/safe-to-call-multiprocessing-from-a-thread-in-python
                ctx = multiprocessing.get_context("forkserver")
                # use one python subprocess: if func releases the GIL
                # it does not matter anyway, if func is full py 1 is enough
                with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                    vals = await loop.run_in_executor(pool, func, self)
            self._func_src_to_idx[src] = len(self._func_src_to_idx)
            self._func_values.append(vals)
            return vals

    def __getstate__(self):
        # enable pickling of Trajecory without call to ready_for_pickle
        # this should make it possible to pass it into a ProcessPoolExecutor
        # and lets us calculate TrajectoryFunction values asyncronously
        # NOTE: this removes everything except the filepaths
        state = self.__dict__.copy()
        state["_h5py_cache"] = None
        state["_h5py_grp"] = None
        state["_func_values"] = []
        state["_func_src_to_idx"] = {}
        return state

    def object_for_pickle(self, group, overwrite):
        # NOTE: we ignore overwrite and assume the group is always empty
        #       (or at least matches this tra and we can add values?)
        # currently overwrite will always be false and we can just ignore it?!
        # and then we can/do also expect group to be empty...?
        state = self.__dict__.copy()
        if self._h5py_grp is not None:
            # we already have a file?
            # lets try to copy?
            group.copy(self._h5py_grp, group)
            state["_h5py_grp"] = None
            state["_h5py_cache"] = None
        # (re) set h5py group such that we use the cache from now on
        self._h5py_grp = group
        self._h5py_cache = TrajectoryFunctionValueCache(self._h5py_grp)
        for src, idx in self._func_src_to_idx.items():
            self._h5py_cache.append(src, self._func_values[idx])
        # clear the 'local' cache and empty state, such that we initialize
        # to empty, next time we will get it all from file directly
        self._func_values = state["_func_values"] = []
        self._func_src_to_idx = state["_func_src_to_idx"] = {}
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
        self._h5py_paths = {"srcs": "FunctionSources",
                            "vals": "FunctionValues"
                            }
        self._srcs_grp = self._root_grp.require_group(self._h5py_paths["srcs"])
        self._vals_grp = self._root_grp.require_group(self._h5py_paths["vals"])

    def __len__(self):
        return len(self._srcs_grp.keys())

    def __iter__(self):
        for idx in range(len(self)):
            yield self._srcs_grp[str(idx)].asstr()[()]

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Keys must be of type str.")
        for idx, k_val in enumerate(self):
            if key == k_val:
                return self._vals_grp[str(idx)][:]
        # if we got until here the key is not in there
        raise KeyError("Key not found.")

    def append(self, src, vals):
        if not isinstance(src, str):
            raise TypeError("Keys (src) must be of type str.")
        if src in self:
            raise ValueError("There are already values stored for src."
                             + " Changing the stored values is not supported.")
        # TODO: do we also want to check vals for type?
        name = str(len(self))
        _ = self._srcs_grp.create_dataset(name, data=src)
        _ = self._vals_grp.create_dataset(name, data=vals)


class TrajectoryConcatenator:
    """
    Create concatenated trajectory from given trajectories and frames.

    The concatenate method takes a list of trajectories and a list of slices,
    returns one trajectory containing only the selected frames in that order.
    Velocities are automatically inverted if the step of a slice is negative,
    this can be controlled via the invert_v_for_negative_step attribute.

    NOTE: We assume that all trajs have the same topolgy
          and attach the the topolgy of the first traj if not told otherwise.
    """

    def __init__(self, invert_v_for_negative_step=True):
        self.invert_v_for_negative_step = invert_v_for_negative_step

    def concatenate(self, trajs, slices, tra_out, top_out=None,
                    overwrite=False):
        """
        Create concatenated trajectory from given trajectories and frames.

        trajs - list of `:class:`Trajectory
        slices - list of (start, stop, step)
        tra_out - output trajectory filepath, absolute or relativ to cwd
        top_out - None or output topology filepath, if None we will take the
                  topology file of the first trajectory in trajs
        overwrite - bool (default=False), if True overwrite existing tra_out,
                    if False and the file exists raise an error
        """
        tra_out = os.path.abspath(tra_out)
        if os.path.exists(tra_out) and not overwrite:
            raise ValueError(f"overwrite=False and tra_out exists: {tra_out}")
        top_out = (trajs[0].topology_file if top_out is None
                   else os.path.abspath(top_out))
        if not os.path.isfile(top_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise ValueError(f"Output topolgy file must exist ({top_out}).")

        # special treatment for traj0 because we need n_atoms for the writer
        u0 = mda.Universe(trajs[0].topology_file, trajs[0].trajectory_file)
        start0, stop0, step0 = slices[0]
        # if the file exists MDAnalysis will silently overwrite
        with mda.Writer(tra_out, n_atoms=u0.trajectory.n_atoms) as W:
            for ts in u0.trajectory[start0:stop0:step0]:
                if self.invert_v_for_negative_step and step0 < 0:
                    u0.atoms.velocities *= -1
                W.write(u0.atoms)
            del u0  # should free up memory and does no harm?!
            for traj, sl in zip(trajs[1:], slices[1:]):
                u = mda.Universe(traj.topology_file, traj.trajectory_file)
                start, stop, step = sl
                for ts in u.trajectory[start:stop:step]:
                    if self.invert_v_for_negative_step and step < 0:
                        u.atoms.velocities *= -1
                    W.write(u.atoms)
                del u
        # return (file paths to) the finished trajectory
        return Trajectory(tra_out, top_out)


class FrameExtractor(abc.ABC):
    # extract a single frame with given idx from a trajectory and write it out
    # simplest case is without modification, but useful modifications are e.g.
    # with inverted velocities, with random Maxwell-Boltzmann velocities, etc.

    @abc.abstractmethod
    def apply_modification(self, universe):
        # this func will is called when the current timestep is at the choosen
        # frame and applies the subclass specific frame modifications to the
        # mdanalysis universe, after this function finishes the frames is
        # written out, i.e. with potential modifications applied
        # no return value is expected or considered,
        # the modifications in the universe are nonlocal anyway
        raise NotImplementedError

    def extract(self, outfile, traj_in, idx, top_out=None, overwrite=False):
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
        top_out - None or output topology filepath, if None we will take the
                  topology file of the input trajectory
        overwrite - bool (default=False), if True overwrite existing tra_out,
                    if False and the file exists raise an error
        """
        outfile = os.path.abspath(outfile)
        if os.path.exists(outfile) and not overwrite:
            raise ValueError(f"overwrite=False and outfile exists: {outfile}")
        top_out = (traj_in.topology_file if top_out is None
                   else os.path.abspath(top_out))
        if not os.path.isfile(top_out):
            # although we would expect that it exists if it comes from an
            # existing traj, we still check to catch other unrelated issues :)
            raise ValueError(f"Output topolgy file must exist ({top_out}).")
        u = mda.Universe(traj_in.topology_file, traj_in.trajectory_file)
        with mda.Writer(outfile, n_atoms=u.trajectory.n_atoms) as W:
            ts = u.trajectory[idx]
            self.apply_modification(u, ts)
            W.write(u.atoms)
        return Trajectory(trajectory_file=outfile, topology_file=top_out)


class NoModificationFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out without modification."""

    def apply_modification(self, universe):
        pass


class InvertedVelocitiesFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out with inverted velocities."""

    def apply_modification(self, universe, ts):
        ts.velocities *= -1


class RandomVelocitiesFrameExtractor(FrameExtractor):
    """Extract a frame from a trajectory, write it out with randomized velocities."""

    def __init__(self, T):
        """Temperature T must be given in degree Kelvin."""
        self.T = T  # in K
        self._rng = np.random.default_rng()

    def apply_modification(self, universe, ts):
        # MDAnalysis uses kJ/mol as energy unit,
        # so we use kB * NA * 10**(-3) to get kB in kJ/(mol * K)
        scale = np.empty((ts.n_atoms, 3), dtype=np.float64)
        s1d = np.sqrt((self.T*constants.k*constants.N_A*10**(-3))
                      / universe.atoms.masses
                      )
        # sigma is the same for all 3 cartesian dimensions
        for i in range(3):
            scale[:, i] = s1d
        ts.velocities = self._rng.normal(loc=0, scale=scale)
