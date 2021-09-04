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
import logging
import os
import io
import pickle
import collections.abc
import h5py
import numpy as np
from pkg_resources import parse_version

from . import _H5PY_PATH_DICT
from .trainset import TrainSet
from ..distributed.pathmovers import ModelDependentPathMover
from .. import __about__


logger = logging.getLogger(__name__)


class BytesStreamtoH5py:
    """
    'Translate' from python bytes objects to arrays of uint8.

    Implement (as required by pickle):
        .write(bytes object) -> int:len(bytes object)
    NOTE: TRUNCATES the dataset to zero length prior to writing.
    """

    def __init__(self, dataset):
        """
        Initialize BytesStreamtoH5py file-like object.

        Parameters:
        -----------
              dataset - an existing 1d h5py datset with dtype=uint8 and
                        maxshape=(None,)
                        ProTip: Can be anything 1d supporting
                        .resize(shape=(new_len,)) and sliced access
        """
        self.dataset = dataset
        self.dataset.resize((0,))

    # make possible to use in with statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # do not catch any exceptions
        pass

    def flush(self):
        # no write buffer, so nothing to do
        pass

    def write(self, byts):
        old_len = len(self.dataset)
        add_len = len(byts)
        self.dataset.resize((old_len+add_len,))
        self.dataset[old_len:old_len+add_len] = np.frombuffer(byts,
                                                              dtype=np.uint8)
        return add_len


# buffered version, seems to be a bit faster(?)
class BytesStreamtoH5pyBuffered:
    """
    'Translate' from python bytes objects to arrays of uint8. Buffered Version.

    Implement (as required e.g. by pickle):
        .write(bytes object) -> int:len(bytes object)
    NOTE: TRUNCATES the dataset to zero length prior to writing.
    """

    def __init__(self, dataset, buffsize=2**29):
        """
        Initialize BytesStreamtoH5pyBuffered file-like object.

        Parameters:
        -----------
              dataset - an existing 1d h5py datset with dtype=uint8 and
                        maxshape=(None,)
                        ProTip: Can be anything 1d supporting
                        .resize(shape=(new_len,)) and sliced access
              buffsize - int, size of internal buffer array measured in bytes,
                         i.e. number of np.uint8,
                         default 2**30 is approximately 1 GiB, see below
                            buffsize = 2**17  # ~= 130 KiB
                            buffsize = 2**27  # ~= 134 MiB
                            buffsize = 2**29  # ~= 530 MiB
                            buffsize = 2**30  # ~= 1GiB
        """
        self.dataset = dataset
        self.dataset.resize((0,))
        self.buffsize = buffsize
        self._buff = np.zeros((self.buffsize,), dtype=np.uint8)
        self._buffpointer = 0

    # make possible to use in with statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # always write buffer to file
        self.close()

    def flush(self):
        # flush write buffers
        self._buffer_to_dset()

    def _buffer_to_dset(self):
        if self._buffpointer > 0:
            # write buffer to file
            old_len = len(self.dataset)
            self.dataset.resize((old_len + self._buffpointer,))
            self.dataset[old_len:old_len + self._buffpointer
                         ] = self._buff[:self._buffpointer]
            self._buffpointer = 0
            # I think we can just overwrite the existing buffer,
            # no need to recreate, just set self._buffpointer to zero
            # self._buff = np.zeros((self.buffsize,), dtype=np.uint8)

    def close(self):
        self._buffer_to_dset()

    def write(self, byts):
        add_len = len(byts)
        if self.buffsize - self._buffpointer >= add_len:
            # fits in buffer -> write to buffer
            self._buff[self._buffpointer:self._buffpointer + add_len
                       ] = np.frombuffer(byts, dtype=np.uint8)
            self._buffpointer += add_len
        else:
            # first write out buffer
            self._buffer_to_dset()
            remains = add_len
            written = 0
            while remains > self.buffsize:
                # write a whole buffer directly to file
                self._buff[:] = np.frombuffer(byts[written:written+self.buffsize],
                                              dtype=np.uint8,
                                              )
                self._buffpointer += self.buffsize
                written += self.buffsize
                remains -= self.buffsize
                self._buffer_to_dset()
            # now what remains should fit into the buffer
            self._buff[:remains] = np.frombuffer(byts[-remains:],
                                                 dtype=np.uint8,
                                                 )
            self._buffpointer += remains

        return add_len


class H5pytoBytesStream:
    """
    'Translate' from arrays of uint8s to python bytes objects.

    Implement (as required by pickle):
        .read(size) -> bytes object with len size
        .readline() -> bytes object with rest of current line
    """

    def __init__(self, dataset, buffsize=2**29):
        """
        Initialize H5pytoBytesStream.

        Parameters:
        -----------
        dataset - existing 1d h5py datset with dtype=uint8 and maxshape=(None,)
                  Tip: Can be anything 1d supporting .resize(shape=(new_len,))
                       and sliced access
        buffsize - int, maximum buffer size/approximate memory footprint
                   measured in bytes, the maximum size of the internal
                   reading cache is (buffsize,) and dtype=uint8 resulting in
                            buffsize = 2**17  # ~= 130 KiB
                            buffsize = 2**27  # ~= 134 MiB
                            buffsize = 2**29  # ~= 530 MiB
                            buffsize = 2**30  # ~= 1GiB

        """
        self.buffsize = buffsize
        self.dataset = dataset
        self._readpointer = 0  # points to where we are in total file
        self._dset_len = len(dataset)
        self._datapointer = 0  # points to where we are in current chunk
        self._fill_data()

    # make possible to use in with statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # do not catch any exceptions
        pass

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == 0:
            # start of the stream (the default);
            # offset should be zero or positive
            self._readpointer = offset
        elif whence == 1:
            # current stream position; offset may be negative
            self._readpointer += offset
        elif whence == 2:
            # end of the stream; offset is usually negative
            self._readpointer = self._dset_len + offset
        else:
            raise ValueError("whence must be 0, 1 or 2.")
        # now fill the buffer
        self._fill_data()

    def tell(self):
        return self._readpointer

    def _fill_data(self):
        if self._dset_len - self._readpointer <= self.buffsize:
            self._data = self.dataset[self._readpointer:]
            self._last_chunk = True
        else:
            self._data = self.dataset[self._readpointer:
                                      self._readpointer + self.buffsize]
            self._last_chunk = False
        # find newlines
        # bytes(b'\n') <-> uint8(10)
        self._line_breaks = np.where(self._data == 10)[0]
        self._datapointer = 0

    def read(self, size=-1):
        # EOF reached
        if self._readpointer > self._dset_len:
            return bytes()
        # the default python file behaviour:
        # if size is None or negative return the rest of the file
        elif (size is None) or (size < 0):
            readstart = self._readpointer
            self._readpointer = self._dset_len
            return self.dataset[readstart:].tobytes()
        # what we do if size given
        else:
            happy = False  # indicating if requested size of data read
            eof = False
            parts = []
            # how many entries we still need to read
            missing = size
            while not happy and not eof:
                readstart = self._datapointer
                if self.buffsize - self._datapointer >= missing:
                    # can satisfy from current chunk
                    self._datapointer += missing
                    self._readpointer += missing
                    parts.append(self._data[readstart:self._datapointer].tobytes())
                    happy = True
                    break
                else:
                    # append rest of current chunk and get new one
                    parts.append(self._data[readstart:].tobytes())
                    # we appended buffsize - current_pointer values
                    missing -= self.buffsize - self._datapointer
                    self._readpointer += self.buffsize - self._datapointer
                    # do not need that, will be zeroed in _fill_data
                    # self._datapointer += self.buffsize - self._datapointer
                    if self._last_chunk:
                        # there is no data left
                        eof = True
                        break
                    else:
                        self._fill_data()
            return bytes().join(parts)

    def readline(self):
        # EOF reached
        if self._readpointer > self._dset_len:
            return bytes()
        newline = False
        eof = False
        parts = []
        while not newline and not eof:
            try:
                next_newline_idx = np.where(self._line_breaks >= self._datapointer)[0][0]
            except IndexError:
                # IndexError means that there is no linebreak left (in chunk)
                # -> append rest of current chunk
                parts.append(self._data[self._datapointer:].tobytes())
                self._readpointer += self.buffsize - self._datapointer
                if self._last_chunk:
                    eof = True
                    break
                else:
                    self._fill_data()
            else:
                # newline found -> append until including newline
                # missing number of elements in new chunk until including newline
                missing = self._line_breaks[next_newline_idx] + 1 - self._datapointer
                parts.append(self._data[self._datapointer:
                                        self._datapointer+missing].tobytes())
                self._datapointer += missing
                self._readpointer += missing
                newline = True
                break

        return bytes().join(parts)


class MutableObjectShelf:
    """
    Interface between a h5py group and pythons pickle.

    Can be used to store arbitrary python objects to h5py files.
    """

    def __init__(self, group):
        self.group = group

    def load(self, buffsize=2**20):
        try:
            dset = self.group['pickle_data']
        except KeyError:
            raise KeyError('No object stored yet.')
        if buffsize is not None:
            with H5pytoBytesStream(dset, buffsize=buffsize) as stream_file:
                obj = pickle.load(stream_file)
        else:
            with H5pytoBytesStream(dset) as stream_file:
                obj = pickle.load(stream_file)
        return obj

    def save(self, obj, overwrite=True, buffsize=2**20):
        exists = True
        try:
            dset = self.group['pickle_data']
        except KeyError:
            # if it does not exist, we get a KeyError and then create it
            # this is more save and more specific then catching the
            # RuntimeError occuring when trying to create an existing dset,
            # since the RuntimeError can mean a lot of stuff
            exists = False
            dset = self.group.create_dataset('pickle_data',
                                             dtype=np.uint8,
                                             maxshape=(None,),
                                             shape=(0,),
                                             chunks=True,
                                             )
        if exists:
            if not overwrite:
                raise RuntimeError('Object exists but overwrite=False.')
            # TODO?: if it exists we assume that it is a dset of the right
            # TODO?: dtype, shape and maxshape. should we check?
        if buffsize is not None:
            with BytesStreamtoH5pyBuffered(dset, buffsize) as stream_file:
                # using pickle protocol 4 means python>=3.4!
                pickle.dump(obj, stream_file, protocol=4)
        else:
            with BytesStreamtoH5py(dset) as stream_file:
                # using pickle protocol 4 means python>=3.4!
                pickle.dump(obj, stream_file, protocol=4)


class AimmdObjectShelf(MutableObjectShelf):
    """
    Specialized MutableObjectShelf for aimmd objects.

    Stores any object with a .from_h5py_group and a .ready_for_pickle method.
    """

    def load(self, buffsize=2**20):
        obj = super().load(buffsize=buffsize)
        obj = obj.complete_from_h5py_group(self.group)
        return obj

    def save(self, obj, overwrite=True, buffsize=2**20, **kwargs):
        # kwargs make it possible to pass aimmd object specific keyword args
        # to the object_for_pickle functions
        obj_to_save = obj.object_for_pickle(self.group,
                                            overwrite=overwrite,
                                            **kwargs)
        super().save(obj=obj_to_save, overwrite=overwrite, buffsize=buffsize)


class RCModelRack(collections.abc.MutableMapping):
    """Dictionary like interface to RCModels stored in an aimmd storage file."""

    def __init__(self, rcmodel_group, storage_directory):
        self._group = rcmodel_group
        # should be an abspath!
        self._storage_directory = storage_directory

    def __getitem__(self, key):
        return AimmdObjectShelf(self._group[key]).load(buffsize=2**25)

    def __setitem__(self, key, value):
        try:
            # make sure the RCmodel group is empty, we overwrite anyway
            # but this avoids issues, e.g. if the density collector tries to
            # write to an old and existing group
            del self._group[key]
        except KeyError:
            pass
        group = self._group.require_group(key)
        AimmdObjectShelf(group).save(obj=value, overwrite=True, buffsize=2**25,
                                     name=key, storage_directory=self._storage_directory,
                                     )

    def __delitem__(self, key):
        del self._group[key]

    def __len__(self):
        return len(self._group.keys())

    def __iter__(self):
        return iter(self._group.keys())


# TODO: DOCUMENT!!!
# distributed TPS storage
class MCstepMemory(collections.abc.Sequence):
    # NOTE: we inherhit from Sequence (instead of MutableSequence) and write a custom .append method
    #       this way we can easily make sure that trial data can not be reset
    # TODO: should we use a `abc.collections.Collection` instead
    #       (to reflect that trials are not necessarily ordered?)
    # stores one MCstep
    # contains a sequence of trial trajectories (2 for TwoWayShooting)
    # *can* contain a path/transition
    def __init__(self, root_grp, modelstore, mcstep=None):
        self._root_grp = root_grp
        self._modelstore = modelstore
        self._h5py_paths = {"path": "path",
                            "trials": "trial_trajectories",
                            "py_data": "obj_shelf_pickle_data",
                            }
        self._tras_grp = self._root_grp.require_group(
                                            self._h5py_paths["trials"]
                                                      )
        if mcstep is not None:
            self.save(mcstep=mcstep)

    def save(self, mcstep):
        if len(self) != 0:
            raise RuntimeError("Can not modify/overwrite saved MCsteps.")
        trajs = mcstep.trial_trajectories
        path = mcstep.path
        mover = mcstep.mover
        for tra in trajs:
            self._append(tra)
        if path is not None:
            self.path = path
        py_grp = self._root_grp.require_group(self._h5py_paths["py_data"])
        if isinstance(mover, ModelDependentPathMover):
            mover_modelstore = mover.modelstore
            # check if they use the same hdf5 group, the RCModelshelf objects
            # do not need to be the same (and most often often are not)
            if mover_modelstore._group != self._modelstore._group:
                logger.error("saving a mcstep with a 'foreign' modelstore")
            mcstep.mover.modelstore = None
        MutableObjectShelf(py_grp).save(obj=mcstep, overwrite=True)
        if isinstance(mover, ModelDependentPathMover):
            # reset the modelstore of the mc.mover in case we use it somewhere else
            mcstep.mover.modelstore = mover_modelstore

    def load(self):
        try:
            py_grp = self._root_grp[self._h5py_paths["py_data"]]
        except KeyError:
            # we get a KeyError if there is no transition
            return None
        else:
            mcstep = MutableObjectShelf(py_grp).load()
        mcstep.trial_trajectories = [t for t in self]
        mcstep.path = self.path
        if isinstance(mcstep.mover, ModelDependentPathMover):
            mcstep.mover.modelstore = self._modelstore
        return mcstep

    @property
    def path(self):
        try:
            tp_grp = self._root_grp[self._h5py_paths["path"]]
        except KeyError:
            # we get a KeyError if there is no transition
            return None
        else:
            return AimmdObjectShelf(tp_grp).load()

    @path.setter
    def path(self, val):
        # TODO: check that value is of type Trajectory?
        if self.path is not None:
            # make sure there is None (yet)
            raise ValueError("Can only set the transition once,"
                             + " i.e. if it has no value.")
        else:
            tp_grp = self._root_grp.require_group(self._h5py_paths["path"])
            # the group should be empty, so lets fails if it is not
            AimmdObjectShelf(tp_grp).save(obj=val, overwrite=False)

    def __len__(self):
        # return number of trial trajectories as length
        return len(self._tras_grp.keys())

    def __getitem__(self, key):
        if isinstance(key, (int, np.int)):
            if key >= len(self):
                raise IndexError(f"Index (was {key}) must be <= len(self).")
            else:
                return AimmdObjectShelf(self._tras_grp[str(key)]).load()
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            ret_val = []
            for idx in range(start, stop, step):
                ret_val += [AimmdObjectShelf(self._tras_grp[str(idx)]).load()]
            return ret_val
        else:
            raise TypeError("Keys must be int or slice.")

    def _append(self, value):
        # append a trajectory to the trajectories associated with this trial
        # TODO: check that value is of type Trajectory?
        single_tra_grp = self._tras_grp.require_group(str(len(self)))
        # group should be empty so overwrite=False to fail if its not
        AimmdObjectShelf(single_tra_grp).save(obj=value, overwrite=False)


class ChainMemory(collections.abc.Sequence):
    # TODO: do we want Sequence behaivour for trials or for MCStates?!
    #       (and have the other available via a method/methods)
    # NOTE: we inherhit from Sequence (instead of MutableSequence) and write a custom .append method
    #       this way we can easily make sure that trial data can not be reset
    # store a single TPS chain
    # should behave like a list of trials?!
    # and have an `accepts` and a `transitions` method?!
    def __init__(self, root_grp, chain_idx, storage):
        self._root_grp = root_grp
        self._chain_idx = chain_idx
        self._storage = storage
        self._h5py_paths = {"MCsteps": "MCsteps",  # the actual datasets
                            "MCstates": "MCstates",  # links to the current (accepted) MC states
                            "modelstore": "RCmodels",  # models used in this chain
                            }
        self._mcsteps_grp = self._root_grp.require_group(
                                                self._h5py_paths["MCsteps"]
                                                        )
        self._mcstates_grp = self._root_grp.require_group(
                                                self._h5py_paths["MCstates"]
                                                         )
        self._models_grp = self._root_grp.require_group(
                                                self._h5py_paths["modelstore"]
                                                       )
        keras_model_store_dir = os.path.join(self._storage._dirname,
                                             (f"{self._root_grp.file.filename}"
                                              + f"_chain{self._chain_idx}"
                                              + "_KerasModelsSaveFiles")
                                             )
        self.modelstore = RCModelRack(rcmodel_group=self._models_grp,
                                      storage_directory=keras_model_store_dir)

    def __len__(self):
        return len(self._mcsteps_grp.keys())

    def __getitem__(self, key):
        if isinstance(key, (int, np.int)):
            if key >= len(self):
                raise IndexError(f"Index (was {key}) must be < len(self).")
            else:
                return MCstepMemory(self._mcsteps_grp[str(key)],
                                    modelstore=self.modelstore).load()
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            # TODO: do we want the generator (i.e. yield)?
            # (the list could get quite big if we have many trials...?)
            # TODO/FIXME: using a generator does not work!!
            # (we would need to use yield also to return single mcsteps
            # and get them with .__next__() or using `for s in steps[0]`
            # which is both strange...we can only write a custom Iterator to
            # return for the slices...
            ret_val = []
            for idx in range(start, stop, step):
                #yield MCstepMemory(self._mcsteps_grp[str(idx)],
                #                   modelstore=self.modelstore).load()
                ret_val += [MCstepMemory(self._mcsteps_grp[str(idx)],
                                         modelstore=self.modelstore).load()]
            return ret_val
        else:
            raise TypeError("Keys must be int or slice.")

    def append(self, mcstep):
        # create a new TPS trial and fill it with the given trajectories + transition
        n = len(self)
        single_step_grp = self._mcsteps_grp.require_group(str(n))
        _ = MCstepMemory(single_step_grp,
                         modelstore=self.modelstore,
                         mcstep=mcstep)
        if mcstep.accepted:
            # add it as active mcstate too
            self._mcstates_grp[str(n)] = single_step_grp
        else:
            # add the previous mcstate as active state again
            self._mcstates_grp[str(n)] = self._mcstates_grp[str(n - 1)]

    def mcstates(self):
        """Return a generator over all active Markov Chain states."""
        for idx in range(len(self._mcstates_grp.keys())):
            yield MCstepMemory(self._mcstates_grp[str(idx)],
                               modelstore=self.modelstore).load()

    def mcstate(self, idx):
        """Return the active Markov Chain state at trial with given idx."""
        if isinstance(idx, (int, np.int)):
            if idx >= len(self._mcstates_grp.keys()):
                raise IndexError(f"No Markov Chain state with index {idx}.")
            return MCstepMemory(self._mcstates_grp[str(idx)],
                                modelstore=self.modelstore).load()
        else:
            raise ValueError("Markov chain state index must be an integer.")


class CentralMemory(collections.abc.Sequence):
    # store N (T)PS chains
    # should behave like a list of chains?!
    def __init__(self, root_grp, storage):
        self._root_grp = root_grp
        self._storage = storage
        self._h5py_paths = {"chains": "PSchains"}
        self._chains_grp = self._root_grp.require_group(
                                                self._h5py_paths["chains"]
                                                        )

    @property
    def n_chains(self):
        return len(self)

    @n_chains.setter
    def n_chains(self, value):
        # can only increase n_chains
        le = len(self)
        if value == le:
            # nothing to do
            return
        if le != 0:
            logger.info("Resetting the number of chains for initialized storage.")
        if value < le:
            raise ValueError("Can only increase number of chains.")
        for i in range(le, value):
            _ = self._chains_grp.create_group(str(i))

    def __len__(self):
        return len(self._chains_grp.keys())

    def __getitem__(self, key):
        if isinstance(key, (int, np.int)):
            if key >= len(self):
                raise IndexError(f"Key must be smaller than n_chains ({len(self)}).")
            return ChainMemory(root_grp=self._chains_grp[str(key)],
                               chain_idx=key, storage=self._storage)
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            ret_val = []
            for idx in range(start, stop, step):
                ret_val += [ChainMemory(root_grp=self._chains_grp[str(idx)],
                                        chain_idx=idx, storage=self._storage)
                            ]
            return ret_val
        else:
            raise TypeError("Keys must be int or slice.")


class Storage:
    """
    Store all aimmd RCModels and data belonging to one TPS simulation.

    Note: Everything belonging to aimmd is stored in the aimmd_data HDF5 group.
          You can store arbitrary data using h5py in the rest of the file
          through accessing it as Storage.file.
    """

    # NOTE: update this below if we introduce breaking API changes!
    # if the current aimmd version is higher than the compatibility_version
    # we expect to be able to read the storage
    # if the storages version is smaller than the current compatibility version
    # we expect to NOT be able to read the storage
    ## introduced h5py storage
    #_compatibility_version = parse_version("0.7")
    ## removed checkpoints, changed the way we store pytorch models
    #_compatibility_version = parse_version("0.8")
    ## renamed arcd -> aimmd
    _compatibility_version = parse_version("0.8.1")

    # TODO: should we require descriptor_dim as input on creation?
    #       to make clear that you can not change it?!
    def __init__(self, fname, mode='a'):
        """
        Initialize (open/create) a Storage.

        Parameters
        ----------
        fname - bytes or str, name of file
        mode - str, mode in which to open file, one of:
            r       : readonly, file must exist
            r+      : read/write, file must exist
            w       : create file, truncate if exists
            w- or x : create file, fail if exists
            a       : read/write if exists, create otherwise (default)

        """
        fexists = os.path.exists(fname)
        self.file = h5py.File(fname, mode=mode)
        self._store = self.file.require_group(_H5PY_PATH_DICT["level0"])
        self._dirname = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), fname)))
        self._central_memory = None
        if ("w" in mode) or ("a" in mode and not fexists):
            # first creation of file: write aimmd compatibility version string
            self._store.attrs["storage_version"] = np.string_(
                                                    self._compatibility_version
                                                              )
            self._store.attrs["aimmd_version"] = np.string_(
                                                    __about__.__version__
                                                           )
            # save the current (i.e. where the file is when we opened it) dirname to attrs
            # we need this to be able to save tensorflow models properly
            self._store.attrs["dirname"] = np.string_(self._dirname)
        else:
            store_version = parse_version(
                            self._store.attrs["storage_version"].decode("ASCII")
                                          )
            if parse_version(__about__.base_version) < store_version:
                raise RuntimeError(
                        "The storage file was written with a newer version of "
                        + "aimmd than the current one. You need at least aimmd "
                        + f"v{str(store_version)}" + " to open it.")
            elif self._compatibility_version > store_version:
                raise RuntimeError(
                        "The storage file was written with an older version of"
                        + " aimmd than the current one. Try installing aimmd "
                        + f"v{str(store_version)}" + " to open it.")
            try:
                stored_dirname = self._store.attrs["dirname"].decode("ASCII")
            except KeyError:
                # make it possible to load 'old' storages without dirname attr set,
                # i.e. try to not break backwards compatibility
                if mode != "r":
                    # storage open with write intent, so just add the attr for dirname
                    self._store.attrs["dirname"] = np.string_(self._dirname)
                    logger.debug("Converted 'old' storage to 'new' format by adding "
                                 + "the 'dirname' attr.")
                else:
                    logger.info("Opened an 'old' storage wihout dirname attribute without "
                                + "write intent. Everything should work as expected, however "
                                + "if you open the storage with write intent it will be "
                                + "converted to the 'new' storage format.")
            else:
                # check if the stored dirname is the same as the current
                # also (possibly) update the stored dirname
                if self._dirname != stored_dirname:
                    # check if dirname changed, i.e. if the file was copied/moved
                    if mode == "r":
                        # no write intent, just check if dirname changed and warn if it did
                        logger.error("The directory containing the storage changed, but we have "
                                     + "no write intent on file, so we can not update it. "
                                     + "KerasRCModel saving might/will not work as expected "
                                     + "if you did not copy the KerasRCmodel directory yourself.")
                    else:
                        # we can just change the path
                        self._store.attrs["dirname"] = np.string_(self._dirname)
                        # but we warn becasue the folder with the models must be copied
                        # TODO/FIXME: automatically copy the folder from old to new location?
                        logger.warn("The directory containing the storage changed, we updated it in the storage."
                                    + "Note that currently you need to copy the KerasRCModels folder yourself.")

        rcm_grp = self.file.require_group(_H5PY_PATH_DICT["rcmodel_store"])
        self.rcmodels = RCModelRack(rcmodel_group=rcm_grp, storage_directory=self._dirname)
        self._empty_cache()  # should be empty, but to be sure

    @property
    def central_memory(self):
        # check if it is there, if yes return
        # (also set our private reference, such that we dont check every time)
        if self._central_memory is not None:
            return self._central_memory
        try:
            cm_grp = self.file[_H5PY_PATH_DICT["distributed_cm"]]
        except KeyError:
            return None
        else:
            self._central_memory = CentralMemory(root_grp=cm_grp, storage=self)
            return self._central_memory

    def initialize_central_memory(self, n_chains):
        """Initialize central_memory for distributed TPS with n_chains."""
        if self.central_memory is not None:
            raise ValueError("CentralMemory already initialized")
        cm_grp = self.file.require_group(_H5PY_PATH_DICT["distributed_cm"])
        self._central_memory = CentralMemory(root_grp=cm_grp, storage=self)
        self.central_memory.n_chains = n_chains

    # make possible to use in with statements
    def __enter__(self):
        return self

    # and automagically close when exiting the with
    def __exit__(self, exeception_type, exception_value, exception_traceback):
        self.close()

    def close(self):
        self._empty_cache()
        self.file.flush()
        self.file.close()

    def _empty_cache(self):
        # empty TrajectoryDensityCollector_cache
        traDC_cache_grp = self.file.require_group(
                                name=_H5PY_PATH_DICT["tra_dc_cache"]
                                                  )
        traDC_cache_grp.clear()

    # TODO  do we even want this implementation?
    #       or should we change to the aimmd-object shelf logic?
    #       having only one trainset makes sense, but feels like a limitation
    #       compared to the TrajectoryDensityCollectors...?
    def save_trainset(self, trainset):
        """
        Save an aimmd.TrainSet.

        There can only be one Trainset in the Storage at a time but you can
        overwrite it as often as you want with TrainSets of different length.
        That is you can not change the number of states or the descriptors
        second axis, i.e. the dimensionality of the descriptor space.
        """
        d_shape = trainset.descriptors.shape
        sr_shape = trainset.shot_results.shape
        w_shape = trainset.weights.shape
        try:
            ts_group = self.file[_H5PY_PATH_DICT["trainset_store"]]
        except KeyError:
            # we never stored a TrainSet here before, so setup datasets
            ts_group = self.file.create_group(_H5PY_PATH_DICT["trainset_store"])
            des_group = ts_group.create_dataset(name='descriptors',
                                                dtype=trainset.descriptors.dtype,
                                                shape=d_shape,
                                                maxshape=(None, d_shape[1]),
                                                )
            sr_group = ts_group.create_dataset(name='shot_results',
                                               dtype=trainset.shot_results.dtype,
                                               shape=sr_shape,
                                               maxshape=(None, sr_shape[1]),
                                               )
            w_group = ts_group.create_dataset(name='weights',
                                              dtype=trainset.weights.dtype,
                                              shape=w_shape,
                                              maxshape=(None,),
                                              )
        else:
            # get existing datsets (TODO: do we need any sanity-checks?)
            des_group = ts_group['descriptors']
            sr_group = ts_group['shot_results']
            w_group = ts_group['weights']
            # resize for current trainset
            des_group.resize(d_shape)
            sr_group.resize(sr_shape)
            w_group.resize(w_shape)
        # now store
        des_group[:] = trainset.descriptors
        sr_group[:] = trainset.shot_results
        w_group[:] = trainset.weights
        py_state = {"n_states": trainset.n_states}
        # now store them
        MutableObjectShelf(ts_group).save(py_state, overwrite=True)

    def load_trainset(self):
        """Load an aimmd.TrainSet."""
        try:
            ts_group = self.file[_H5PY_PATH_DICT["trainset_store"]]
        except KeyError:
            raise KeyError('No TrainSet in file.')
        descriptors = ts_group['descriptors'][:]
        shot_results = ts_group['shot_results'][:]
        weights = ts_group['weights'][:]
        # try to load descriptor_transform and states
        py_state = MutableObjectShelf(ts_group).load()
        return TrainSet(py_state["n_states"],
                        descriptors=descriptors,
                        shot_results=shot_results,
                        weights=weights,
                        )
