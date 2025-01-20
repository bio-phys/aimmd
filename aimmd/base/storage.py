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
import asyncmd
import numpy as np
from pkg_resources import parse_version

from . import _H5PY_PATH_DICT
from .trainset import TrainSet
from ..distributed.pathmovers import ModelDependentPathMover
from .. import __version__


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
class MCstepMemory(MutableObjectShelf):
    # stores one MCstep
    # contains a sequence of trial trajectories, e.g. 2 for TwoWayShooting
    # *can* contain a path/transition
    def __init__(self, grp, modelstore, mcstep=None):
        super().__init__(group=grp)
        self._modelstore = modelstore
        if mcstep is not None:
            self.save(mcstep=mcstep)

    def save(self, mcstep):
        mover = mcstep.mover
        if isinstance(mover, ModelDependentPathMover):
            mover_modelstore = mover.modelstore
            # check if they use the same hdf5 group, the RCModelshelf objects
            # do not need to be the same (and most often often are not)
            if mover.modelstore is not None:
                # movers that have been pickled can have modelstore=None
                if mover_modelstore._group != self._modelstore._group:
                    logger.error("saving a mcstep with a 'foreign' modelstore")
            mcstep.mover.modelstore = None
        super().save(obj=mcstep, overwrite=False, buffsize=2**22)
        if isinstance(mover, ModelDependentPathMover):
            # reset the modelstore of the mc.mover in case we use it somewhere else
            mcstep.mover.modelstore = mover_modelstore

    def load(self):
        mcstep = super().load(buffsize=2**22)
        if isinstance(mcstep.mover, ModelDependentPathMover):
            mcstep.mover.modelstore = self._modelstore
        return mcstep


class MCStepCollection(collections.abc.Sequence):
    # TODO: do we want Sequence behavior for trials or for MCStates?!
    #       (and have the other available via a method/methods)
    # NOTE: we inherhit from Sequence (instead of MutableSequence) and write a
    #       custom .append method, this way we can easily make sure that trial
    #       data can not be reset
    # store a single TPS chain
    # should behave like a list of trials?!
    # and have an `accepts` and a `transitions` method?!
    def __init__(self, root_grp, modelstore):
        self._root_grp = root_grp
        self._modelstore = modelstore
        self._h5py_paths = {"MCsteps": "MCsteps",  # the actual datasets
                            "MCstates": "MCstates",  # links to the current (accepted) MC states
                            }
        self._mcsteps_grp = self._root_grp.require_group(
                                                self._h5py_paths["MCsteps"]
                                                        )
        self._mcstates_grp = self._root_grp.require_group(
                                                self._h5py_paths["MCstates"]
                                                         )

    def __len__(self):
        return len(self._mcsteps_grp.keys())

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key >= len(self):
                raise IndexError(f"Index (was {key}) must be < len(self).")
            else:
                return MCstepMemory(self._mcsteps_grp[str(key)],
                                    modelstore=self._modelstore,
                                    ).load()
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
                                         modelstore=self._modelstore,
                                         ).load()]
            return ret_val
        else:
            raise TypeError("Keys must be int or slice.")

    def append(self, mcstep):
        # add given mcstep to self, add it as active mcstate if it is accepted
        n = len(self)
        single_step_grp = self._mcsteps_grp.require_group(str(n))
        _ = MCstepMemory(single_step_grp,
                         modelstore=self._modelstore,
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
                               modelstore=self._modelstore,
                               ).load()

    def mcstate(self, idx):
        """Return the active Markov Chain state at trial with given idx."""
        if isinstance(idx, (int, np.integer)):
            if idx >= len(self._mcstates_grp.keys()):
                raise IndexError(f"No Markov Chain state with index {idx}.")
            return MCstepMemory(self._mcstates_grp[str(idx)],
                                modelstore=self._modelstore,
                                ).load()
        else:
            raise ValueError("Markov chain state index must be an integer.")


class MCStepCollectionBundle(collections.abc.Sequence):
    # class to access a number of MCStepcollections as storage attribute
    # behaves like a list of MCStepCollections
    # allows (re)setting the number of collections, only increase possible to
    # make sure no collections need to be deleted, but I (hejung) think this
    # is something we anyway only set once when we create the store at the
    # begining of a simulation
    # TODO?
    # should this behave like a MCStepCollection if there is only one stored?
    # we could get that behavior by making the sequence a storage property and
    # checking if it has len==1 before returning, but then we can not access
    # self.n_collections if len==1
    def __init__(self, group, modelstore) -> None:
        self._group = group
        self._modelstore = modelstore

    def __len__(self) -> int:
        return len(self._group.keys())

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key >= len(self):
                raise IndexError("Key must be smaller than number of "
                                 f"MCStepCollections ({len(self)}).")
            return MCStepCollection(root_grp=self._group[str(key)],
                                    modelstore=self._modelstore)
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            ret_val = []
            for idx in range(start, stop, step):
                ret_val += [MCStepCollection(root_grp=self._group[str(idx)],
                                             modelstore=self._modelstore)
                            ]
            return ret_val
        else:
            raise TypeError("Keys must be int or slice.")

    @property
    def n_collections(self):
        return len(self)

    @n_collections.setter
    def n_collections(self, value):
        # can only increase n_collections
        le = len(self)
        if value == le:
            # nothing to do
            return
        if le != 0:
            logger.info("Resetting the number of MCStepCollections for "
                        "initialized storage.")
        if value < le:
            raise ValueError("Can only increase number of collections.")
        for i in range(le, value):
            _ = self._group.create_group(str(i))

    def delete_all_collections(self):
        """
        Delete all MCStep collections and set self.n_collections to zero.

        Can be useful if we are using Brain.reinitialize_from_workdir and
        want to keep the cached CV values (and dont care for the old storage).
        The mcstep_collections new need to be initialized as usual by setting
        `self.n_collections` to the desired value.
        """
        for i in range(0, len(self)):
            del self._group[str(i)]


class ChainSamplerStore(MutableObjectShelf):
    # class to store single PathChainSampler objects
    def __init__(self, group, mcstep_collection, modelstore):
        super().__init__(group=group)
        self.mcstep_collection = mcstep_collection
        self.modelstore = modelstore

    def save(self, obj, buffsize=2 ** 22):
        # set the stuff we cannot save to None after keeping a reference
        if obj.mcstep_collection._root_grp != self.mcstep_collection._root_grp:
            logger.error("Saving a PathChainSampler associated with a "
                         "different MCStepCollection.")
        pcs_step_collection = obj.mcstep_collection
        obj.mcstep_collection = None
        pcs_modelstore = obj.modelstore  # TODO: test and warn as below?
        obj.modelstore = None
        mover_modelstores = []
        for mover in obj.movers:
            if isinstance(mover, ModelDependentPathMover):
                if mover.modelstore._group != self.modelstore._group:
                    logger.error("Saving a mover in the PathSamplingChain with"
                                 " a different modelstore.")
                mover_modelstores += [mover.modelstore]
                mover.modelstore = None
            else:
                mover_modelstores += [None]
        # NOTE: we leave step set and then just pickle the mcstep
        #       (this will ge rid of the modelstore for the movers but we dont
        #        use the movers from the steps for restarting, so this does not
        #        matter)
        # save the stripped down saveable object
        super().save(obj, overwrite=True, buffsize=buffsize)
        # and rebuild the object with the non-saveable objects
        obj.mcstep_collection = pcs_step_collection
        obj.modelstore = pcs_modelstore
        for mover, mover_ms in zip(obj.movers, mover_modelstores):
            if isinstance(mover, ModelDependentPathMover):
                mover.modelstore = mover_ms
        return

    def load(self, buffsize=2 ** 22):
        obj = super().load(buffsize)
        obj.mcstep_collection = self.mcstep_collection
        # NOTE: we load the object with the mcstep set (the mover attached to
        #       the step will not have a modelstore set anymore because it has
        #       been pickled)
        for mover in obj.movers:
            if isinstance(mover, ModelDependentPathMover):
                mover.modelstore = self.modelstore
        obj.modelstore = self.modelstore
        return obj


class ChainSamplerStoreBundle(collections.abc.Sequence):
    # helper class to store a bunch of PathChainSamplers
    def __init__(self, group, mcstep_collections, modelstore) -> None:
        self.group = group
        # mcstep_collections is a list of mcstep collections, one for each
        # PathSamplingChain in Brain (can be the same entry for all though)
        self.mcstep_collections = mcstep_collections
        self.modelstore = modelstore

    def __len__(self):
        return len(self.group.keys())

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            if key >= len(self):
                raise IndexError(f"Key must be smaller than n_chains ({len(self)}).")
            return ChainSamplerStore(group=self.group[str(key)],
                                     mcstep_collection=self.mcstep_collections[key],
                                     modelstore=self.modelstore,)
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            ret_val = []
            for idx in range(start, stop, step):
                ret_val += [ChainSamplerStore(group=self.group[str(idx)],
                                              mcstep_collection=self.mcstep_collections[idx],
                                              modelstore=self.modelstore,
                                              )
                            ]
            return ret_val
        else:
            raise TypeError("Keys must be int or slice.")

    @property
    def n_samplers(self):
        return len(self)

    @n_samplers.setter
    def n_samplers(self, value):
        # this just sets the number of groups we have for ChainSamplerStores
        # saving/loading always works by iterating over existing entries/groups
        # and calling the ChainSamplerStore save and load methods
        le = len(self)
        if value > le:
            for i in range(le, value):
                _ = self.group.create_group(str(i))
        elif value < le:
            for i in range(value, le):
                del self.group[str(i)]
        elif value == le:
            # nothing to do
            return


class BrainStore(MutableObjectShelf):
    # class to save distributed Brain objects
    def __init__(self, group, storage):
        super().__init__(group)
        self._storage = storage
        self._h5py_paths = {
            "samplers": "PSChainSamplers",
            # this is the mapping between PSChainSampler and MCStep collection
            # it is a 1d int array, each sampler has the idx of its collection
            "sampler_to_stepcollection": "PSChainSamplers_to_MCStepCollections",
            }
        self._sampler_stores_grp = self.group.require_group(
                                                self._h5py_paths["samplers"]
                                                            )

    @property
    def _sampler_to_stepcollection(self):
        try:
            sampler_to_step_dset = self.group[
                                self._h5py_paths["sampler_to_stepcollection"]
                                              ]
        except KeyError:
            raise KeyError("No brain has been stored here yet.")
        else:
            return sampler_to_step_dset[:].copy()

    @_sampler_to_stepcollection.setter
    def _sampler_to_stepcollection(self, val):
        # TODO: check val for basic sanity?
        try:
            # check if there is something
            _ = self.group[self._h5py_paths["sampler_to_stepcollection"]]
        except KeyError:
            # TODO: do we want to do anything if there is an old value? or just delete...
            pass
        else:
            del self.group[self._h5py_paths["sampler_to_stepcollection"]]
        self.group.create_dataset(name=self._h5py_paths["sampler_to_stepcollection"],
                                  data=val,
                                  dtype=np.int64)

    def save(self, obj, buffsize=2 ** 22):
        storage = obj.storage
        if storage is not self._storage:
            logger.error("Saving a brain from a different storage!")
        obj.storage = None
        model = obj.model
        obj.model = None
        tasks = obj.tasks
        obj.tasks = None
        samplers = obj.samplers
        sampler_to_stepcollection = obj.sampler_to_mcstepcollection
        mcstep_collections = [self._storage.mcstep_collections[idx]
                              for idx in sampler_to_stepcollection]
        # save the mapping for when we load the brain
        self._sampler_to_stepcollection = sampler_to_stepcollection
        sampler_stores = ChainSamplerStoreBundle(
                                        group=self._sampler_stores_grp,
                                        mcstep_collections=mcstep_collections,
                                        modelstore=self._storage.rcmodels,
                                                 )
        sampler_stores.n_samplers = len(samplers)
        for sstore, sampler in zip(sampler_stores, samplers):
            sstore.save(sampler)
        obj.samplers = None
        # save the stripped down brain
        super().save(obj, overwrite=True, buffsize=buffsize)
        # and rebuild the brain to working state
        obj.storage = storage
        obj.model = model
        obj.tasks = tasks
        obj.samplers = samplers

    def load(self, model, tasks, buffsize=2 ** 22):
        obj = super().load(buffsize)
        obj.storage = self._storage
        obj.model = model
        obj.tasks = tasks
        sampler_to_stepcollection = self._sampler_to_stepcollection
        mcstep_collections = [self._storage.mcstep_collections[idx]
                              for idx in sampler_to_stepcollection]
        sampler_stores = ChainSamplerStoreBundle(
                                        group=self._sampler_stores_grp,
                                        mcstep_collections=mcstep_collections,
                                        modelstore=self._storage.rcmodels,
                                                 )
        # TODO: reset sampler_to_mcstepcollection?!
        #       I (hejung) think we dont need to as it is saved with the brain
        #       (and therefore now restored) anyway
        obj.samplers = [ss.load() for ss in sampler_stores]
        return obj


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
    # Note that we did not increase the compatibility version for distributed,
    # because we can read the 'old' (v0.8.1) storages (distributed storage can
    # even be added to 'old' pre-distributed storage together with ops stuff,
    # except that we can only ever store one trainingset)

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
        if ("w" in mode) or ("a" in mode and not fexists):
            # first creation of file: write aimmd compatibility version string
            self._store.attrs["storage_version"] = np.string_(
                                                    self._compatibility_version
                                                              )
            self._store.attrs["aimmd_version"] = np.string_(__version__)
            # save the current (i.e. where the file is when we opened it) dirname to attrs
            # we need this to be able to save tensorflow models properly
            self._store.attrs["dirname"] = np.string_(self._dirname)
        else:
            store_version = parse_version(
                            self._store.attrs["storage_version"].decode("ASCII")
                                          )
            if parse_version(__version__) < store_version:
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
                        # but we warn because the folder with the models must be copied
                        # TODO/FIXME: automatically copy the folder from old to new location?
                        logger.warn("The directory containing the storage changed, we updated it in the storage."
                                    + "Note that currently you need to copy the KerasRCModels folder yourself.")

        rcm_grp = self.file.require_group(_H5PY_PATH_DICT["rcmodel_store"])
        self.rcmodels = RCModelRack(rcmodel_group=rcm_grp, storage_directory=self._dirname)
        self._mcstep_collections = None
        # register this storage file as trajectory value cache with asyncmd
        if mode != "r":
            # we have write intent on the file, so get/create the trajectory value cache
            self._distributed_traj_val_cache = self.file.require_group(
                                _H5PY_PATH_DICT["distributed_traj_val_cache"]
                                                                       )
            try:
                cur_cache = asyncmd.config._GLOBALS["H5PY_CACHE"]
            except KeyError:
                # no cache set, so we set this file
                asyncmd.config.register_h5py_cache(
                                h5py_group=self._distributed_traj_val_cache,
                                make_default=True,
                                                   )
            else:
                logger.warning("Resetting the asyncmd h5py trajectory value "
                               + f"cache. Was {cur_cache}.")
        else:
            # no write intent, so we warn about it
            logger.warning("Opening storage without write intent, asyncmd "
                           + "trajectory value caching will not be performed "
                           + "in h5py (but most likely as seperate npz files)."
                           )
        # empty the density collector caches (should be empty, but to be sure)
        self._empty_cache()

    def load_brain(self, model, tasks):
        brain = BrainStore(group=self.file.require_group(_H5PY_PATH_DICT["distributed_brainstore"]),
                           storage=self).load(model=model, tasks=tasks)
        return brain

    def save_brain(self, brain):
        BrainStore(group=self.file.require_group(_H5PY_PATH_DICT["distributed_brainstore"]),
                   storage=self
                   ).save(obj=brain)

    @property
    def mcstep_collections(self):
        if self._mcstep_collections is None:
            self._mcstep_collections = MCStepCollectionBundle(
                                            group=self.file.require_group(_H5PY_PATH_DICT["distributed_mcstepcollections"]),
                                            modelstore=self.rcmodels,
                                                              )
        return self._mcstep_collections

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
