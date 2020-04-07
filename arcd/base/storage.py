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
import logging
import os
import pickle
import collections.abc
import h5py
import numpy as np
from pkg_resources import parse_version
from openpathsampling import CollectiveVariable as OPSCollectiveVariable
from openpathsampling import Volume as OPSVolume
from .trainset import TrainSet


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

    Implement (as required by pickle):
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

    def write(self, byts):
        add_len = len(byts)
        if self.buffsize - self._buffpointer >= add_len:
            # fits in buffer -> write to buffer
            self._buff[self._buffpointer:self._buffpointer + add_len
                       ] = np.frombuffer(byts, dtype=np.uint8)
            self._buffpointer += add_len
        else:
            # write buffer to file
            old_len = len(self.dataset)
            self.dataset.resize((old_len + self._buffpointer,))
            self.dataset[old_len:old_len + self._buffpointer
                         ] = self._buff[:self._buffpointer]
            # I think we can just overwrite the existing buffer,
            # no need to recreate, just set self._buffpointer to zero
            # self._buff = np.zeros((self.buffsize,), dtype=np.uint8)
            self._buffpointer = 0
            # write data either
            if add_len < self.buffsize/2:
                # to new buffer if 'small enough'
                self._buff[:add_len] = np.frombuffer(byts, dtype=np.uint8)
                self._buffpointer += add_len
            else:
                # TODO/FIXME: I think this can go way beyond buffsize?!
                # TODO/FIXME: since we make an array out of it
                # or directly to file if 'too big'
                self.dataset.resize((old_len + self._buffpointer + add_len,))
                self.dataset[old_len + self._buffpointer:
                             ] = np.frombuffer(byts, dtype=np.uint8)

        return add_len

    def close(self):
        # write buffer to file
        old_len = len(self.dataset)
        self.dataset.resize((old_len + self._buffpointer,))
        self.dataset[old_len:old_len + self._buffpointer
                     ] = self._buff[:self._buffpointer]


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

    @property
    def readpointer(self):
        return self._readpointer

    @readpointer.setter
    def readpointer(self, val):
        self._readpointer = int(val)
        self._fill_data()

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

    def load(self):
        try:
            dset = self.group['pickle_data']
        except KeyError:
            raise KeyError('No object stored yet.')

        with H5pytoBytesStream(dset) as stream_file:
            obj = pickle.load(stream_file)
        return obj

    def save(self, obj, overwrite=True):
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
        # TODO: use buffered version or not?
        #with BytesStreamtoH5py(dset) as stream_file:
        with BytesStreamtoH5pyBuffered(dset) as stream_file:
            # using pickle protocol 4 means python>=3.4!
            pickle.dump(obj, stream_file, protocol=4)


class ArcdObjectShelf(MutableObjectShelf):
    """
    Specialized MutableObjectShelf for arcd objects.

    Stores any object with a .from_h5py_group and a .ready_for_pickle method.
    """

    def load(self):
        obj = super().load()
        obj = obj.complete_from_h5py_group(self.group)
        return obj

    def save(self, obj, overwrite=True):
        obj_to_save = obj.object_for_pickle(self.group, overwrite=overwrite)
        super().save(obj=obj_to_save, overwrite=overwrite)


class RCModelRack(collections.abc.MutableMapping):
    """Dictionary like interface to RCModels stored in an arcd storage file."""

    def __init__(self, rcmodel_group):
        self._group = rcmodel_group

    def __getitem__(self, key):
        return ArcdObjectShelf(self._group[key]).load()

    def __setitem__(self, key, value):
        group = self._group.require_group(key)
        ArcdObjectShelf(group).save(obj=value, overwrite=True)

    def __delitem__(self, key):
        del self._group[key]

    def __len__(self):
        return len(self._group.keys())

    def __iter__(self):
        return iter(self._group.keys())


class Storage:
    """
    Store all arcd RCModels and data belonging to one TPS simulation.

    Note: Everything belonging to arcd is stored in the arcd_data HDF5 group.
          You can store arbitrary data using h5py in the rest of the file
          through accessing it as Storage.file.
    """

    # setup dictionary mapping descriptive strings to 'paths' in HDF5 file
    h5py_path_dict = {"level0": "/arcd_data"}  # toplevel arcd group
    h5py_path_dict["cache"] = h5py_path_dict["level0"] + "/cache"  # cache
    h5py_path_dict.update({  # these depend on cache and level0 to be defined
        "rcmodel_store": h5py_path_dict["level0"] + "/RCModels",
        "trainset_store": h5py_path_dict["level0"] + "/TrainSet",
        "tra_dc_cache": h5py_path_dict["cache"] + "/TrajectoryDensityCollectors",
                           })
    # if the current arcd version is higher than the compatibility_version
    # we expect to be able to read the storage
    _compatibility_version = parse_version("0.7")  # introduced h5py storage

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
        self._store = self.file.require_group(self.h5py_path_dict["level0"])
        if ("w" in mode) or ("a" in mode and not fexists):
            # first creation of file: write arcd compatibility version string
            self._store.attrs["storage_version"] = np.string_(self._compatibility_version)
        else:
            store_version = parse_version(self._store.attrs["storage_version"])
            if self._compatibility_version < store_version:
                raise RuntimeError(
                        "The storage file was written with a newer version of "
                        + "arcd than the current one. You need at least arcd "
                        + "v{:s}".format(str(store_version)) + " to open it.")
        rcm_grp = self.file.require_group(self.h5py_path_dict["rcmodel_store"])
        self.rcmodels = RCModelRack(rcmodel_group=rcm_grp)
        self._empty_cache()  # should be empty, but to be sure

    def __del__(self):
        self._empty_cache()
        self.file.flush()  # TODO: do we need this? or does close() do it?
        self.file.close()

    def _empty_cache(self):
        # empty TrajectoryDensityCollector_cache
        traDC_cache_grp = self.file.require_group(
                                name=self.h5py_path_dict["tra_dc_cache"]
                                                  )
        traDC_cache_grp.clear()

    # TODO  do we even want this implementation?
    #       or should we change to the arcd-object shelf logic?
    #       having only one trainset makes sense, but feels like a limitation
    #       compared to the TrajectoryDensityCollectors...?
    def save_trainset(self, trainset):
        """
        Save an arcd.TrainSet.

        We try to pickle descriptor_transform and states, except if they are
        ops objects. In that case we save the names and they can be reloaded by
        passing the storage to load_trainset.
        There can only be one Trainset in the Storage at a time but you can
        overwrite it as often as you want with TrainSets of different length.
        That is you can not change the number of states or the descriptors
        second axis, i.e. the dimensionality of the descriptor space.
        """
        d_shape = trainset.descriptors.shape
        sr_shape = trainset.shot_results.shape
        w_shape = trainset.weights.shape
        try:
            ts_group = self.file[self.h5py_path_dict["trainset_store"]]
        except KeyError:
            # we never stored a TrainSet here before, so setup datasets
            ts_group = self.file.create_group(
                                        self.h5py_path_dict["trainset_store"]
                                              )
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
        # we try to handle states and descriptor_transform sensibly:
        # if states are OPS volumes we will replace by name to later reload
        # from OPS storage; same for descriptor_transform, i.e. if it is an OPS
        # collectivevariable replace by name
        # everything else we will just try to pickle as usual python objects
        # if we found any ops objects we set 'ops_objects' to True to know when loading
        # first prepare descriptor_transform and states for storing
        py_state = {"ops_objects": False}
        if isinstance(trainset.descriptor_transform, OPSCollectiveVariable):
            py_state["ops_objects"] = True
            py_state["descriptor_transform"] = trainset.descriptor_transform.name
        else:
            py_state["descriptor_transform"] = trainset.descriptor_transform
        states = []
        for s in trainset.states:
            if isinstance(s, OPSVolume):
                states.append(s.name)
                py_state["ops_objects"] = True
            else:
                states.append(s)
        py_state["states"] = states
        # now store them
        MutableObjectShelf(ts_group).save(py_state, overwrite=True)

    def load_trainset(self, states=None, descriptor_transform=None, ops_storage=None):
        """
        Load an arcd.TrainSet.

        Parameters:
        -----------
        states - None or list of states, if given overwrite states for TrainSet
        descriptor_transform - None or descriptor_transform function, if given
                               overwrites descriptor_transform for TrainSet
        ops_storage - None or ``openpathsampling.Storage``, if given will try
                      to reload states and descriptor_transform for TrainSet
                      from there

        Note: Passing states or descriptor_transform takes precedence over what
              we load from the ``openpathsampling.Storage``.
        """
        try:
            ts_group = self.file[self.h5py_path_dict["trainset_store"]]
        except KeyError:
            raise KeyError('No TrainSet in file.')
        descriptors = ts_group['descriptors'][:]
        shot_results = ts_group['shot_results'][:]
        weights = ts_group['weights'][:]
        # try to load descriptor_transform and states
        py_state = MutableObjectShelf(ts_group).load()
        des_trans_loaded = py_state["descriptor_transform"]
        states_loaded = py_state["states"].copy()
        if (ops_storage is not None) and py_state["ops_objects"]:
            # we only replace the states and descriptor_transform if we believe
            # they are from ops, this way we can still combine 'normal'
            # pickleable state and descriptor_transform functions with ops objs
            if isinstance(des_trans_loaded, str):
                des_trans_loaded = ops_storage.cvs.find(des_trans_loaded)
            for i, s in enumerate(states_loaded):
                if isinstance(s, str):
                    states_loaded[i] = ops_storage.volumes.find(s)
        elif py_state["ops_objects"]:
            logger.warn("TrainSet was saved with ops objects (states and/"
                        + "or descriptor_transform) but no ops_storage "
                        + "was passed to load_trainset(). You will have to"
                        + " reset them manually. They are still the names "
                        + "of the ops objects at savetime.")
        # now sort out if we were given something that takes precedence
        if descriptor_transform is None:
            # FIXME: this way we can never overwrite the descriptor_transform
            #        to None if it has a value in storage!
            descriptor_transform = des_trans_loaded
        if states is None:
            states = states_loaded
        # and return the created trainset
        return TrainSet(states=states,
                        descriptor_transform=descriptor_transform,
                        descriptors=descriptors,
                        shot_results=shot_results,
                        weights=weights,
                        )
