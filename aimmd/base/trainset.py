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
import numpy as np
from collections.abc import Iterable, Iterator
from . import Properties


logger = logging.getLogger(__name__)


class TrainSet(Iterable):
    """Stores shooting results and the corresponding descriptors."""

    # TODO: do we need weights for the points?
    def __init__(self, n_states,
                 descriptors=None, shot_results=None, weights=None):
        """
        Create a TrainSet.

        n_states - int, number of states, i.e. number of possible outcomes
        descriptors - None or numpy.ndarray [shape=(n_points, n_dim)],
                      if given trainset is initialized with these descriptors
        shot_results - None or numpy.ndarray [shape=(n_points, n_states)],
                       if given trainset is initialized with these shot_results
        weights - None or numpy.ndarray [shape=(n_points,)],
                  if given we will use this as weights for the training points
        """
        self.n_states = n_states
        self._tp_idxs = [[i, j] for i in range(n_states)
                         for j in range(i + 1, n_states)]

        if ((descriptors is not None) and (shot_results is not None)):
            descriptors = np.asarray(descriptors, dtype=np.float64)
            shot_results = np.asarray(shot_results, dtype=np.float64)
            if shot_results.shape[0] != descriptors.shape[0]:
                raise ValueError("'descriptors' and 'shot_results' must contain an"
                                 + ' equal number of points /have the same '
                                 + 'first dimension.')
            if weights is not None:
                weights = np.asarray(weights, dtype=np.float64)
                if weights.shape[0] != descriptors.shape[0]:
                    raise ValueError("If given 'weights' and 'descriptors' must"
                                     + ' contain the same number of points '
                                     + '/have the same first dimension.')
            else:
                # assume equal weights for all given points
                weights = np.ones((descriptors.shape[0],), dtype=np.float64)
        else:
            descriptors = np.empty((0, 0), dtype=np.float64)
            shot_results = np.empty((0, 0), dtype=np.float64)
            weights = np.empty((0,), dtype=np.float64)

        self._descriptors = descriptors
        self._shot_results = shot_results
        self._weights = weights
        self._fill_pointer = shot_results.shape[0]

    @property
    def shot_results(self):
        """Return states reached for each point."""
        return self._shot_results[:self._fill_pointer]

    @property
    def descriptors(self):
        """Return descriptor coordinates for each point."""
        return self._descriptors[:self._fill_pointer]

    @property
    def weights(self):
        """Return weights for each point."""
        return self._weights[:self._fill_pointer]

    @property
    def transitions(self):
        """Calculate number of transitions for each point."""
        return sum(self._shot_results[:self._fill_pointer, i]
                   * self._shot_results[:self._fill_pointer, j]
                   for i, j in self._tp_idxs)

    def __len__(self):
        """Return number of points in TrainSet."""
        return self._fill_pointer

    def __getitem__(self, key):
        """Return a new TrainSet with a subset of points."""
        if isinstance(key, int):
            # catch out of bounds access
            if key >= len(self):
                raise KeyError('key >= len(self)')
            elif key < -len(self):
                raise KeyError('key < -len(self)')
            # but allow negative keys and treat them correctly
            elif key < 0:
                key = len(self) + key
            # slice to preserve dimensionality
            descriptors = self._descriptors[:self._fill_pointer][key:key+1]
            shots = self._shot_results[:self._fill_pointer][key:key+1]
            weights = self._weights[:self._fill_pointer][key:key+1]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            descriptors = self._descriptors[start:stop:step]
            shots = self._shot_results[start:stop:step]
            weights = self._weights[start:stop:step]
        elif isinstance(key, np.ndarray):
            descriptors = self._descriptors[:self._fill_pointer][key]
            shots = self._shot_results[:self._fill_pointer][key]
            weights = self._weights[:self._fill_pointer][key]
        else:
            raise KeyError('keys must be int, slice or np.ndarray.')

        return TrainSet(self.n_states,
                        descriptors=descriptors, shot_results=shots,
                        weights=weights)

    # TODO: do we need __setitem__ ??
    def __setitem__(self, key):
        raise NotImplementedError

    def iter_batch(self, batch_size=None, shuffle=True):
        """Iterate over the (shuffled) TrainSet in chunks of batch_size."""
        return TrainSetIterator(self, batch_size, shuffle)

    def __iter__(self):
        """
        Iterate over the shuffled TrainSet in chunks of 128 points.

        Use self.iter_batch() if you want to control batch_size or shuffle.
        """
        return TrainSetIterator(self, 128, True)

    def _extend_if_needed(self, descriptor_dim, add_entries=100):
        """Extend internal storage arrays if next step would not fit."""
        # need descriptors dimensionality to extend the array
        # at least if we do not know it yet, i.e. at first resize
        shadow_len = self._shot_results.shape[0]
        if shadow_len == 0:
            # no points yet, just create the arrays
            self._shot_results = np.zeros((add_entries, self.n_states),
                                          dtype=np.float64)
            self._descriptors = np.zeros((add_entries, descriptor_dim),
                                         dtype=np.float64)
            self._weights = np.zeros((add_entries, ), dtype=np.float64)
        elif shadow_len <= self._fill_pointer + 1:
            # no space left for the next point, extend
            self._shot_results = np.concatenate(
                (self._shot_results,
                 np.zeros((add_entries, self.n_states), dtype=np.float64)
                 )
                                                )
            self._descriptors = np.concatenate(
                (self._descriptors,
                 np.zeros((add_entries, descriptor_dim), dtype=np.float64)
                 )
                                               )
            self._weights = np.concatenate(
                (self._weights, np.zeros((add_entries,), dtype=np.float64))
                                           )

    def append_point(self, descriptors, shot_results, weight=1.):
        """
        Append the given 1d-arrays of descriptors and shot_results.

        descriptors - np.ndarray with shape (descriptor_dim,)
        shot_results - np.ndarray with shape (n_states,)
        weight - float, (unnormalized) weight of the point
        """
        self._extend_if_needed(descriptors.shape[0])
        self._shot_results[self._fill_pointer] = shot_results
        self._descriptors[self._fill_pointer] = descriptors
        self._weights[self._fill_pointer] = weight
        self._fill_pointer += 1


class TrainSetIterator(Iterator):
    """Iterate over TrainSet in batches, possibly shuffle before iterating."""

    def __init__(self, trainset, batch_size, shuffle):
        self.i = 0
        self.max_i = len(trainset)
        if batch_size is None:
            # Note: we use None to denote we dont care,
            #       but if possible use the full trainset
            batch_size = len(trainset)
        elif np.isinf(batch_size):
            # Note: we use inf to mean take the full trainset always
            #       even if we then run into out of memory issues
            batch_size = len(trainset)
        self.batch_size = batch_size
        self.trainset = trainset
        self.idxs = (np.random.permutation(self.max_i) if shuffle
                     else np.arange(self.max_i)  # just a range if no shuffle
                     )

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.max_i:
            # nothing remains
            raise StopIteration
        elif self.i + self.batch_size > self.max_i:
            # only part of a batch
            start = self.i
            stop = self.max_i
        else:
            # more than or exactly a full batch remaining
            start = self.i
            stop = self.i + self.batch_size

        self.i += self.batch_size
        des = self.trainset.descriptors[self.idxs[start:stop]]
        shots = self.trainset.shot_results[self.idxs[start:stop]]
        ws = self.trainset.weights[self.idxs[start:stop]]
        return {Properties.descriptors: des,
                Properties.shot_results: shots,
                Properties.weights: ws,
                }
