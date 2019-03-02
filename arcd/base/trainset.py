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
import numpy as np
from collections.abc import Iterable, Iterator


logger = logging.getLogger(__name__)


class TrainSet(Iterable):
    """
    Store shooting results (i.e. states reached) and the corresponding
    descriptors.
    """

    # TODO: do we need weights for the points?
    def __init__(self, states, descriptor_transform=None,
                 descriptors=None, shot_results=None):
        self.states = states
        n_states = len(states)
        self._tp_idxs = [[i, j] for i in range(n_states)
                         for j in range(i + 1, n_states)]
        # TODO: maybe default to a lambda func just taking xyz of a OPS traj?
        # TODO: instead of the None value we have now?
        self.descriptor_transform = descriptor_transform

        if ((descriptors is not None) and (shot_results is not None)):
            descriptors = np.asarray(descriptors, dtype=np.float64)
            shot_results = np.asarray(shot_results, dtype=np.float64)
            if shot_results.shape[0] != descriptors.shape[0]:
                raise ValueError('descriptors and shot_results must contain an'
                                 + ' equal number of points /have the same '
                                 + 'first dimension.')
        else:
            descriptors = np.empty((0, 0), dtype=np.float64)
            shot_results = np.empty((0, 0), dtype=np.float64)

        self._descriptors = descriptors
        self._shot_results = shot_results
        self._fill_pointer = shot_results.shape[0]

    @property
    def shot_results(self):
        return self._shot_results[:self._fill_pointer]

    @property
    def descriptors(self):
        return self._descriptors[:self._fill_pointer]

    @property
    def transitions(self):
        return sum(self._shot_results[:self._fill_pointer, i]
                   * self._shot_results[:self._fill_pointer, j]
                   for i, j in self._tp_idxs)

    def __len__(self):
        return self._fill_pointer

    def __getitem__(self, key):
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
            return (self._descriptors[:self._fill_pointer][key:key+1],
                    self._shot_results[:self._fill_pointer][key:key+1])
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return (self._descriptors[start:stop:step],
                    self._shot_results[start:stop:step])
        else:
            raise KeyError('keys must be int or slice.')

    # TODO: do we need __setitem__ ??
    def __setitem__(self, key):
        raise NotImplementedError

    def iter_batch(self, batch_size=64, shuffle=True):
        """Returns an Iterator over the whole TrainSet in chunks of batch size."""
        return TrainSetIterator(self, batch_size, shuffle)

    def __iter__(self):
        """
        This simply uses batch_size=64 and shuffle=True.
        See iter_batch if you need to control batch_size and shuffle.
        """
        return TrainSetIterator(self, 64, True)

    def _extend_if_needed(self, descriptor_dim, add_entries=100):
        """Extend internal storage arrays if next step would not fit."""
        # need descriptors dimensionality to extend the array
        # at least if we do not know it yet, i.e. at first resize
        shadow_len = self._shot_results.shape[0]
        if shadow_len == 0:
            # no points yet, just create the arrays
            self._shot_results = np.zeros((add_entries, len(self.states)), dtype=np.float64)
            self._descriptors = np.zeros((add_entries, descriptor_dim), dtype=np.float64)
        elif shadow_len <= self._fill_pointer + 1:
            # no space left for the next point, extend
            self._shot_results = np.concatenate(
                (self._shot_results, np.zeros((add_entries, len(self.states)), dtype=np.float64))
                                                )
            self._descriptors = np.concatenate(
                (self._descriptors, np.zeros((add_entries, descriptor_dim), dtype=np.float64))
                                               )

    def append_ops_mcstep(self, mcstep, ignore_invalid=False):
        """
        Appends the results and descriptors extracted from given OPS MCStep.
        """
        try:
            details = mcstep.change.canonical.details
            shooting_snap = details.shooting_snapshot
        # TODO: warn or pass? if used togehter with other TP generation schemes
        # than shooting, pass is the right thing to do,
        # otherwise this should never happen anyway...but then it might be good
        # to know if it does... :)
        except AttributeError:
            # wrong kind of move (no shooting_snapshot)
            # this could actually happen if we use arcd in one simulation
            # together with other TPS/TIS schemes
            logger.warn('Tried to add a MCStep that has no shooting_snapshot.')
        except IndexError:
            # very wrong kind of move (no trials!)
            # I think this should never happen?
            logger.error('Tried to add a MCStep that contains no trials.')
        else:
            # find out which states we reached
            trial_traj = mcstep.change.canonical.trials[0].trajectory
            init_traj = details.initial_trajectory
            test_points = [s for s in [trial_traj[0], trial_traj[-1]]
                           if s not in [init_traj[0], init_traj[-1]]]
            shot_results = np.array([sum(int(state(pt)) for pt in test_points)
                                     for state in self.states])
            total_count = sum(shot_results)

            # TODO: for now we assume TwoWayShooting, because otherwise we can not redraw v!
            # warn if no states were reached, dont add the point except ignore_invalid=True,
            # it makes no contribution to the loss since terms are 0,
            # this makes the 'harmonic loss' from multi-domain models blow up,
            # also some regularization schemes will overregularize by miscounting
            if total_count < 2 and not ignore_invalid:
                logger.warn('Total states reached is < 2. This probably means there are '
                            + 'uncommited trajectories. Will not add the point.')
                return
            # get and possibly transform descriptors
            # descriptors is a 1d-array, since we use a snap and no traj in CV
            descriptors = self.descriptor_transform(shooting_snap)
            if not np.all(np.isfinite(descriptors)):
                logger.warn('There are NaNs or infinities in the training '
                            + 'descriptors. \n We used numpy.nan_to_num() to'
                            + ' proceed. You might still want to have '
                            + '(and should have) a look @ \n'
                            + 'np.where(np.isinf(descriptors): '
                            + str(np.where(np.isinf(descriptors)))
                            + 'and np.where(np.isnan(descriptors): '
                            + str(np.where(np.isnan(descriptors))))
                descriptors = np.nan_to_num(descriptors)
            # add shooting results and transformed descriptors to training set
            self.append_point(descriptors, shot_results)

    def append_point(self, descriptors, shot_results):
        """
        Append the given 1d-arrays of descriptors and shot_results.

        descriptors - np.ndarray with shape (descriptor_dim,)
        shot_results - np.ndarray with shape (n_states,)
        """
        self._extend_if_needed(descriptors.shape[0])
        self._shot_results[self._fill_pointer] = shot_results
        self._descriptors[self._fill_pointer] = descriptors
        self._fill_pointer += 1


class TrainSetIterator(Iterator):
    """Iterate over TrainSet in batches, possibly shuffle before iterating."""

    def __init__(self, trainset, batch_size, shuffle):
        self.i = 0
        self.max_i = len(trainset)
        self.batch_size = batch_size
        if shuffle:
            # shuffle before iterating
            shuffle_idxs = np.random.permutation(self.max_i)
            self.descriptors = trainset.descriptors[shuffle_idxs]
            self.shot_results = trainset.shot_results[shuffle_idxs]
        else:
            self.descriptors = trainset.descriptors
            self.shot_results = trainset.shot_results

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
            # more than a full batch remaining
            start = self.i
            stop = self.i + self.batch_size

        self.i += self.batch_size
        return (self.descriptors[start:stop],
                self.shot_results[start:stop])
