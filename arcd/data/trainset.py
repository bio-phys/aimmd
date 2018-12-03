#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:48:43 2018

@author: Hendrik Jung

This file is part of ARCD.

ARCD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ARCD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ARCD.  If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import collections
import numpy as np
from pyaudi import gdual_vdouble as gdual


logger = logging.getLogger(__name__)


class TrainSetBase:
    """
    TODO
    """
    def __init__(self, n_states, coords, shot_results, weights=None):
        coords = np.asarray(coords)
        shot_results = np.asarray(shot_results)
        if not coords.shape[0] == shot_results.shape[0]:
            raise ValueError('coords and shot_results must have the same '
                             + 'first dimension.')
        if weights:
            weights = np.asarray(weights)
            if not weights.shape[0] == coords.shape[0]:
                raise ValueError('If given, weights must have the same '
                                 + 'first dimension as coords and '
                                 + 'shot_results.')
        else:
            weights = np.ones((coords.shape[0],))
        self._coords = coords
        self._shot_results = shot_results
        self._weights = weights
        self._tp_idxs = [[i, j] for i in range(n_states)
                         for j in range(i + 1, n_states)]

    def __getitem__(self, key):
        if isinstance(key, int):
            # slices of length 1 to preserve dim
            return [self._coords[key:key + 1],
                    self._shot_results[key:key + 1],
                    self._weights[key:key + 1]]
        elif isinstance(key, slice):
            # real slices
            start, stop, step = key.indices(len(self))
            return [self._coords[start:stop:step],
                    self._shot_results[start:stop:step],
                    self._weights[start:stop:step]]
        else:
            raise KeyError('Key must be int or slice.')

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        val = np.asarray(val)
        if val.shape[0] == len(self):
            self._weights = val
        else:
            raise ValueError('weights need to have the same first dimension '
                             + 'as coords and shot_results.')

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, val):
        raise NotImplementedError('Set at creation time and/or use '
                                  + 'self.add_step() for consistency.')

    @property
    def shot_results(self):
        return self._shot_results

    @shot_results.setter
    def shot_results(self, val):
        raise NotImplementedError('Set at creation time and/or use '
                                  + 'self.add_step() for consistency.')

    @property
    def transitions(self):
        return sum([self._shot_results[:, i] * self._shot_results[:, j]
                    for i, j in self._tp_idxs])

    @transitions.setter
    def transitions(self, val):
        raise NotImplementedError('Indirectly set at creation time via '
                                  + 'shot_results and/or use self.add_step() '
                                  + 'for consistency.')

    def __len__(self):
        return len(self._shot_results)


class TSmixin_OPS:
    """
    TODO
    """
    def __init__(self, coords_transform, states,
                 coords=None, shot_results=None, weights=None,
                 ):
        self.coords_transform = coords_transform
        self.states = states
        self._tp_idxs = [[i, j] for i in range(len(states))
                         for j in range(i + 1, len(states))]
        if coords is not None:
            coords = np.asarray(coords)
            shot_results = np.asarray(shot_results)
            if not coords.shape[0] == shot_results.shape[0]:
                raise ValueError('coords and shot_results must have the same '
                                 + 'first dimension.')
            if weights:
                weights = np.asarray(weights)
                if not weights.shape[0] == coords.shape[0]:
                    raise ValueError('If given, weights must have the same '
                                     + 'first dimension as coords and '
                                     + 'shot_results.')
            else:
                weights = np.ones((coords.shape[0],))
            self._coords = coords
            self._shot_results = shot_results
            self._weights = weights
        else:
            self._coords = np.zeros((0, 0))
            self._shot_results = np.zeros((0, 0))
            self._weights = np.ones((0,))

    # TODO
    #def pickle_enable(self)
    #def pickle_restore(self, ops_storage)

    def add_mcstep(self, step):
        try:
            details = step.change.canonical.details
            shooting_snap = details.shooting_snapshot
        # TODO: warn or pass? if used togehter with other TP generation schemes
        # than shooting, pass is the right thing to do,
        # otherwise this should never happen anyway...but then it might be good
        # to know if it does... :)
        except AttributeError:
            # wrong kind of move (no shooting_snapshot)
            pass
        except IndexError:
            # very wrong kind of move (no trials!)
            pass
        else:
            # find out which states we reached
            trial_traj = step.change.canonical.trials[0].trajectory
            init_traj = details.initial_trajectory
            test_points = [s for s in [trial_traj[0], trial_traj[-1]]
                           if s not in [init_traj[0], init_traj[-1]]]
            total = collections.Counter(
                {state: sum([int(state(pt)) for pt in test_points])
                 for state in self.states})
            total_count = sum(total.values())
            # warn if no states were reached, add the point in any case,
            # it makes no contribution to the loss since all terms are 0, but
            # some regularization schemes will overregularize by miscounting
            if total_count < 1:
                logger.warn('Total states reached is < 1. This means we added '
                            + 'uncommited trajectories.')
            # add shooting results and transformed coordinates to training set
            coords = self.coords_transform(shooting_snap)
            if not np.all(np.isfinite(coords)):
                logger.warn('There are NaNs or infinities in the training '
                            + 'coordinates. \n We used numpy.nan_to_num() to'
                            + ' proceed. You might still want to have '
                            + '(and should have) a look @ \n'
                            + 'np.where(np.isinf(coords): '
                            + str(np.where(np.isinf(coords)))
                            + 'and np.where(np.isnan(coords): '
                            + str(np.where(np.isnan(coords))))
                coords = np.nan_to_num(coords)
            new_len = len(self) + 1
            self._shot_results = np.resize(self._shot_results,
                                           new_shape=(new_len,
                                                      len(self.states))
                                           )
            self._coords = np.resize(self._coords,
                                     new_shape=(new_len, coords.shape[0]))
            self._weights = np.resize(self._weights, new_shape=(new_len,))
            self._weights[-1] = 1.
            for i, state in enumerate(self.states):
                self._shot_results[-1, i] = total[self.states[i]]
            self._coords[-1] = coords


class TSmixin_dCGPy:
    """
    TODO
    """
    @property
    def trainform(self):
        return ([gdual(self._coords[:, i])
                 for i in range(self._coords.shape[1])],
                self.shot_results)


class TSmixin_keras:
    """
    TODO
    """
    @property
    def trainform(self):
        return self

    def __iter__(self):
        self._iter = -self.batch_size
        # only iterate over entries which have weight >= min_weight
        retain_idxs = np.where(self._weights >= self.min_weight)[0]
        # shuffle before iterating
        shuffle_idxs = np.random.permutation(len(retain_idxs))
        self._i_coords = self._coords[retain_idxs][shuffle_idxs]
        self._i_shot_results = self._shot_results[retain_idxs][shuffle_idxs]
        self._i_weights = self._weights[retain_idxs][shuffle_idxs]
        return self

    def __next__(self):
        self._iter += self.batch_size
        if self._iter >= len(self._i_shot_results):
            raise StopIteration
        if (self._iter + self.batch_size) > len(self._i_shot_results):
            # return whats left
            return [self._i_coords[self._iter:],
                    self._i_shot_results[self._iter:],
                    self._i_weights[self._iter:]]
        # return a full batch if we got until here
        return [self._i_coords[self._iter:self._iter + self.batch_size],
                self._i_shot_results[self._iter:self._iter + self.batch_size],
                self._i_weights[self._iter:self._iter + self.batch_size]]


class TrainSetdCGPy(TSmixin_dCGPy, TrainSetBase):
    """
    TODO
    """


class TrainSetdCGPy_OPS(TSmixin_dCGPy, TSmixin_OPS, TrainSetBase):
    """
    TODO
    """


class TrainSetKeras(TSmixin_keras, TrainSetBase):
    """
    TODO
    """
    def __init__(self, n_states, coords, shot_results, weights=None,
                 batch_size=32, min_weight=0.0005):
        super().__init__(n_states, coords, shot_results, weights)
        self.batch_size = batch_size
        self.min_weight = min_weight


class TrainSetKeras_OPS(TSmixin_keras, TSmixin_OPS, TrainSetBase):
    """
    TODO
    """
    def __init__(self, coords_transform, states,
                 coords=None, shot_results=None, weights=None,
                 batch_size=32, min_weight=0.0005):
        super().__init__(coords_transform, states,
                         coords, shot_results, weights)
        self.batch_size = batch_size
        self.min_weight = min_weight
