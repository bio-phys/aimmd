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
    def __init__(self, coords_cv, states,
                 coords=None, shot_results=None, weights=None,
                 ):
        self.coords_cv = coords_cv
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

    def add_step(self, step):
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
            # it makes no contribution to training, since all terms are 0
            # some regularization schemes will 'overregularize' by miscounting
            if total_count < 1:
                logger.warn('Total states reached is < 1. This means we added '
                            + 'uncommited trajectories.')
            # add shooting results and transformed coordinates to training set
            coords = self.coords_cv(shooting_snap)
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


class TrainSetdCGPy(TrainSetBase):
    """
    TODO
    """
    @property
    def trainform(self):
        return ([gdual(self._coords[:, i])
                 for i in range(self._coords.shape[1])],
                self.shot_results)


class TrainSetKeras(TrainSetBase):
    """
    TODO
    """
    

