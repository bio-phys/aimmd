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
from abc import ABC, abstractmethod
from openpathsampling.engines.snapshot import BaseSnapshot as OPSBaseSnapshot
from openpathsampling.engines import Trajectory as OPSTrajectory


logger = logging.getLogger(__name__)


class WrapperBase(ABC):
    """
    TODO
    """
    def __init__(self, model):
        self.model = model
        n_out = self._get_n_out(model)
        if n_out == 1:
            self.p = self._p_binom
            self.z_sel = self.__call__
        else:
            self.p = self._p_multinom
            self.z_sel = self._z_sel_multinom

    def __call__(self, coords):
        return self._q(coords)

    @abstractmethod
    def _get_n_out(self, model):
        # should return the number of outputs for a given model
        pass

    @abstractmethod
    def _q(self, coords):
        # should take a 2d numpy array, dim=(n_points, n_dim)
        # should return the unnormalized log committment probability/ies
        pass

    def _p_binom(self, coords):
        q = self(coords)
        return 1/(1 + np.exp(-q))

    def _p_multinom(self, coords):
        exp_q = np.exp(self(coords))
        return exp_q / np.sum(exp_q, axis=1)

    def _z_sel_multinom(self, coords):
        """
        This expression is zero if (and only if) x is at the point of
        maximaly conceivable p(TP|x), i.e. all p_i are equal.
        z_{sel}(x) always lies in [0, 25], where z_{sel}(x)=25 implies
        p(TP|x)=0. We can therefore select the point for which this
        expression is closest to zero as the optimal SP.
        """
        p = self._p_multinom(coords)
        # the prob to be on any TP is 1 - the prob to be on no TP
        # to be on no TP means beeing on a "self transition" (A->A, etc.)
        reactive_prob = 1 - np.sum(p * p, axis=1)
        # scale z_sel to [0, 25]
        return (
                (25 / (1 - 1 / self.n_out))
                * (1 - 1 / self.n_out - reactive_prob)
                )


class OPSWrapperBase(WrapperBase):
    """
    TODO
    """
    def __init__(self, model, coords_cv):
        super().__init__(model)
        self.coords_cv = coords_cv

    def __call__(self, trajectory, convert_ops=True):
        if convert_ops:
            if isinstance(trajectory, OPSBaseSnapshot):
                trajectory = OPSTrajectory([trajectory])
            return self._q(self.coords_cv(trajectory))
        else:
            return self._q(trajectory)


class KerasWrapperMixin:
    """
    Mixin class to wrap keras models
    """
    def _get_n_out(self, model):
        return model.output_shape[1]

    def _q(self, coords):
        return self.model.predict(coords)


class KerasOPSWrapper(KerasWrapperMixin, OPSWrapperBase):
    """
    TODO
    """


class KerasWrapper(KerasWrapperMixin, WrapperBase):
    """
    TODO
    """

# TODO: dCGPy wrapper, dCGPy OPS wrapper
