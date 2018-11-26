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

logger = logging.getLogger(__name__)


class WrapperBase(ABC):
    """
    TODO
    """
    def __init__(self, n_out=1):
        self.n_out = n_out
        if n_out == 1:
            self.p = self._p_binom
            self.z_sel = self.__call__
        else:
            self.p = self._p_multinom
            self.z_sel = self._z_sel_multinom

    def __call__(self, coords):
        return self.q(coords)

    @abstractmethod
    def q(self, coords):
        # should return the unnormalized log probability/ies
        pass

    def _p_binom(self, coords):
        q = self(coords)
        return 1/(1 + np.exp(-q))

    def _p_multinom(self, coords):
        exp_q = np.exp(self(coords))
        return exp_q / np.sum(exp_q, axis=1)

    def _z_sel_multinom(self, coords):
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
    def __init__(self, coords_cv, n_out=1):
        super().__init__(n_out)
        self.coords_cv = coords_cv

    def __call__(self, trajectory):
        return super.__call__(self.coords_cv(trajectory))

# TODO: keras NN OPS wrapper, (NN wrapper, dCGPy wrapper, dCGPy OPS wrapper)