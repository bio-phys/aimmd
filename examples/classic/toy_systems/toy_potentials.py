"""
This file contains (2-dimensional) potential energy surfaces to be used with the
openpathsampling toy engine.
"""
import numpy as np
import openpathsampling.engines.toy as toys


class XYDiagpot(toys.PES):
    """
    Implementation of the Hummer-Szabo-potential.

    V(x,y) = b ((x^2 - 1)^2 + (x - y)^2)
    """
    def __init__(self, b: float):
        """
        Initialize the potential.

        Parameters
        ----------
        b : float
            The potentials b parameter, i.e. the barrier height (in kT / 10).
        """
        self.b = b
        self._local_dVdx = None

    def __repr__(self):
        return f"XYDiagpot with barrier height {self.b}."

    def to_dict(self):
        dct = super().to_dict()
        dct["b"] = self.b
        return dct

    def V(self, sys):
        """
        V(x,y) = b ((x^2 - 1)^2 + (x - y)^2)
        """
        x = sys.positions[0]
        y = sys.positions[1]
        return self.b * ((x**2 - 1)**2 + (x - y)**2)

    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)
        x = sys.positions[0]
        y = sys.positions[1]
        self._local_dVdx[0] = 2 * self.b * (2 * x**3 - x - y)
        self._local_dVdx[1] = - 2 * self.b * (x - y)
        return self._local_dVdx
