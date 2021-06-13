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
from dcgpy import kernel_set_gdual_vdouble as kernel_set
from dcgpy import expression_weighted_gdual_vdouble as dcgpy_expression
from pyaudi import gdual_vdouble as gdual


logger = logging.getLogger(__name__)


def initialize_random_expression(n_in, n_out,
                                 rows=1, cols=15, levels_back=16, arity=2,
                                 kernels=None, seed=None):
    """
    Initialize a randomized weighted dCGPy expression.

    Returns a weighted gdual dCGPy expression.

    Parameters:
    -----------
    n_in - int, number of input coordinates
    n_out - int, number of outputs
    rows - int, default=1, number of rows in the CGP expression
    cols - int, default=15, number of cloumns in the CGP expression
    levels_back - int, default=16, how far back in the graph a node can get
                  input from, a sensitive choice is cols + 1
    arity - int, default=2, the arity of the nodes
    kernels - None or list of str, if None all possible values are used:
              'sum', 'diff', 'mul', 'div', 'sig', 'sin', 'log', 'exp',
              they stand for: +, -, *, /, sigmoid, sine, natural logarithm,
                             exponential function
    seed - int, the random seed determining the active genes of the expression,
           if None will be choosen at random with numpy.random.randint()

    """
    if kernels is None:
        kernels = ['sum', 'diff', 'mul', 'div', 'sig', 'sin', 'log', 'exp']
    if seed is None:
        seed = np.random.randint(533533)
    # transform strings to dCGPy kernels
    kernels = kernel_set(kernels)()
    # initialize random expression
    ex = dcgpy_expression(n_in, n_out, rows=rows, cols=cols,
                          levels_back=levels_back, arity=arity,
                          kernels=kernels, seed=seed)
    # randomize the weights
    ar = ex.get_arity()
    r = ex.get_rows()
    for j in range(r * ex.get_cols()):
        if isinstance(ar, list):
            a = ar[j // r]
        else:
            a = ar
        for k in range(a):
            ex.set_weight(j + n_in, k, gdual([np.random.normal(0, 1)]))
    return ex
