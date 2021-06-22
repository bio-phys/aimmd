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
import pyaudi as ad
import sympy as sp
import numpy as np
from pyaudi import gdual_vdouble as gdual


logger = logging.getLogger(__name__)


def binom_loss(expression, x, shot_results):
    # we expect shot results to be a 2d np array
    rc = expression(x)[0]
    n = np.sum(shot_results)
    # shot_results[:,1] is n_B
    # the RC gives progress towards B, i.e. p_B = 1 / (1 + exp(-rc))
    return (gdual(shot_results[:, 0]) * ad.log(1. + ad.exp(rc))
            + gdual(shot_results[:, 1]) * ad.log(1. + ad.exp(-rc))
            ) / n


def multinom_loss(expression, x, shot_results):
    # we expect shot_results to be a 2d np array
    rcs = expression(x)
    n = np.sum(shot_results)
    lnZ = ad.log(sum([ad.exp(rc) for rc in rcs]))
    return (sum([(lnZ - rc) * gdual(shot_results[:, i])
                for i, rc in enumerate(rcs)])
            / n
            )


# complexity penalties
def operation_count(expression, fact=0.0005):
    n = expression.get_n()
    m = expression.get_m()
    try:
        ex_sp = expression.simplify(['x' + str(i) for i in range(n)],
                                    subs_weights=True)
    except sp.PolynomialDivisionFailed:
        # TODO/FIXME: this is a dirty fix for a sympy/dcgp-python error
        #             sometimes the simplyfy method of dcgp-python fails
        #             because the underlying sympy methods for the construction
        #             and simplyfication of the expression sometimes fail
        #             with this error
        #             returning nan makes sure we can not accept these
        #             expressions but still finish the optimization
        return float('nan')
    else:
        c = sum([ex_sp[i].count_ops() for i in range(m)])
        return c * fact


def active_genes_count(expression, fact=0.0005):
    return len(expression.get_active_genes()) * fact


# weight regularizations
def _get_active_weights(expression):
    # gets and return the list of active weights
    arity = expression.get_arity()
    an = expression.get_active_nodes()
    n = expression.get_n()
    r = expression.get_rows()
    aw_idxs = []
    for k in range(len(an)):
        if an[k] >= n:
            if isinstance(arity, list):
                # new dcgpy has arity as a vector
                a = arity[(an[k] - n) // r]
            else:
                # 'old' dcgpy, arity is the same for all nodes
                a = arity
            for l in range(a):
                aw_idxs.append((an[k] - n) * a + l)
    ws = expression.get_weights()
    return [ws[i] for i in aw_idxs]


def l1_regularization(expression, fact=0.0005):
    active_weights = _get_active_weights(expression)
    return fact * sum([ad.abs(aw) for aw in active_weights])


def l2_regularization(expression, fact=0.0005):
    active_weights = _get_active_weights(expression)
    return fact * ad.sqrt(sum([aw * aw
                               for aw in active_weights]))
