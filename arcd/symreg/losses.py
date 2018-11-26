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
import pyaudi as ad
from pyaudi import gdual_vdouble as gdual


logger = logging.getLogger(__name__)


def binom_loss(expression, x, shot_results):
    # we expect shot results to be a 2d np array
    rc = expression(x)[0]
    n = len(rc.constant_cf)
    # shot_results[:,1] is n_B
    # the RC gives progress towards B, i.e. p_B = 1 / (1 + exp(-rc))
    return (gdual(shot_results[:, 0]) * ad.log(1. + ad.exp(rc))
            + gdual(shot_results[:, 1]) * ad.log(1. + ad.exp(-rc))
            ) / n


def multinom_loss(expression, x, shot_results):
    # we expect shot_results to be a 2d np array
    rcs = expression(x)
    n = len(rcs[0].constant_cf)
    lnZ = ad.log(sum([ad.exp(rc) for rc in rcs]))
    return (sum([(lnZ - rc) * gdual(shot_results[:, i])
                for i, rc in enumerate(rcs)])
            / n
            )


# complexity penalties
def operation_count(expression, fact=0.0005):
    n = expression.get_n()
    m = expression.get_m()
    ex_sp = expression.simplify(['x' + str(i) for i in range(n)],
                                subs_weights=True)
    c = sum([ex_sp[i].count_ops() for i in range(m)])
    return c * fact


def active_genes_count(expression, fact=0.0005):
    return len(expression.get_active_genes()) * fact


# weight regularizations
def l1_regularization(active_weights, fact=0.0005):
    return fact * sum([ad.abs(aw) for aw in active_weights])


def l2_regularization(active_weights, fact=0.0005):
    return fact * ad.sqrt(sum([aw * aw
                               for aw in active_weights]))
