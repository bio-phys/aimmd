#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:22:53 2018

@author: Hendrik Jung

This file is part of ARCD. This file also contains code adapted from
the dCGPy examples (https://github.com/darioizzo/d-CGP) originally
from Dario Izzo, Francesco Biscani, Alessio Mereta.
dCGPy is GPL licensed.

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
import math
import logging
#import pyaudi
import numpy as np
import pyaudi as ad
from pyaudi import gdual_vdouble as gdual


logger = logging.getLogger(__name__)


def binom_loss(expression, x, shot_results):
    # we expect shot results to be a 2d np array
    rc = expression(x)[0]
    # shot_results[:,1] is n_B
    # the RC gives progress towards B, i.e. p_B = 1 / (1 + exp(-rc))
    return (gdual(shot_results[:, 0])*ad.log(1. + ad.exp(rc))
            + gdual(shot_results[:, 1])*ad.log(1. + ad.exp(-rc))
            )


def multinom_loss(expression, x, shot_results):
    # we expect shot_results to be a 2d np array
    rcs = expression(x)
    lnZ = ad.log(sum([ad.exp(rc) for rc in rcs]))
    return sum([(lnZ - rc) * gdual(shot_results[:, i])
                for i, rc in enumerate(rcs)])


def regularized_binom_loss(expression, x, shot_results, regularization=0.01):
    # we expect shot results to be a 2d np array
    rc = expression(x)[0]
    # regularize per point to make regularization independent of trainset size
    reg_term = (regularization * len(expression.get_active_genes())
                * len(rc.constant_cf)
                )
    # shot_results[:,1] is n_B
    # the RC gives progress towards B, i.e. p_B = 1 / (1 + exp(-rc))
    return (gdual(shot_results[:, 0])*ad.log(1. + ad.exp(rc))
            + gdual(shot_results[:, 1])*ad.log(1. + ad.exp(-rc))
            + reg_term)


def regularized_multinom_loss(expression, x, shot_results,
                              regularization=0.01):
    # we expect shot_results to be a 2d np array
    rcs = expression(x)
    # regularize per point to make regularization independent of trainset size
    reg_term = (regularization * len(expression.get_active_genes())
                * len(rcs[0].constant_cf)
                )
    lnZ = ad.log(sum([ad.exp(rc) for rc in rcs]))
    return (sum([(lnZ - rc) * gdual(shot_results[:, i])
                for i, rc in enumerate(rcs)])
            + reg_term)


def optimize_expression(expression, offsprings, max_gen, xt, yt, loss_function,
                        newtonParams, keep_weights=False):
    """
xt and yt are directly passed to the loss function. In most cases, xt should be
a list of pyaudi weighted gduals for the gradient calculations, while yt can be
a numpy array contaning 'just' reference values. The loss function must take 3
parameters: the dCGPy expression, xt and yt (in that order).


Adapted from weighted symbolic regression example:
https://github.com/darioizzo/d-CGP

Originally from Dario Izzo, Francesco Biscani, Alessio Mereta.
dCGPy is GPL licensed.
    """
    # The offsprings chromosome, loss and weights
    chromosome = [1] * offsprings
    loss = [1] * offsprings
    weights = [1] * offsprings
    # Init the best as the initial expression
    best_chromosome = expression.get()
    best_weights = expression.get_weights()
    best_loss = sum(loss_function(expression, xt, yt).constant_cf)
    if math.isnan(best_loss):
        # if initial expression loss is NaN we set loss to inf
        # such that we later take the first nonNan expression
        best_loss = float('inf')

    # Main loop over generations
    for g in range(max_gen):
        for i in range(offsprings):
            expression.set(best_chromosome)
            expression.set_weights(best_weights)
            expression.mutate_active(i)
            if keep_weights and i == 0:
                # if we did not mutate any genes and do not randomize the
                # weights this allows us to optimize further if we did not take
                # enough newton steps last round,
                # but it will make us stuck in local minima in parameter space
                new_parms = newtonParams.copy()
                new_parms.update({'randomize_weights': False})
                newton(expression, loss_function, xt, yt, **new_parms)
            else:
                newton(expression, loss_function, xt, yt, **newtonParams)
            # get the loss
            loss[i] = sum(loss_function(expression, xt, yt).constant_cf)
            chromosome[i] = expression.get()
            weights[i] = expression.get_weights()
        for i in range(offsprings):
            if not math.isnan(loss[i]) and loss[i] <= best_loss:
                expression.set(chromosome[i])
                expression.set_weights(weights[i])
                best_chromosome = chromosome[i]
                best_loss = loss[i]
                best_weights = weights[i]

    expression.set(best_chromosome)
    expression.set_weights(best_weights)
    return best_loss, best_chromosome, best_weights


# This is used to sum over the component of a vectorized coefficient,
# accounting for the fact that if its dimension is 1, then it could represent
# [a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a ...] with [a]
def collapse_vectorized_coefficient(x, N):
    """
    Original from weighted symbolic regression example:
    https://github.com/darioizzo/d-CGP

    by Dario Izzo, Francesco Biscani, Alessio Mereta.
    dCGPy is GPL licensed.
    """
    if len(x) == N:
        return sum(x)
    return x[0] * N


# Newton's method for minimizing the error function f w.r.t. the weights of the
# dCGP expression. We take a specified amount of steps, each by choosing
# randomly between n_min and n_max weights
def newton(ex, f, x, yt, steps, n_weights=[2, 3], randomize_weights=True):
    """
    Adapted from weighted symbolic regression example:
    https://github.com/darioizzo/d-CGP

    Originally from Dario Izzo, Francesco Biscani, Alessio Mereta.
    dCGPy is GPL licensed.
    """

    n = ex.get_n()
    r = ex.get_rows()
    c = ex.get_cols()
    a = ex.get_arity()
    #v = np.zeros(r * c * a)

    # random initialization of weights
    if randomize_weights:
        w=[]
        for i in range(r*c):
            for j in range(a):
                w.append(gdual([np.random.normal(0,1)]))
        ex.set_weights(w)
        #wi = ex.get_weights()

    # get active weights
    an = ex.get_active_nodes()
    is_active = [False] * (n + r * c) # bool vector of active nodes
    for k in range(len(an)):
        is_active[an[k]] = True
    aw=[] # list of active weights
    for k in range(len(an)):
        if an[k] >= n:
            for l in range(a):
                aw.append([an[k], l]) # pair node/ingoing connection
    # check that we have enough active weights
    if len(aw)<n_weights[0]:
        if len(aw) < 1:
            return
        # otherwise reset minimum number of updated weights
        n_weights[0] = len(aw)

    for i in range(steps):
        w = ex.get_weights() # initial weights
        # random choice of the weights w.r.t. which we'll minimize the error
        num_vars = np.random.randint(n_weights[0], min(n_weights[1], len(aw)) + 1) # number of weights
        awidx = np.random.choice(len(aw), num_vars, replace = False) # indexes of chosen weights
        ss = [] # symbols
        for j in range(len(awidx)):
            ss.append("w" + str(aw[awidx[j]][0]) + "_" + str(aw[awidx[j]][1]))
            idx = (aw[awidx[j]][0] - n) * a + aw[awidx[j]][1]
            w[idx] = gdual(w[idx].constant_cf, ss[j], 2)
        ex.set_weights(w)

        # compute the error
        E = f(ex, x, yt)
        Ei = sum(E.constant_cf)

        # get gradient and Hessian
        dw = np.zeros(len(ss))
        H = np.zeros((len(ss),len(ss)))
        for k in range(len(ss)):
            dw[k] = collapse_vectorized_coefficient(E.get_derivative({"d"+ss[k]: 1}), len(x[0].constant_cf))
            H[k][k] = collapse_vectorized_coefficient(E.get_derivative({"d"+ss[k]: 2}), len(x[0].constant_cf))
            for l in range(k):
                H[k][l] = collapse_vectorized_coefficient(E.get_derivative({"d"+ss[k]: 1, "d"+ss[l]: 1}), len(x[0].constant_cf))
                H[l][k] = H[k][l]

        det = np.linalg.det(H)
        if det == 0: # if H is singular
            continue

        # compute the updates
        updates = - np.linalg.inv(H) @ dw

        # update the weights
        for k in range(len(updates)):
            idx = (aw[awidx[k]][0] - n) * a + aw[awidx[k]][1]
            ex.set_weight(aw[awidx[k]][0], aw[awidx[k]][1], w[idx] + updates[k])
        wfe = ex.get_weights()
        for j in range(len(awidx)):
            idx = (aw[awidx[j]][0] - n) * a + aw[awidx[j]][1]
            wfe[idx] = gdual(wfe[idx].constant_cf)
        ex.set_weights(wfe)

        # if error increased restore the initial weights
        Ef = sum(f(ex, x, yt).constant_cf)
        if not Ef < Ei:
            for j in range(len(awidx)):
                idx = (aw[awidx[j]][0] - n) * a + aw[awidx[j]][1]
                w[idx] = gdual(w[idx].constant_cf)
            ex.set_weights(w)
