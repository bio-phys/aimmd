"""
This file is part of AIMMD. This file also contains code adapted from
the dCGPy examples (https://github.com/darioizzo/d-CGP) originally
from Dario Izzo, Francesco Biscani, Alessio Mereta.
dCGPy is GPL licensed.

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
import math
import logging
import numpy as np
import pyaudi as ad
from pyaudi import gdual_vdouble as gdual
from .losses import _get_active_weights


logger = logging.getLogger(__name__)


def optimize_expression(expression, offsprings, max_gen, xt, yt, loss_function,
                        complexity_regularization=None,
                        weight_regularization=None,
                        max_gen_sans_improvement=500,
                        newtonParams={'steps': 500},
                        keep_weights=False):
    """
Optimizes the given expression INPLACE.

xt and yt are directly passed to the loss function. In most cases, xt should be
a list of pyaudi weighted gduals for the gradient calculations, while yt can be
a numpy array contaning 'just' reference values. The loss function must take 3
parameters: the dCGPy expression, xt and yt (in that order).
complexity regularization and weight regularization must take only the expression
as input and return a loss value. They are added to the loss but evaluated only when
needed, i.e. complexity regularization once after every mutation and weight
regularization for every newton step.

Adapted from weighted symbolic regression example:
https://github.com/darioizzo/d-CGP

Originally from Dario Izzo, Francesco Biscani, Alessio Mereta.
dCGPy is GPL licensed.
    """
    # build loss functions dict
    # the rationale is:
    # complexity regularization is a term that does not depend on the
    # value of the weights, it is constant for one specific form of expression,
    # therefore we can ignore it when doing the newton steps
    # weight regularization depends on the value of the weights and we do newton steps
    # to decrease both the loss and regularization value simulataneously,
    # therefore weight regularizations take the currently optimized weights as pyaudi numbers
    # to get the derivatives
    loss_functions = {}
    if complexity_regularization and weight_regularization:
        loss_functions['full'] = lambda ex, x, y, aw: (sum(loss_function(ex, x, y).constant_cf)
                                                       + complexity_regularization(ex)
                                                       + weight_regularization(aw)
                                                       )
        loss_functions['newton'] = lambda ex, x, y, aw: (loss_function(ex, x, y)
                                                         + weight_regularization(aw) / len(x[0].constant_cf)
                                                         )
    elif complexity_regularization:
        loss_functions['full'] = lambda ex, x, y, aw: (sum(loss_function(ex, x, y).constant_cf)
                                                       + complexity_regularization(ex)
                                                       )
        loss_functions['newton'] = lambda ex, x, y, aw: loss_function(ex, x, y)
    elif weight_regularization:
        loss_functions['full'] = lambda ex, x, y, aw: (sum(loss_function(ex, x, y).constant_cf)
                                                       + weight_regularization(aw)
                                                       )
        loss_functions['newton'] = lambda ex, x, y, aw: (loss_function(ex, x, y)
                                                         + weight_regularization(aw) / len(x[0].constant_cf)
                                                         )
    else:
        loss_functions['full'] = lambda ex, x, y, aw: sum(loss_function(ex, x, y).constant_cf)
        loss_functions['newton'] = lambda ex, x, y, aw: loss_function(ex, x, y)

    # The offsprings chromosome, loss and weights
    chromosome = [None] * offsprings
    loss = [None] * offsprings
    weights = [None] * offsprings
    gens_sans_improvement = 0
    # Init the best as the initial expression
    best_chromosome = expression.get()
    best_weights = expression.get_weights()
    aw_list = _get_active_weights(expression)
    best_loss = loss_functions['full'](expression, xt, yt, aw_list)
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
                newton(expression, loss_functions['newton'], xt, yt, **new_parms)
            else:
                newton(expression, loss_functions['newton'], xt, yt, **newtonParams)
            # get the loss
            aw_list = _get_active_weights(expression)
            loss[i] = loss_functions['full'](expression, xt, yt, aw_list)
            chromosome[i] = expression.get()
            weights[i] = expression.get_weights()
        improvement = False  # whether we see any improvement in the fitness for any offspring
        for i in range(offsprings):
            if not math.isnan(loss[i]) and loss[i] <= best_loss:
                expression.set(chromosome[i])
                expression.set_weights(weights[i])
                # check if the old and new best chromosome are the same
                # if yes we will +1 gens_sans_improvement
                same_chromosome = True
                for bc, c in zip(best_chromosome, chromosome[i]):
                    if bc != c:
                        same_chromosome = False
                        break
                for bw, w in zip(best_weights, weights[i]):
                    if bw != w:
                        same_chromosome = False
                        break
                if not same_chromosome:
                    best_chromosome = chromosome[i]
                    best_loss = loss[i]
                    best_weights = weights[i]
                    improvement = True
        if improvement:
            gens_sans_improvement = 0
        else:
            gens_sans_improvement += 1
        # check if we terminate because we have not seen any improvement for too long
        if gens_sans_improvement >= max_gen_sans_improvement:
            print("Terminating optimization because the fitness has not improved for "
                  + f"{gens_sans_improvement} generations (>= {max_gen_sans_improvement}).")
            break

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
    ar = ex.get_arity()
    #v = np.zeros(r * c * a)

    # random initialization of weights
    if randomize_weights:
        w = []
        for i in range(r*c):
            if isinstance(ar, list):
                a = ar[i // r]  # arity of the current column
            else:
                a = ar
            for j in range(a):
                w.append(gdual([np.random.normal(0,1)]))
        ex.set_weights(w)
        #wi = ex.get_weights()

    # get active weights
    an = ex.get_active_nodes()
    is_active = [False] * (n + r * c) # bool vector of active nodes
    for k in range(len(an)):
        is_active[an[k]] = True
    aw = [] # list of active weights
    for k in range(len(an)):
        if an[k] >= n:
            if isinstance(ar, list):
                # arity list starts after the input nodes
                a = ar[(an[k] - n) // r]
            else:
                a = ar
            for l in range(a):
                aw.append([an[k], l]) # pair node/ingoing connection
    # check that we have enough active weights
    if len(aw) < n_weights[0]:
        if len(aw) < 1:
            return
        # otherwise reset minimum number of updated weights
        n_weights[0] = len(aw)

    for i in range(steps):
        w = ex.get_weights()  # initial weights
        # random choice of the weights w.r.t. which we'll minimize the error
        num_vars = np.random.randint(n_weights[0], min(n_weights[1], len(aw)) + 1) # number of weights
        awidx = np.random.choice(len(aw), num_vars, replace=False) # indexes of chosen weights
        ss = []  # symbols
        opt_weights = []
        for j in range(len(awidx)):
            ss.append("w" + str(aw[awidx[j]][0]) + "_" + str(aw[awidx[j]][1]))
            if isinstance(ar, list):
                # the number of weights before the current weight is given by the sum of the arities in the cols before
                # plus the sum of the arities of the nodes in the current col up to current row/node
                n_previous = (sum(ar[:(aw[awidx[j]][0] - n) // r]) * r
                              + ar[(aw[awidx[j]][0] - n) // r] * ((aw[awidx[j]][0] - n) % r)
                              )
                idx = n_previous + aw[awidx[j]][1]
            else:
                # arity is the same for all nodes,
                # number of previous weights is simply numebr of nodes times arity
                idx = (aw[awidx[j]][0] - n) * a + aw[awidx[j]][1]
            w[idx] = gdual(w[idx].constant_cf, ss[j], 2)
            opt_weights.append(w[idx])
        ex.set_weights(w)

        # compute the error
        E = f(ex, x, yt, opt_weights)
        Ei = sum(E.constant_cf)

        # get gradient and Hessian
        dw = np.zeros(len(ss))
        H = np.zeros((len(ss), len(ss)))
        for k in range(len(ss)):
            dw[k] = collapse_vectorized_coefficient(E.get_derivative({"d"+ss[k]: 1}), len(x[0].constant_cf))
            H[k][k] = collapse_vectorized_coefficient(E.get_derivative({"d"+ss[k]: 2}), len(x[0].constant_cf))
            for l in range(k):
                H[k][l] = collapse_vectorized_coefficient(E.get_derivative({"d"+ss[k]: 1, "d"+ss[l]: 1}), len(x[0].constant_cf))
                H[l][k] = H[k][l]

        det = np.linalg.det(H)
        if det == 0:  # if H is singular
            continue

        # compute the updates
        updates = - np.linalg.inv(H) @ dw

        # update the weights
        for k in range(len(updates)):
            if isinstance(ar, list):
                # the number of weights before the current weight is given by the sum of the arities in the cols before
                # plus the sum of the arities of the nodes in the current col up to current row/node
                n_previous = (sum(ar[:(aw[awidx[k]][0] - n) // r]) * r
                              + ar[(aw[awidx[k]][0] - n) // r] * ((aw[awidx[k]][0] - n) % r)
                              )
                idx = n_previous + aw[awidx[k]][1]
            else:
                # arity is the same for all nodes,
                # number of previous weights is simply numebr of nodes times arity
                idx = (aw[awidx[k]][0] - n) * a + aw[awidx[k]][1]
            ex.set_weight(aw[awidx[k]][0], aw[awidx[k]][1], w[idx] + updates[k])
        wfe = ex.get_weights()
        for j in range(len(awidx)):
            if isinstance(ar, list):
                # the number of weights before the current weight is given by the sum of the arities in the cols before
                # plus the sum of the arities of the nodes in the current col up to current row/node
                n_previous = (sum(ar[:(aw[awidx[j]][0] - n) // r]) * r
                              + ar[(aw[awidx[j]][0] - n) // r] * ((aw[awidx[j]][0] - n) % r)
                              )
                idx = n_previous + aw[awidx[j]][1]
            else:
                # arity is the same for all nodes,
                # number of previous weights is simply numebr of nodes times arity
                idx = (aw[awidx[j]][0] - n) * a + aw[awidx[j]][1]
            wfe[idx] = gdual(wfe[idx].constant_cf)
        ex.set_weights(wfe)

        # if error increased restore the initial weights
        Ef = sum(f(ex, x, yt, wfe).constant_cf)
        if not Ef < Ei:
            for j in range(len(awidx)):
                if isinstance(ar, list):
                    # the number of weights before the current weight is given by the sum of the arities in the cols before
                    # plus the sum of the arities of the nodes in the current col up to current row/node
                    n_previous = (sum(ar[:(aw[awidx[j]][0] - n) // r]) * r
                                  + ar[(aw[awidx[j]][0] - n) // r] * ((aw[awidx[j]][0] - n) % r)
                                  )
                    idx = n_previous + aw[awidx[j]][1]
                else:
                    # arity is the same for all nodes,
                    # number of previous weights is simply numebr of nodes times arity
                    idx = (aw[awidx[j]][0] - n) * a + aw[awidx[j]][1]
                w[idx] = gdual(w[idx].constant_cf)
            ex.set_weights(w)
