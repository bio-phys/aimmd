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
from functools import reduce
import torch
import numpy as np


class HMC(torch.optim.Optimizer):
    """
    Hamiltonian Monte Carlo optimizer using leap-frog integration.

    Note that this optimizer 'only' makes sure that the neural network weights
    after every iteration are samples from the posterior distribution defined
    by interpreting the loss function as negative loglikelihood of the
    parameters given the data. It does not store these samples in any way.

    Parameters
    ----------
    params - torch model parameters as for other optimizers,
             NOTE: there can only be one parameter group
    lr - float, actually epsilon, the leap-frog stepsize
    tau - int, number of leap-frog steps
    weight_decay - float, strength of L2 normalization


    lr_func - function that returns epsilon for this MCstep when called
    tau_func - function that returns number of leap-frog steps when called
    NOTE: if lr_func and/or tau_func are given lr and/or tau are ignored

    """

    def __init__(self, params, lr=1e-3, tau=500, weight_decay=0,
                 lr_func=None, tau_func=None):
        # NOTE: lr is actually epsilon, i.e. the leapfrog step-size,
        # the corresponding lr for gradient descent would be lr=0.5 * epsilon**2
        # TODO: param reasonability checks?
        defaults = dict(lr=lr, tau=tau, weight_decay=weight_decay,
                        lr_func=lr_func, tau_func=tau_func
                        )
        super(HMC, self).__init__(params, defaults)
        # NOTE/TODO: only supports one param group! (same as for LBFGS)
        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    # same as for LBFGS
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda tot, p: tot + p.numel(), self._params, 0)
        return self._numel_cache

    # same as for LBFGS
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # for variance estimation and weight decay
    def _gather_flat_params(self):
        # using .data breaks autograd (which is intended)
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # TODO: atm this is obsolete,
    # as changing the mass matrix on the fly violates detailed balance...
    # maybe we can use this to estimate in a preliminary run,
    # store the estimate and then change the optimizer for production?
    def _update_variance_estimate(self, count, mean, sig2):
        """
        Online variance estimation.

        The sample variance can be calculated as sig2 / (count - 1)

        Returns
        -------
        count - number of samples
        means - sample means
        sig2 - accumulated squared distance from the mean

        """
        flat_params = self._gather_flat_params()
        count += 1
        delta = flat_params.sub(mean)
        mean.add_(1./count, delta)
        delta2 = flat_params.sub(mean)
        sig2.add_(delta.mul(delta2))
        return count, mean, sig2

    # TODO: also obsolete atm, see above!
    def _update_covariance_estimate(self, count, means, COV_mat):
        """
        Online covariance estimation.

        The covariance matrix can be calculated as COV_mat / (count - 1)

        Returns
        -------
        count - number of samples
        means - sample means
        COV_mat - accumulated comoments: \sum_i (x_i - \bar{x}) (y_i - \bar{y})

        """
        # TODO/FIXME: the covariances returned by this function are not exactly positive semi-definite
        flat_params = self._gather_flat_params()
        count += 1
        delta = flat_params.sub(means)  # dx with old means
        means.add_(1./count, delta)  # update means
        delta2 = flat_params.sub(means)  # dx with new means
        N = self._numel()
        for i in range(N):
            update = delta[i:].mul(delta2[i])
            # this updates the upper triangle + diagonal of the COV matrix
            COV_mat[i, i:].add_(update)
            # this updates only the lower triangle
            COV_mat[i+1:, i].add_(update[1:])
        return count, means, COV_mat

    # same as _add_grad for LBFGS
    def _weight_step(self, step_size, momenta):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(step_size, momenta[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        assert len(self.param_groups) == 1  # only one param group!
        weight_decay = self.param_groups[0]['weight_decay']
        eps_func = self.param_groups[0]['lr_func']
        tau_func = self.param_groups[0]['tau_func']
        if eps_func is not None:
            epsilon = eps_func()
        else:
            epsilon = self.param_groups[0]['lr']
        eps_half = epsilon / 2.
        if tau_func is not None:
            tau = tau_func()
        else:
            tau = self.param_groups[0]['tau']
        # register global state as state for first param
        # the reasoning is the same as for LBFGS: it helps load_state_dict
        state = self.state[self._params[0]]
        # initialize state if in first step
        state.setdefault('hamilton_steps_tot', 0)
        state.setdefault('accepts', [])
        state.setdefault('dHs', [])
        state.setdefault('Hs', [])
        state.setdefault('Ts', [])
        state.setdefault('Vs', [])
        steps_tot = state.setdefault('steps_tot', 0)
        steps_tot += 1
        # store current weights (using .data breaks autograd which is intended)
        initial_params = [torch.empty_like(p.data).copy_(p.data) for p in self._params]
        # get initial loss and gradients
        initial_loss = closure()
        flat_grad = self._gather_flat_grad()
        if weight_decay != 0:
            flat_params = self._gather_flat_params()
            flat_grad.add_(weight_decay, flat_params)

        # initialize momenta and calculate initial H
        momenta = torch.empty_like(flat_grad).normal_(mean=0, std=1)
        T = float(momenta.dot(momenta)) / 2.
        V = float(initial_loss)
        if weight_decay != 0:
            V += weight_decay * float(flat_params.dot(flat_params)) / 2
        H = T + V
        # simple leap-frog algorithm in phase space
        # one half step in momenta before loop,
        # such that we can make full steps in the loop
        momenta.sub_(eps_half, flat_grad)
        for t in range(1, tau+1):
            # full step in weights
            # unit masses -> momenta = velocities
            self._weight_step(epsilon, momenta)
            # get new gradients
            loss = closure()
            flat_grad = self._gather_flat_grad()
            if weight_decay != 0:
                flat_params = self._gather_flat_params()
                flat_grad.add_(weight_decay, flat_params)
            if t < tau:
                # full step in momenta except if in last iteration
                momenta.sub_(epsilon, flat_grad)
        # do half step in momenta at the end
        momenta.sub_(eps_half, flat_grad)
        # reevaluate H: accept/reject
        T_new = float(momenta.dot(momenta)) / 2.
        V_new = float(loss)
        if weight_decay != 0:
            V_new += weight_decay * float(flat_params.dot(flat_params)) / 2

        H_new = T_new + V_new
        dH = H_new - H
        if dH <= 0:
            accept = 1
        elif np.random.ranf() < np.exp(-dH):
            accept = 1
        else:
            accept = 0
        # bookkeeping and setting correct state for next step
        state['steps_tot'] = steps_tot
        state['accepts'].append(accept)
        state['dHs'].append(dH)
        state['Hs'].append([H, H_new])
        state['Ts'].append([T, T_new])
        state['Vs'].append([V, V_new])
        if accept:
            # accept
            state['hamilton_steps_tot'] += tau
            loss_out = loss
        else:
            # reject
            # reset weights
            with torch.no_grad():
                for ip, p in zip(initial_params, self._params):
                    p.copy_(ip)
            loss_out = initial_loss

        return loss_out
