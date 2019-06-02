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


    lr_func - function that returns epsilon for this MCstep when called
    tau_func - function that returns number of leap-frog steps when called
    NOTE: if lr_func and/or tau_func are given lr and/or tau are ignored

    """

    def __init__(self, params, lr=1e-3, tau=500, weight_decay=0, mass_est=False,
                 mass_est_min_samples=10, mass_est_sample_interval=1,
                 lr_func=None, tau_func=None):
        # NOTE: lr is actually epsilon, i.e. the leapfrog step-size, the corresponding lr in a simple gradient descent setting is lr=0.5 * epsilon**2
        # TODO: param reasonability checks?
        defaults = dict(lr=lr, tau=tau, weight_decay=weight_decay,
                        mass_est=mass_est,
                        mass_est_min_samples=mass_est_min_samples,
                        mass_est_sample_interval=mass_est_sample_interval,
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
        assert len(self.param_groups) == 1
        # TODO/FIXME: changing the masses on the fly violates detailed balance
        # at least if we do not account for it in the accept/reject
        mass_est = self.param_groups[0]['mass_est']
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
        if mass_est:
            # get current mass estimate
            mass_est_min_samples = self.param_groups[0]['mass_est_min_samples']
            mass_est_sample_interval = self.param_groups[0]['mass_est_sample_interval']
            inv_mass_mat = state.get('inv_mass_mat', None)
            if inv_mass_mat is None:
                # make sure we always have masses, even without an estimate
                inv_mass_mat = torch.diagflat(torch.ones(self._numel(), dtype=flat_grad.dtype, device=flat_grad.device))
                state['inv_mass_mat'] = inv_mass_mat

        # initialize momenta and calculate initial H
        momenta = torch.empty_like(flat_grad).normal_(mean=0, std=1)
        # TODO/FIXME: it seems that without the correction it works better?
        # -> the issue here were probably the non positive definite covariance/mass matrices
        if mass_est:
            T = float(momenta.dot(inv_mass_mat.matmul(momenta))) / 2.
        else:
            T = float(momenta.dot(momenta)) / 2.
        
        V = float(initial_loss)
        if weight_decay != 0:
            V += weight_decay * float(flat_params.dot(flat_params)) / 2

        H = T + V
        for t in range(tau):
            # leap-frog algorithm in phase space
            # half-step in momenta
            momenta.sub_(eps_half, flat_grad)
            # full step in weights
            if mass_est:
                self._weight_step(epsilon, inv_mass_mat.matmul(momenta))
            else:
                # unit masses -> momenta = velocities
                self._weight_step(epsilon, momenta)
            # get new gradients
            loss = closure()
            flat_grad = self._gather_flat_grad()
            if weight_decay != 0:
                flat_params = self._gather_flat_params()
                flat_grad.add_(weight_decay, flat_params)
            # other half-step in momenta
            momenta.sub_(eps_half, flat_grad)
        
        # reevaluate H: accept/reject
        if mass_est:
            T_new = float(momenta.dot(inv_mass_mat.matmul(momenta))) / 2.
        else:
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
            out = loss
        else:
            # reject
            # reset weights
            with torch.no_grad():
                for ip, p in zip(initial_params, self._params):
                    p.copy_(ip)
            out = initial_loss
        # estimate weight covariance/mass matrix at the end of each HMC step, should give a better sample for the first round?!
        if mass_est:
            if steps_tot % mass_est_sample_interval == 0:
                # on the fly estimate of variance of weights to set as inverse masses
                me_count = state.get('mass_est_count', 0)
                me_mean = state.get('mass_est_mean', None)
                me_COV_mat = state.get('mass_est_COV_mat', None)
                #me_sig2 = state.get('mass_est_sig2', None)
                if me_mean is None:
                    me_mean = torch.zeros_like(flat_grad)
                    me_COV_mat = torch.zeros_like(inv_mass_mat)
                    #me_sig2 = torch.zeros_like(flat_grad)
                me_count, me_mean, me_COV_mat = self._update_covariance_estimate(me_count, me_mean, me_COV_mat)
                #me_count, me_mean, me_sig2 = self._update_mass_estimate(me_count, me_mean, me_sig2)
                if me_count >= mass_est_min_samples:
                    # TODO/FIXME:
                    # actually mass matrices are positive semi-definite!
                    # covariance matrices should be so too, but ours are not exactly, because of numerics? a bug?
                    # for now we impose positive semi-definititeness by decomposition, setting alls lambda < 0 to 0 and then recomposition
                    #lambdas, Q = torch.symeig(me_COV_mat / (me_count - 1), eigenvectors=True)
                    #lambdas[lambdas < 0.] = 0.
                    # update mass estimate inplace
                    #inv_mass_mat.copy_(Q.matmul(torch.diag(lambdas).matmul(Q.t())))
                    inv_mass_mat.copy_(me_COV_mat / (me_count - 1))
                    #inv_mass_mat = me_sig2.diagflat() / (me_count - 1)
                    state['inv_mass_mat'] = inv_mass_mat
                # store current estimates
                state['mass_est_count'] = me_count
                state['mass_est_mean'] = me_mean
                #tate['mass_est_sig2'] = me_sig2        
                state['mass_est_COV_mat'] = me_COV_mat
        
        return out
