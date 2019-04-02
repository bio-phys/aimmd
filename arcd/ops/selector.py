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
from openpathsampling.shooting import ShootingPointSelector


logger = logging.getLogger(__name__)


class RCModelSelector(ShootingPointSelector):
    """
    Select 'ideal' shooting points learned by the corresponding model.

    The points are 'ideal' in the sense that p(TP|x) is maximal there.
    Can be used together with `CommittorModelTrainer`.

    Parameters
    ----------
    model - :class:`arcd.base.RCModel` a wrapped model predicting RC values
    states - list of :class:`openpathsampling.Volume`, one for each state
    distribution - string specifying the SP selection distribution,
                   either 'gaussian' or 'lorentzian'
                   'gaussian': p_{sel}(x) ~ exp(-alpha * z_{sel}(x)**2)
                   'lorentzian': p_{sel}(x) ~ gamma**2
                                              / (gamma**2 + z_{sel}(x)**2)
    scale - float, 'softness' parameter of the selection distribution,
            higher values result in a boader spread of SPs around the TSE,
            1/alpha for 'gaussian' and gamma for 'lorentzian'

    Notes
    -----
    We use the z_sel function of the RCModel as input for the SP selection.

    """

    def __init__(self, model, states=None,
                 distribution='lorentzian', scale=1.):
        """Initialize a RCModelSelector."""
        super(RCModelSelector, self).__init__()
        self.model = model
        if states is None:
            logger.warning('Consider passing the states to speed up'
                           + ' accepting/rejecting.')
        self.states = states
        self.distribution = distribution
        self.scale = scale

    @classmethod
    def from_dict(cls, dct):
        """Will be called by OPS when loading self from storage."""
        # atm we set model = None,
        # since we can not arbitrary models in OPS storages
        # but if used with an arcd.TrainingHook it will (re)set the model
        # to the one saved besides the OPS storage
        obj = cls(None,
                  dct['states'],
                  distribution=dct['distribution'],
                  scale=dct['scale'])
        logger.warning('Restoring RCModelSelector without model.'
                       + 'If used together with arcd.TrainingHook you can '
                       + 'ignore this warning, otherwise please take care of '
                       + 'resetting the model yourself.')
        return obj

    def to_dict(self):
        """Will be called by OPS when saving self to storage."""
        dct = {}
        dct['distribution'] = self._distribution
        dct['scale'] = self.scale
        dct['states'] = self.states
        return dct

    @property
    def distribution(self):
        """Return the name of the shooting point selection distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, val):
        if val == 'gaussian':
            self._f_sel = lambda z: self._gaussian(z)
            self._distribution = val
        elif val == 'lorentzian':
            self._f_sel = lambda z: self._lorentzian(z)
            self._distribution = val
        else:
            raise ValueError('Distribution must be one of: '
                             + '"gaussian" or "lorentzian"')

    def _lorentzian(self, z):
        return self.scale / (self.scale**2 + z**2)

    def _gaussian(self, z):
        return np.exp(-z**2/self.scale)

    def f(self, snapshot, trajectory):
        """Return the unnormalized proposal probability of a snapshot."""
        z_sel = self.model.z_sel(snapshot)
        if not np.all(np.isfinite(z_sel)):
            logger.warning('The model predicts NaNs or infinities. '
                           + 'We used np.nan_to_num to proceed')
            z_sel = np.nan_to_num(z_sel)
        # casting to python float solves the problem that
        # metropolis_acceptance is not saved !
        ret = float(self._f_sel(z_sel))
        if ret == 0.:
            if self.sum_bias(trajectory) == 0.:
                return 1.
        return ret

    def probability(self, snapshot, trajectory):
        """Return proposal probability of the snapshot for this trajectory."""
        # only evaluate costly symmetry functions if needed,
        # if trajectory is no TP it has weight 0 and p_pick = 0 for all points
        self_transitions = [1 < sum(s(p)
                                    for p in [trajectory[0], trajectory[-1]])
                            for s in self.states]
        if any(self_transitions):
            return 0.
        # trajectory is a TP since it is no self-transition, calculate p_pick
        sum_bias = self.sum_bias(trajectory)
        if sum_bias == 0.:
            return 1./len(trajectory)
        return self.f(snapshot, trajectory) / sum_bias

    def sum_bias(self, trajectory):
        """
        Return the partition function of proposal probabilities for trajectory.
        """
        # casting to python float solves the problem that
        # metropolis_acceptance is not saved !
        return float(np.sum(self._biases(trajectory)))

    def _biases(self, trajectory):
        z_sels = self.model.z_sel(trajectory)
        if not np.all(np.isfinite(z_sels)):
            logger.warning('The model predicts NaNs or infinities. '
                           + 'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        return self._f_sel(z_sels)

    def pick(self, trajectory):
        """Return the index of the chosen snapshot within trajectory."""
        biases = self._biases(trajectory)
        sum_bias = np.sum(biases)
        if sum_bias == 0.:
            logger.error('Model not able to give educated guess.\
                         Choosing based on luck.')
            # we can not give any meaningfull advice and choose at random
            return np.random.randint(len(trajectory))

        rand = np.random.random() * sum_bias
        idx = 0
        prob = biases[0]
        while prob <= rand and idx < len(biases):
            idx += 1
            prob += biases[idx]

        # let the model know which SP we chose
        self.model.register_sp(trajectory[idx])
        return idx
