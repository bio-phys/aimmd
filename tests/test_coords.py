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
import mdtraj as md
import numpy as np
from aimmd.coords._symmetry import integrate_cos_binom
import aimmd.coords.symmetry as sym


class Test_symmetry:
    def test_cv(self):
        # setup topology
        top = md.Topology()

        cha = top.add_chain()
        res0 = top.add_residue('REACT', cha)
        top.add_atom('react', md.element.argon, res0, serial=0)

        chb = top.add_chain()
        res1 = top.add_residue('SOLV', chb)
        top.add_atom('solv1', md.element.arsenic, res1, serial=1)

        chc = top.add_chain()
        res2 = top.add_residue('SOLV', chb)
        top.add_atom('solv2', md.element.arsenic, res2, serial=2)

        chc = top.add_chain()
        res3 = top.add_residue('SOLV', chc)
        top.add_atom('solv3', md.element.arsenic, res3, serial=3)

        chd = top.add_chain()
        res4 = top.add_residue('SOLV', chd)
        top.add_atom('solv4', md.element.arsenic, res4, serial=4)

        # setup trajectory
        xyz = np.array([[[0., 0., 0.], [1., 0., 0.],
                         [1./np.sqrt(2), 1./np.sqrt(2), 0.],
                         [0., 1., 0.], [-1., 0., 0.]]])
        rs = [1., 1., 1., 1.]  # solvent distances from reactant atom
        rs_angs = [[1., 1., np.pi/4.], [1., 1., np.pi/2.], [1., 1., np.pi],
                   [1., 1., np.pi/4.], [1., 1., 3*np.pi/4],
                   [1., 1., np.pi/2.]]  # r1, r2, angle triples
        traj = md.Trajectory(xyz=xyz, topology=top)

        # symmetry function params
        mol_idx = np.array([0], dtype=np.int64)
        solv_idxs = [[np.array([1, 2, 3, 4], dtype=np.int64)]]
        zetas = [1, 2, 4, 8, 16, 32, 64, 128]
        # eta, r_s, zeta, lambda
        g_parms = [np.array([[1., 1.], [5., 1.], [1., 0.], [5., 0.]]),
                   np.array([[1., 1., z, l] for z in zetas for l in [+1, -1]]
                            + [[5., 1., z, l] for z in zetas for l in [+1, -1]]
                            + [[1., 0., z, l] for z in zetas for l in [+1, -1]]
                            + [[5., 0., z, l] for z in zetas for l in [+1, -1]]
                            )
                   ]
        cutoff = 10.
        n_per_solv = [[1.]]
        rho_solv = [1.]
        # we have only one frame
        test_vals = sym.transform(traj, mol_idx, solv_idxs, g_parms, cutoff,
                                  n_per_solv, rho_solv)[0]
        validation_vals = np.zeros((len(g_parms[0])+len(g_parms[1]),))
        g2_parms = g_parms[0]
        for i, parms in enumerate(g2_parms):
            for r in rs:
                validation_vals[i] += self.g_2(r, parms[0], parms[1])
        g5_parms = g_parms[1]
        for i, parms in enumerate(g5_parms):
            for r1, r2, ang in rs_angs:
                validation_vals[i+len(g2_parms)] += self.g_5(r1, r2, ang,
                                                             parms[0],
                                                             parms[1],
                                                             parms[2],
                                                             parms[3])

        assert np.allclose(test_vals, validation_vals)

    def g_2(self, r, eta, r_s, rho_solv=1., n_per_solv=1.):
        val = np.exp(-eta * (r - r_s)**2)
        if r_s == 0.:
            # it is a sphere around the origin
            number_correction = (rho_solv * n_per_solv * 4./3. * np.pi
                                 * (2. / (np.sqrt(2.*eta)))**2)
        else:
            number_correction = ((rho_solv * n_per_solv
                                  * 8. * np.sqrt(2) * np.pi * r_s**2)
                                 / np.sqrt(eta))
        return val/number_correction

    def g_5(self, r1, r2, ang, eta, r_s, zeta, lamb, rho_solv=1, n_per_solv=1.):
        val = (1. + lamb*np.cos(ang))**zeta * np.exp(-eta*((r1 - r_s)**2
                                                     + (r2 - r_s)**2)
                                                     )
        pow_zeta = 2.**(1. - zeta)  # prefactor for angular part
        if r_s == 0.:
            # it is some sort of sphere around the origin
            # angular part not yet integrated, 2/3 instead of 4/3
            number_correction = (rho_solv * n_per_solv * 2./3. * np.pi
                                 * (2. / (np.sqrt(2.*eta)))**2)
        else:
            number_correction = ((rho_solv * n_per_solv
                                  * 4. * np.sqrt(2) * r_s**2)
                                 / np.sqrt(eta))
        number_correction *= integrate_cos_binom(zeta)  # * pow_zeta
        # actualy we have n * (n - 1)/2 pairs/angles
        # but this approximation is guaranteed to be non-negative
        number_correction *= number_correction/2.
        val /= (number_correction*pow_zeta)
        return val
