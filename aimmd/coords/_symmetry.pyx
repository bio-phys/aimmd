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
cimport cython
cimport numpy as cnp
from libc cimport math
from cpython cimport list

ctypedef cnp.float64_t float64_t
ctypedef cnp.float32_t float32_t
ctypedef cnp.int64_t int64_t
ctypedef cnp.int32_t int32_t

import numpy as np
import mdtraj as md
from cython.parallel import prange


def sf(mdtra, mol_idxs, solv_idxs, g_parms, cutoff,
       n_per_solv, rho_solv=[33.], alpha_cutoff=150.):
    """
    Calculate the values of a set of symmetry functions for each reactant atom.
    Calculates g2 and g5.

    See "Atom-centered symmetry functions for constructing high-dimensional
        neural network potentials" J. Behler, J. Chem. Phys. 134, 074106 (2011)
    and "Neural networks for local structure detection in polymorphic systems",
        P. Geiger and C. Dellago J. Chem. Phys. 139, 164105 (2013)
    for theory and nomenclature.
    We use here the fermi cutoff from the second paper and modified g5
    in our own way, namely we use (r_ik/ij - r_s)**2 instead of (r_ik/ij)**2

    Parameters
    ----------
    mdtra - :class:`mdtraj.Trajectory` object
    mol_idxs, solv_idxs - output of `symmetry.generate_indices()`
    g_parms - iterable of g2_parms, g5_parms
        g2_parms - list of eta, r_s values
        g5_parms - list of eta, r_s, zeta, lambda value quadrouples
    cutoff - float cutoff radius in nm
    n_per_solv - list of lists of float, how many atoms of that kind there are
                 per solvent molecule
    rho_solv - list of number density values for solvent molecules in 1/nm**3
    alpha_cutoff - float measure of 'abruptness' of cutoff

    Results
    -------
    np.array of symmetry function values for each reactant atom,
    shape = (n_frames, n_atoms_reactant * n_symmetry_funcs),
    where n_symmetry_funcs = number_of_solvent_atom_types
                            * (len(g2_parms) + len(g5_parms)),
    Second axis structure:
        [solv_resname_1[solv_type_1[reactant_atom_1[g20, g21, ..., g5n]]],
                         ...,
        solv_resname_1[solv_type_1[reactant_atom_n[g20, g21, ..., g5n]]],
        solv_resname_1[solv_type_2[reactant_atom_1[g20, g21, ..., g5n]]],
                        ...,
        solv_resname_1[solv_type_n[reactant_atom_n[g20, g21, ..., g5n]]]
                        ...,
        solv_resname_n[solv_type_n[reactant_atom_n[g20, g21, ..., g5n]]]
        ],
    """
    # TODO: check inputs for consistency!
    # i.e. g2/g5 params shape, n_per_solv compatible with solv_idxs shape, etc
    return np.concatenate([symmetry_functions_by_solv(mdtra, mol_idxs,
                                                      solv_by_typ,
                                                      g_parms, cutoff,
                                                      n_per_solv[j][i],
                                                      rho_solv[j],
                                                      alpha_cutoff)
                           for j, solv_by_molname in enumerate(solv_idxs)
                           for i, solv_by_typ in enumerate(solv_by_molname)
                           ], axis=1)


@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef cnp.ndarray[float64_t, ndim=2] symmetry_functions_by_solv(
                                        mdtra,
                                        cnp.ndarray[int64_t, ndim=1] mol_idxs,
                                        cnp.ndarray[int64_t, ndim=1] solv_idxs,
                                        list g_parms, float64_t cutoff,
                                        float64_t n_per_solv,
                                        float64_t rho_solv,
                                        float64_t alpha_cutoff):
    """
    Calculates symmetry functions for one solvent type.

    Parameters
    ----------
    mdtra - :class:`mdtraj.Trajectory` object
    mol_idxs, solv_idx - 1d numpy.ndarray, dtype=int64,
                         i.e. 2 mdtraj selection arrays
    g_parms - iterable of g2_parms, g5_parms
        g2_parms - list of eta, r_s values
        g5_parms - list of eta, r_s, zeta, lambda value quadrouples
    cutoff - float cutoff radius in nm
    n_per_solv - how many atoms of that kind there are per solvent molecule
    rho_solv - number density of solvent molecules in 1/nm**3
    alpha_cutoff - float measure of 'abruptness' of cutoff
    """
    # local variables
    cdef Py_ssize_t n_frames = mdtra.n_frames
    cdef Py_ssize_t n_mol = mol_idxs.shape[0]  # total number of reactant atoms
    cdef Py_ssize_t n_solv  # number of solvent molecules in cutoff in frame f
    cdef Py_ssize_t n_g2, n_g4, n_g5, n_sf
    cdef cnp.ndarray[float64_t, ndim=2] g2_parms, g5_parms
    cdef float64_t eta, r_s, zeta, lamb  # unpacked g2/5 parms
    # TODO: FIXME: if gx_parms.shape[1] is not correct this has
    # funny out of bounds reads as a consequence...
    # reults in no error but very wrong values!
    g2_parms, g5_parms = (np.array(parms) for parms in g_parms)
    n_g2 = g2_parms.shape[0]
    n_g5 = g5_parms.shape[0]
    n_sf = n_g2 + n_g5

    cdef float64_t r_c_fermi = cutoff - 1./math.sqrt(alpha_cutoff)
    cdef cnp.ndarray[int64_t, ndim=1] solv_in_cutoff_f
    cdef cnp.ndarray[int32_t, ndim=2] pairs, intra_pairs, triples
    cdef cnp.ndarray[float32_t, ndim=2] r_inter, r_intra, angles

    cdef float32_t Rij, Rik, Rjk, Aijk
    cdef float64_t cos_val, cos_term, f_cut_ij, f_cut_ik, f_cut_jk, exp_val_g5

    # loop variables
    cdef Py_ssize_t m, f, s_at # solvent_typ, reactant_molecule, frame, solvent_atom
    cdef Py_ssize_t i, j, k # use i for gX_param set number, j and k for solvent atoms
    cdef Py_ssize_t offset # used for getting ijk angle and jk intra-solv distance indexs
    cdef Py_ssize_t n_intra # number of possible unique intra solvent pairs (and also solv-react-solv triples)

    # output
    cdef cnp.ndarray[float64_t, ndim=2] output = np.zeros((n_frames, n_mol*n_sf), dtype=np.float64)

    # get indices of all atoms that are closer than cutoff to any reactant atom
    cdef list solv_in_cutoff = md.compute_neighbors(traj=mdtra, cutoff=cutoff,
                                                    query_indices=mol_idxs,
                                                    haystack_indices=solv_idxs)

    for f in range(n_frames):
        solv_in_cutoff_f = solv_in_cutoff[f]
        n_solv = solv_in_cutoff_f.shape[0]
        n_intra = int(n_solv * (n_solv - 1) / 2)
        # make solvent-reactant pairs, solvent-reactant-solvent triples
        # and make solvent-solvent intra_pairs
        pairs = mdtra.topology.select_pairs(mol_idxs, solv_in_cutoff_f)
        intra_pairs = mdtra.topology.select_pairs(solv_in_cutoff_f, solv_in_cutoff_f)

        # compute distances and angles
        r_inter = md.compute_distances(mdtra[f], pairs)
        r_intra = md.compute_distances(mdtra[f], intra_pairs)
        # calculate angles from law of cosines !!

        # loop over all atoms to calculate symmetry function values
        for m in prange(n_mol, nogil=True, schedule='static'):
            for j in range(n_solv):
                Rij = r_inter[0, m*n_solv + j]
                if Rij <= cutoff:
                    f_cut_ij = 1./(1. + math.exp(alpha_cutoff*(Rij - r_c_fermi)))
                    # G2 loop
                    for i in range(n_g2):
                        eta = g2_parms[i, 0]
                        r_s = g2_parms[i, 1]
                        output[f, m*n_sf + i] += f_cut_ij * math.exp(-eta * (Rij - r_s) * (Rij - r_s))
                    # END G2 loop

                    # G5 angular symmetry func
                    # calculate offset = sum_over_j(n_solv - 1 - j)
                    # = j*n_solv - j - sum_over_j(j)
                    # = j*n_solv - j - j*(j-1)/2
                    # it puts us to the index where atoms jk pairs/triples start
                    offset = j*n_solv - j - (j*(j-1)/2)
                    for k in range(j+1, n_solv):
                        Rik = r_inter[0, m*n_solv + k]
                        # k starts at j+1, but we want to count from zero to j
                        # so we use k - j - 1
                        Rjk = r_intra[0, offset + k - j - 1]
                        if Rik <= cutoff:
                            # we have non-zero G5 contrib
                            cos_val = ((Rij*Rij + Rik*Rik - Rjk*Rjk)
                                       / (2. * Rij * Rik))  # law of cosines
                            f_cut_ik = 1./(1. + math.exp(alpha_cutoff*(Rik - r_c_fermi)))
                            # G5 loop
                            for i in range(n_g5):
                                eta = g5_parms[i, 0]
                                r_s = g5_parms[i, 1]
                                zeta = g5_parms[i, 2]
                                lamb = g5_parms[i, 3]
                                cos_term = (1. + lamb * cos_val)**zeta
                                exp_val_g5 = math.exp(-eta*((Rij - r_s) * (Rij - r_s)
                                                            + (Rik - r_s) * (Rik - r_s)
                                                            )
                                                      )
                                output[f, m*n_sf + n_g2 + i] += (cos_term * exp_val_g5
                                                                 * f_cut_ij * f_cut_ik)
                            # END G5 loop
                    # END k loop
            # END j LOOP
        # END m loop
    # END f loop
    # volume scaling and prefactors for g4 and g5
    cdef float64_t fact
    cdef float64_t number_correction, sqrt_eta
    cdef float64_t pow_zeta
    for i in range(n_g2):
        eta = g2_parms[i, 0]
        r_s = g2_parms[i, 1]
        #sqrt_eta = math.sqrt(eta)
        #number_correction = ((math.sqrt(math.pi)*(2. * eta * r_s**2 + 1.)*(math.erf(sqrt_eta * r_s) + 1.))
        #                     / (4. * sqrt_eta**3)
        #                     + (r_s * math.exp(-eta * r_s**2)) / (2. * eta)
        #                     )  # radial integral 0 -> inf
        #number_correction *= 4. * math.pi  # angular integral
        #number_correction *= rho_solv * n_per_solv
        # correction factor for water number density in nm
        # rho ~ 33 /nm**3 for water
        # V ~ 4 pi * r_s**2 * width, i.e. a midpoint approximation of the integral
        # width ~ 4 sigma = 4 sqrt(1/(2*eta)) because it is a gaussian
        if r_s == 0.:
            # it is a sphere around the origin
            number_correction = (rho_solv * n_per_solv * 4./3. * math.pi
                                 * (2. / (math.M_SQRT2 * math.sqrt(eta)))**2)
        else:
             number_correction = ((rho_solv * n_per_solv
                                   * 8. * math.M_SQRT2 * math.pi * r_s**2)
                                  / math.sqrt(eta))
        output[:, i::n_sf] /= number_correction

    for i in range(n_g5):
        eta = g5_parms[i, 0]
        r_s = g5_parms[i, 1]
        zeta = g5_parms[i, 2]
        # correction factor for solvent number density in nm
        # rho_water ~ 33 /nm**3
        # V ~ 2 * r_s**2 * width * integral[(cos(phi) + 1)**zeta]_0_to_2pi
        # i.e. a midpoint approximation of the r integral
        # and the angle integral solved
        # width ~ 4 sigma = 4 sqrt(1/(2*eta)) because it is a gaussian
        pow_zeta = math.exp2(1. - zeta) # prefactor for angular part
        # volume correction :
        if r_s == 0.:
            # it is some sort of sphere around the origin
            # angular part not yet integrated, 2/3 instead of 4/3
            number_correction = (rho_solv * n_per_solv * 2./3. * math.pi
                                 * (2. / (math.M_SQRT2 * math.sqrt(eta)))**2)
        else:
            number_correction = ((rho_solv * n_per_solv
                                  * 4. * math.M_SQRT2 * r_s**2)
                                 / math.sqrt(eta))

        #sqrt_eta = math.sqrt(eta)
        #number_correction = ((math.sqrt(math.pi)*(2. * eta * r_s**2 + 1.)*(math.erf(sqrt_eta * r_s) + 1.))
        #                     / (4. * sqrt_eta**3)
        #                     + (r_s * math.exp(-eta * r_s**2)) / (2. * eta)
        #                     )  # radial integral 0 -> inf
        #number_correction *= 2. # integrate theta 0 -> pi
        #number_correction *= rho_solv * n_per_solv

        number_correction *= integrate_cos_binom(zeta)# * pow_zeta
        # actualy we have n * (n - 1)/2 pairs/angles
        # but this approximation is guaranteed to be non-negative
        number_correction *= number_correction/2.
        output[:, n_g2 + i::n_sf] /= (number_correction*pow_zeta)
        #output[:, n_g2 + n_g4 + i::n_sf] *= pow_zeta

    return output


def integrate_cos_power(n):
    """
    calculates the integral cos(x)**n dx
    from x=0 to x=2pi recursively
    """
    if not (n % 2 == 0):
        return 0.
    if n == 0:
        return 2.*math.pi
    if n == 2:
        return math.pi
    return (n-1)/n * integrate_cos_power(n-2)

from scipy.special import binom


def integrate_cos_binom(n):
    """
    calculates the integral (1 + cos(x))**n dx
    from x=0 to x=2pi recursively
    by using the binomial series representation
    (1 + cos(x))**n = sum_k=0^n binom_coeff(n,k) cos(x)**k
    """
    res = 0.
    upper = int(n) + 1
    for i in range(upper):
        res += binom(n,i) * integrate_cos_power(i)
    return res
