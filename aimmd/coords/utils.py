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
from . import symmetry, internal


logger = logging.getLogger(__name__)


def get_involved(idx, ic_parms={}, sf_parms={}, solvent_atoms=None,
                 solvent_resname=None):
    """
    For a given idx returns the type of collective variable (internal ('IC'),
    symmetry ('SF') or custom cv ('CustomCV')) for a CV built by concatenating
    IC, SF and custom CV in that order if present.
    For IC and SF returns all information that the get_involved(idx) of the
    corresponding submodules give. For custom cv we return the idx into the
    array of custom cv.
    Have a look at the submodule docstrings for more.
    """
    n_ic = 0
    n_sf = 0
    n_sf_per_at = 0

    if ic_parms:
        N_ic = [len(ic_parms['pairs']), len(ic_parms['triples']),
                2*len(ic_parms['quadruples'])]
        n_ic = sum(N_ic)

    if sf_parms:
        n_sf_per_at = sum([len(sf_parms['g_parms'][0]),
                           len(sf_parms['g_parms'][1])])
        if not (solvent_atoms and solvent_resname):
            raise ValueError('solvent_atoms and solvent_resname must be given if sf_parms is given.')
        # finally calculate n_sf
        n_solv_atoms = [len(atoms) for atoms in solvent_atoms]
        n_mol = len(sf_parms['mol_idxs'])
        n_sf_per_resname = [n_sf_per_at * n_mol * N for N in (n_solv_atoms)]
        n_sf = np.cumsum(n_sf_per_resname)[-1]

    if idx < n_ic:
        return 'IC', internal.get_involved(idx, ic_parms['pairs'],
                                           ic_parms['triples'],
                                           ic_parms['quadruples'])
    elif idx < n_ic + n_sf:
        return 'SF', symmetry.get_involved(idx - n_ic, sf_parms['g_parms'],
                                           sf_parms['mol_idxs'], solvent_atoms,
                                           solvent_resname)
    else:
        return 'CustomCV', idx - (n_ic + n_sf)
