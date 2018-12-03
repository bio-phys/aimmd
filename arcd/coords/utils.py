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
from . import symmetry, internal


logger = logging.getLogger(__name__)


def transform(mdtra, ic_parms={}, sf_parms={}, custom_cv_func=None,
              custom_cv_parms={}):
    """
    Helper function: returns the concatenated output of internal coordinates
    symmetry functions and a custom defined cv_function (custom_cv_func).
    The function parameters of the wanted functions should be passed as dicts,
    where key=parameter_name and value=parameter_value.
    The custom_cv_function needs to operate on mdtraj trajectories, similar to
    internal-coordinates and symmetry-functions, i.e. it will be passed
    a mdtraj trajectory as first argument and then params by keyword.

    Ordering: internal_coords, symmerty_funcs, custom_cv (if present)
    See the respective function docstrings for more on the parameters.
    """
    import numpy as np
    from arcd.coordinates import internal, symmetry
    out = []
    if ic_parms:
        out.append(internal.transform(mdtra, **ic_parms))
    if sf_parms:
        # TODO: I don't think we need this?
        #sf_defaults = {'rho_solv': [33.], 'alpha_cutoff': 150.}
        #sf_defaults.update(sf_parms)
        out.append(symmetry.transform(mdtra, **sf_parms))
    if custom_cv_func:
        out.append(custom_cv_func(mdtra, **custom_cv_parms))

    return np.concatenate(out, axis=1)


def get_involved(idx, ic_parms={}, sf_parms={}, solvent_atoms=None,
                 solvent_resname=None):
    """
    For a given idx returns the type of collective variable (internal ('IC'),
    symmetry ('SF') or custom cv ('CustomCV')) and all
    information that the get_involved(idx) of the corresponding sub collective
    variable gives. Except for custom cv where just the idx into the
    array of custom cv is given.
    Have a look at their docstrings for more.
    """
    n_ic = 0
    n_sf = 0
    n_sf_per_at = 0

    if ic_parms:
        N_ic = [len(ic_parms['pairs']), len(ic_parms['triples']),
                2*len(ic_parms['quadrouples'])]
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
                                           ic_parms['quadrouples'])
    elif idx < n_ic + n_sf:
        return 'SF', symmetry.get_involved(idx - n_ic, sf_parms['g_parms'],
                                           sf_parms['mol_idxs'], solvent_atoms,
                                           solvent_resname)
    else:
        return 'CustomCV', idx - (n_ic + n_sf)
