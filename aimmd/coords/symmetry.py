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
from functools import reduce
from ._symmetry import sf


logger = logging.getLogger(__name__)


# TODO: standardtize the outputs of the get_involved function(s)
def get_involved(idx, g_parms, mol_idxs, solvent_atoms, solvent_resname):
    """
    Parameters
    ----------
    idx - index to the symmetry function coordinates array
    g_parms - symmerty function parameters as given to coordinate function
    mol_idx - reactant/protein indexes as given to coordinate function
    solvent_atoms - list of solvent atom names (solvent_atnames) or
                    list of solvent atom elements (solvent_atelements)
                    as given to generate_indices()
    solvent_resname - list of solvent resnames as given to generate_indices()

    Returns for a given index to symmetry_function array:
    type of symmetry func (str; 'G2' or 'G5'),
    symmetry function parameters (list of float value pair),
    reactant atom on which it is centered (int),
    solvent resname from solvent_resname list (str),
    and the solvent type (str) describing
    the atom name or element of the solvent as in solvent_atnames/atelements
    """
    n_solv_atoms = [len(atoms) for atoms in solvent_atoms]
    n_mol = len(mol_idxs)
    g2_parms, g5_parms = g_parms
    n_g2 = len(g2_parms)
    n_g5 = len(g5_parms)
    n_sf = n_g2 + n_g5

    n_sf_per_resname = [n_sf * n_mol * N for N in n_solv_atoms]
    bounds = np.cumsum(n_sf_per_resname)
    try:
        solv_rname_idx = next(i for i, bound_idx in enumerate(bounds)
                              if bound_idx > idx)
    except StopIteration:
        # we reached the end of the list without finding a matching index
        raise ValueError('Something is wrong, either idx is out of range or we got malformated/wrong variables besides idx')
    bounds = [0] + list(bounds)  # we add a zero at the start for the next line to work
    carry = idx - bounds[solv_rname_idx]
    solv_idx = int(carry / (n_mol * n_sf))
    carry = carry % (n_mol * n_sf)
    at_idx = mol_idxs[int(carry / n_sf)]
    sf_idx = carry % n_sf

    if sf_idx < n_g2:
        sf_type = 'G2'
        sf_parms = g2_parms[sf_idx]
    elif n_g2 <= sf_idx < (n_g2 + n_g5):
        sf_type = 'G5'
        sf_parms = g5_parms[sf_idx - n_g2]

    return (sf_type, sf_parms, at_idx, solvent_resname[solv_rname_idx],
            solvent_atoms[solv_rname_idx][solv_idx])


def generate_indices(topology, solvent_resname, solvent_atnames=None,
                     solvent_atelements=None, exclude_attypes=None,
                     reactant_selection=None):
    """
    Generate indices for symmetry function calculation.

    Parameters
    ----------
    topology - :class:`mdtraj.Topology` object
    solvent_resname - list of str, residue names of solvent,
                      symmetry funcs will be calculated for all other atoms
    solvent_atnames - None or a list of lists of atom names, if given we will treat
                      solvent atoms differently by name
    solvent_atelements - None or a list of lists of element symbols, if given will
                         differentiate solvent atoms by element
    exclude_attypes - None or list of str, atom types to exclude from reactant,
                      e.g. exclude_attypes = ['H'] will select only
                      non-hydrogen atoms to center symmetry functions on
    reactant_selection - None or a mdtraj selection string to select the
                         reactant atom on which to center the symmetry functions,
                         if None the reactant will be everything that is
                         not resname solvent_resname, possibly minus ignored
                         atom types

    NOTE: Either solvent_atnames or solvent_atelements must be given.
          Only atoms matching resname AND atom element/name will be included
          in solv_idxs (and inculded in the symmetry fucntion calculation)

    Returns
    -------
    mol_idxs, solv_idxs
    mol_idxs - list of atom indices on which we will center the symmetry
               functions, everything that is not resname 'solvent_resname'
    solv_idxs - list of lists of solvent atom indices, outer list by atom type,
                or by atom name if solvent_atnames given, we will differentiate
                solvent atom 'species' by the outer list
    """
    def remove_attypes(sel_str, exclude_attypes):
        # add excluded elements to selection string
        sel_str += ' and not ('
        for t in exclude_attypes[:-1]:
            sel_str += 'type ' + t + ' or '
        sel_str += 'type ' + exclude_attypes[-1] + ')'
        return sel_str

    # reactant selection
    if reactant_selection:
        # selection string given, remove attypes if necessary
        if exclude_attypes:
            sel_str = remove_attypes(reactant_selection, exclude_attypes)
        else:
            sel_str = reactant_selection
    else:
        # no selection string -> reactant is everything not solvent
        sel_str = 'not resname {:s}'.format(solvent_resname[0])
        for rname in solvent_resname[1:]:
            sel_str += ' and not resname {:s}'.format(rname)
        if exclude_attypes:
            sel_str = remove_attypes(sel_str, exclude_attypes)
    # finally use the selection string
    mol_idxs = topology.select(sel_str)

    # solvent selection
    if solvent_atelements is not None:
        # we differentiate solvent atoms by chemical element
        solv_idxs = [[topology.select('resname {:s} and type {:s}\
                                     '.format(rname, t))
                      for t in solvent_atelements[i]]
                     for i, rname in enumerate(solvent_resname)]
    elif solvent_atnames is not None:
        # differentiate solvent atoms by their name
        solv_idxs = [[topology.select('resname {:s} and name {:s}\
                                     '.format(rname, n))
                      for n in set(solvent_atnames[i])]
                     for i, rname in enumerate(solvent_resname)]
    else:
        raise ValueError('Either solvent_atelements or solvent_attypes must be given')

    # make sure that reactant atoms are not included in solvent too
    filtered_solv_idxs = []
    for solv_idxs_by_resname in solv_idxs:
        filtered_solv_by_resname = []
        for idxs in solv_idxs_by_resname:
            masks = []
            for mol_i in mol_idxs:
                masks.append(np.where(idxs == mol_i,
                                      np.full_like(idxs, False, dtype=np.bool),
                                      np.full_like(idxs, True, dtype=np.bool)))
            mask = reduce(lambda x, y: np.logical_and(x, y), masks)
            if not np.all(mask):
                logger.warn('Some indices are part of solvent and reactant,'
                            + ' we removed them from solvent. Indices: '
                            + str(idxs[np.logical_not(mask)]))
            filtered_solv_by_resname.append(idxs[mask])
        filtered_solv_idxs.append(filtered_solv_by_resname)

    return mol_idxs, filtered_solv_idxs


transform = sf
