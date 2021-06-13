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
import networkx as nx


logger = logging.getLogger(__name__)


# TODO: standardtize the outputs of the get_involved function(s)
def get_involved(idx, pairs, triples, quadruples):
    """
    For a given index to internal coordinates array
    return the atom indices of the involved atoms
    """
    if idx < len(pairs):
        return pairs[idx]
    elif idx < len(pairs) + len(triples):
        idx -= len(pairs)
        return triples[idx]
    else:
        idx -= len(pairs)
        idx -= len(triples)
        if idx % 2 == 0:
            cos_sin = 'sin'
        else:
            cos_sin = 'cos'
        return cos_sin, quadruples[int(idx/2)]


def generate_indices(topology, source_idx, exclude_atom_names=None):
    """
    Generates the index pairs, triples and quadroples needed to calculate
    distances, angles and dihedrals for internal coordinate representation of a
    single molecule. We do this by going along the (directed) bondgraph from
    source atom until we encounter another leaf.

    NOTE: Source has to be a terminal atom, i.e. one with just one bond,
          otherwise we could miss some angles and dihedrals

    Parameters
    ----------
    topology - :class:`mdtraj.Topology` object
    source_idx - the atom index of the atom which should be the source atom,
                 i.e. at the origin of the coordinate representation
    exclude_atom_names - None or list of string, if given we will not create
                         any bond, angle or dihedral that would include one of
                         the listed atom names, usefull to e.g. exclude all Hs

    Returns
    -------
    pairs, triples, quadruples - tuple of lists of lists with index
                                  pairs/triples/quadruples
    """
    pairs = []
    triples = []
    quadruples = []
    bfs_bondgraph = nx.bfs_tree(topology.to_bondgraph(),
                                topology.atom(source_idx))
    succ_dict = dict(nx.bfs_successors(bfs_bondgraph,
                                       topology.atom(source_idx)))
    for origin_at, neighbour_ats in succ_dict.items():
        for middle_at in neighbour_ats:
            if exclude_atom_names is not None:
                if middle_at.name in exclude_atom_names:
                    # skip any bond, angle or dihedral including this atom
                    continue
            pairs.append([origin_at.index, middle_at.index])
            if middle_at in succ_dict.keys():
                # the middle atom has neighbours,
                # we define angles over all of them
                for target_at in succ_dict[middle_at]:
                    if exclude_atom_names is not None:
                        if target_at.name in exclude_atom_names:
                            # skip any angle or dihedral including this atom
                            continue
                    triples.append([origin_at.index,
                                    middle_at.index,
                                    target_at.index])
                    if target_at in succ_dict.keys():
                        # if the target_at has at least one neighbour
                        # we can define a dihedral over the four atoms
                        # any one of the neighbours is sufficient to fix
                        # the rotation
                        dihed_ats = succ_dict[target_at]
                        if dihed_ats:  # first check if dihed_ats list is empty
                            dihed_at = None
                            if exclude_atom_names is not None:
                                # sort out if any of the atom is not excluded
                                for at in dihed_ats:
                                    if at.name not in exclude_atom_names:
                                        dihed_at = at
                            else:
                                # no exclusions, we just take the first one
                                dihed_at = dihed_ats[0]
                            if dihed_at is not None:
                                quadruples.append([origin_at.index,
                                                    middle_at.index,
                                                    target_at.index,
                                                    dihed_at.index])

    return pairs, triples, quadruples


# TODO: do we want to scale the bondlength?
# TODO: we would need to define l_0 and a sigma for every bond/ bond_type
def ic(mdtra, pairs, triples, quadruples):
    """
    Calculates internal coordinate representation consisting of distances,
    angles and dihedrals given by pairs, triples and quadruples.

    NOTE: Angular coordinates are 1/2*(cos(angle)+1) to capture the
    periodicity and scale to [0,1]. We make use of the symmerty here
    and omit the sin part because the cosine specifies the physical situation.
    Dihedral angles have two coordinates,
    1/2*(sin(dihed)+1) and 1/2*(cos(dihed)+1), because the symmetry we used
    for the angles is not valid in general for arbitrary substituents.

    Parameters
    ----------
    mdtra - :class:`mdtraj.Trajectory` object
    pairs, triples,
    quadruples     - lists of lists with index pairs/triples/quadruples,
                      output of `ic_generate_idxs(topology, source_idx)`

    Returns
    -------
    internal_coords - np.array of internal coordinate values,
                      shape = (n_frames, n_pairs + n_triples + n_quadruples)
    """
    import mdtraj as md
    import numpy as np

    dists = md.compute_distances(mdtra, pairs)
    # sometimes the angles are NaN because of floating point inaccuracy
    # this means that mdtraj took arccos(x) with |x| > 1,
    # we do not know if +1 or -1, so mapping NaN to 0 is not really correct...
    # it should be 0 or pi depending on sign
    # but at least cos(0) = cos(pi)
    angles = md.compute_angles(mdtra, triples)
    diheds = md.compute_dihedrals(mdtra, quadruples)
    if not np.all(np.isfinite(angles)):
        logger.warn('Used np.nan_to_num() on the angles because they were not finite.')
        angles = np.nan_to_num(angles)
    if not np.all(np.isfinite(diheds)):
        logger.warn('Used np.nan_to_num() on the dihedrals because they were not finite.')
        diheds = np.nan_to_num(diheds)

    angles = 0.5*(1. + np.cos(angles))

    diheds_out = np.zeros((diheds.shape[0], diheds.shape[1]*2))
    diheds_out[:, ::2] = 0.5*(1. + np.sin(diheds))
    diheds_out[:, 1::2] = 0.5*(1. + np.cos(diheds))

    return np.concatenate([dists, angles, diheds_out], axis=1)


transform = ic
