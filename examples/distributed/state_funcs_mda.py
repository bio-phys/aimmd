#!/usr/bin/env python3
"""
State functions for alanine dipetide.

Needs to be separate module/import to be able to use multiprocessing from the notebooks.
"""
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals


def alpha_R(traj, skip=1):
    u = mda.Universe(traj.structure_file, *traj.trajectory_files,)
    psi_ag = u.select_atoms("index 6")  #"resname ALA and name N")
    psi_ag += u.select_atoms("index 8")  #"resname ALA and name CA")
    psi_ag += u.select_atoms("index 14")  #"resname ALA and name C")
    psi_ag += u.select_atoms("index 16")  #"resname NME and name N")
    phi_ag = u.select_atoms("index 4")  #"resname ACE and name C")
    phi_ag += u.select_atoms("index 6")  #"resname ALA and name N")
    phi_ag += u.select_atoms("index 8")  #"resname ALA and name CA")
    phi_ag += u.select_atoms("index 14")  #"resname ALA and name C")
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    # (otherwise we might need to wait until the next garbage collection and
    #  if that happens to be after we want to apply this func to many trajs
    #  we will run into the number of open files limit)
    u.trajectory.close()
    # phi: -pi -> 0
    # psi: > -50 but smaller 30 degree
    deg = 180/np.pi
    state[(phi <= 0) & (-50/deg <= psi) & (psi <= 30/deg)] = True
    return state


def C7_eq(traj, skip=1):
    u = mda.Universe(traj.structure_file, *traj.trajectory_files,)
    psi_ag = u.select_atoms("index 6")  #"resname ALA and name N")
    psi_ag += u.select_atoms("index 8")  #"resname ALA and name CA")
    psi_ag += u.select_atoms("index 14")  #"resname ALA and name C")
    psi_ag += u.select_atoms("index 16")  #"resname NME and name N")
    phi_ag = u.select_atoms("index 4")  #"resname ACE and name C")
    phi_ag += u.select_atoms("index 6")  #"resname ALA and name N")
    phi_ag += u.select_atoms("index 8")  #"resname ALA and name CA")
    phi_ag += u.select_atoms("index 14")  #"resname ALA and name C")
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # make sure MDAnalysis closes the underlying trajectory files directly
    # (otherwise we might need to wait until the next garbage collection and
    #  if that happens to be after we want to apply this func to many trajs
    #  we will run into the number of open files limit)
    u.trajectory.close()
    # phi: -pi -> 0
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state


def generate_atomgroups_for_ic(molecule):
    """
    Generate atomgroups describing all bonds, angles and dihedrals of given molecule.

    Parameters:
    -----------
    molecule - `MDAnalysis.AtomGroup` (preferably) of one continous molecule

    Returns:
    --------
    bonds, angles, dihedrals - lists of `MDAnalysis.AtomGroup` of len 2, 3 and 4
    """
    bonds = [mda.AtomGroup([], molecule.universe) for _ in range(2)]
    angles = [mda.AtomGroup([], molecule.universe) for _ in range(3)]
    dihedrals = [mda.AtomGroup([], molecule.universe) for _ in range(4)]
    for b in molecule.bonds:
        for i, at in enumerate(b.atoms):
            bonds[i] += at
    for a in molecule.angles:
        for i, at in enumerate(a.atoms):
            angles[i] += at
    for d in molecule.dihedrals:
        for i, at in enumerate(d.atoms):
            dihedrals[i] += at

    return bonds, angles, dihedrals


def descriptor_func_ic(traj, molecule_selection="protein", skip=1, use_SI=True):
    """Calculate symmetry invariant internal coordinate representation for molecule_selection."""
    u = mda.Universe(traj.structure_file, *traj.trajectory_files,)
    molecule = u.select_atoms(molecule_selection)
    bonds, angles, dihedrals = generate_atomgroups_for_ic(molecule)
    bond_vals = np.empty((len(u.trajectory[::skip]), len(bonds[0])), dtype=np.float64)
    angle_vals = np.empty((len(u.trajectory[::skip]), len(angles[0])), dtype=np.float64)
    dihedral_vals = np.empty((len(u.trajectory[::skip]), len(dihedrals[0])), dtype=np.float64)
    for f, ts in enumerate(u.trajectory):
        calc_bonds(bonds[0].positions, bonds[1].positions, box=ts.dimensions, result=bond_vals[f])
        calc_angles(*(angles[i].positions for i in range(3)), box=ts.dimensions, result=angle_vals[f])
        calc_dihedrals(*(dihedrals[i].positions for i in range(4)), box=ts.dimensions, result=dihedral_vals[f])
    u.trajectory.close()

    # capture periodicity
    angle_vals = 0.5 * (1. + np.cos(angle_vals))
    dihedrals_out = np.empty((dihedral_vals.shape[0], dihedral_vals.shape[1] * 2))
    dihedrals_out[:, ::2] = 0.5 * (1. + np.sin(dihedral_vals))
    dihedrals_out[:, 1::2] = 0.5 * (1. + np.cos(dihedral_vals))

    if use_SI:
        # mdanalysis uses \AA
        bond_vals /= 10.

    return np.concatenate((bond_vals, angle_vals, dihedrals_out), axis=1)


def descriptor_func_psi_phi(traj, skip=1):
    """Only psi and phi angle as internal coords. Actually cos and sin for both of them."""
    u = mda.Universe(traj.structure_file, *traj.trajectory_files,)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    phi = np.empty((len(u.trajectory[::skip]), 1), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]), 1), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f, 0] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f, 0] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    u.trajectory.close()

    return np.concatenate((psi, phi), axis=1)
    #return 1 + 0.5*np.concatenate([np.sin(psi), np.cos(psi), np.sin(phi), np.cos(phi)], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Calculate CV values for alanine dipeptide",
                                     )
    parser.add_argument("structure_file", type=str)
    parser.add_argument("trajectory_files", type=str, nargs="+")
    parser.add_argument("output_file", type=str)
    parser.add_argument("-f", "--function", type=str,
                        default="descriptors",
                        choices=["alphaR", "C7eq", "descriptors_ic", "descriptors_psi_phi"])
    parser.add_argument("-s", "--skip", type=int, default=1)
    parser.add_argument("-si", "--use-SI", dest="use_SI", type=bool, default=True)
    parser.add_argument("-ms", "--molecule-selection", dest="molecule_selection", type=str, default="protein",
                        help="molecule selection string for internal coordinate representation")
    args = parser.parse_args()
    # NOTE: since args is a namespace args.trajectory_file will be the path to
    #       the trajectory file, i.e. we can pass args instead of an
    #       aimmd.Trajectory to the functions above
    if args.function == "descriptors_ic":
        vals = descriptor_func_ic(args, molecule_selection=args.molecule_selection,
                                  skip=args.skip, use_SI=args.use_SI)
    elif args.function == "descriptors_psi_phi":
        vals = descriptor_func_psi_phi(args, skip=args.skip)
    elif args.function == "alphaR":
        vals = alpha_R(args, skip=args.skip)
    elif args.function == "C7eq":
        vals = C7_eq(args, skip=args.skip)

    np.save(args.output_file, vals)
