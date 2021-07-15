#!/usr/bin/env python3
"""
State functions for alanine dipetide.

Needs to be separate module/import to be able to use multiprocessing from the notebooks.
"""
import os
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals


def alpha_R(traj):
    u = mda.Universe(traj.structure_file, traj.trajectory_file)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    state = np.full((len(u.trajectory),), False, dtype=bool)
    phi = np.empty((len(u.trajectory),), dtype=np.float64)
    psi = np.empty((len(u.trajectory),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimension)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimension)
    # phi: -pi -> 0
    # psi: > -50 but smaller 30 degree
    deg = 180/np.pi
    state[(phi <= 0) & (-50/deg <= psi) & (psi <= 30/deg)] = True
    return state


def C7_eq(traj, scratch_dir):
    u = mda.Universe(traj.structure_file, traj.trajectory_file)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    state = np.full((len(u.trajectory),), False, dtype=bool)
    phi = np.empty((len(u.trajectory),), dtype=np.float64)
    psi = np.empty((len(u.trajectory),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimension)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimension)
    # phi: -pi -> 0
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state


def descriptor_func(traj, scratch_dir):
    # TODO: make this a real descriptor func!!
    u = mda.Universe(traj.structure_file, traj.trajectory_file)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    phi = np.empty((len(u.trajectory),), dtype=np.float64)
    psi = np.empty((len(u.trajectory),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimension)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimension)
    
    return np.concatenate([psi, phi], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Calculate CV values for alanine dipeptide",
                                     )
    parser.add_argument("structure_file", type=str)
    parser.add_argument("trajectory_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("-f", "--function", type=str,
                        default="descriptors",
                        choices=["alphaR", "C7eq", "descriptors"])
    args = parser.parse_args()
    # NOTE: since args is a namespace args.trajectory_file will be the path to
    #       the trajectory file, i.e. we can pass args instead of an
    #       aimmd.Trajectory to the functions above
    if args.function == "descriptors":
        vals = descriptor_func(args)
    elif args.function == "alphaR":
        vals = alpha_R(args)
    elif args.function == "C7eq":
        vals = C7_eq(args)

    np.save(args.output_file, vals)
