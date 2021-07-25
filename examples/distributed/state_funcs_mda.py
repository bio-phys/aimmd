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


def alpha_R(traj, skip=1):
    # NOTE: use refresh_offsets=True such that we do not run into any trouble when
    #       opening the same traj at the same time from two different processes/universes
    #       to avoid reading a possibly corrupted/in the process of beeing created offsets
    #       file we just rebuild all offsets
    u = mda.Universe(traj.structure_file, traj.trajectory_file, refresh_offsets=True)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # phi: -pi -> 0
    # psi: > -50 but smaller 30 degree
    deg = 180/np.pi
    state[(phi <= 0) & (-50/deg <= psi) & (psi <= 30/deg)] = True
    return state


def C7_eq(traj, skip=1):
    u = mda.Universe(traj.structure_file, traj.trajectory_file, refresh_offsets=True)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    state = np.full((len(u.trajectory[::skip]),), False, dtype=bool)
    phi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]),), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    # phi: -pi -> 0
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state


def descriptor_func_ic(traj, skip=1):
    # TODO: write this!
    raise NotImplementedError


def descriptor_func_psi_phi(traj, skip=1):
    """Only psi and phi angle as internal coords. Actually cos and sin for both of them."""
    u = mda.Universe(traj.structure_file, traj.trajectory_file, refresh_offsets=True)
    psi_ag = u.select_atoms("index 6 or index 8 or index 14 or index 16")
    phi_ag = u.select_atoms("index 4 or index 6 or index 8 or index 14")
    # empty arrays to fill
    phi = np.empty((len(u.trajectory[::skip]), 1), dtype=np.float64)
    psi = np.empty((len(u.trajectory[::skip]), 1), dtype=np.float64)
    for f, ts in enumerate(u.trajectory[::skip]):
        phi[f, 0] = calc_dihedrals(*(at.position for at in phi_ag), box=ts.dimensions)
        psi[f, 0] = calc_dihedrals(*(at.position for at in psi_ag), box=ts.dimensions)
    
    return 1 + 0.5*np.concatenate([np.sin(psi), np.cos(psi), np.sin(phi), np.cos(phi)], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Calculate CV values for alanine dipeptide",
                                     )
    parser.add_argument("structure_file", type=str)
    parser.add_argument("trajectory_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("-f", "--function", type=str,
                        default="descriptors",
                        choices=["alphaR", "C7eq", "descriptors_ic", "descriptors_psi_phi"])
    parser.add_argument("-s", "--skip", type=int, default=1)
    args = parser.parse_args()
    # NOTE: since args is a namespace args.trajectory_file will be the path to
    #       the trajectory file, i.e. we can pass args instead of an
    #       aimmd.Trajectory to the functions above
    if args.function == "descriptors_ic":
        vals = descriptor_func_ic(args, skip=args.skip)
    elif args.function == "descriptors_psi_phi":
        vals = descriptor_func_psi_phi(args, skip=args.skip)
    elif args.function == "alphaR":
        vals = alpha_R(args, skip=args.skip)
    elif args.function == "C7eq":
        vals = C7_eq(args, skip=args.skip)

    np.save(args.output_file, vals)
