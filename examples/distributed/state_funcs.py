#!/usr/bin/env python3
"""
State functions for alanine dipetide.

Needs to be separate module/import to be able to use multiprocessing from the notebooks.
"""
import os
import argparse
import aimmd
import numpy as np
import mdtraj as mdt


def alpha_R(traj, scratch_dir):
    traj = mdt.load(traj.trajectory_file,
                    # mdt can not work with tprs, so we use theinitial gro for now
                    top=os.path.join(scratch_dir, "gmx_infiles/conf.gro"),
                    )
    psi = mdt.compute_dihedrals(traj, indices=[[6, 8, 14, 16]])[:, 0]
    phi = mdt.compute_dihedrals(traj, indices=[[4, 6, 8, 14]])[:, 0]
    state = np.full_like(psi, False, dtype=bool)
    # phi: -pi -> 0
    # psi: > -50 but smaller 30 degree
    deg = 180/np.pi
    state[(phi <= 0) & (-50/deg <= psi) & (psi <= 30/deg)] = True
    return state


def C7_eq(traj, scratch_dir):
    traj = mdt.load(traj.trajectory_file,
                    # mdt can not work with tprs, so we use theinitial gro for now
                    top=os.path.join(scratch_dir, "gmx_infiles/conf.gro"),
                    )
    psi = mdt.compute_dihedrals(traj, indices=[[6, 8, 14, 16]])[:, 0]
    phi = mdt.compute_dihedrals(traj, indices=[[4, 6, 8, 14]])[:, 0]
    state = np.full_like(psi, False, dtype=bool)
    # phi: -pi -> 0
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state


def descriptor_func_ic(traj, scratch_dir):
    """All internal coordinates (bond-length, angles, dihedrals) as descriptors."""
    traj = mdt.load(traj.trajectory_file,
                    # mdt can not work with tprs, so we use theinitial gro for now
                    top=os.path.join(scratch_dir, "gmx_infiles/conf.gro"),
                    )
    pairs, triples, quadruples = aimmd.coords.internal.generate_indices(traj.topology, source_idx=1)
    descriptors = aimmd.coords.internal.transform(traj, pairs=pairs, triples=triples, quadruples=quadruples)

    return descriptors


def descriptor_func_psi_phi(traj, scratch_dir):
    """Only psi and phi angle as internal coords. Actually cos and sin for both of them."""
    traj = mdt.load(traj.trajectory_file,
                    # mdt can not work with tprs, so we use theinitial gro for now
                    top=os.path.join(scratch_dir, "gmx_infiles/conf.gro"),
                    )
    psi = mdt.compute_dihedrals(traj, indices=[[6, 8, 14, 16]])
    phi = mdt.compute_dihedrals(traj, indices=[[4, 6, 8, 14]])
    # make sure the return value lies \in [0,1]
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
    parser.add_argument("-sd", "--scratch_dir", type=str,
                        default="/home/think/scratch/aimmd_distributed")
    args = parser.parse_args()
    # NOTE: since args is a namespace args.trajectory_file will be the path to
    #       the trajectory file, i.e. we can pass args instead of an
    #       aimmd.Trajectory to the functions above
    if args.function == "descriptors_ic":
        vals = descriptor_func_ic(args, args.scratch_dir)
    elif args.function == "descriptors_psi_phi":
        vals = descriptor_func_psi_phi(args, args.scratch_dir)
    elif args.function == "alphaR":
        vals = alpha_R(args, args.scratch_dir)
    elif args.function == "C7eq":
        vals = C7_eq(args, args.scratch_dir)

    np.save(args.output_file, vals)
