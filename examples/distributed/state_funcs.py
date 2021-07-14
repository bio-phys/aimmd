#!/usr/bin/env python3
"""
State functions for alanine dipetide.

Needs to be separate module/import to be able to use multiprocessing from the notebooks.
"""
import os
import argparse
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


def descriptor_func(traj, scratch_dir):
    # TODO: make this a real descriptor func!!
    traj = mdt.load(traj.trajectory_file,
                    # mdt can not work with tprs, so we use theinitial gro for now
                    top=os.path.join(scratch_dir, "gmx_infiles/conf.gro"),
                    )
    psi = mdt.compute_dihedrals(traj, indices=[[6, 8, 14, 16]])
    phi = mdt.compute_dihedrals(traj, indices=[[4, 6, 8, 14]])

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
    parser.add_argument("-sd", "--scratch_dir", type=str,
                        default="/home/think/scratch/aimmd_distributed")
    args = parser.parse_args()
    # NOTE: since args is a namespace args.trajectory_file will be the path to
    #       the trajectory file, i.e. we can pass args instead of an
    #       aimmd.Trajectory to the functions above
    if args.function == "descriptors":
        vals = descriptor_func(args, args.scratch_dir)
    elif args.function == "alphaR":
        vals = alpha_R(args, args.scratch_dir)
    elif args.function == "C7eq":
        vals = C7_eq(args, args.scratch_dir)

    np.save(args.output_file, vals)
