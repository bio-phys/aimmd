"""
State functions for alanine dipetide.

Needs to be separate module/import to be able to use multiprocessing from the notebooks.
"""
import os
import numpy as np
import mdtraj as mdt


def alpha_R(traj, scratch_dir):
    traj = mdt.load(traj.trajectory_file,
                    # mdt can not work with tprs, so we use theinitial gro for now
                    top=os.path.join(scratch_dir, "gmx_infiles/conf.gro"),
                    )
    psi = mdt.compute_dihedrals(traj, indices=[[6,8,14,16]])[:, 0]
    phi = mdt.compute_dihedrals(traj, indices=[[4,6,8,14]])[:, 0]
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
    psi = mdt.compute_dihedrals(traj, indices=[[6,8,14,16]])[:, 0]
    phi = mdt.compute_dihedrals(traj, indices=[[4,6,8,14]])[:, 0]
    state = np.full_like(psi, False, dtype=bool)
    # phi: -pi -> 0 
    # psi: 120 -> 200 degree
    deg = 180/np.pi
    state[(phi <= 0) & ((120/deg <= psi) | (-160/deg >= psi))] = True
    return state
