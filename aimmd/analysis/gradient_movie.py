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
import mdtraj as md
import openpathsampling as paths


logger = logging.getLogger(__name__)


class GradientMovieMaker:
    """
    Create pdb-movie sequences with RC gradient info in Bfactors.

    Parameters:
    -----------
        model - :class:`aimmd.RCModel`
        descriptor_transform - :class:`openpathsampling.MDTrajFunctionCV`
        topology - :class:`mdtraj.Topology`
        towards_state - int or None, state w.r.t. which we calculate gradients,
                        only relevant for multi state RCModels
        n_frames - int, number of frames for gradient swinging motion movies
        amplitude - float, maximal amplitude for gradient swinging movies

    """

    def __init__(self, model, descriptor_transform, topology,
                 towards_state=None, n_frames=100, amplitude=0.025):
        """Initialize GradientMovieMaker."""
        if model.n_out > 1:
            if towards_state is None:
                towards_state = 0
                logger.warn("Using a multistate RCModel without setting "
                            + "towards_state. Defaulting to towards_state=0.")
        # we will ignore self.towards_state if self.model.n_out == 1
        self.towards_state = towards_state
        self.n_frames = n_frames
        self.amplitude = amplitude
        self.model = model
        self.model_sqrt_eps = None

        if not isinstance(descriptor_transform, paths.MDTrajFunctionCV):
            raise ValueError("descriptor_transform must be an openpathsampling"
                             + ".MDTrajFunctionCV.")
        self.descriptor_transform = descriptor_transform
        self.descriptor_transform_sqrt_eps = None

        if not isinstance(topology, md.Topology):
            raise ValueError("Topology must be a mdtraj.Topology.")
        self.topology = topology

    def gradients_model_q(self, descriptors):
        """
        Calculate self.model.q gradients w.r.t. given descriptors.

        For multistate models the RC towards state self.towards_state is used.
        Uses the symmetric difference quotient grad = (f(x+h) - f(x-h)) / 2h
        with h = self.model_sqrt_eps * descriptors, self.model_sqrt_eps
        will be automatically initialized to sqrt(eps) if not set,
        where eps is the machine epsilon of the model reaction coordinate.
        """
        if self.model_sqrt_eps is None:
            q = self.model.q(descriptors, use_transform=False)
            self.model_sqrt_eps = np.sqrt(np.finfo(q.dtype).eps)

        # s_num is the output of the model we take for grads
        if self.model.n_out == 1:
            s_num = 0
        else:
            s_num = self.towards_state
        n_point = descriptors.shape[0]
        n_dim = descriptors.shape[1]
        h = self.model_sqrt_eps * descriptors
        # make sure h > 0
        h[h == 0.] = self.model_sqrt_eps * 0.01
        grads = np.zeros((n_point, n_dim))
        for d in range(n_dim):
            val_pl = descriptors.copy()
            val_pl[:, d] += h[:, d]
            val_min = descriptors.copy()
            val_min[:, d] -= h[:, d]
            grads[:, d] = (
                        self.model.q(val_pl, use_transform=False)[:, s_num]
                        - self.model.q(val_min, use_transform=False)[:, s_num]
                           ) / (2 * h[:, d])
        return grads

    def gradients_descriptor_transform(self, traj, atom_indices=None):
        # TODO: this is the slowest part!
        """
        Calculate descriptor_transform gradients w.r.t. to traj coordinates.

        Parameters:
        -----------
        traj - mdtraj or openpathsampling trajectory with one frame,
               [at least we ignore all other frames]
        atom_indices - 1d numpy.array of atom indices,
                       if given will calculate gradients only for those atoms

        Uses the symmetric difference quotient grad = (f(x+h) - f(x-h)) / 2h
        with h = self.descriptor_transform_sqrt_eps * xyz,
        self.descriptor_transform_sqrt_eps will be automatically initialized
        to sqrt(eps) if not set,
        where eps is the machine epsilon of the descriptor_transform output.

        """
        if isinstance(traj, paths.Trajectory):
            traj = traj.to_mdtraj()
        d = self.descriptor_transform.cv_callable(
                                    traj, **self.descriptor_transform.kwargs
                                                  )
        n_descript = d.shape[1]
        n_atoms = traj.xyz.shape[1]
        if self.descriptor_transform_sqrt_eps is None:
            traj_eps = np.finfo(traj.xyz.dtype).eps
            descript_eps = np.finfo(d.dtype).eps
            if traj_eps <= descript_eps:
                eps = descript_eps
            else:
                eps = traj_eps
            self.descriptor_transform_sqrt_eps = np.sqrt(eps)
        if atom_indices is None:
            logger.warn("Consider giving atom_indices to improve performance.")
            atom_indices = np.arange(n_atoms)
        h = self.descriptor_transform_sqrt_eps * traj.xyz[0]
        # make sure h > 0
        h[h == 0.] = self.descriptor_transform_sqrt_eps * 0.01
        gradients = np.zeros((n_atoms, 3, n_descript))
        for at in atom_indices:
            xyz_probe = np.zeros((6, n_atoms, 3))
            xyz_probe[:] = traj.xyz[0]
            for j in range(3):
                xyz_probe[2*j, at, j] += h[at, j]
                xyz_probe[2*j + 1, at, j] -= h[at, j]
            tra_probe = md.Trajectory(xyz_probe, self.topology)
            tra_probe.unitcell_vectors = np.array(
                                [traj.unitcell_vectors[0]
                                 for _ in range(tra_probe.n_frames)]
                                                  )
            descript = self.descriptor_transform.cv_callable(
                                tra_probe, **self.descriptor_transform.kwargs
                                                             )
            for j in range(3):
                gradients[at, j, :] = (descript[2*j] - descript[2*j + 1]
                                       ) / (2 * h[at, j])

        return gradients

    def anchor_mols_from_atom_indices(self, atom_indices):
        """Get the set of molecules containing atom_indices."""
        molecules = self.topology.find_molecules()
        anchor_mols = []
        for mol in molecules:
            for idx in atom_indices:
                if self.topology.atom(idx) in mol:
                    anchor_mols.append(mol)
                    # no need to go through all the other atoms
                    # if one is already part of the molecule
                    break
        return anchor_mols

    def movie_around_xyz(self, xyz, outfile, unitcell_vectors,
                         atom_indices=None, anchor_mols=None,
                         overwrite=True, normalize=True,
                         gradient_proportional=True,
                         mass_normalized=True):
        """
        Write out pdb gradient movie around given xyz coordinates.

        Parameters:
        -----------
        xyz - numpy.array, shape=(1, n_atoms, 3)
        outfile - str, filename the pdb movie will be written to
        unitcell_vectors - mdtraj unitcell_vectors for the simulation box,
                           numpy.array, shape=(3,3), i.e. for a single frame,
                           required to apply minimum image convention,
                           to put molecules back in box and to center the movie
        atom_indices - 1d numpy.array of atom indices,
                       if given will calculate gradients only for those atoms
        anchor_mols - list of mdtraj molecules to center the movie on,
                      will be guessed from atom_indices if None
        overwrite - bool, wheter to overwrite existing files with given name
        normalize - bool, wheter to normalize the calcuated Bfactors to [0., 99.99],
                    the PDB-format requires them to be in (-10, 100)
        gradient_proportional - bool, wheter to output the movie additionally
                                with atom movements proportional to the
                                gradient magnitudes
        mass_normalized - bool, whether to output the movie additionally with
                          atom movements proportional to the gradient magnitude
                          but scaled with the inverse atom mass

        """
        if (anchor_mols is None) and (atom_indices is not None):
            anchor_mols = self.anchor_mols_from_atom_indices(atom_indices)
        tra = md.Trajectory(xyz, self.topology)
        tra.unitcell_vectors = np.array([unitcell_vectors])
        descriptors = self.descriptor_transform.cv_callable(
                                    tra, **self.descriptor_transform.kwargs
                                                            )
        dq_ddescript = self.gradients_model_q(descriptors)
        ddescript_dx = self.gradients_descriptor_transform(tra, atom_indices)
        dq_dx = np.sum(dq_ddescript[0] * ddescript_dx, axis=-1)
        if mass_normalized:
            # get the masses if we need them
            masses = np.array([a.element.mass for a in self.topology.atoms])
        # we divide by the zero gradient values and then replace the NaNs
        # with zeros where the gradient was zero
        norm = np.sqrt(np.sum(dq_dx**2, axis=-1, keepdims=True))
        with np.errstate(invalid='ignore'):
            dq_dx_unit = np.true_divide(dq_dx, norm)
        dq_dx_unit[np.isnan(dq_dx_unit)] = 0
        if gradient_proportional:
            dq_dx_prop = dq_dx / np.max(norm)

        xyz_out = np.zeros((self.n_frames, xyz.shape[1], 3))
        xyz_out[:] = xyz[0]
        omega = 2 * np.pi / self.n_frames
        t_frame = np.arange(self.n_frames).reshape(self.n_frames, 1, 1)
        xyz_out += dq_dx_unit * self.amplitude * np.sin(omega * t_frame)
        tra_out = md.Trajectory(xyz_out, self.topology)
        # promote box_vector to tra length
        unitcell_vectors = np.array([unitcell_vectors
                                     for _ in range(self.n_frames)])
        tra_out.unitcell_vectors = unitcell_vectors
        # unwrap and possibly anchor tra
        tra_out = tra_out.image_molecules(anchor_molecules=anchor_mols)
        # enforce befactors to be in (-10, 100), this is a PDB format requirement
        # i.e. we normalize to be in [0, 99.99]
        bfacts = np.sqrt(np.sum(dq_dx**2, axis=-1))
        if normalize:
            bfacts *= 99.99/np.max(bfacts)
        elif np.max(bfacts) >= 100:
            logger.warn('The Bfactors must be < 100, but some calculated values'
                        + ' are higher. Normalizing to [0., 99.99].')
            bfacts *= 99.99/np.max(bfacts)
        tra_out.save_pdb(outfile, force_overwrite=overwrite,
                         bfactors=bfacts
                         )
        if gradient_proportional:
            # write out the gradient proportional movement trajectory
            # calculation is the same as before, just different dq_dx magnitude
            xyz_out = np.zeros((self.n_frames, xyz.shape[1], 3))
            xyz_out[:] = xyz[0]
            xyz_out += dq_dx_prop * self.amplitude * np.sin(omega * t_frame)
            tra_out = md.Trajectory(xyz_out, self.topology)
            tra_out.unitcell_vectors = unitcell_vectors
            tra_out = tra_out.image_molecules(anchor_molecules=anchor_mols)
            # (very naively) handle file ending...works only for '.pdb'
            if outfile.endswith('.pdb'):
                outfile_prop = (outfile[:-4]
                                + '_gradient_proportional_movements.pdb')
            tra_out.save_pdb(outfile_prop, force_overwrite=overwrite,
                             bfactors=bfacts
                             )
        if mass_normalized:
            # write out the gradient proportional movement trajectory
            # calculation is the same as before, just different dq_dx magnitude
            xyz_out = np.zeros((self.n_frames, xyz.shape[1], 3))
            xyz_out[:] = xyz[0]
            xyz_out += (dq_dx_prop / np.reshape(masses, (masses.shape[0], 1))
                        * self.amplitude * np.sin(omega * t_frame))
            tra_out = md.Trajectory(xyz_out, self.topology)
            tra_out.unitcell_vectors = unitcell_vectors
            tra_out = tra_out.image_molecules(anchor_molecules=anchor_mols)
            # (very naively) handle file ending...works only for '.pdb'
            if outfile.endswith('.pdb'):
                outfile_mass_norm = (outfile[:-4]
                                     + '_mass_normalized_gradient_proportional_movements.pdb')
            # use mass normalized bfactors here too
            bfacts /= masses
            if normalize:
                # renormalize to be in [0, 99.99]
                bfacts *= 99.99/np.max(bfacts)
            tra_out.save_pdb(outfile_mass_norm, force_overwrite=overwrite,
                             bfactors=bfacts
                             )

    def movie_around_snapshot(self, snap, outfile, atom_indices=None,
                              anchor_mols=None, overwrite=True):
        """
        Write out pdb gradient movie around given OPS snapshot.

        Parameters:
        -----------
        snap - openpathsampling snapshot
        outfile - str, filename the pdb movie will be written to
        atom_indices - 1d numpy.array of atom indices,
                       if given will calculate gradients only for those atoms
        anchor_mols - list of mdtraj molecules to center the movie on,
                      will be guessed from atom_indices if None
        overwrite - bool, wheter to overwrite existing files with given name

        """
        # this looks a bit strange but makes sure we cast from simtk.Quantities
        # to numpy arrays before adding the additional dim
        xyz = np.array([np.array(snap.coordinates)])
        unitcell_vectors = np.array(snap.box_vectors)
        self.movie_around_xyz(xyz, outfile,
                              atom_indices=atom_indices,
                              unitcell_vectors=unitcell_vectors,
                              anchor_mols=anchor_mols,
                              overwrite=overwrite
                              )

    def color_by_gradient(self, traj, outfile, atom_indices=None,
                          anchor_mols=None, overwrite=True,
                          normalize=True,
                          mass_normalized=True,
                          normalize_per_frame=False,
                          single_frames=False):
        """
        Write magnitude of gradients into Bfactors at each frame in outfile.

        traj - mdtraj trajectory or openpathsampling trajectory or snapshot
        outfile - str, filename the pdb movie will be written to
        atom_indices - 1d numpy.array of atom indices,
                       or list of numpy.arrays (one for each frame),
                       if given will calculate gradients only for those atoms
        anchor_mols - list of mdtraj molecules to center the movie on,
                      will be guessed from atom_indices if None
        overwrite - bool, wheter to overwrite existing files with given name
        normalize - bool, wheter to normalize the calcuated Bfactors to [0., 99.99],
                    the PDB-format requires them to be in (-10, 100)
        mass_normalized - bool, whether to additionally write out a trajectory
                          with gradients normalized by the mass of each atom
        normalize_per_frame - bool, wheter to additionally write out a
                              trajectory with gradients normalized per frame
        single_frames - bool, wheter to output a series of single frame pbds
                        suffixed by frame number

        """
        def strip_pdb_suffix(outfile):
            if outfile.endswith('.pdb'):
                outfile = outfile[:-4]
            return outfile

        def write_single_frames(tra, Bfactors, outfile, overwrite):
            for i, f in enumerate(tra):
                outfile = strip_pdb_suffix(outfile)
                outname = (outfile
                           + '_{:03d}'.format(i)
                           + '.pdb')
                f.save_pdb(outname, force_overwrite=overwrite,
                           bfactors=Bfactors[i])

        if isinstance(traj, paths.BaseSnapshot):
            traj = paths.Trajectory([traj])
        if isinstance(traj, paths.Trajectory):
            traj = traj.to_mdtraj()
        if atom_indices is not None:
            if isinstance(atom_indices[0], (int, np.integer)):
                atom_indices = np.asarray(atom_indices, dtype=np.int_)
                # make it possible to pass separate atom_indices for each frame
                # or just one array but then take it for the whole tra
                atom_indices = [atom_indices for _ in range(len(traj))]
            else:
                # cast to numpy ints and arrays
                atom_indices = [np.asarray(idxs, dtype=np.int_)
                                for idxs in atom_indices]
            if anchor_mols is None:
                # our best guess at the anchor molecules is taking all molecules
                # any of the atom indices for any frame points to
                at_idx_set = set(np.concatenate([idxs for idxs in atom_indices],
                                                axis=0)
                                 )
                anchor_mols = self.anchor_mols_from_atom_indices(at_idx_set)
        if mass_normalized:
            # get the masses if we need them
            masses = np.array([a.element.mass for a in self.topology.atoms])

        Bfactors = []
        if mass_normalized:
            Bfactors_mass_normalized = []
        if normalize_per_frame:
            Bfactors_normalized = []
        descriptors = self.descriptor_transform.cv_callable(
                                    traj, **self.descriptor_transform.kwargs
                                                            )
        dq_ddescript = self.gradients_model_q(descriptors)
        for f in range(traj.n_frames):
            ddescript_dx = self.gradients_descriptor_transform(traj[f:f+1],
                                                               atom_indices[f])
            dq_dx = np.sum(dq_ddescript[f] * ddescript_dx, axis=-1)
            Bfactors.append(np.sqrt(np.sum(dq_dx**2, axis=-1)))
            if mass_normalized:
                Bfactors_mass_normalized.append(Bfactors[-1] / masses)
            if normalize_per_frame:
                norm = 99.99/np.max(np.sqrt(np.sum(dq_dx**2, axis=-1)))
                Bfactors_normalized.append(np.sqrt(np.sum(dq_dx**2, axis=-1))
                                           / norm
                                           )

        Bfactors = np.array(Bfactors)
        # normalize Bfactors to PDB-fromat requirement
        if normalize:
            Bfactors *= 99.99/np.max(Bfactors)
        elif np.max(Bfactors) >= 100:
            logger.warn('The Bfactors must be < 100, but calculated gradients'
                        + ' are higher. Normalizing to [0., 99.99].')
            Bfactors *= 99.99/np.max(Bfactors)
        traj_out = traj.image_molecules(anchor_molecules=anchor_mols)
        if mass_normalized:
            Bfactors_mass_normalized = np.array(Bfactors_mass_normalized)
            if normalize:
                Bfactors_mass_normalized *= 99.99 / np.max(Bfactors_mass_normalized)
            elif np.max(Bfactors_mass_normalized) >= 100:
                logger.warn('The Bfactors must be < 100, but calculated gradients'
                            + ' are higher. Normalizing to [0., 99.99].')
                Bfactors_mass_normalized *= 99.99/np.max(Bfactors_mass_normalized)
        if normalize_per_frame:
            Bfactors_normalized = np.array(Bfactors_normalized)
        # NOTE: visualizing the created movie can be done by writing the
        # Bfactor values for every frame into the user field in VMD
        # the user field is dynamically read every frame and allows for
        # different colors in different frames,
        # see the VMD script @ examples/resources/pdbbfactor.tcl
        if single_frames:
            write_single_frames(traj_out, Bfactors, outfile, overwrite)
            if mass_normalized:
                outfile_mod = strip_pdb_suffix(outfile) + '_mass_normalized'
                write_single_frames(traj_out, Bfactors_mass_normalized,
                                    outfile_mod, overwrite)
            if normalize_per_frame:
                outfile_mod = strip_pdb_suffix(outfile) + '_frame_normalized'
                write_single_frames(traj_out, Bfactors_normalized,
                                    outfile_mod, overwrite)
        else:
            traj_out.save_pdb(outfile, force_overwrite=overwrite,
                              bfactors=Bfactors)
            if mass_normalized:
                outfile_mod = strip_pdb_suffix(outfile) + '_mass_normalized.pdb'
                traj_out.save_pdb(outfile_mod, force_overwrite=overwrite,
                                  bfactors=Bfactors_mass_normalized)
            if normalize_per_frame:
                outfile_mod = strip_pdb_suffix(outfile) + '_frame_normalized.pdb'
                traj_out.save_pdb(outfile, force_overwrite=overwrite,
                                  bfactors=Bfactors_normalized)
