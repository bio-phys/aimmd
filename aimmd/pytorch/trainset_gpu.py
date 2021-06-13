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
import torch
from ..base.trainset import TrainSet


class TrainSetTorchGPU(TrainSet):
    """A TrainSet that directly stores everything as torch tensors on GPU."""

    def __init__(self, states, descriptor_transform=None,
                 descriptors=None, shot_results=None, torch_device=None):
        # TODO: this is the same as for Trainset...can we deduplicate?
        self.states = states
        n_states = len(states)
        self._tp_idxs = [[i, j] for i in range(n_states)
                         for j in range(i + 1, n_states)]
        # TODO: maybe default to a lambda func just taking xyz of a OPS traj?
        # TODO: instead of the None value we have now?
        self.descriptor_transform = descriptor_transform

        # this is torch/gpu specific
        if not torch.cuda.is_available():
            raise ValueError('No cuda devices available. '
                             + 'Use a "normal" TrainSet instead.')
        if torch_device is not None:
            self.torch_device = torch_device
        else:
            self.torch_device = 'cuda'
        if ((descriptors is not None) and (shot_results is not None)):
            descriptors = torch.tensor(descriptors, device=self.torch_device)
            shot_results = torch.tensor(shot_results, device=self.torch_device)
            if shot_results.shape[0] != descriptors.shape[0]:
                raise ValueError('descriptors and shot_results must contain an'
                                 + ' equal number of points /have the same '
                                 + 'first dimension.')
        else:
            descriptors = torch.empty((0, 0), device=self.torch_device)
            shot_results = torch.empty((0, 0), device=self.torch_device)

        self._descriptors = descriptors
        self._shot_results = shot_results
        self._fill_pointer = shot_results.shape[0]

    @property
    def shot_results(self):
        return self._shot_results[:self._fill_pointer].cpu().numpy()

    @property
    def descriptors(self):
        return self._descriptors[:self._fill_pointer].cpu().numpy()

    @property
    def transitions(self):
        return sum(self._shot_results[:self._fill_pointer, i]
                   * self._shot_results[:self._fill_pointer, j]
                   for i, j in self._tp_idxs).cpu().numpy()

    def _extend_if_needed(self, descriptor_dim, add_entries=100):
        """
        Extend internal storage arrays if next step would not fit.
        """
        # need descriptors dimensionality to extend the array
        # at least if we do not know it yet, i.e. at first resize
        shadow_len = self._shot_results.shape[0]
        if shadow_len == 0:
            # no points yet, just create the arrays
            self._shot_results = torch.zeros((add_entries, len(self.states)),
                                             device=self.torch_device)
            self._descriptors = torch.zeros((add_entries, descriptor_dim),
                                            device=self.torch_device)
        elif shadow_len <= self._fill_pointer + 1:
            # no space left for the next point, extend
            new_len = shadow_len + add_entries
            self._shot_results.resize_((new_len, len(self.states)))
            self._descriptors.resize_((new_len, descriptor_dim))

    def append_point(self, descriptors, shot_results):
        """
        Append the given 1d-arrays of descriptors and shot_results.

        descriptors - np.ndarray with shape (descriptor_dim,)
        shot_results - np.ndarray with shape (n_states,)
        """
        self._extend_if_needed(descriptors.shape[0])
        self._fill_pointer += 1
        self._shot_results[self._fill_pointer] = torch.tensor(
                                                    shot_results,
                                                    device=self.torch_device
                                                              )
        self._descriptors[self._fill_pointer] = torch.tensor(
                                                    descriptors,
                                                    device=self.torch_device
                                                            )
