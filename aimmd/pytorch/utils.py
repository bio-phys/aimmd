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
import torch


logger = logging.getLogger(__name__)


def get_closest_pytorch_device(location):
        """
        Check if location is an available pytorch.device.

        Otherwise returns the pytorch device that is `closest` to location.
        Adopted from pytorch.serialization.validate_cuda_device
        """
        if isinstance(location, torch.device):
            location = str(location)
        if not isinstance(location, str):
            raise ValueError("location should be a string or torch.device")
        if 'cuda' in location:
            if not torch.cuda.is_available():
                # no cuda, go to CPU
                logger.info('Restoring on CPU, since CUDA is not available.')
                return torch.device('cpu')
            if location[5:] == '':
                device = 0
            else:
                device = max(int(location[5:]), 0)
            if device >= torch.cuda.device_count():
                # other cuda device ID
                logger.info('Restoring on a different CUDA device.')
                # TODO: does this choose any cuda device or always No 0 ?
                return torch.device('cuda')
            # if we got until here we can restore on the same CUDA device we
            # saved from
            return torch.device('cuda:'+str(device))
        else:
            # we trained on cpu before
            # TODO: should we try to go to GPU if it is available?
            return torch.device('cpu')


def optimizer_state_to_device(sdict, device):
    """
    Helper function to move all tensors in optimizer state dicts to device.

    This enables saving/loading models on machines with and without GPU.
    """
    for state in sdict['state'].values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return sdict
