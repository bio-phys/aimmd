"""
This file is part of ARCD.

ARCD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ARCD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ARCD. If not, see <https://www.gnu.org/licenses/>.
"""
import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):
    """
    Simple feedforward network with 4 hidden layers.
    n_in - number of input coordinates
    n_out - number of log probabilities to output,
            i.e. 1 for 2-state and N for N-State,
            make sure to use the correct loss
    """
    def __init__(self, n_in, n_out=1, act=F.elu):
        super().__init__()
        self.act = act
        self.lay0 = nn.Linear(n_in, n_in)
        self.lay1 = nn.Linear(n_in, n_in)
        self.lay2 = nn.Linear(n_in, n_in)
        self.lay3 = nn.Linear(n_in, n_in)
        self.out_lay = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = self.act(self.lay0(x))
        x = self.act(self.lay1(x))
        x = self.act(self.lay2(x))
        x = self.act(self.lay3(x))
        x = self.out_lay(x)
        return x
