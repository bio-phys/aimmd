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


# NOTE ON PYTORCH MODELS
# every ANN here needs to have a self.call_kwargs dictionary
# containing the kwargs needed to instantiate the ANN
# the reason is the way we save and restore the RCmodels with pytorch ANNs
# the dict enables the reinstantiation as cls(**call_kwargs)
# which we use when loading a previuosly saved RCmodel


class FFNet(nn.Module):
    """Simple feedforward network with a variable number of hidden layers."""

    def __init__(self, n_in, n_hidden, n_out=1, activation=F.elu):
                 #TODO/FIXME: is there a reason we allow for kwargs we do not use??
                 # tests pass without it ;)
                 #**kwargs):
        """
        Initialize FFNet.

        n_in - number of input coordinates
        n_hidden - list of ints, number of hidden units per layer
        n_out - number of log probabilities to output,
                i.e. 1 for 2-state and N for N-State,
                make sure to use the correct loss
        activation - activation function or list of activation functions,
                     if one function it is used for all hidden layers,
                     if a list the length must match the number of hidden layers
        """
        super().__init__()
        self.call_kwargs = {'n_in': n_in,
                            'n_hidden': n_hidden,
                            'n_out': n_out,
                            'activation': activation,
                            }
        if not isinstance(activation, list):
            activation = [activation] * len(n_hidden)
        self.activation = activation
        n_units = [n_in] + list(n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_units[i], n_units[i+1])
                                            for i in range(len(n_units)-1)
                                            ])
        self.out_lay = nn.Linear(n_units[-1], n_out)

    def forward(self, x):
        for act, lay in zip(self.activation, self.hidden_layers):
            x = act(lay(x))
        # last layer without any activation function,
        # we always predict log probabilities
        x = self.out_lay(x)
        return x


class PreActivationResidualUnit(nn.Module):
    """
    Full pre-activation residual unit as proposed in
    'Identity Mappings in Deep Residual Networks' by He et al (arXiv:1603.05027)

    """
    def __init__(self, n_units, n_skip=4, activation=F.elu, norm_layer=None):
        """
        n_units - number of units per layer
        n_skip - number of layers to skip with the identity connection
        activation - activation function class
        norm_layer - normalization layer class
        """
        super().__init__()
        self.call_kwargs = {'n_units': n_units,
                            'n_skip': n_skip,
                            'activation': activation,
                            'norm_layer': norm_layer,
                            }
        self.layers = nn.ModuleList([nn.Linear(n_units, n_units)
                                     for _ in range(n_skip)])
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.norm_layers = nn.ModuleList([norm_layer(n_units)
                                          for _ in range(n_skip)])
        # TODO: do we want to be able to use different activations?
        # i.e. should we use a list of activation functions?
        self.activation = activation

    def forward(self, x):
        identity = x
        for lay, norm in zip(self.layers, self.norm_layers):
            x = lay(self.activation(norm(x)))
        x = x + identity
        return x


class ResNet(nn.Module):
    """
    Variable depth residual neural network
    """
    def __init__(self, n_out, n_units, n_blocks=4, block_kwargs=None):
        """
        n_out - number of outputs/ log probabilities to predict
        n_units - number of units per hidden layer [==number of inputs]
        n_blocks - number of residual blocks
        block_kwargs - None or list of dicts with ResUnit instatiation kwargs,
                       if None we will use the 'PreActivationResidualUnit' with defaults
        """
        if block_kwargs is None:
            block_kwargs = [{'class': PreActivationResidualUnit}
                            for _ in range(n_blocks)]
        #self.block_list = 
