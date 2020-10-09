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
import math
import torch.nn as nn
import torch.nn.functional as F


# NOTE ON PYTORCH MODELS
# every ANN here needs to have a self.call_kwargs dictionary
# containing the kwargs needed to instantiate the ANN
# the reason is the way we save and restore the RCmodels with pytorch ANNs
# the dict enables the reinstantiation as cls(**call_kwargs)
# which we use when loading a previously saved RCmodel


class ModuleStack(nn.Module):
    """
    arcd specialized wrapper around pytorch ModuleList.

    Adds a last linear layer to predict log-probabilities and additionally
    keeps track of the single modules call_kwargs to facilitate saving/loading.
    """

    def __init__(self, n_out, modules, modules_call_kwargs=None):
        """
        Initialize ModuleStack.

        n_out - number of log probabilities to output,
                i.e. 1 for 2-state and N for N-State,
                make sure to use the correct loss
        modules - list of initialized arcd network parts, will be used in the
                  given order to construct the network from in to out,
                  additionally a last linear layer with n_out units is added
                  to predict the log-probabilities
        modules_call_kwargs - [default None] list of dicts, classes and
                              call_kwargs of the single modules,
                              will be automatically inferred if modules is given
                              and is mainly used for automatic reinstanitiation
                              when loading/saving the model
        """
        super().__init__()
        # see if we use (saved) modules_call_kwargs
        # or if we extract it from given modules
        if modules_call_kwargs is None:
            modules_call_kwargs = []
            for module in modules:
                modules_call_kwargs.append({'class': module.__class__,
                                            'call_kwargs': module.call_kwargs,
                                            })
        else:
            # TODO: should we perform basic sanity checks for formatting etc?
            # or we just let it throw whatever errors we might create :)
            modules = [mck['class'](**mck['call_kwargs'])
                       for mck in modules_call_kwargs]

        self.module_list = nn.ModuleList(modules)
        self.log_predictor = nn.Linear(modules[-1].n_out, n_out)
        self.n_out = n_out
        self.call_kwargs = {'n_out': n_out,
                            # we save None, since we recreate from modules_call_kwargs
                            'modules': None,
                            'modules_call_kwargs': modules_call_kwargs,
                            }

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        x = self.log_predictor(x)
        return x


class FFNet(nn.Module):
    """Simple feedforward network with a variable number of hidden layers."""

    def __init__(self, n_in, n_hidden, activation=F.elu):
        """
        Initialize FFNet.

        n_in - number of input coordinates
        n_hidden - list of ints, number of hidden units per layer
        activation - activation function or list of activation functions,
                     if one function it is used for all hidden layers,
                     if a list the length must match the number of hidden layers
        """
        super().__init__()
        self.call_kwargs = {'n_in': n_in,
                            'n_hidden': n_hidden,
                            'activation': activation,
                            }
        self.n_out = n_hidden[-1]
        if not isinstance(activation, list):
            activation = [activation] * len(n_hidden)
        self.activation = activation
        n_units = [n_in] + list(n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_units[i], n_units[i+1])
                                            for i in range(len(n_units)-1)
                                            ])

    def forward(self, x):
        for act, lay in zip(self.activation, self.hidden_layers):
            x = act(lay(x))
        return x


class SNN(nn.Module):
    """
    Self-normalizing neural network.

    Literature:
    'Self-Normalizing Neural Networks' by Klambauer et al (arXiv:1706.02515)

    """

    def __init__(self, n_in, n_hidden, dropout={}):
        """
        Initialize SNN.

        n_in - number of input coordinates
        n_hidden - list of ints, number of hidden units per layer
        dropout - dict, {'idx': p_drop}, i.e.
                  keys give the index of the hidden layer AFTER which
                  AlphaDropout is applied and the respective dropout
                  probabilities are given by the values
        """
        super().__init__()
        self.call_kwargs = {'n_in': n_in,
                            'n_hidden': n_hidden,
                            'dropout': dropout,
                            }
        self.n_out = n_hidden[-1]
        n_units = [n_in] + list(n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_units[i], n_units[i+1])
                                            for i in range(len(n_units)-1)
                                            ])
        # TODO: do we have a better way of doing this?
        # we could use None instead of the Identity,
        # but then we would have to check in forward if it is None
        # now we can just apply without 'thinking'
        dropout_list = [nn.Identity() for _ in range(len(self.hidden_layers))]
        for key, val in dropout.items():
            idx = int(key)
            dropout_list[idx] = nn.AlphaDropout(val)
        self.dropout_layers = nn.ModuleList(dropout_list)
        self.activation = nn.SELU()
        self.reset_parameters()  # initialize weights

    def forward(self, x):
        for h, d in zip(self.hidden_layers, self.dropout_layers):
            x = d(self.activation(h(x)))
        return x

    def reset_parameters(self):
        # properly initialize weights
        # TODO? for the biases we keep the pytorch standard, i.e.
        # uniform \in [-1/sqrt(N_in), + 1/sqrt(N_in)]
        # NOTE: I think we do not need to check:
        # we can only have nn.Linear layers in there
        for lay in self.hidden_layers:
            fan_out = lay.out_features
            nn.init.normal_(lay.weight, mean=0., std=1./math.sqrt(fan_out))


class PreActivationResidualUnit(nn.Module):
    """
    Full pre-activation residual unit.

    Literature:
    'Identity Mappings in Deep Residual Networks' by He et al (arXiv:1603.05027)

    """

    def __init__(self, n_units, n_skip=4, activation=F.elu, norm_layer=None):
        """
        Initialize PreActivationResidualUnit.

        n_units - number of units per layer
        n_skip - number of layers to skip with the identity connection
        activation - activation function class
        norm_layer - normalization layer class
        """
        super().__init__()
        self.n_out = n_units
        self.call_kwargs = {'n_units': n_units,
                            'n_skip': n_skip,
                            'activation': activation,
                            'norm_layer': norm_layer,
                            }  # I think we do not need this here...
        self.layers = nn.ModuleList([nn.Linear(n_units, n_units)
                                     for _ in range(n_skip)])
        if norm_layer is None:
            # TODO: is this really what we want?!
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
    Variable depth residual neural network.

    """

    def __init__(self, n_units, n_blocks=4, block_class=None, block_kwargs=None):
        """
        n_units - number of units per hidden layer [==number of inputs]
        n_blocks - number of residual blocks
        block_class - None or class or list of classes,
                      if None we will use the PreActivationResidualunit,
                      if one class, we will use it for all Residualblocks,
                      if a list it must be of len n_blocks, for each Block,
                      we will use the class given in the list
        block_kwargs - None or dict or list of dicts with ResUnit instatiation kwargs,
                       if None we will use the default values for each block class,
                       if a dict we will use the same dict as kwargs for all blocks,
                       if a list it must be of length n_blocks and contain kwargs
                       for each block

        """
        super().__init__()
        self.n_out = n_units
        self.call_kwargs = {'n_units': n_units,
                            'n_blocks': n_blocks,
                            'block_class': block_class,
                            'block_kwargs': block_kwargs,
                            }
        if block_class is None:
            block_class = PreActivationResidualUnit
        if not isinstance(block_class, (list, tuple)):
            # make it a list
            block_class = [block_class for _ in range(n_blocks)]
        if block_kwargs is None:
            # make a list with empty dicts for kwargs
            block_kwargs = [{} for _ in range(n_blocks)]
        if isinstance(block_kwargs, dict):
            # we assume it is the same for all blocks since it is just one dict
            block_kwargs = [block_kwargs for _ in range(n_blocks)]

        self.block_list = nn.ModuleList([clas(n_units=n_units, **kwargs)
                                         for clas, kwargs in zip(block_class,
                                                                 block_kwargs)
                                         ])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x
