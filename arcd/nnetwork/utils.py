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
from keras.models import Model, load_model
from keras.utils import CustomObjectScope
from keras import layers
from keras import backend as K
from .losses import binomial_loss, multinomial_loss


def create_snn(ndim, hidden_parms, optimizer, n_states, multi_state=True):
    """
    Creates a Keras SNN for committor predictions.
    The network takes as input a coordinatevector of length ndim and
    predicts rc[coords], having dim=1 or dim=n_states, influencing
    the choice of loss function, i.e. binomial or multinomial loss.
    It will be compiled with a lossfunction that expects the number of states
    reached from these coordinates (a length N_states array [n_a, n_b, ..., n_x])
    as y_true.
    For training coords (x-train) must be a 2d array (shape=(batch_size, ndim))
    while y_true (n_a, n_b, ..., n_x) must be of shape(batch_size, N_states)

    Parameters
    ----------
    ndim - int, number of input coordinates for SNN
    hidden_parms - list of dicts (len >= 1),
                   dicts are passed as kwargs to the hidden layer corresponding
                   to the list entry,
                   except for key 'units_factor', which determines the number
                   of units in the layer as int(ndim * units_factor),
                   if given the additional key 'dropout' with a float value
                   (or None) will result in a Dropout layer beeing applied
                   AFTER the corresponding layer in the hidden_params dicts,
                   i.e. [{parms1}, {parms2}] gives you 2 hidden layers,
                   the first with parms1 and the second with parms2,
                   if parms1 includes dropout you will get h -> drop -> h
                   for the hidden layers
    optimizer - `keras.optimizer` object, will be passed as optimizer to
                training_model.compile()
    n_states - int, number of states for multistate TPS,
               will be ignored if multi_state=False
    multistate - Bool, if False the neural network will output one single rc,
                 and for the loss we will assume p_B = 1/(1 + exp(-rc)),
                 if True the neural network will output the RCs
                 towards all states and we will use a multinomial loss

    Returns
    -------
    a compiled keras model
    """
    def drop_from_parms(parms):
        try:
            drop = parms['dropout']
            del parms['dropout']
        except KeyError:
            drop = None
        return drop, parms

    def fix_snn_parms(parms):
        parms['activation'] = 'selu'
        parms['kernel_initializer'] = 'lecun_normal'
        parms['bias_initializer'] = 'lecun_normal'
        return parms

    def apply_hidden_unit(inp, drop, layer_parms):
        h = layers.Dense(**layer_parms)(inp)
        if drop:
            h = layers.AlphaDropout(drop)(h)
        return h

    # INPUTS
    coords = layers.Input(shape=(ndim,),
                          name='coords',
                          dtype=K.floatx())

    # LAYERS
    # preprocess options dictionaries
    for d in hidden_parms:
        fact = d['units_factor']  # number of hidden units = units_fact * ndim
        del d['units_factor']
        d['units'] = int(fact*ndim)
    # hidden layers
    drop, parms = drop_from_parms(hidden_parms[0])
    parms = fix_snn_parms(parms)
    h = apply_hidden_unit(coords, drop, parms)
    for parms in hidden_parms[1:]:
        drop, parms = drop_from_parms(parms)
        parms = fix_snn_parms(parms)
        h = apply_hidden_unit(h, drop, parms)

    # last layer(s) + model building
    if multi_state:
        # multinomial output
        rc = layers.Dense(n_states,
                          activation='linear',
                          name='rc')(h)
        model = Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=multinomial_loss)
    else:
        # we use a binomial output
        rc = layers.Dense(1,
                          activation='linear',
                          name='rc')(h)
        model = Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=binomial_loss)

    return model


def create_resnet(ndim, hidden_parms, optimizer, n_states, multi_state=True):
    """
    Creates a Keras ResNet for committor prediction.
    The network takes as input a coordinatevector of length ndim and
    predicts rc[coords], having dim=1 or dim=n_states, influencing
    the choice of loss function, i.e. binomial or multinomial loss.
    It will be compiled with a lossfunction that expects the number of states
    reached from these coordinates (a length N_states array [n_a, n_b, ..., n_x])
    as y_true.
    For training coords (x-train) must be a 2d array (shape=(batch_size, ndim))
    while y_true (n_a, n_b, ..., n_x) must be of shape(batch_size, N_states)

    Parameters
    ----------
    ndim - int, number of input coordinates for ANN
    hidden_parms - list of dicts (len >= 1),
                   dicts are passed as kwargs to the hidden layer corresponding
                   to the list entry,
                   except for:
                   key 'units_factor', which determines the number
                   of units in the layer as int(ndim * units_factor),
                   key 'residual_n_skip', which determines the number of hidden
                   layers in that residual unit
                   key 'batch_norm' results in preactivation batch
                   normaliziation beeing applied, float values between 0
                   and 1 determine the fraction of units beeing dropped out
                   key 'dropout' with a float value will result in a Dropout
                   layer beeing applied AFTER the corresponding ResUnit
                   in the hidden_params dicts, but NOT to the shortcut
                   connection i.e. [{parms1}, {parms2}] gives you 2 ResUnits,
                   the first with parms1 and the second with parms2,
                   if parms1 includes dropout you will get:
                       ResUnit1 -> drop + input_shortcut -> ResUnit2
    optimizer - `keras.optimizer` object, will be passed as optimizer to
                training_model.compile()
    n_states - int, number of states for multistate TPS,
               will be ignored if multi_state=False
    multistate - Bool, if False the neural network will output one single rc,
                 and for the loss we will assume p_B = 1/(1 + exp(-rc)),
                 if True the neural network will output the RCs
                 towards all states and we will use a multinomial loss

    Returns
    -------
    a compiled keras model
    """
    def drop_residual_batch_from_parms(parms):
        try:
            drop = parms['dropout']
            del parms['dropout']
        except KeyError:
            drop = None
        try:
            residual = parms['residual_n_skip']
            del parms['residual_n_skip']
        except KeyError:
            residual = 0
        try:
            batch = parms['batch_norm']
            del parms['batch_norm']
        except KeyError:
            batch = False
        return drop, residual, batch, parms

    def apply_residual_unit(inp, n_lay, drop, batch, layer_parms):
        """
        inp - a keras tensor
        drop - float or None, fraction droped out after the last layer
        batch - bool, whether to aply Batch normalization
        n_lay - number of hidden layers that this residual unit skips
        layer_params - parameters for hidden layers
        snn - bool, if True we use AlphaDropout
        """
        try:
            activation = layer_parms['activation']
            layer_parms['activation'] = 'linear'
        except KeyError:
            raise ValueError('Layers that will be used in a residual unit '
                             + 'must have an activation. '
                             + '(You can give "linear".)')
        # TODO: dropout before since it is pre-activation!?
        if batch:
            h = layers.BatchNormalization()(inp)
            h = layers.Activation(activation)(h)
        else:
            h = layers.Activation(activation)(inp)
        h = layers.Dense(**layer_parms)(h)
        for i in range(1, n_lay):
            if batch:
                h = layers.BatchNormalization()(h)
            h = layers.Activation(activation)(h)
            h = layers.Dense(**layer_parms)(h)
        if drop:
            h = layers.Dropout(drop)(h)
        # add the two branches
        out = layers.Add()([inp, h])
        return out

    def apply_hidden_unit(inp, drop, batch, layer_parms):
        h = layers.Dense(**layer_parms)(inp)
        if batch:
            h = layers.BatchNormalization(h)
        if drop:
            h = layers.Dropout(drop)(h)
        return h

    # INPUTS
    coords = layers.Input(shape=(ndim,),
                          name='coords',
                          dtype=K.floatx())

    # HIDDEN LAYERS
    # preprocess options dictionaries
    for d in hidden_parms:
        fact = d['units_factor']  # number of hidden units = units_fact * ndim
        del d['units_factor']
        d['units'] = int(fact*ndim)
    # hidden layers
    drop, res, batch, parms = drop_residual_batch_from_parms(hidden_parms[0])
    if res:
        h = apply_residual_unit(coords, res, drop, batch, parms)
    else:
        h = apply_hidden_unit(coords, drop, batch, parms)
    for parms in hidden_parms[1:]:
        drop, res, batch, parms = drop_residual_batch_from_parms(parms)
        if res:
            h = apply_residual_unit(h, res, drop, batch, parms)
        else:
            h = apply_hidden_unit(h, drop, batch, parms)

    # last layer(s) + model building
    if multi_state:
        # multinomial output
        rc = layers.Dense(n_states,
                          activation='linear',
                          name='rc')(h)
        model = Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=multinomial_loss)
    else:
        # we use a binomial output
        # output is the reaction coordinate value
        rc = layers.Dense(1,
                          activation='linear',
                          name='rc')(h)
        model = Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=binomial_loss)

    return model


def load_keras_model(filename):
    """
    Loads a model from a given keras hdf5 model file.
    Takes care of setting the custom loss function.
    """
    with CustomObjectScope({'binomial_loss': binomial_loss,
                            'multinomial_loss': multinomial_loss}):
        model = load_model(filename)
    return model