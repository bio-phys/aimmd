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
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from . import layers as custom_layers


# NOTE ON LOSS FUNCTIONS
# keras build in loss functions return a vector,
# where each entry corresponds to a training point (see keras losses source)
# however training uses the K.mean() over that vector as loss if the loss
# function is passed as loss=loss_FX to model.compile()
# if you use a tensor operation as loss and add it via model.add_loss(),
# training uses this tensor as loss, i.e. we have to take care of taking
# the mean ourselfes
# this distinction must happen somwhere in the serializing of the loss if
# passed through model.compile()
# Additionally only models with named loss functions can be reloaded
# but we still have to give the lossFX to a keras.utils.CustomObjectScope
# because it is a custom named lossFx
def binomial_loss(y_true, y_pred):
    """
    Maximum likeliehood loss function for TPS with random velocities or
    equivalently diffusive dynamics. We use the log-likeliehood derived from
    a binomial distribution.
    ln L = ln(\prod_i L_i) = \sum_i ln(L_i)
    where the loss per point, L_i = n_a * ln(1- p_B) + n_B * ln(p_B)
    We expect the ann to output the reaction coordinate rc and construct p_B
    as p_B = 1/(1 + exp(-rc))

    NOTE: this is NOT normalized in any way.
    Parameters
    ----------
    We expect y_true to be an array with shape (N, 2), where N is the number
    of shooting points. y_true[0,0] = n_a and y_true[0,1] = n_b for shot 0.
    We expect y-pred to be the predicted reaction coordinate value.
    """
    # note that we are 'vendor-locked' here using tensorflow to handle the NaNs
    # i.e. we lose the compability with other DL frameworks taht keras offers
    t1 = y_true[:, 0] * K.log(1. + K.exp(y_pred[:, 0]))
    t2 = y_true[:, 1] * K.log(1. + K.exp(-y_pred[:, 0]))
    zeros = tf.zeros_like(t1)
    return (tf.where(tf.equal(y_true[:, 0], 0), zeros, t1)
            + tf.where(tf.equal(y_true[:, 1], 0), zeros, t2)
            )


def binomial_loss_normed(y_true, y_pred):
    """
    Maximum likeliehood loss function for TPS with random velocities or
    equivalently diffusive dynamics. We use the log-likeliehood derived from
    a binomial distribution.
    ln L = ln(\prod_i L_i) = \sum_i ln(L_i)
    where the loss per point, L_i = n_a * ln(1- p_B) + n_B * ln(p_B)
    We expect the ann to output the reaction coordinate rc and construct p_B
    as p_B = 1/(1 + exp(-rc))

    NOTE: this is normalized per shot.
    Parameters
    ----------
    We expect y_true to be an array with shape (N, 2), where N is the number
    of shooting points. y_true[0,0] = n_a and y_true[0,1] = n_b for shot 0.
    We expect y-pred to be the predicted reaction coordinate value.
    """
    rc = y_pred[:, 0]
    n_a = y_true[:, 0]
    n_b = y_true[:, 1]
    return ((n_b*K.log(1. + K.exp(-rc)) + n_a*K.log(1. + K.exp(rc)))
            / (n_a + n_b))


def multinomial_loss(y_true, y_pred):
    """
    Maximum likelihood loss function for multiple state TPS.

    NOTE: this is NOT normalized in any way.
    Parameters
    ----------
    We expect y_true to be an array with shape (N, N_states), where N is the
    number of shooting points. y_true[0,0] = n_a and y_true[0,1] = n_b etc
    for shot 0.
    We expect y-pred to be proportional to ln(p).
    This is equivalent to binomial_loss if N_states = 2.
    """
    zeros = tf.zeros_like(y_pred)
    ln_Z = K.log(K.sum(K.exp(y_pred), axis=-1, keepdims=True))
    return K.sum(tf.where(tf.equal(y_true, 0),
                          zeros, (ln_Z - y_pred) * y_true),
                 axis=-1)


def multinomial_loss_normed(y_true, y_pred):
    """
    Maximum likelihood loss function for multiple state TPS.

    NOTE: this is normalized per shot.
    Parameters
    ----------
    We expect y_true to be an array with shape (N, N_states), where N is the
    number of shooting points. y_true[0,0] = n_a and y_true[0,1] = n_b etc
    for shot 0.
    We expect y-pred to be proportional to ln(p).
    This is equivalent to binomial_loss if N_states = 2.
    """
    ln_Z = K.log(K.sum(K.exp(y_pred), axis=-1, keepdims=True))
    return K.sum((ln_Z - y_pred) * y_true, axis=-1) / K.sum(y_true, axis=-1)


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
        try:
            fact = d['units_factor']  # number of hidden units = units_fact * ndim
            del d['units_factor']
            d['units'] = int(fact*ndim)
        except KeyError:
            # test access so we get the KeyError
            t = d['units']
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
        model = tf.keras.models.Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=multinomial_loss)
    else:
        # we use a binomial output
        rc = layers.Dense(1,
                          activation='linear',
                          name='rc')(h)
        model = tf.keras.models.Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=binomial_loss)

    return model


def create_resnet(ndim, hidden_parms, optimizer, n_states, multi_state=True,
                  learn_norm=None, norm_lay_kwargs={}, dropout_inputs=None):
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
                   normaliziation beeing applied
                   key 'dropout' with a float value between 0 and 1,
                   determining the fraction of units beeing dropped out, will
                   result in a Dropout layer beeing applied AFTER the corresponding
                   ResUnit in the hidden_params dicts, but NOT to the shortcut
                   connection i.e. [{parms1}, {parms2}] gives you 2 ResUnits,
                   the first with parms1 and the second with parms2,
                   if parms1 includes dropout you will get:
                       ResUnit1 -> drop + input_shortcut -> ResUnit2,
    optimizer - `keras.optimizer` object, will be passed as optimizer to
                training_model.compile()
    n_states - int, number of states for multistate TPS,
               will be ignored if multi_state=False
    multistate - Bool, if False the neural network will output one single rc,
                 and for the loss we will assume p_B = 1/(1 + exp(-rc)),
                 if True the neural network will output the RCs
                 towards all states and we will use a multinomial loss
    learn_norm - None or string, if str one of: '1for1' or '1forAll',
                 if '1for1' we use one hidden unit with one input per output
                 if '1forAll' we use the same hidden unit paremters elementwise
                 on all inputs
    norm_lay_kwargs - dict, keyword arguments directly passed to the InputNorm
                      layer, which supports the same kwargs as a Dense layer
                      except for 'units' (not necessary)
                      and 'partial_norm', which controls to which inputs the
                      InputNorm is applied; if it is present it is interpreted
                      as integer index specifying up to which input the Norm is applied,
                      every input with a higher index will be left unchanged and
                      concatenated with the normalized inputs before the first hidden layer,
                      e.g. if partial_norm=1 we will normalize only the first input
    dropout_inputs - fraction inputs to drop out before first layer of the network

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
        try:
            fact = d['units_factor']  # number of hidden units = units_fact * ndim
            del d['units_factor']
            d['units'] = int(fact*ndim)
        except KeyError:
            # test if it is there
            t = d['units']
    # hidden layers
    drop, res, batch, parms = drop_residual_batch_from_parms(hidden_parms[0])
    h = coords
    if dropout_inputs is not None:
        h = layers.Dropout(dropout_inputs)(h)
    if learn_norm:
        # preprocess norm layer kwargs
        try:
            act = norm_lay_kwargs['activation']
            del norm_lay_kwargs['activation']
        except KeyError:
            # fallback to sigmoid
            act = 'sigmoid'
        try:
            partial_norm = int(norm_lay_kwargs['partial_norm'])
            del norm_lay_kwargs['partial_norm']
        except KeyError:
            partial_norm = False
        # sort out which kind of norm layer we use
        if learn_norm == '1for1':
            norm_lay = custom_layers.InputNorm1for1(activation=act, **norm_lay_kwargs)
        elif learn_norm == '1forAll':
            norm_lay = custom_layers.InputNorm1forAll(activation=act, **norm_lay_kwargs)
        else:
            raise ValueError("'learn_norm' must be one of '1for1' or '1forAll'.")
        # and finally use it
        if partial_norm:
            slice1_lay = layers.Lambda(lambda x: x[:, 0:partial_norm])
            slice2_lay = layers.Lambda(lambda x: x[:, partial_norm:])
            h_normed = norm_lay(slice1_lay(h))
            h = layers.concatenate(inputs=[h_normed, slice2_lay(h)], axis=-1)
        else:
            h = norm_lay(h)
    if res:
        h = apply_residual_unit(h, res, drop, batch, parms)
    else:
        h = apply_hidden_unit(h, drop, batch, parms)

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
        model = tf.keras.models.Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=multinomial_loss)
    else:
        # we use a binomial output
        # output is the reaction coordinate value
        rc = layers.Dense(1,
                          activation='linear',
                          name='rc')(h)
        model = tf.keras.models.Model(inputs=coords, outputs=rc)
        model.compile(optimizer=optimizer, loss=binomial_loss)

    return model


def load_keras_model(filename):
    """
    Load a model from a given keras hdf5 model file.
    Takes care of setting the custom loss function.
    """
    with CustomObjectScope({'binomial_loss': binomial_loss,
                            'multinomial_loss': multinomial_loss,
                            'InputNorm1forAll': custom_layers.InputNorm1forAll,
                            'InputNorm1for1': custom_layers.InputNorm1for1,
                            }):
        model = tf.keras.models.load_model(filename)
    return model
