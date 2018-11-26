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
from keras import backend as K


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

    Parameters
    ----------
    We expect y_true to be an array with shape (N, 2), where N is the number
    of shooting points. y_true[0,0] = n_a and y_true[0,1] = n_b for shot 0.
    We expect y-pred to be the predicted reaction coordinate value.
    """
    rc = y_pred[:, 0]
    n_a = y_true[:, 0]
    n_b = y_true[:, 1]
    return n_b*K.log(1. + K.exp(-rc)) + n_a*K.log(1. + K.exp(rc))


def multinomial_loss(y_true, y_pred):
    """
    Maximum likelihood loss function for multiple state TPS.
    Parameters
    ----------
    We expect y_true to be an array with shape (N, N_states), where N is the
    number of shooting points. y_true[0,0] = n_a and y_true[0,1] = n_b etc
    for shot 0.
    We expect y-pred to be proportional to ln(p).
    This is equivalent to binomial_loss if N_states = 2.
    """
    ln_Z = K.log(K.sum(K.exp(y_pred), axis=-1, keepdims=True))
    return K.sum((ln_Z - y_pred) * y_true, axis=-1)
