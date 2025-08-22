# This file is part of aimmd
#
# aimmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# aimmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with aimmd. If not, see <https://www.gnu.org/licenses/>.
"""
This file contains functions and classes (re)used internally in aimmd.

These functions and classes are not (thought to be) exposed to the users but
instead intended to be (re)used in newly added asyncmd code.

Currently in here are:

- attach_kwargs_to_object: a function to attach kwargs to an object as properties
  or attributes. This does type checking and warns when previously unset things
  are set. It is used, e.g., in the distributed.CommittorSimulation class.
"""
import logging


def attach_kwargs_to_object(obj, *, logger: logging.Logger,
                            **kwargs: dict,
                            ) -> None:
    """
    Set all kwargs as object attributes/properties, error on mismatching type.

    Warn when we set an unknown (i.e. previously undefined attribute/property)

    Parameters
    ----------
    obj : object
        The object to attach the kwargs to.
    logger: logging.Logger
        The logger to use for logging.
    **kwargs : dict
        Zero to N keyword arguments.
    """
    dval = object()
    for kwarg, value in kwargs.items():
        if (cval := getattr(obj, kwarg, dval)) is not dval:
            if isinstance(value, type(cval)):
                # value is of same type as default so set it
                setattr(obj, kwarg, value)
            else:
                raise TypeError(f"Setting attribute {kwarg} with "
                                + f"mismatching type ({type(value)}). "
                                + f" Default type is {type(cval)}."
                                )
        else:
            # not previously defined, so warn that we ignore it
            logger.warning("Ignoring unknown keyword-argument %s.", kwarg)
