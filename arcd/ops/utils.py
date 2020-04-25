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
import logging
from .selector import RCModelSelector


logger = logging.getLogger(__name__)


# TODO: can we write a nice function that works for some selectors only?
#       i.e. that is selector specific and lets the user choose the selector?
def set_rcmodel_in_all_selectors(model, simulation):
    """
    Replace all RCModelSelectors models with the given model.

    Useful for restarting TPS simulations, since the arcd RCModels can not be
    saved by ops together with the RCModelSelector.
    """
    for move_group in simulation.move_scheme.movers.values():
        for mover in move_group:
            if isinstance(mover.selector, RCModelSelector):
                mover.selector.model = model
