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


logger = logging.getLogger(__name__)


from . import symreg, ops, coords
from .base.trainset import TrainSet
from .__about__ import (__version__, __title__, __author__,
                        __license__, __copyright__
                        )

try:
    from . import pytorch
except ImportError:
    logger.warn("pytorch not available")

try:
    from . import keras
except ImportError:
    logger.warn("keras not available")
