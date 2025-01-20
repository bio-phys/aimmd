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


logger = logging.getLogger(__name__)


from ._version import __version__, __git_hash__
from .base import rcmodel
from .base.trainset import TrainSet
from .base.storage import Storage
from .base.utils import (emulate_production_from_trainset,
                         emulate_production_from_storage,
                         )
from . import ops, coords, analysis


try:
    from . import pytorch
except (ImportError, ModuleNotFoundError):
    logger.warning("Pytorch not available")

try:
    from . import symreg
except(ImportError, ModuleNotFoundError):
    logger.warning('dCGPy not found. SymReg will not be available.')

try:
    from . import keras
except (ImportError, ModuleNotFoundError):
    logger.warning("Tensorflow/Keras not available")
