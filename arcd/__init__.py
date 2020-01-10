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


from .__about__ import (__version__, __title__, __author__,
                        __license__, __copyright__
                        )
from . import ops, coords, analysis
from .base.trainset import TrainSet
from .base.utils import (emulate_production_from_trainset,
                         emulate_production_from_storage,
                         load_model,
                         load_model_with_storage,
                         )

try:
    from . import pytorch
except (ModuleNotFoundError, ImportError):
    logger.warning("Pytorch not available")

try:
    from . import symreg
except(ModuleNotFoundError, ImportError):
    logger.warning('dCGPy not found. SymReg will not be available.')

try:
    from . import keras
except (ModuleNotFoundError, ImportError):
    logger.warning("Tensorflow/Keras not available")
