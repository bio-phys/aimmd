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
from . import networks
from .optim import HMC
# TODO: should we really make the Model "base" classes available?
#       i.e. the ones without a train decision function attached?
#       Or can we expect an import from a user who writes her own train decision? :)
from .rcmodel import (PytorchRCModel,
                      EEScalePytorchRCModel,
                      EEScalePytorchRCModelAsync,
                      EERandPytorchRCModel,
                      EERandPytorchRCModelAsync,
                      # TODO: async versions for the other models?!
                      EnsemblePytorchRCModel,
                      EEScaleEnsemblePytorchRCModel,
                      EERandEnsemblePytorchRCModel,
                      MultiDomainPytorchRCModel, EEMDPytorchRCModel)
