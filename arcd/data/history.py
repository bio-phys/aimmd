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

# TODO: save and load!
class History:
    """
    TODO
    """
    def __init__(self):
        self.expected_efficiency = []
        self.expected_committors = []


class KerasTrainerHistory(History):
    """
    TODO
    """
    def __init__(self):
        super().__init__()
        self.loss = []
        self.loss_per_batch = []
        self.training_decision = []
