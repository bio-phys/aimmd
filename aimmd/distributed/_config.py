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
# same layout idea as in asyncmd, so a verbatim copy of the comment there:
# NOTE: This file **only** contains the dictionaries with the values
#       and **no** functions to set them, the funcs all live in 'config.py'.
#       The idea here is that we can then without any issues import additional
#       stuff (like the config functions from 'slurm.py') in 'config.py'
#       without risking circular imports becasue all asyncmd files should only
#       need to import the _CONFIG and _SEMAPHORES dicts from '_config.py'.

_SEMAPHORES = {}
