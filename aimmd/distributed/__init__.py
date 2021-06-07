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
# TODO: is the the best place for our semaphore(s)?
# TODO?: introduce a maximum for the number of slurm jobs? _SEM_MAX_SLURM_JOB
# NOTE: this is actually not that useful as you could run multiple python
#       processes with arcd on the same cluster, and then this could not work as intended
import os
import asyncio


def set_max_process(num=None, max_num=None):
    """
    Set the maximum number of concurrent python processes.
    If num is None, default to os.cpu_count().
    """
    # TODO: I think we should use less as default, maybe 0.25*cpu_count()?
    # and limit to 30-40?, i.e never higher even if we have 1111 cores?
    global _SEM_MAX_PROCESS
    if num is None:
        logical_cpu_count = os.cpu_count()
        if logical_cpu_count is not None:
            num = int(logical_cpu_count / 4)
        else:
            # fallback if os.cpu_count() can not determine the number of cpus
            # play it save and not have more than 2?
            # TODO: think about a good number!
            num = 2
    if max_num is not None:
        num = min((num, max_num))
    _SEM_MAX_PROCESS = asyncio.Semaphore(num)


# TODO/FIXME: calling this function again does not change the semaphore?!
#             i.e. we can only set max_processes when intitially importing?!
#             since this really sux we should think about passing the/a semaphore to
#             the sampling simulation?
set_max_process()


# ensure that only one Chain can access the central model at a time
_SEM_BRAIN_MODEL = asyncio.Semaphore(1)


# make stuff from submodules available (after defining the semaphores)
from .trajectory import Trajectory, TrajectoryFunctionWrapper
from .mdconfig import MDP
from .mdengine import GmxEngine, SlurmGmxEngine
from .logic import MCstep, Brain, TwoWayShootingPathMover, CommittorSimulation
