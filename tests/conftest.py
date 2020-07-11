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
import pytest
# TODO/BUG: weird stuff: if not importing mdtraj before RCModel
# the tests segfault....!?
import mdtraj
import numpy as np
import openpathsampling as paths
import openpathsampling.engines.toy as toys
from openpathsampling.engines import Trajectory as OPSTrajectory
from functools import reduce
from arcd.base.rcmodel import RCModel


@pytest.fixture
def ops_toy_sim_setup():
    # construct PES
    n_harmonics = 20
    pes_list = []
    pes_list += [toys.OuterWalls(sigma=[0.2, 1.0] + [0. for _ in range(n_harmonics)],
                                 x0=[0.0, 0.0] + [0. for _ in range(n_harmonics)])
                 ]
    pes_list += [toys.Gaussian(A=-.7,
                               alpha=[12.0, 12.0] + [0. for _ in range(n_harmonics)],
                               x0=[-.75, -.5] + [0. for _ in range(n_harmonics)])
                 ]
    pes_list += [toys.Gaussian(A=-.7,
                               alpha=[12.0, 12.0] + [0. for _ in range(n_harmonics)],
                               x0=[.75, .5] + [0. for _ in range(n_harmonics)])
                 ]
    pes_list += [toys.HarmonicOscillator(A=[0., 0.] + [1./2. for _ in range(n_harmonics)],
                                         omega=[0., 0.] + [0.2, 0.5] + [10.*np.random.ranf() for _ in range(n_harmonics-2)],
                                         x0=[0. for _ in range(n_harmonics + 2)])
                 ]
    pes = reduce(lambda x, y: x+y, pes_list)
    # topology and integrator
    topology = toys.Topology(n_spatial=2 + n_harmonics,
                             masses=np.array([1.0 for _ in range(2 + n_harmonics)]),
                             pes=pes,
                             n_atoms=1
                             )
    integ = toys.LangevinBAOABIntegrator(dt=0.02, temperature=0.1, gamma=2.5)
    options = {'integ': integ,
               'n_frames_max': 5000,
               'n_steps_per_frame': 1
               }
    toy_eng = toys.Engine(options=options,
                          topology=topology
                          )
    toy_eng.initialized = True
    template = toys.Snapshot(coordinates=np.array([[-0.75, -0.5] + [0. for _ in range(n_harmonics)]]),
                             velocities=np.array([[0.0, 0.0] + [0. for _ in range(n_harmonics)]]),
                             engine=toy_eng
                             )
    toy_eng.current_snapshot = template
    # collective variables, states and initial TP
    def circle(snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)
    opA = paths.CoordinateFunctionCV(name="opA", f=circle, center=[-0.75, -0.5])
    opB = paths.CoordinateFunctionCV(name="opB", f=circle, center=[0.75, 0.5])
    stateA = paths.CVDefinedVolume(opA, 0.0, 0.15).named('StateA')
    stateB = paths.CVDefinedVolume(opB, 0.0, 0.15).named('StateB')
    descriptor_transform = paths.FunctionCV('descriptor_transform', lambda s: s.coordinates[0], cv_wrap_numpy_array=True)
    initAB = paths.Trajectory([toys.Snapshot(coordinates=np.array([[-0.75 + i/700., -0.5 + i/1000] + [0. for _ in range(n_harmonics)]]),
                                             velocities=np.array([[1.0, 0.0] + [0. for _ in range(n_harmonics)]]),
                                             engine=toy_eng
                                             )
                               for i in range(1001)
                               ]
                              )
    # velocity randomizer setup
    beta = integ.beta
    modifier = paths.RandomVelocities(beta=beta, engine=toy_eng)

    return {'states': [stateA, stateB],
            'descriptor_transform': descriptor_transform,
            'initial_TP': initAB,
            'engine': toy_eng,
            'template': template,
            'modifier': modifier,
            'cv_ndim': n_harmonics + 2,
            }


@pytest.fixture
def oneout_rcmodel():
    return OneOutRCModel(lambda x: -x)


@pytest.fixture
def oneout_rcmodel_notrans():
    return OneOutRCModel(None)


@pytest.fixture
def oneout_rcmodel_opstrans():
    def transform(x):
        if isinstance(x, OPSTrajectory):
            return np.array([[0.]])
        else:
            return np.array([[-200.]])
    return OneOutRCModel(transform)


@pytest.fixture
def twoout_rcmodel():
    return TwoOutRCModel(lambda x: -x)


@pytest.fixture
def twoout_rcmodel_notrans():
    return TwoOutRCModel(None)


class OneOutRCModel(RCModel):
    def __init__(self, transform=None):
        # work on numpy arrays directly, but test the use of transform
        # using transform should exchange p_A with p_B
        super().__init__(descriptor_transform=transform)
        self.n_train = 0

    @property
    def n_out(self):
        return 1

    def train_hook(self, trainset):
        # just to have something to check if
        # and how often we would have trained
        self.n_train += 1

    def _log_prob(self, descriptors):
        # so we have a return value that has the correct shape
        return np.sum(descriptors, axis=1, keepdims=True)

    def test_loss(self, trainset):
        # just to be able to instantiate
        raise NotImplementedError


class TwoOutRCModel(RCModel):
    def __init__(self, transform):
        super().__init__(descriptor_transform=transform)
        self.n_train = 0

    @property
    def n_out(self):
        return 2

    def train_hook(self, trainset):
        # just to have something to check if
        # and how often we would have trained
        self.n_train += 1

    def _log_prob(self, descriptors):
        # so we have a return value that has the correct shape
        q_A = np.sum(descriptors, axis=1, keepdims=True)
        # take q_B = -q_A such that p_A + p_B = 1
        q_B = -np.sum(descriptors, axis=1, keepdims=True)
        return np.concatenate((q_A, q_B), axis=1)

    def test_loss(self, trainset):
        # just to be able to instantiate
        raise NotImplementedError
