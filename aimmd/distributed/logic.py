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
import os
import abc
import logging


logger = logging.getLogger(__name__)


# TODO: DOCUMENT!
class BrainTask(abc.ABC):
    """All BrainTasks should subclass this."""
    def __init__(self, interval=1):
        self.interval = interval

    @abc.abstractmethod
    def run(self, brain, chain_result):
        # TODO: find a smart way to pass the chain result (if we even want to?)
        pass


class Brain:
    """
    The 'brain' of the simulation.

    Attributes
    ----------
        model - the committor model
        trainset - the trainingset storing all shooting results of all TPS sims
        states - iterable of state(functions)
        descriptor_transform - function transforming trajectory frames to
                               model input (descriptor) space
        n_chain - number of separate Markov chains/TPS simulations
        tasks - list of `BrainTask` objects,
                tasks will be checked if they should run in the order they are
                in the list after any one TPS sim has finished a trial,
                note that tasks will only be ran at their specified intervals

    """
    # TODO: docstring + remove obsolete notes when we are done
    # should do basically what the hooks do in the ops case:
    #   - call the models train_hook, i.e. let it decide if it wants to train
    #   - keep track of the current model/ save it etc
    #   - keep track of its workers/ store their results in the central trainset
    #   - 'controls' the central arcd-storage (with the 'main' model and the trainset in it)
    #   - density collection should also be done (from) here
    #   NOTE: we run the tasks at specified frequencies like the hooks in ops
    #   NOTe: opposed to ops our tasks are an ordered list that gets done in deterministic order (at least the checking)
    #   Note: (this way one could also see the last task that is run as a pre-step task...?)
    #   TODO: make it possible to pass task state?
    chain_directory_prefix = "chain_"
    def __init__(self, model, trainset, states, descriptor_transform, n_chain,
                 workdir, tasks=[], **kwargs):
        # TODO: descriptor_transform and states?!
        self.model = model
        self.trainset = trainset
        self.states = states
        self.descriptor_transform = descriptor_transform
        self.n_chain = n_chain
        self.workdir = workdir
        self.tasks = tasks
        # TODO: is this what we want? or rather check type if it exists,
        #       possibly warn or reject if type missmatch, but set all
        #       and not only the exisiting ones?!
        # make it possible to set all existing attributes via kwargs
        dval = object()
        for kwarg, value in kwargs.items():
            if getattr(self, kwarg, dval) is not dval:
                setattr(self, kwarg, value)
        # and we do all setup of counters etc after to make sure they are what
        # we expect
        self.total_trials = 0
        # TODO: chain-setup
        # self.chains = [TPSChain()]

    def object_for_pickle(self, group, overwrite):
        # currently overwrite will always be True
        return self

    def complete_from_h5py_group(self, group):
        return self

    def run_tasks(self, chain_result):
        # TODO: find a smart way to pass the chain result (if we even want to?)
        for t in self.tasks:
            if self.total_trials % t.interval == 0:
                t.run(brain=self, chain_result=chain_result)

    # TODO: do we need/want weights? better func name?
    def seed_initial_transitions(self, transitions, weights):
        # should put a transition into every TPS sim
        # TODO: transitions should be a list of 'trajectory' objects?
        # TODO: define that Trajectory object
        return NotImplementedError


class TPSChain:
    # the single TPS simulation object:
    #   - keeps track of a single markov chain
    #   - communicate 'only' descriptors of current TP to Brain for SP selection (get a frame index + selecting model back)
    #     (the brain needs to know about the descriptors of the selected SP for expected eff calculation)
    #   - keeps track of the model that selected the SPs (one arcd-storage per TPS sim)
    #   - could/should do accept/reject on its own?!
    #   - prepare the run input files for gmx, i.e. write out the struct corresponding to selected SP idx with random velocities
    #   - start/organize the running of the trials (via a dedicated object, possibly specialized to the queing sys)
    #   - cut and concatenate if its a TP (again possibly via dedicated helper objects?)
    #   - organizes the calculation of descriptors for the trajectory if it is a TP
    #   - (re)-organizes the chains folder structure according to orders from brain (accept/reject of TPs, start new trials, etc)
    #     chain folder structure should be similar to our MGA2 runs with symlinks
    # TODO: make it possible to run post-processing 'hooks'?!
    def __init__(self, brain, workdir):
        #TODO: should this even take a/the brain on init?
        self.brain = brain
        self.workdir = workdir
        self._trial_num = 0
        self._accepts = []  # list of zeros and ones, one entry per trial
        self._transitions = []  # list of 0 and 1, same as above

    @property
    def n_trials(self):
        return self._trial_num

    @property
    def n_accepts(self):
        return sum(self._accepts)

    @property
    def n_transitions(self):
        return sum(self._transitions)

    # TODO: maybe better than the bit below are these two funcs?
    def initialize_from_directory(self, workdir):
        # initializes self from a given workdir
        # should set all internal state, i.e. trial num etc
        return NotImplementedError
    def ensure_consistent_directory_structure(self):
        # should ensure that the sturcture of workdir is consistent with internal state
        # check for correct number of trials, possibly oder of accepts/rejects if we store that?
        return NotImplementedError

    def ensure_directory_structure(self):
        # TODO: could be useful, but do we want/need it?
        # small todo: better name :)
        return NotImplementedError


#TODO: TPSTrial class (for running the single trials)

#TODO?!: write a TwoWaySlurmTrajectoryPropagator! (run fw and bw trajectories simultaneously)

#TODO: write a TrajectoryPropagatorUntilState
#       - take an initialized engine? + workdir etc
#       - or only the options and the engine class?