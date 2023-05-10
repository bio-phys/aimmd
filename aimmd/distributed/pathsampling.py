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
# logic code to steer a distributed (T)PS simulations
import os
import abc
import typing
import asyncio
import logging
import numpy as np

from asyncmd.mdengine import EngineError, EngineCrashedError
from asyncmd.trajectory.propagate import MaxStepsReachedError

from ._config import _SEMAPHORES
from .. import TrainSet
from .pathmovers import (MCstep, PathMover, ModelDependentPathMover)
from .utils import accepted_trajs_from_aimmd_storage


logger = logging.getLogger(__name__)


# TODO: DOCUMENT!
class BrainTask(abc.ABC):
    """All BrainTasks should subclass this."""
    def __init__(self, interval=1):
        self.interval = interval

    @abc.abstractmethod
    def run(self, brain, mcstep: MCstep, sampler_idx):
        # TODO: find a smart way to pass the sampler result (if we even want to?)
        pass


class SaveTask(BrainTask):
    def __init__(self, storage, model, trainset: TrainSet,
                 interval=100, name_prefix="Central_RCModel"):
        super().__init__(interval=interval)
        self.storage = storage
        self.model = model
        self.trainset = trainset
        self.name_prefix = name_prefix

    async def run(self, brain, mcstep: MCstep, sampler_idx):
        # this only runs when total_steps % interval == 0
        # i.e. we can just save when we run
        async with _SEMAPHORES["BRAIN_MODEL"]:
            self.storage.save_trainset(self.trainset)
            savename = f"{self.name_prefix}_after_step{brain.total_steps}"
            self.storage.rcmodels[savename] = self.model


class TrainingTask(BrainTask):
    # add stuff to trainset + call model trainhook
    def __init__(self, model, trainset: TrainSet, interval=1):
        super().__init__(interval=interval)
        self.trainset = trainset
        self.model = model

    async def run(self, brain, mcstep: MCstep, sampler_idx):
        try:
            states_reached = mcstep.states_reached
            shooting_snap = mcstep.shooting_snap
            predicted_committors_sp = mcstep.predicted_committors_sp
        except AttributeError:
            # wrong kind of move?!
            logger.warning("Tried to add a step that was no shooting snapshot")
        else:
            descriptors = await self.model.descriptor_transform(shooting_snap)
            async with _SEMAPHORES["BRAIN_MODEL"]:
                # descriptors is 2d but append_point expects 1d
                self.trainset.append_point(descriptors=descriptors[0],
                                           shot_results=states_reached)
                # append the committor prediction for the SP at the time of
                # selection
                self.model.expected_p.append(predicted_committors_sp)
                # call the train hook every time, the model 'decides' on its
                # own if it trains
                self.model.train_hook(self.trainset)


# TODO: DOCUMENT
class DensityCollectionTask(BrainTask):
    # do density collection
    def __init__(self, model, first_collection=100,
                 recreate_interval=500, interval=10,
                 ):
        super().__init__(interval=interval)
        self.model = model
        self.first_collection = first_collection
        self.recreate_interval = recreate_interval
        self._last_collection = None  # starting step values for collections
        self._has_never_run = True

    async def run(self, brain, mcstep: MCstep, sampler_idx):
        if brain.storage is None:
            logger.error("Density collection/adaptation is currently only "
                         + "possible for simulations with attached storage."
                         )
            return
        if self._has_never_run:
            # get the start values from a possible previous simulation
            self._last_collection = [len(mcstep_collection)
                                     for mcstep_collection in brain.storage.mcstep_collections
                                     ]
            # minus the step we just finished before we ran
            collection_idx = brain.sampler_to_mcstepcollection[sampler_idx]
            self._last_collection[collection_idx] -= 1
            self._has_never_run = False
        dc = self.model.density_collector
        if brain.total_steps - self.first_collection >= 0:
            if ((brain.total_steps % self.interval == 0)
                or (brain.total_steps - self.first_collection == 0)):
                trajs, counts = accepted_trajs_from_aimmd_storage(
                                                storage=brain.storage,
                                                per_mcstep_collection=False,
                                                starts=self._last_collection,
                                                                  )
                async with _SEMAPHORES["BRAIN_MODEL"]:
                    await dc.add_density_for_trajectories(model=self.model,
                                                          trajectories=trajs,
                                                          counts=counts,
                                                          )
                # remember the start values for next time
                self._last_collection = [len(mcstep_collection)
                                         for mcstep_collection in brain.storage.mcstep_collections
                                         ]
        # Note that this below is an if because reevaluation should be
        # independent of adding new TPs in the same MCStep
        if brain.total_steps % self.recreate_interval == 0:
            # reevaluation time
            async with _SEMAPHORES["BRAIN_MODEL"]:
                await dc.reevaluate_density(model=self.model)


# TODO: rename chains -> samplers
#       reset and rename chain_directory_prefix -> sampler_directory_prefix
#       ...
#       re the todo below: name would now be ChainSamplerBundle (since the PathSamplingChain is now ChainSampler)

# (old) TODO: better name!? 'PathSamplingBundle'? 'PathSamplingChainBundle'?
#       then we should also rename the storage stuff from 'central_memory'
#       to 'distributed'? 'chainbundle'?
class Brain:
    """The 'brain' of the simulation."""
    # TODO: docstring + remove obsolete notes when we are done
    # tasks should do basically what the hooks do in the ops case:
    #   - call the models train_hook, i.e. let it decide if it wants to train
    #   - keep track of the current model/ save it etc
    #   - keep track of its workers/ store their results in the central trainset
    #   - 'controls' the central aimmd-storage (with the 'main' model and the trainset in it)
    #   - density collection should also be done (from) here
    #   NOTE: we run the tasks at specified frequencies like the hooks in ops
    #   NOTe: opposed to ops our tasks are an ordered list that gets done in deterministic order (at least the checking)
    #   Note: (this way one could also see the last task that is run as a pre-step task...?)
    #   TODO: make it possible to pass task state?
    sampler_directory_prefix = "sampler"

    def __init__(self, model, workdir, storage, movers_per_sampler,
                 sampler_to_mcstepcollection,
                 mover_weights_per_sampler=None, tasks=[], **kwargs):
        """
        Initialize an `aimmd.distributed.Brain`.

        Parameters:
        ----------
        model - the committor model
        workdir - a directory
        storage - aimmd.Storage
        movers_per_sampler - list of list of (initialized) PathMovers
        sampler_to_mcstepcollection - list of integers, one for each Sampler,
                                      the int indicates the index of the
                                      mcstepcollection this sampler adds its
                                      produced steps to
        mover_weights_per_sampler - None or list of list of floats, entries must
                                  be probabilities, i.e. must sum to 1
                                  if None we will take equal probabilities for
                                  all movers
        tasks - list of `BrainTask` objects,
                tasks will be checked if they should run in the order they are
                in the list after any one TPS sim has finished a trial,
                note that tasks will only be ran at their specified intervals
        """
        # TODO: we expect movers/mover_weights to be lists of lists?
        #       one for each sampler, this is lazy and put the setup burden on the user!
        # TODO: descriptor_transform and states?!
        self.model = model
        self.workdir = os.path.abspath(workdir)
        self.storage = storage
        self.tasks = tasks
        # TODO: sanity check?
        self.sampler_to_mcstepcollection = sampler_to_mcstepcollection
        # make it possible to set all existing attributes via kwargs
        # check the type for attributes with default values
        dval = object()
        for kwarg, value in kwargs.items():
            cval = getattr(self, kwarg, dval)
            if cval is not dval:
                if isinstance(value, type(cval)):
                    # value is of same type as default so set it
                    setattr(self, kwarg, value)
                else:
                    raise TypeError(f"Setting attribute {kwarg} with "
                                    + f"mismatching type ({type(value)}). "
                                    + f" Default type is {type(cval)}."
                                    )
        # and we do all setup of counters etc after to make sure they are what
        # we expect
        self._sampler_idxs_for_steps = []  # one sampler idx for each step done in the order they have finished
        # sampler-setup
        swdirs = [os.path.join(self.workdir, f"{self.sampler_directory_prefix}_{i}")
                  for i in range(len(movers_per_sampler))]
        # make the dirs (TODO: this fails if the dirs already exist!)
        [os.mkdir(d) for d in swdirs]
        self.storage.mcstep_collections.n_collections = len(movers_per_sampler)
        if mover_weights_per_sampler is None:
            # let each PathChainSampler generate equal weigths for its movers
            mover_weights_per_sampler = [None for _ in range(len(movers_per_sampler))]
        collection_per_sampler = [self.storage.mcstep_collections[idx]
                                  for idx in self.sampler_to_mcstepcollection
                                  ]
        self.samplers = [PathChainSampler(workdir=wdir,
                                          mcstep_collection=mcstep_col,
                                          modelstore=self.storage.rcmodels,
                                          sampler_idx=sampler_idx,
                                          movers=movs,
                                          mover_weights=mov_ws,
                                          )
                         for sampler_idx, (wdir, mcstep_col, movs, mov_ws)
                         in enumerate(zip(swdirs, collection_per_sampler,
                                          movers_per_sampler,
                                          mover_weights_per_sampler)
                                      )
                         ]

    @property
    def total_steps(self):
        return len(self._sampler_idxs_for_steps)

    @property
    def accepts(self):
        """
        List of 1 (accept) and 0 (reject) of length total_steps.

        Accepts are over all samplers in the order the steps finished.
        """
        counters = [0 for _ in self.samplers]
        accepts_per_sampler = [c.accepts for c in self.samplers]
        accepts = []
        for cidx in self._sampler_idxs_for_steps:
            accepts += [accepts_per_sampler[cidx][counters[cidx]]]
            counters[cidx] += 1
        return accepts

    # TODO: DOCUMENT!
    @classmethod
    def samplers_from_moverlist(cls, model, workdir, storage, n_sampler,
                                movers_cls, movers_kwargs,
                                samplers_use_same_stepcollection=False,
                                mover_weights=None,
                                tasks=[], **kwargs):
        """
        Initialize `self` with n_sampler PathChainSamplers with given movers.

        Convienience function to set up a brain with multiple identical samplers,
        each sampler is created with movers defined by movers_cls, movers_kwargs
        (and the optional mover_weights).
        If samplers_use_same_stepcollection = True all sampler will use the 
        same mcstepcollection, i.e. the first one.
        If it is False each sampler will use its own, i.e.
        `sampler_to_mcstepcollection = [i for i in range(n_sampler)]`.
        All other arguments are directly passed to `self.__init__()`.
        """
        movers_per_sampler = [[mov(**kwargs) for mov, kwargs in zip(movers_cls,
                                                                    movers_kwargs
                                                                    )
                               ]
                              for _ in range(n_sampler)
                              ]
        mover_weights_per_sampler = [mover_weights] * n_sampler
        if samplers_use_same_stepcollection:
            sampler_to_mcstepcollection = [0 for _ in range(n_sampler)]
        else:
            sampler_to_mcstepcollection = [i for i in range(n_sampler)]
        return cls(model=model, workdir=workdir, storage=storage,
                   movers_per_sampler=movers_per_sampler,
                   sampler_to_mcstepcollection=sampler_to_mcstepcollection,
                   mover_weights_per_sampler=mover_weights_per_sampler,
                   tasks=tasks, **kwargs,
                   )

    # TODO: better func name? (seed_initial_paths() or seed_initial-mcsteps()?)
    def seed_initial_paths(self, trajectories, weights=None, replace=True):
        """
        Initialize all PathChainSamplers from given trajectories.

        Creates initial MonteCarlo steps for each PathChainSampler containing
        one of the given transitions drawn at random (with given weights).

        Parameters:
        ----------
        trajectories - list of `aimmd.distributed.Trajectory`
        weights - None or list of weights, one for each trajectory
        replace - bool, whether to draw the trajectories with replacement
        """
        # TODO: should we check/make the movers check if the choosen traj
        #       satistfies the correct ensemble?!
        if any(c.current_step is not None for c in self.samplers):
            raise ValueError("Can only seed if all managed samplers have no "
                             + "current_step set.")
        if weights is not None:
            if len(weights) != len(trajectories):
                raise ValueError("trajectories and weights must have the same"
                                 + f" length, but have {len(trajectories)} and"
                                 + f" {len(weights)} respectively.")
            # normalize to probabilities
            weights = np.array(weights) / np.sum(weights)
        # draw the idxs for the trajectories
        traj_idxs = np.random.choice(np.arange(len(trajectories)),
                                     size=len(self.samplers),
                                     replace=replace,
                                     p=weights)
        # put a (dummy) MCstep with the trajectory into every PathSampling sim
        for idx, sampler in zip(traj_idxs, self.samplers):
            # assume that we can get dir from first trajectory_file, i.e. dont
            # check (and dont care) if the structure_file is somewhere else
            sdir, _ = os.path.split(trajectories[idx].trajectory_files[0])
            s = MCstep(mover=None, stepnum=0, directory=sdir,  # required
                       # our initial seed path for this sampler
                       path=trajectories[idx],
                       # initial step must be an accepted MCstate
                       accepted=True,
                       p_acc=1,
                       )
            sampler.current_step = s
            # save the initial step to storage
            sampler.mcstep_collection.append(s)

    # TODO?NOTE: is this todo still relevant? or are the functions we have on
    #            storage side enough, i.e. can the two funcs below go?
    # TODO:! write this to save the brain and its chains!
    #        only need to take care of the movers (but those should be pickleable?!)
    #        and the chainstores + brain.storage
    #        and the model and the tasks?
    #        then we can use the usual AimmdShelfs, i.e. pickle
    def object_for_pickle(self, group, overwrite):
        # currently overwrite will always be True
        return self

    def complete_from_h5py_group(self, group):
        return self

    async def run_for_n_accepts(self, n_accepts):
        # run for n_accepts in total over all samplers
        acc = 0
        sampler_tasks = [asyncio.create_task(s.run_step(self.model))
                         for s in self.samplers]
        while acc < n_accepts:
            done, pending = await asyncio.wait(sampler_tasks,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                # iterate over all done results, because there can be multiple
                # done sometimes?
                sampler_idx = sampler_tasks.index(result)
                # await 'result' to get the actual result out of the wrapped
                # task, i.e. what a call to `result.result()` contains
                # but this also raises all exceptions that might have been
                # raised by the task and suppressed so fair, i.e. check for
                # `result.exception() is None` and raise if not
                mcstep = await result
                self._sampler_idxs_for_steps += [sampler_idx]
                if mcstep.accepted:
                    acc += 1
                # run tasks/hooks
                await self._run_tasks(mcstep=mcstep,
                                      sampler_idx=sampler_idx,
                                      )
                # remove old task from list and start next step in the sampler
                # that just finished
                _ = sampler_tasks.pop(sampler_idx)
                sampler_tasks.insert(sampler_idx, asyncio.create_task(
                                     self.samplers[sampler_idx].run_step(self.model)
                                                                      )
                                     )
        # now that we have enough accepts finish all steps that are still pending
        done, pending = await asyncio.wait(sampler_tasks,
                                           return_when=asyncio.ALL_COMPLETED)
        for result in done:
            sampler_idx = sampler_tasks.index(result)
            mcstep = await result
            self._sampler_idxs_for_steps += [sampler_idx]
            if mcstep.accepted:
                acc += 1
            # run tasks/hooks
            await self._run_tasks(mcstep=mcstep,
                                  sampler_idx=sampler_idx,
                                  )
        # save self at the end
        self.storage.save_brain(self)

    async def run_for_n_steps(self, n_steps):
        # run for n_steps total in all samplers combined
        sampler_tasks = [asyncio.create_task(s.run_step(self.model))
                         for s in self.samplers]
        n_started = len(sampler_tasks)
        while n_started < n_steps:
            done, pending = await asyncio.wait(sampler_tasks,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                # iterate over all done results, because there can be multiple
                # done sometimes?
                sampler_idx = sampler_tasks.index(result)
                mcstep = await result
                self._sampler_idxs_for_steps += [sampler_idx]
                # run tasks/hooks
                await self._run_tasks(mcstep=mcstep,
                                      sampler_idx=sampler_idx,
                                      )
                # remove old task from list and start next step in the sampler
                # that just finished
                _ = sampler_tasks.pop(sampler_idx)
                sampler_tasks.insert(sampler_idx, asyncio.create_task(
                                     self.samplers[sampler_idx].run_step(self.model)
                                                                      )
                                     )
                n_started += 1

        # enough started, finish steps that are still pending
        done, pending = await asyncio.wait(sampler_tasks,
                                           return_when=asyncio.ALL_COMPLETED)
        for result in done:
            sampler_idx = sampler_tasks.index(result)
            mcstep = await result
            self._sampler_idxs_for_steps += [sampler_idx]
            # run tasks/hooks
            await self._run_tasks(mcstep=mcstep,
                                  sampler_idx=sampler_idx,
                                  )
        # save self at the end
        self.storage.save_brain(self)

    async def _run_tasks(self, mcstep, sampler_idx):
        for t in self.tasks:
            if self.total_steps % t.interval == 0:
                await t.run(brain=self, mcstep=mcstep,
                            sampler_idx=sampler_idx)


# TODO: rename to (Path)ChainSampler?
#       rename chainstore -> mcstep_collection?
#       
class PathChainSampler:
    # the single TPS simulation object:
    #   - keeps track of a single markov chain

    mcstep_foldername_prefix = "mcstep_"  # name = prefix + str(stepnum)
    mcstate_name_prefix = "mcstate_"  # name for symlinks to folders with
                                      # accepted step at stepnum
    max_retries_on_crash = 10  # maximum number of *retries* on MD engine crash
                               # i.e. setting to 1 means *retry* once on crash
    wait_time_on_crash = 60  # time to wait for cleanup of 'depending threads'
                             # after a crash, 'depending threads' are e.g. the
                             # conjugate trials in a two way simulation,
                             # in seconds!
    # TODO: make this saveable!? together with its brain!
    # TODO: make it possible to run post-processing 'hooks'?!
    # TODO: initialize from directory structure?

    def __init__(self, workdir: str, mcstep_collection, modelstore,
                 sampler_idx: int, movers: list[PathMover],
                 mover_weights: typing.Optional["list[float]"] = None):
        self.workdir = os.path.abspath(workdir)
        self.mcstep_collection = mcstep_collection
        self.modelstore = modelstore
        self.sampler_idx = sampler_idx
        self.movers = movers
        for mover in self.movers:
            # set the store for all ModelDependentpathMovers
            # this way we can initialize them without a store
            # also set the sampler_idx such that they can save the rcmodels
            if isinstance(mover, ModelDependentPathMover):
                mover.modelstore = self.modelstore
                mover.sampler_idx = self.sampler_idx
        if mover_weights is None:
            self.mover_weights = [1/len(movers) for _ in range(len(movers))]
        else:
            self.mover_weights = mover_weights
        self._rng = np.random.default_rng()
        self._accepts = []  # list of zeros and ones, one entry per trial
        self._current_step = None
        self._stepnum = 0

    @property
    def current_step(self):
        return self._current_step

    # TODO/FIXME: set self._step_num, self._accepts etc?!
    @current_step.setter
    def current_step(self, step):
        self._current_step = step

    @property
    def n_steps(self):
        return self._stepnum

    @property
    def n_accepts(self):
        return sum(self._accepts)

    @property
    def accepts(self):
        return self._accepts.copy()

    async def run_step(self, model):
        # run one MCstep from current_step
        # takes a model for SP selection
        # first choose a mover
        instep = self.current_step
        if instep is None:
            raise ValueError("current_step must be set.")
        self._stepnum += 1
        done = False
        n_crash = 0
        n_maxlen = 0
        mover = self._rng.choice(self.movers, p=self.mover_weights)
        while not done:
            step_dir = os.path.join(self.workdir,
                                    f"{self.mcstep_foldername_prefix}"
                                    + f"{self._stepnum}"
                                    )
            os.mkdir(step_dir)
            try:
                outstep = await mover.move(instep=instep, stepnum=self._stepnum,
                                           wdir=step_dir, model=model)
            except MaxStepsReachedError as e:
                # error raised when any trial takes "too long" to commit
                logger.error(f"Sampler {self.sampler_idx}: MaxStepsReachedError, "
                             + "retrying MC step from scratch.")
                os.rename(step_dir, step_dir + f"_max_len{n_maxlen+1}")
                n_maxlen += 1
            except EngineCrashedError as e:
                # catch error raised when gromacs crashes
                if n_crash < self.max_retries_on_crash:
                    logger.error(f"Sampler {self.sampler_idx}: MD engine crashed"
                                 + f"for the {n_crash + 1}th time, "
                                 + "retrying MC step for another "
                                 + f"{self.max_retries_on_crash - n_crash} "
                                 + "times.")
                    # wait a bit for everything to finish the cleanup
                    await asyncio.sleep(self.wait_time_on_crash)
                    # move stepdir and retry
                    os.rename(step_dir, step_dir + f"_crash{n_crash+1}")
                    n_crash += 1
                else:
                    # reached maximum tries, raise the error and crash the sampling :)
                    raise e from None
            else:
                done = True

        self.mcstep_collection.append(outstep)
        outstep.save()  # write it to a pickle file next to the trajectories
        if outstep.accepted:
            self._accepts.append(1)
            self.current_step = outstep
            os.symlink(step_dir, os.path.join(self.workdir,
                                              f"{self.mcstate_name_prefix}"
                                              + f"{self._stepnum}"
                                              )
                       )
        else:
            self._accepts.append(0)
            # link the old state as current/accepted state
            os.symlink(instep.directory,
                       os.path.join(self.workdir, f"{self.mcstate_name_prefix}"
                                                  + f"{self._stepnum}"
                                    )
                       )
        return outstep
