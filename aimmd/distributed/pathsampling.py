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
import pickle
import shutil
import typing
import asyncio
import logging
import datetime
import functools
import numpy as np

from asyncmd.mdengine import EngineError, EngineCrashedError
from asyncmd.trajectory.propagate import MaxStepsReachedError
from asyncmd.trajectory.functionwrapper import TrajectoryFunctionWrapper

from ._config import _SEMAPHORES
from .. import TrainSet
from .pathmovers import (MCstep, PathMover, ModelDependentPathMover)
from .utils import accepted_trajs_from_aimmd_storage


logger = logging.getLogger(__name__)


class BrainTask(abc.ABC):
    """
    Abstract base class for all `BrainTask`s.

    `BrainTask`s are run at the specifed interval after a step is done. They
    can be used to keep track of simulation results and alter the behavior of
    the simulation easily. See e.g. the `SaveTask` or `TrainingTask` (which
    adds the steps to a training set and trains the model). They are similar
    to openpathsampling hooks.
    """

    def __init__(self, interval: int = 1):
        self.interval = interval

    @abc.abstractmethod
    async def run(self, brain, mcstep: MCstep, sampler_idx: int):
        """
        This method is called by the `Brain` every `interval` steps.

        It is called with the brain performing the simulation, the mcstep that
        just finished and the sampler index if the sampler that did the step.
        """
        raise NotImplementedError


class SaveTask(BrainTask):
    """Save the model and trainset at given interval (in steps) to storage."""

    def __init__(self, storage, model, trainset: TrainSet,
                 interval: int = 100, name_prefix: str = "Central_RCModel"):
        """
        Initialize a :class:`SaveTask`.

        Parameters
        ----------
        storage : aimmd.Storage
            The storage to save to.
        model : aimmd.rcmodel.RCModel
            The reaction coordinate model to save.
        trainset : TrainSet
            The trainset to save.
        interval : int, optional
            Save interval in simulation steps, by default 100
        name_prefix : str, optional
            Save name prefix for the reaction coordinate model, savename will
            be `f"{name_prefix}_after_step{step_num}"`,
            by default "Central_RCModel"
        """
        super().__init__(interval=interval)
        self.storage = storage
        self.model = model
        self.trainset = trainset
        self.name_prefix = name_prefix

    async def run(self, brain, mcstep: MCstep, sampler_idx: int):
        """This method is called by the `Brain` every `interval` steps."""
        # this only runs when total_steps % interval == 0
        # i.e. we can just save when we run
        async with _SEMAPHORES["BRAIN_MODEL"]:
            self.storage.save_brain(brain=brain)
            self.storage.save_trainset(self.trainset)
            savename = f"{self.name_prefix}_after_step{brain.total_steps}"
            self.storage.rcmodels[savename] = self.model


class TrainingTask(BrainTask):
    """
    Update trainingset and train model.

    This task adds the shooting results of the finished steps to the given
    trainingset and train the model (or better: let the model decide if it
    wants to train).
    """

    def __init__(self, model, trainset: TrainSet):
        """
        Initialize a :class:`TrainingTask`.

        Parameters
        ----------
        model : aimmd.rcmodel.RCModel
            The reaction coordinate model to train.
        trainset : TrainSet
            The trainset to which we add the shooting results.
        """
        # interval must be 1 otherwise we dont add every step to the trainset
        super().__init__(interval=1)
        self.trainset = trainset
        self.model = model

    async def run(self, brain, mcstep: MCstep, sampler_idx: int):
        """This method is called by the `Brain` every `interval` steps."""
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


class DensityCollectionTask(BrainTask):
    """
    Perform density collection and update the estimate as requested.

    Density collection keeps track of the density of potential shooting points
    projected into the space of the committor. This enables flattening the
    shooting point distribution and the biasing towards the transition state is
    not influenced by non-uniform distribution of points in committor space.

    This class supports different `modes` which refer to the distribution that
    should be flattened (and which should match the the shooting point
    distribution):

        - "p_x_TP" flattens the density of points along transitions, p(x|TP),
          useful if your shooting points are selected from the last accepted
          transition

        - "custom" flattens the density of points from a given list of
          trajectories, optionally with associated weights provided as a list
          of np.ndarrays. They are provided at initialization as `trajectories`
          and `trajectory_weights` respectively.
    """

    def __init__(self, model, first_collection: int = 100,
                 recreate_interval: int = 500, mode: str = "p_x_TP",
                 trajectories=None, trajectory_weights=None,
                 ):
        """
        Initialize a :class:`DensityCollectionTask`.

        Parameters
        ----------
        model : aimmd.rcmodel.RCModel
            The reaction coordinate model for which we should do density
            collection.
        first_collection : int, optional
            After how many simulation steps we should do the first collection,
            by default 100
        recreate_interval : int, optional
            After how many simulation steps should we use the current model to
            update the estimate of the density (because the model predictions
            have changed sufficiently enough), by default 500
        mode : str, optional
            Which distribution of points should we try to flatten, can be one
            of "p_x_TP" or "custom", by default "p_x_TP"
        trajectories : list[Trajectory], optional
            Only relevant for mode "custom", the configurations of the ensemble
            of points we want to flatten, by default None
        trajectory_weights : list[np.ndarray], optional
            Only relevant for mode "custom", the optional weights for the
            configurations provided as `trajectories`, by default None
        """
        # we use interval=1 because we check for first_collection and recreate
        super().__init__(interval=1)
        self.model = model
        self.first_collection = first_collection
        self.recreate_interval = recreate_interval
        self._last_collection = None  # starting step values for collections
        self._has_never_run = True
        self.mode = mode
        # TODO: check that trajectories and trajectory_weights are not None if
        #       we are in "custom" mode?!
        self._trajectories = trajectories
        self._trajectory_weights = trajectory_weights

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val: str):
        allowed = ["p_x_TP", "custom"]
        if val not in allowed:
            raise ValueError(f"`mode` must be one of {allowed} (was {val}).")
        self._mode = val

    async def _run_p_x_TP(self, brain, mcstep: MCstep, sampler_idx: int):
        if brain.storage is None:
            logger.error("Density collection/adaptation is currently only "
                         "possible for simulations with attached storage."
                         )
            return
        if self._has_never_run:
            # get the start values from a possible previous simulation
            self._last_collection = [
                    len(mcstep_collection)
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
                self._last_collection = [
                    len(mcstep_collection)
                    for mcstep_collection in brain.storage.mcstep_collections
                                         ]
            # Note that this below is an if because reevaluation should be
            # independent of adding new TPs in the same MCStep
            if brain.total_steps % self.recreate_interval == 0:
                # reevaluation time
                async with _SEMAPHORES["BRAIN_MODEL"]:
                    await dc.reevaluate_density(model=self.model)

    async def _run_custom(self, brain, mcstep: MCstep, sampler_idx: int):
        dc = self.model.density_collector
        if brain.total_steps - self.first_collection >= 0:
            if brain.total_steps - self.first_collection == 0:
                async with _SEMAPHORES["BRAIN_MODEL"]:
                    await dc.add_density_for_trajectories(
                                            model=self.model,
                                            trajectories=self._trajectories,
                                            counts=self._trajectory_weights,
                                                          )
                self._has_never_run = False
        if brain.total_steps % self.recreate_interval == 0:
            if self._has_never_run:
                # if we have not added the trajectories yet we need to do that
                async with _SEMAPHORES["BRAIN_MODEL"]:
                    await dc.add_density_for_trajectories(
                                            model=self.model,
                                            trajectories=self._trajectories,
                                            counts=self._trajectory_weights,
                                                          )
                self._has_never_run = False
            else:
                # otherwise we can just reevaluate with the current model
                async with _SEMAPHORES["BRAIN_MODEL"]:
                    await dc.reevaluate_density(model=self.model)

    async def run(self, brain, mcstep: MCstep, sampler_idx):
        """This method is called by the `Brain` every `interval` steps."""
        if self.mode == "p_x_TP":
            await self._run_p_x_TP(brain=brain, mcstep=mcstep,
                                   sampler_idx=sampler_idx)
        elif self.mode == "custom":
            await self._run_custom(brain=brain, mcstep=mcstep,
                                   sampler_idx=sampler_idx)


class StorageCheckpointTask(BrainTask):
    """
    Create checkpoints of the :class:`aimmd.Storage` used in the TPS simulation.

    This class creates copies (checkpoints) of the :class:`aimmd.Storage` used
    with the TPS simulation in regular intervals. It uses a turnover system
    inspired by gromacs, i.e. it creates a checkpoint and if a checkpoint
    already exists it will move that checkpoint to checkpoint_previous before
    creating the next checkpoint. This ensures that even if something
    unexpected happens during the checkpoint creation there will always be a
    useable checkpoint copy.
    """

    def __init__(self,
                 storage,
                 interval: int = 50,
                 checkpoint_suffix: str = ".ckpt",
                 checkpoint_prev_suffix: str = "_prev",
                 ):
        """
        Initialize a :class:`StorageCheckpointTask`.

        Parameters
        ----------
        storage : aimmd.Storage
            The :class:`aimmd.Storage` to create checkpoints of.
        interval : int, optional
            The interval (in Monte Carlo steps) in which a checkpoint should be
            created, by default 50
        checkpoint_suffix : str, optional
            The suffix/string to append to the name of the storage to deduce
            the checkpoint name, by default ".ckpt"
        checkpoint_prev_suffix : str, optional
            The suffix/string to append to the name of the checkpoint to deduce
            the checkpoint previous name, by default "_prev"
        """
        super().__init__(interval=interval)
        self.storage = storage
        self.checkpoint_suffix = checkpoint_suffix
        self.checkpoint_prev_suffix = checkpoint_prev_suffix

    async def run(self, brain, mcstep: MCstep, sampler_idx: int):
        """
        This method is called by the `Brain` every `interval` steps.

        In this Task it is used to create the checkpoint and potentially move
        the last checkpoint to previous checkpoint.
        """
        # Flush the storage buffer, ensuring that all data is written to the underlying file
        self.storage.file.flush()
        # Get the name of the file used for storage
        fname = self.storage.file.filename
        _, tail = os.path.split(fname)
        # Use storage._dirname to find the directory the storage and checkpoint
        # should live in
        fname = os.path.join(self.storage._dirname, tail)
        checkpoint_fname = fname + self.checkpoint_suffix

        # Check if a checkpoint file already exists
        if os.path.isfile(fname + self.checkpoint_suffix):
            # If there's also a previous checkpoint file, remove it
            prev_checkpoint_fname = fname + self.checkpoint_suffix + self.checkpoint_prev_suffix
            if os.path.isfile(prev_checkpoint_fname):
                os.remove(prev_checkpoint_fname)
                logger.debug("Removed previous checkpoint %s.",
                             prev_checkpoint_fname,
                             )
            # Rename the current checkpoint file to mark it as the previous one
            os.rename(fname + self.checkpoint_suffix, prev_checkpoint_fname)
            logger.info("Moved last checkpoint (%s) to previous checkpoint (%s).",
                        checkpoint_fname, prev_checkpoint_fname)
        # Actually copy the current state to the checkpoint file
        shutil.copy2(fname, fname + self.checkpoint_suffix)
        # Inform the user about the checkpointing action
        logger.info("Copied storage file %s to create checkpoint %s.",
                    fname, checkpoint_fname,
                    )


# TODO: DOCUMENT! Directly write a paragraph of documentation for use with sphinx!?
#       We would need:
#        - a few words on the folder structure with mcsteps and symlinks to the
#          accepts
#       - a few words on the saved pickles for each step (which contain the
#         mcstep object)
#       - the idea that the brain steers many samplers and how to define them
#         using the PathMover lists (and weights), i.e. that a random mover is
#         drawn (with given weight) and that this mover then does the move to
#         get the Markov chain one step forward (thereby keeping detailed
#         balance if every mover keeps it on its own)
class Brain:
    """
    The `Brain` of the path sampling simulation.

    This is the central user-facing object to run and analyze your simulation.
    It controls multiple `PathChainSampler` objects simultaneously, running
    trials in all of them at the same time. Depending on the settings, each
    `PathChainSampler` represents and samples its own Markov Chain or multiple/
    all samplers add their steps to the same collection with a known weight.
    The possible trial moves performed in each `PathChainSampler` depend on the
    list of `movers` (and `mover_weights`) associated with it.

    See the classmethod :meth:`samplers_from_moverlist` to setup a `Brain` with
    a given number of identical `PathChainSampler`s.

    Attributes
    ----------
    sampler_directory_prefix : str
        Prefix for sampler directories, by default "sampler_"
    """

    # TODO: make it possible to pass task state? (Do we even need that?)
    sampler_directory_prefix = "sampler_"

    def __init__(self, model, workdir, storage, sampler_to_mcstepcollection,
                 movers_per_sampler, mover_weights_per_sampler=None, tasks=[],
                 **kwargs):
        """
        Initialize a :class:`Brain`.

        Parameters
        ----------
        model : aimmd.rcmodel.RCModel
            The reaction coordinate model (potentially) selecting shooting
            points.
        workdir : str
            The directory in which the simulation should take place.
        storage : aimmd.Storage
            The storage to save the results to.
        sampler_to_mcstepcollection : list[int]
            A list with one entry for each `PathChainSampler`, the int
            indicates the index of the mcstepcollection this sampler uses to
            store its produced MCsteps
        movers_per_sampler : list[list[PathMover]]
            The outer list contains a list for each PathChainSampler, defining
            the movers this sampler should use. I.e. the outer list has the
            same length as we have/want `PathChainSampler`s, the inner lists
            can be of different lenght for each sampler and contain different
            `PathMover`s (just take care that the weights match the movers).
        mover_weights_per_sampler : None or list[list[float]]
            The outer list contains a list for each PathChainSampler, defining
            the weights for the movers in this sampler. For each sampler the
            entries in the list must be probabilities, i.e. must sum to 1.
            If None (or if one of the list for one of the samplers is None), we
            will use equal probabilities for all movers (in this sampler).
        tasks : list[BrainTask]
            List of `BrainTask` objects to run at their specified intervals,
            tasks will be checked if they should run in the order they are in
            the list after any one TPS sim has finished a trial.
        """
        # TODO: do we want descriptor_transform and states here at a central
        #       place?! Has the benefit that we dont need to pass it to every
        #       mover, but the big drawback of assuming we only ever want to do
        #       *T*PS with this class
        self._check_model(model=model, storage=storage)
        self.model = model
        self.workdir = os.path.relpath(workdir)
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
        # Keep track of which sampler did which step, we just have a list with
        # one sampler idx for each step done in the order they have finished
        self._sampler_idxs_for_steps = []
        # sampler-setup
        swdirs = [
            os.path.join(self.workdir, f"{self.sampler_directory_prefix}{i}")
            for i in range(len(movers_per_sampler))
                  ]
        # make the dirs (dont fail if the dirs already exist!)
        [os.makedirs(d, exist_ok=True) for d in swdirs]
        # we create as many stepcollections as the maximum index in
        # sampler_to_mcstepcollection makes us assume
        n_collections = max(sampler_to_mcstepcollection) + 1  # 0 based index
        self.storage.mcstep_collections.n_collections = n_collections
        if mover_weights_per_sampler is None:
            # let each PathChainSampler generate equal weigths for its movers
            mover_weights_per_sampler = [
                        None for _ in range(len(movers_per_sampler))
                                         ]
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

    def _check_model(self, model, storage):
        """Basic sanity checks for model before TPS simulation.

        Checks:
            - that the model has a descriptor transform and if it is async
            - that the model has states and if they are callable and async
            - the model.density_collector.cache file is set to storage
        """
        # if we warn about anything not beeing set as expected we should also
        # tell the user about model.ee_params (which will then most likely also
        #  be at their defaults)
        any_warned = False
        # descriptor_transform check
        if not isinstance(model.descriptor_transform, TrajectoryFunctionWrapper):
            if model.descriptor_transform is None:
                # not set at all, i.e. to None
                warn_str = "The model has no `descriptor_transform` set."
            else:
                # it is set but not to what we expect
                warn_str = "The model has a `descriptor_transform` that is of "
                warn_str += f"type {type(model.descriptor_transform)}."
            logger.warning(("%s In most cases the descriptor transform should "
                            "be a `asyncmd.trajectory.functionwrapper."
                            "TrajectoryFunctionWrapper` (subclass). "
                            "If your model can not operate on `asyncmd."
                            "Trajectory` objects this TPS simulation will most"
                            " likely crash."),
                           warn_str
                           )
            any_warned = True
        # states check (not so important, in practice we only infer the number
        #               of outputs for the model from the number of states)
        if model.states is None:
            logger.warning("model.states is not set. This will lead to "
                           "unexpected behavior and it is recommended to set "
                           "the states to `asyncmd.trajectory.functionwrapper."
                           ".TrajectoryFunctionwrapper subclasses."
                           )
            any_warned = True
        elif not all(isinstance(s, TrajectoryFunctionWrapper)
                     for s in model.states
                     ):
            logger.warning("Not all model.states are `asyncmd.trajectory."
                           "functionwrapper.TrajectoryFunctionwrapper` "
                           "subclasses. This might lead to unexpected behavior"
                           " and it is recommended to set them to appropriate"
                           "values."
                           )
            any_warned = True
        # density collector cache file
        # TODO: what is the best thing to do here?
        #       check if it is set to the 'correct' file (i.e. storage),
        #       -> If it is None we set it to storage and warn about it
        #       -> If it is set to a value we dont (re)set it but warn if that
        #          value is not storage?
        if not ((model.density_collector.cache_file is storage)
                or (model.density_collector.cache_file is storage.file)
                ):
            # Note: check for the h5-file and the storage class object,
            #       it could be either of the two
            if model.density_collector.cache_file is None:
                warn_str = "`model.density_collector.cache_file` is not set."
            else:
                warn_str = "`model.density_collector.cache_file` is not set to"
                warn_str += "`storage`."
            logger.warning(("%s If this was not intended it is recommended to "
                            "set `model.density_collector.cache_file` to the "
                            "same value as `storage` to avoid unexpected "
                            "side-effects."
                            ), warn_str,
                           )
            any_warned = True
        # If we warned about model.descriptor_transform, model.states or
        # model.density_collector.cache_file we also should let the users know
        # to check the ee_params (which are potentially at their defaults)
        if any_warned:
            logger.warning("It is likely that you also want to check the"
                           "`model.ee_params` dictionary and potentially "
                           "modify the default values in it to ensure that the"
                           "iterative training is controlled properly."
                           )

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

    @classmethod
    def samplers_from_moverlist(cls, model, workdir, storage, n_sampler,
                                movers_cls, movers_kwargs, mover_weights=None,
                                samplers_use_same_stepcollection=False,
                                tasks=[], **kwargs):
        """
        Initialize :class:`Brain` with n_sampler identical `PathChainSampler`s.

        This is a convienience function to set up a brain with multiple
        identical samplers, each sampler is created with the movers defined by
        `movers_cls`, `movers_kwargs` (and the optional `mover_weights`).
        If `samplers_use_same_stepcollection = True`, all sampler will use the
        same mcstepcollection with index 0, i.e. the first one.
        If it is False each sampler will use its own mcstepcollection, i.e.
        `sampler_to_mcstepcollection = [i for i in range(n_sampler)]`.
        All other arguments are directly passed to `Brain.__init__()`.

        Parameters
        ----------
        model : aimmd.rcmodel.RCModel
            The reaction coordinate model (potentially) selecting shooting
            points.
        workdir : str
            The directory in which the simulation should take place.
        storage : aimmd.Storage
            The storage to save the results to.
        n_sampler : int
            The number of (identical) `PathChainSampler`s to create.
        movers_cls : list[PathMover_classes]
            A list of (uninitialzed) `PathMover` classes defining the sampling
            scheme.
        movers_kwargs : list[dict]
            The keyword arguments used to initialize each of the mover classes
            in the list above.
        mover_weights : list[float] or None
            A list defining the weights for the movers. The entries in the list
            must be probabilities, i.e. must sum to 1. If None, we will use
            equal probabilities for all movers. By default None.
        samplers_use_same_stepcollection : bool
            Whether all `PathChainSampler`s use the same mcstepcollection, if
            False each sampler will use its own collection, by default False.
        tasks : list[BrainTask]
            List of `BrainTask` objects to run at their specified intervals,
            tasks will be checked if they should run in the order they are in
            the list after any one TPS sim has finished a trial.
        """
        movers_per_sampler = [
            [mov(**kwargs) for mov, kwargs in zip(movers_cls, movers_kwargs)]
            for _ in range(n_sampler)
                              ]
        mover_weights_per_sampler = [mover_weights] * n_sampler
        if samplers_use_same_stepcollection:
            sampler_to_mcstepcollection = [0] * n_sampler
        else:
            sampler_to_mcstepcollection = list(range(n_sampler))
        return cls(model=model, workdir=workdir, storage=storage,
                   movers_per_sampler=movers_per_sampler,
                   sampler_to_mcstepcollection=sampler_to_mcstepcollection,
                   mover_weights_per_sampler=mover_weights_per_sampler,
                   tasks=tasks, **kwargs,
                   )

    def seed_initial_paths(self, trajectories, weights=None, replace=True):
        """
        Initialize all `PathChainSampler`s from given trajectories.

        Creates initial MonteCarlo steps for each PathChainSampler containing
        one of the given transitions drawn at random (with the given weights).

        Parameters
        ----------
        trajectories : list[asyncmd.Trajectory]
            The input paths to use.
        weights : None or list[float]
            The weights to use, must have one entry for each trajectory. If
            None we will use equal weights for all trajectories.
        replace : bool, optional
            Whether to draw the trajectories with replacement, by default True.
        """
        # TODO: should we check/make the movers check if the choosen traj
        #       satistfies the correct ensemble?! (For this we would need to
        #       know the states and maybe more, which has the caveats discussed
        #       above at Brain.__init__)
        if any(c.current_step is not None for c in self.samplers):
            raise ValueError("Can only seed if all managed samplers have no "
                             + "current_step set.")
        if self.total_steps > 0:
            raise ValueError("Can only seed initial steps if no steps have "
                             "been finished yet, but "
                             f"total_steps={self.total_steps}."
                             )
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
        for sidx, (idx, sampler) in enumerate(zip(traj_idxs, self.samplers)):
            # save the first dummy step to its own dir as step 0!
            # we just create the dir and save only the picklesuch that we can
            # know about the initial traj/path for the chain when using
            # reinitialize_from_workdir
            step_dir = os.path.join(
                        self.workdir,
                        f"{self.sampler_directory_prefix}{sidx}",
                        f"{sampler.mcstep_foldername_prefix}{sampler._stepnum}"
                                    )
            os.mkdir(step_dir)
            s = MCstep(mover=None, stepnum=0, directory=step_dir,  # required
                       # our initial seed path for this sampler
                       path=trajectories[idx],
                       # initial step must be an accepted MCstate
                       accepted=True,
                       p_acc=1,
                       )
            # set the step as current step and adds it to samplers storage
            # also save it as step zero
            sampler._store_finished_step(step=s, save_step_pckl=True,
                                         make_symlink=False,
                                         is_step_zero=True,
                                         # ^^^ do not add to sampler._accepts
                                         )

    async def reinitialize_from_workdir(self, run_tasks: bool = True,
                                        finish_steps: bool = False,
                                        resave_mcstep_pckl: bool = False,
                                        ):
        """
        Reinitialize the `Brain` from an existing working/simulation directory.

        This method first finds all finished steps and adds them to the `Brain`
        and storage in the order they have originally been produced (using the
        mtime from the operating system). It then searches for any unfinished
        steps and finishes them before adding them to the storage and `Brain`.

        If `run_tasks` is True (the default) the `Brain` will run all attached
        `BrainTask`s (like density collection and training) for all steps it
        adds as if they would have been just simulated.

        This method can be used to restart a crashed or otherwise prematurely
        terminated simulation to a new storage. In this case the `Brain` should
        be initialized exactly as at the start of the originial simulation when
        calling this function, i.e. with `movers`, `workdir`, etc. set but no
        steps have been performed yet.
        This method can also be used to restart a simulation with a different
        descriptor_transform and/or reaction coordinate model architecture.
        Note that also in this case the `Brain` needs to be initialized very
        similar to what you originaly had set when starting the simulation.
        For more details on what can be changed and what not please consult the
        example notebooks and the documentation.

        Parameters
        ----------
        run_tasks : bool, optional
            If we should run the `BrainTask`s, by default True
        finish_steps : bool, optional
            If we should finish the unfinished steps, by default False.
            If False, we will put the unfinished steps into the respective
            samplers and finish/continue them as soon as we run again.
            NOTE, that it can be beneficial to set finish_steps=False if you
            intend to add more steps than the unfinished ones.
            You would then call `run_for_n_steps` or `run_for_n_accepts`
            directly after this function with the amount of steps/accepts you
            want to add. The benefit of this strategy is that the new and the
            unfinished steps will then be done concurrently, instead of waiting
            for the unfinished steps to be done and then after they are all
            done start the new steps.
        resave_mcstep_pckl : bool, optional
            Whether we should rewrite the MCStep pickle file, by default False.
            This can be useful e.g. to update all python objects to a new
            version of aimmd/asyncmd or to update all filepaths/workdirs after
            moving the simulation.
            Note that the previous mcstep pickle files will be moved and not
            overwritten.

        Raises
        ------
        ValueError
            If the `Brain` has already started another simulation, i.e. if
            `self.total_steps>0`
        """
        if self.total_steps > 0:
            raise ValueError("Can only reinitialize brain object if no steps "
                             "have been done yet, but total_steps="
                             f"{self.total_steps}.")
        # first we find all finished steps for each sampler
        # then we order them by finish-time, add them to storage in that order
        steps_by_samplers = [s._finished_steps_from_workdir()
                             for s in self.samplers]
        steptimes_by_samplers = [sorted(steps.keys())
                                 for steps in steps_by_samplers]
        steptimes_sorted = sorted(sum(steptimes_by_samplers, []))
        # add the finished steps in the right order
        while len(steptimes_sorted) > 0:
            st = steptimes_sorted.pop(0)
            # find the index of the sampler it comes from
            for sidx, steptimes in enumerate(steptimes_by_samplers):
                if st in steptimes:
                    step = steps_by_samplers[sidx][st]
                    # remove this time from this sampler (so we would get the
                    #  next sampler with exactly the same time if that ever
                    #  happens)
                    steptimes_by_samplers[sidx].remove(st)
                    # Note that our initial seeded steps have mover=None,
                    # they should not trigger stepnum increase or task
                    # execution and are also not included in the list to
                    # mapp steps to samplers
                    if step.mover is not None:
                        # NOTE: same order as when the step is intitially
                        #       produced, stepnum is increased at the start of
                        #       make_step, tasks are run in/directly after the
                        #       step finishes, then it is stored and finally
                        #       processed/added to self
                        # increase the samplers step counter
                        self.samplers[sidx]._stepnum += 1
                        # run tasks (if we should)
                        if run_tasks:
                            await self._run_tasks(mcstep=step, sampler_idx=sidx)
                        # store the finished step for the sampler it came from
                        self.samplers[sidx]._store_finished_step(
                                            step=step,
                                            save_step_pckl=resave_mcstep_pckl,
                                            make_symlink=False,
                                                                 )
                        # add the step to self
                        self._sampler_idxs_for_steps += [sidx]
                    else:
                        # store the zeroth step, do not add to sampler._accepts
                        self.samplers[sidx]._store_finished_step(
                                            step=step,
                                            save_step_pckl=resave_mcstep_pckl,
                                            make_symlink=False,
                                            is_step_zero=True,
                                                                 )

        info_str = f"After adding all finished steps we have a total of {self.total_steps} steps."
        if finish_steps:
            info_str += " Now working on the unfinished ones."
        else:
            info_str += " Note that potential unfinished steps will only be "
            info_str += "finished when calling `Brain.run_for_n_steps()` or "
            info_str += "Brain.run_for_n_accepts()`."
        print(info_str)
        if not finish_steps:
            # get out of here if we dont want to finish the steps (but do more)
            return
        # now we should only have unfinished steps left
        # we run them by running finish_step in each sampler, it finishes and
        # returns the step or returns None if no unfinished steps are present
        sampler_tasks = [asyncio.create_task(
                            s.finish_step(
                                model=self.model,
                                brain_task_callback=functools.partial(self._run_tasks,
                                                                      sampler_idx=s_idx)
                                          )
                                             )
                         for s_idx, s in enumerate(self.samplers)]
        # run them all and get the results in the order they are done
        pending = sampler_tasks
        while len(pending) > 0:
            done, pending = await asyncio.wait(pending,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                sampler_idx = sampler_tasks.index(result)
                mcstep = await result
                if mcstep is None:
                    # no unfinished steps in this sampler, go to next result
                    continue
                self._sampler_idxs_for_steps += [sampler_idx]

    async def run_for_n_accepts(self, n_accepts: int,
                                print_progress: typing.Optional[int] = None):
        """
        Run simulation until `n_accepts` accepts have been produced.

        `n_accepts` is counted over all samplers. Note that the concept of
        accepts might not make too much sense for your sampling scheme, e.g.
        when shooting from points with know equilibrium weights.

        Parameters
        ----------
        n_accepts : int
            Total number of accepts to produce.
        print_progress : typing.Optional[int], optional
            Print a short progress summary every `print_progress` steps,
            print nothing if `print_progress=None`, by default None
        """
        # run for n_accepts in total over all samplers
        acc = 0
        sampler_tasks = [asyncio.create_task(
                            s.make_step(
                                model=self.model,
                                brain_task_callback=functools.partial(self._run_tasks,
                                                                      sampler_idx=s_idx)
                                        )
                                             )
                         for s_idx, s in enumerate(self.samplers)]
        n_done = 0
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
                n_done += 1
                if print_progress is not None:
                    if n_done % print_progress == 0:
                        pstr = f"{n_done} steps done."
                        pstr += f" Produced {acc} accepts so far in this run."
                        print(pstr)
                # remove old task from list and start next step in the sampler
                # that just finished
                _ = sampler_tasks.pop(sampler_idx)
                sampler_tasks.insert(
                                sampler_idx,
                                asyncio.create_task(
                                    self.samplers[sampler_idx].make_step(
                                        model=self.model,
                                        brain_task_callback=functools.partial(self._run_tasks,
                                                                              sampler_idx=sampler_idx)
                                                                         )
                                                    )
                                     )
        # now that we have enough accepts finish all that are still pending
        # get the remaining steps in the order the steps finish
        pending = sampler_tasks
        while len(pending) > 0:
            done, pending = await asyncio.wait(pending,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                sampler_idx = sampler_tasks.index(result)
                mcstep = await result
                self._sampler_idxs_for_steps += [sampler_idx]
                if mcstep.accepted:
                    acc += 1
                n_done += 1
                if print_progress is not None:
                    if n_done % print_progress == 0:
                        pstr = f"{n_done} steps done."
                        pstr += f" Produced {acc} accepts so far in this run."
                        print(pstr)

    async def run_for_n_steps(self, n_steps: int,
                              print_progress: typing.Optional[int] = None):
        """
        Run path sampling simulation for `n_steps` steps.

        `n_steps` is the total number of steps in all samplers combined.

        Parameters
        ----------
        n_steps : int
            Total number of steps (Monte Carlo trials) to run.
        print_progress : typing.Optional[int], optional
            Print a short progress summary every `print_progress` steps,
            print nothing if `print_progress=None`, by default None
        """
        if n_steps < len(self.samplers):
            # Do not start a step in every sampler, but only as many as
            # requested. Check which samplers contain unfinished steps and do
            # those first.
            # NOTE: if we have more unfinished steps than we are requested to
            #       do steps, we will still have unfinished steps afterwards
            n_incomplete = sum(s.contains_partial_step for s in self.samplers)
            # how many samplers we need to start fresh to get to n_steps
            # samplers running including all unfinished
            n_to_start_new = max((n_steps - n_incomplete, 0))
            sampler_tasks = []
            n_started = 0
            s_idx = 0
            while n_started < n_steps:
                sampler = self.samplers[s_idx]
                brain_task_callback = functools.partial(self._run_tasks,
                                                        sampler_idx=s_idx)
                if sampler.contains_partial_step:
                    sampler_tasks += [asyncio.create_task(
                                        sampler.make_step(
                                            model=self.model,
                                            brain_task_callback=brain_task_callback,
                                                          )
                                                          )
                                      ]
                    n_started += 1
                elif n_to_start_new > 0:
                    # we still got fresh samplers to start
                    sampler_tasks += [asyncio.create_task(
                                        sampler.make_step(
                                            model=self.model,
                                            brain_task_callback=brain_task_callback,
                                                          )
                                                          )
                                      ]
                    n_started += 1
                    n_to_start_new -= 1
                else:
                    # dont start more fresh samplers, instead put a dummy task
                    # into sampler_tasks
                    sampler_tasks += [asyncio.create_task(asyncio.sleep(0))]
                # always increase sampler idx for next iter
                s_idx += 1
        else:
            # more steps to do than we have samplers, so start a step in every
            # sampler and then go into the while loop below
            sampler_tasks = [asyncio.create_task(
                                s.make_step(
                                    model=self.model,
                                    brain_task_callback=functools.partial(self._run_tasks,
                                                                          sampler_idx=s_idx)
                                            )
                                                 )
                             for s_idx, s in enumerate(self.samplers)]
            n_started = len(sampler_tasks)
        # we only go into the loop below if the user requested more steps than
        # we have samplers, otherwise we will skip to the last loop directly
        # (because then n_started = n_steps)
        n_done = 0
        while n_started < n_steps:
            done, pending = await asyncio.wait(sampler_tasks,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                # iterate over all done results, because there can be multiple
                # done sometimes?
                sampler_idx = sampler_tasks.index(result)
                mcstep = await result
                if mcstep is None:
                    # bail out because this mcstep was a dummy sampler, i.e. None
                    continue
                self._sampler_idxs_for_steps += [sampler_idx]
                n_done += 1
                if print_progress is not None:
                    if n_done % print_progress == 0:
                        pstr = f"{n_done} (of {n_steps}) steps done."
                        pstr += f" Produced {sum(self.accepts[-n_done:])}"
                        pstr += " accepts so far in this run."
                        print(pstr)
                # remove old task from list and start next step in the sampler
                # that just finished
                _ = sampler_tasks.pop(sampler_idx)
                sampler_tasks.insert(
                                sampler_idx,
                                asyncio.create_task(
                                    self.samplers[sampler_idx].make_step(
                                        model=self.model,
                                        brain_task_callback=functools.partial(self._run_tasks,
                                                                              sampler_idx=sampler_idx)
                                                                         )
                                                    )
                                     )
                n_started += 1
        # enough started, finish steps that are still pending
        # get them in the order the steps finish
        pending = sampler_tasks
        while len(pending) > 0:
            done, pending = await asyncio.wait(pending,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                sampler_idx = sampler_tasks.index(result)
                mcstep = await result
                if mcstep is None:
                    # bail out because this mcstep is a dummy sampler, i.e. None
                    continue
                self._sampler_idxs_for_steps += [sampler_idx]
                n_done += 1
                if print_progress is not None:
                    if n_done % print_progress == 0:
                        pstr = f"{n_done} (of {n_steps}) steps done."
                        pstr += f" Produced {sum(self.accepts[-n_done:])}"
                        pstr += " accepts so far in this run."
                        print(pstr)

    async def _run_tasks(self, mcstep, sampler_idx):
        for t in self.tasks:
            if self.total_steps % t.interval == 0:
                await t.run(brain=self, mcstep=mcstep,
                            sampler_idx=sampler_idx)
        return mcstep


# TODO: DOCUMENT!
class PathChainSampler:
    # Keeps track of a single markov chain or produces samples for a shared
    # ensemble, but in both cases runs independent of other samplers

    mcstep_foldername_prefix = "mcstep_"  # name = prefix + str(stepnum)
    mcstate_name_prefix = "mcstate_"  # name for symlinks to folders with
                                      # accepted step at stepnum
    max_retries_on_crash = 10  # maximum number of *retries* on MD engine crash
                               # i.e. setting to 1 means *retry* once on crash
    wait_time_on_crash = 60  # time to wait for cleanup of 'depending threads'
                             # after a crash, 'depending threads' are e.g. the
                             # conjugate trials in a two way simulation,
                             # in seconds!
    restart_info_fname = ".restart_info.pckl"
    # TODO: make it possible to run post-processing 'hooks'/tasks?!
    #       this would be similar to what we have with the Brain but here we
    #       would pass in the MCstep as done by the PathMover and then let the
    #       hook/task alter it (e.g. add/calculate additional data) and then
    #       pass on the ammended MCstep to the Brain for storing/training/etc

    def __init__(self, workdir: str, mcstep_collection, modelstore,
                 sampler_idx: int, movers: list[PathMover],
                 mover_weights: typing.Optional["list[float]"] = None):
        self.workdir = os.path.relpath(workdir)
        self.mcstep_collection = mcstep_collection
        self.modelstore = modelstore
        self.sampler_idx = sampler_idx
        self.movers = movers
        # NOTE: option to always accept the first/next transition generated
        #       this is a bit hidden (and needs to be explicitly enabled by the
        #       user for each Sampler), because it is a dangerous option.
        #       It can however be very useful to quickly start a new simulation
        #       with extended state definitions from a previous TPS simulation.
        #       Note that we set p_acc = inf for the step that we forced to be
        #       accepted, such that they can be found in the analysis.
        self.always_accept_next_TP = False
        for mover in self.movers:
            mover.sampler_idx = self.sampler_idx
            # set the store for all ModelDependentpathMovers
            # this way we can initialize them without a store
            # also set the sampler_idx such that they can save the rcmodels
            if isinstance(mover, ModelDependentPathMover):
                mover.modelstore = self.modelstore
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
        """The current active MCStep of the `PathChainSampler`."""
        return self._current_step

    # TODO/FIXME: set self._step_num, self._accepts etc?!
    @current_step.setter
    def current_step(self, step):
        self._current_step = step

    @property
    def n_steps(self) -> int:
        """The total number of MCSteps this `PathChainSampler` did."""
        return self._stepnum

    @property
    def n_accepts(self) -> int:
        """The total number of accepted MCSteps this `PathChainSampler` did."""
        return sum(self._accepts)

    @property
    def accepts(self) -> list[int]:
        """
        The history of accepts of this `PathChainSampler` as ordered list.

        The list will always be of length `self.n_steps` and contains a zero or
        a one for each MCStep, depending on if it was accepted (one) or not
        accepted (zero).
        """
        return self._accepts.copy()

    @property
    def contains_partial_step(self) -> bool:
        """
        Whether this `PathChainSampler` currently contains an unfinished step.
        """
        # check if there is an unfinished step in this sampler, which can only
        # happen when we did reinitialize the Brain from workdir using
        # reinitialize_from_workdir and passed finish_steps=False
        # _make_step increases the _stepnum, so we are looking for the next step
        step_dir = os.path.join(self.workdir,
                                f"{self.mcstep_foldername_prefix}"
                                + f"{self._stepnum+1}"
                                )
        restart_file = os.path.join(step_dir, self.restart_info_fname)
        return os.path.isfile(restart_file)

    def _finished_steps_from_workdir(self):
        # we iterate over all possible stepdir, this might be a bit wasteful
        # if there is other stuff in the workdir (TODO: break the loop when we
        #  encounter the first non finished step?)
        # we can have only maximally as many steps done as we have entries
        max_num = len(os.listdir(self.workdir))
        all_possible_stepdirs = [os.path.join(self.workdir,
                                              self.mcstep_foldername_prefix
                                              + f"{stepnum}")
                                 for stepnum in range(max_num + 1)]
        all_steps_by_time = {}
        for stepdir in all_possible_stepdirs:
            if os.path.isdir(stepdir):
                # the step is only done if we have an MCStep pickle file
                if os.path.isfile(os.path.join(stepdir,
                                               MCstep.default_savename,
                                               )
                                  ):
                    s = MCstep.load(directory=stepdir)
                    mtime = os.path.getmtime(os.path.join(stepdir,
                                                          s.default_savename)
                                             )
                    mtime = datetime.datetime.fromtimestamp(mtime)
                    all_steps_by_time[mtime] = s

        return all_steps_by_time

    def _store_finished_step(self, step: MCstep,
                             save_step_pckl: bool = False,
                             make_symlink: bool = False,
                             instep: typing.Optional[MCstep] = None,
                             is_step_zero: bool = False,
                             ):
        self.mcstep_collection.append(step)
        if save_step_pckl:
            pickle_fname = os.path.join(step.directory, step.default_savename)
            if os.path.isfile(pickle_fname):
                # move any potential previous pickle files to make sure we can
                # go back if we need to
                num = 1
                new_fname = pickle_fname + str(num)
                while os.path.isfile(new_fname):
                    num += 1
                    new_fname = pickle_fname + str(num)
                logger.info("MCStep pickle file (%s) exists already, "
                            "moving to %s before writing the new pickle file.",
                            pickle_fname, new_fname,
                            )
                os.rename(pickle_fname, new_fname)
            # and write the pickle file next to the trajectories
            # at pickle_fname
            step.save()
        if step.accepted:
            if not is_step_zero:
                self._accepts.append(1)
            self.current_step = step
            if make_symlink:
                fd = os.open(self.workdir, os.O_RDONLY)
                os.symlink(os.path.relpath(step.directory, self.workdir),
                           f"{self.mcstate_name_prefix}{self._stepnum}",
                           dir_fd=fd,
                           )
                os.close(fd)
        else:
            if not is_step_zero:
                # we should never have a zeroth step that is not accepted, but
                # to be sure... :)
                self._accepts.append(0)
            if make_symlink:
                # link the old state as current/accepted state
                fd = os.open(self.workdir, os.O_RDONLY)
                os.symlink(os.path.relpath(instep.directory, self.workdir),
                           f"{self.mcstate_name_prefix}{self._stepnum}",
                           dir_fd=fd,
                           )
                os.close(fd)

    async def finish_step(self,
                          model,
                          brain_task_callback: typing.Optional[typing.Coroutine] = None,
                          ):
        """
        Finish the current MCStep (if any).

        This method will only finish partially finished steps and not start any
        new MCSteps, i.e. if `self.contains_partial_step == True` it will
        finish the partial step and return it, otherwise it will return None.

        Parameters
        ----------
        model : aimmd.RCModel
            The reaction coordinate (committor) model used to bias the shooting
            point selection.
        brain_task_callback : typing.Optional[typing.Coroutine], optional
            The callback to run the `BrainTask`s, will be run after finishing
            (but before saving) the MCstep. It must accept the finished MCStep
            as the only argument.
            By default None, which means run no callback/BrainTasks

        Returns
        -------
        MCStep or None
            Returns either the finished MCStep or None (if no partially
            finished MCStep exists).
        """
        # _make_step increases the _stepnum, so we are looking for the next step
        step_dir = os.path.join(self.workdir,
                                f"{self.mcstep_foldername_prefix}"
                                + f"{self._stepnum+1}"
                                )
        restart_file = os.path.join(step_dir, self.restart_info_fname)
        if os.path.isfile(restart_file):
            with open(restart_file, "rb") as pf:
                restart_info = pickle.load(pf)
            instep = restart_info["instep"]
            mover = restart_info["mover"]
            # (re)set the sampler idx
            mover.sampler_idx = self.sampler_idx
            if isinstance(mover, ModelDependentPathMover):
                # (re)set the modelstore
                mover.modelstore = self.modelstore
            # finish the step and return it
            return await self._make_step(model=model, instep=instep,
                                         mover=mover,
                                         brain_task_callback=brain_task_callback,
                                         continuation=True)
        # no restart file means no unfinshed step
        # we return None
        return None

    async def make_step(self,
                        model,
                        brain_task_callback: typing.Optional[typing.Coroutine] = None,
                        ):
        """
        Make one Monte Carlo step from current step.

        Parameters
        ----------
        model : RCModelAsync
            The reaction coordinate/committor model passed to the mover, e.g.
            used to select an optiomal shooting point for a shooting move.
        brain_task_callback : typing.Optional[typing.Coroutine], optional
            The callback to run the `BrainTask`s, will be run after finishing
            (but before saving) the MCstep. It must accept the finished MCStep
            as the only argument.
            By default None, which means run no callback/BrainTasks

        Returns
        -------
        MCstep
            The produced Monte Carlo step.
        """
        # run one MCstep from current_step
        # takes a model for SP selection
        # NOTE: we check if there is an unfinished step in here and do finsh it
        #       if yes (this enables us to call run_for_n_steps method on a
        #       brain with some unfinished steps, do all wanted steps in
        #       parallel, i.e unfinshed and new without having to wait for the
        #       unfinished steps first)
        if self.contains_partial_step:
            # we found a restart file, load it and finish the step
            # note that finish step will in this case always return a MC step
            # (instead of None), because there is a restart file, such that
            # make_step will always return a valid mcstep as expected
            return await self.finish_step(model, brain_task_callback)
        # no restart file found, so do a step from scratch
        instep = self.current_step
        if instep is None:
            logger.warning("Sampler %d: Instep is None."
                           " This will only work with sampling schemes that"
                           " generate their own shooting points.",
                           self.sampler_idx)
        # choose a mover
        mover = self._rng.choice(self.movers, p=self.mover_weights)
        # and make/run the actual step
        return await self._make_step(model=model, instep=instep, mover=mover,
                                     brain_task_callback=brain_task_callback,
                                     continuation=False)

    async def _make_step(self, model, mover: PathMover,
                         instep: typing.Optional[MCstep],
                         brain_task_callback: typing.Optional[typing.Coroutine],
                         continuation: bool):
        self._stepnum += 1
        step_dir = os.path.join(self.workdir,
                                f"{self.mcstep_foldername_prefix}{self._stepnum}"
                                )
        if continuation:
            n_crash = 0
            n_maxlen = 0
            # get the number of times we crashed/reached max length
            while os.path.isdir(step_dir + f"_max_len{n_maxlen+1}"):
                n_maxlen += 1
            while os.path.isdir(step_dir + f"_crash{n_crash+1}"):
                n_crash += 1
        else:
            n_crash = 0
            n_maxlen = 0
        done = False
        while not done:
            if not continuation:
                os.mkdir(step_dir)
                # write restart info to every newly created stepdir
                with open(os.path.join(step_dir, self.restart_info_fname),
                          "wb") as pf:
                    pickle.dump({"instep": instep, "mover": mover}, pf)
            # and do the actual step
            try:
                outstep = await mover.move(instep=instep, stepnum=self._stepnum,
                                           wdir=step_dir, model=model,
                                           continuation=continuation,
                                           )
            except MaxStepsReachedError as e:
                # error raised when any trial takes "too long" to commit
                logger.error("Sampler %d: MaxStepsReachedError, retrying MC "
                             "step from scratch.", self.sampler_idx,
                             )
                os.rename(step_dir, step_dir + f"_max_len{n_maxlen+1}")
                n_maxlen += 1
                continuation = False
            except EngineCrashedError as e:
                # catch error raised when gromacs crashes
                if n_crash < self.max_retries_on_crash:
                    logger.error("Sampler %d: MD engine crashed for the %dth "
                                 "time, retrying for another %d times.",
                                 self.sampler_idx, n_crash + 1,
                                 self.max_retries_on_crash - n_crash,
                                 )
                    # wait a bit for everything to finish the cleanup
                    await asyncio.sleep(self.wait_time_on_crash)
                    # move stepdir and retry
                    os.rename(step_dir, step_dir + f"_crash{n_crash+1}")
                    n_crash += 1
                    continuation = False
                else:
                    # reached maximum tries, raise the error and crash the sampling :)
                    raise e from None
            else:
                # remove the restart file
                os.remove(os.path.join(step_dir, self.restart_info_fname))
                done = True

        # check if we should accept unconditionally, i.e. if it is a TP but
        # would not have been accepted otherwise (see self.__init__ for more)
        if self.always_accept_next_TP and (outstep.path is not None):
            # can (and should) only accept steps that generated a new transition
            # always set to False to make sure the next step is only accepted
            # if the MC criterion allows it, even if this one would be accepted
            self.always_accept_next_TP = False
            if not outstep.accepted:
                # modify step only if it would not have been accepted anyway
                outstep.accepted = True
                outstep.p_acc = np.inf

        # Call the brains BrainTaks (if any)
        if brain_task_callback is not None:
            outstep = await brain_task_callback(mcstep=outstep)

        self._store_finished_step(outstep, save_step_pckl=True,
                                  make_symlink=True, instep=instep,
                                  )
        return outstep
