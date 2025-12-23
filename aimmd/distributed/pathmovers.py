# This file is part of aimmd
#
# aimmd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# aimmd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with aimmd. If not, see <https://www.gnu.org/licenses/>.
"""
This file contains the implementations of various PathMovers.

PathMovers generate new trial paths and accept/reject them, i.e. they move the
MarkovChain from one MCStep to the next, or alternatively - depending on if the
SP selection scheme supports it - PathMovers can generate new trial paths and
directly assign them a weight in the sampled path ensemble without rejecting
any trials.
"""
import functools
import os
import abc
import asyncio
import logging
import typing
import numpy as np

from asyncmd import Trajectory
from asyncmd.trajectory.convert import (FrameExtractor, RandomVelocitiesFrameExtractor,
                                        InvertedVelocitiesFrameExtractor,
                                        TrajectoryConcatenator,
                                        )
from asyncmd.trajectory.propagate import (
                            ConditionalTrajectoryPropagator,
                            InPartsTrajectoryPropagator,
                            construct_tp_from_plus_and_minus_traj_segments,
                                          )
from asyncmd.utils import nstout_from_mdconfig

from ._config import _SEMAPHORES
from .dataclasses import MDEngineSpec, MCstep
from .spselectors import (SPSelector,
                          RCModelSPSelectorFromTraj,
                          )
from ..tools import attach_kwargs_to_object as _attach_kwargs_to_object

if typing.TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from ..base.rcmodel import RCModelAsyncMixin


logger = logging.getLogger(__name__)


class PathMover(abc.ABC):
    """
    Abstract base class defining the API for all :class:`PathMover`.

    This :class:`PathMover` can optionally use a model in the trial generation,
    e.g., to bias the selection of shooting points. This is controlled via the
    ``trial_generation_uses_model`` attribute (by default ``True``).

    If a model is used, this class takes care of saving and retrieving the model
    state to ensure the same reaction coordinate model is used for the whole trial,
    i.e., SP selection and accept/reject/ensemble weight assignment).
    To save unnecessary saves of unneeded models, no saving is performed when
    ``trial_generation_uses_model`` is ``False``.
    """
    savename_prefix = "RCModel"
    delete_cached_model = True  # whether to delete the saved model after step is done
    trial_generation_uses_model = True  # when this is True we save/load the model

    def __init__(self, *, modelstore=None, sampler_idx: int | None = None, **kwargs) -> None:
        # every PathMover uses its own numpy newstyle RNG for accept/reject
        self._rng = np.random.default_rng()
        self.sampler_idx = sampler_idx
        # NOTE: The modelstore is an aimmd.storage.RCModelRack,
        #       we set it to None by default because it will be set through
        #       PathChainSampler.__init__ for all movers it owns to the
        #       rcmodel-store associated with it,
        #       same for sampler_idx (which is used to create a unique savename
        #       for the saved models)
        self.modelstore = modelstore
        _attach_kwargs_to_object(obj=self, logger=logger, **kwargs)

    # NOTE: (this is used in MCStep.__str__ so it should not be too long)
    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    # TODO: reformulate or remove the bit below? And then make it part of __getstate__ docstring?!
    # NOTE: we take care of the modelstore in storage (and
    #       PathChainSampler.__init__) to enable us to set the mover object as
    #       MCstep attribute (instead of an identifying string) and still be
    #       able to be pickle the steps.
    #       When saving a MCstep we ensure that the modelstore is the correct
    #       (as in associated with that mcstep_collection) RCModel store and
    #       when loading a MCstep we set the movers .modelstore attribute
    #       to the modelstore associated with the mcstep_collection.
    #       For pickling we just set the modelstore to None when pickling,
    #       this is why e.g. the PathChainSampler also sets the modelstore for
    #       all the movers it loads to continue steps.
    #       Subclasses can still use __getstate__ and __setstate__ to create
    #       their runtime attributes but need to call the super classes
    #       __getstate__ and __setstate__ as usual (see the TwoWayShooting for
    #       an example).

    def __getstate__(self):
        state = self.__dict__.copy()
        state["modelstore"] = None
        return state

    def __setstate__(self, state: dict):
        self.__dict__ = state

    def _model_name(self, stepnum):
        return (f"{self.savename_prefix}_in_chainsampler{self.sampler_idx}"
                + f"_at_step{stepnum}")

    def store_model(self, model: "RCModelAsyncMixin", stepnum: int) -> None:
        """
        Store the given ``model`` for ``stepnum``.

        Parameters
        ----------
        model : RCModelAsyncMixin
            The model to store.
        stepnum : int
            The step number in the MC chain, i.e. the number of steps performed
            in the sampler performing this step.

        Raises
        ------
        RuntimeError
            If :attr:`modelstore` is ``None``.
        """
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to store a model.")
        model_name = self._model_name(stepnum=stepnum)
        self.modelstore[model_name] = model

    def get_model(self, stepnum: int) -> "RCModelAsyncMixin":
        """
        Load the stored model for given ``stepnum``.

        Parameters
        ----------
        stepnum : int
            The step number in the MC chain.

        Returns
        -------
        RCModelAsyncMixin
            The loaded model used for the given stepnum.

        Raises
        ------
        RuntimeError
            If :attr:`modelstore` is ``None``.
        """
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to load a model.")
        model_name = self._model_name(stepnum=stepnum)
        return self.modelstore[model_name]

    def delete_model(self, stepnum: int) -> None:
        """
        Delete the stored model for a given ``stepnum``.

        Parameters
        ----------
        stepnum : int
            The step number in the MC chain to delete the model for.

        Raises
        ------
        RuntimeError
            If :attr:`modelstore` is ``None``.
        """
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to delete a model.")
        model_name = self._model_name(stepnum=stepnum)
        del self.modelstore[model_name]

    async def move(self, instep: MCstep, stepnum: int, workdir: str, *,
                   model: "RCModelAsyncMixin",
                   continuation: bool = False) -> MCstep:
        """
        Perform a move in the MC chain, i.e. generate a new MCstep.

        NOTE: Subclasses should overwrite the :meth:`generate_move` method, to
              enable reuse of the same save/delete logic for every (model-dependant)
              pathmover.

        Parameters
        ----------
        instep : MCstep
            The input :class:`MCstep`, i.e. the previous step in the MC chain.
        stepnum : int
            The step number of the current step in the MC chain. This is the number
            of steps performed in the sampler performing this step.
        workdir : str
            The (working) directory to use for this :class:`MCstep`.
        model : RCModelAsyncMixin
            The reaction coordinate model to use during the step.
        continuation : bool, optional
            Whether to (try to) continue an existing step from the files found
            in the ``workdir``, by default False.

        Returns
        -------
        MCstep
            The newly generated :class:`MCstep`.
        """
        if self.trial_generation_uses_model:
            model = await self._pre_move(stepnum=stepnum, model=model,
                                         continuation=continuation,
                                         )
        outstep = await self.generate_move(instep=instep, stepnum=stepnum, workdir=workdir,
                                           model=model, continuation=continuation,
                                           )
        await self._post_move(stepnum=stepnum)
        return outstep

    async def _pre_move(self, stepnum: int, *,
                        model: "RCModelAsyncMixin", continuation: bool,
                        ) -> "RCModelAsyncMixin":
        # NOTE: need to use the same model for the whole MC step independently
        # of what other samplers do in the meantime, so we store the passed model
        # and then load it in _move to have our own copy of the model that does
        # not change during our whole trial move
        if not continuation:
            # only store the model if this is the first time we try this step,
            # otherwise we just return what we stored before
            async with _SEMAPHORES["BRAIN_MODEL"]:
                self.store_model(model=model, stepnum=stepnum)
        # release the Semaphore, return the loaded model for the step os we have
        # a separate copy for selection/accept/reject
        return self.get_model(stepnum=stepnum)

    async def _post_move(self, stepnum):
        if self.delete_cached_model:
            self.delete_model(stepnum=stepnum)

    @abc.abstractmethod
    async def generate_move(self, instep: MCstep, stepnum: int, workdir: str, *,
                            model: "RCModelAsyncMixin", continuation: bool,
                            ) -> MCstep:
        """
        Abstract method to be implemented in subclasses. Will be called by :meth:`move`.

        This method must perform the actual move in path space using the given
        copy of the model, while all model saving/copying is performed in
        :meth:`move`.
        This method will be passed in the copy of the model to use for the whole
        remainder of the step.

        Note that all movers should be able to continue an interrupted step. At
        least if their steps are computationally expensive, otherwise it might
        be ok to cleanup and start a new step from scratch when ``continuation``
        is ``True``, but then the mover might have to clean up files from a previous
        run of the same mover in the same workdir.
        """
        raise NotImplementedError


class ShootingPathMover(PathMover):
    """
    Base for all shooting :class:`PathMover`.

    Contains common methods to provide the shooting point(s).
    """
    forward_deffnm = "forward"  # engine deffnm for forward shots
    backward_deffnm = "backward"  # same for backward shots
    path_filename = "path"  # filename for produced transitions
    # shooting point for the forward/backward shot will be named e.g.
    # $forward_deffnm$shooting_point_suffix
    shooting_point_suffix = "_SP"
    # concatenated output trajectories of the forward/backward shot
    # will be named e.g. $forward_deffnm$concatenated_trajectory_suffix
    concatenated_trajectory_suffix = "_concatenated"
    # remove the trajectory parts when the trial is done (also logs)
    remove_temporary_trajectories = True

    def __init__(self, states, md_engine_spec: MDEngineSpec, *,
                 sp_selector: SPSelector, **kwargs) -> None:
        """
        Parameters
        ----------
        states : list[asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper]
            State functions (stopping conditions) to use, passed to Propagator.
        md_engine_spec : MDEngineSpec
            Description/Specification of the MD engine (including parameters)
            used in the trial propagation. See :class:`MDEngineSpec` for
            what is included.
        sp_selector : SPSelector, optional
            The shooting point selector to use to provide the (forward) SP.
        """
        # NOTE
        # The modelstore and the sampler_idx will be set by
        # `PathSamplingChain.__init__()` to the rcmodel-store associated with
        # the chainstore of the chain this sampler will be used with and the index
        # it has in the Brain, respectively
        super().__init__(modelstore=None, sampler_idx=None, **kwargs)
        self.states = states
        self.md_engine_spec = md_engine_spec
        self.sp_selector = sp_selector
        # and build the frame extractors
        self._frame_extractor_fw, self._frame_extractor_bw = (
            self.build_frame_extractors()
        )
        # and propagator classes
        self._propagators = self.build_propagators()

    @abc.abstractmethod
    def build_frame_extractors(self) -> "tuple[FrameExtractor, FrameExtractor | None]":
        """
        Initialize and return the forward and (optionally) backward frame extractors.

        Returns
        -------
        tuple[FrameExtractor, FrameExtractor | None]
            The forward and the backward frame extractor.
        """

    @abc.abstractmethod
    def build_propagators(
        self,
    ) -> "Sequence[ConditionalTrajectoryPropagator | InPartsTrajectoryPropagator]":
        """
        Initialize and return the trajectory propagators for this PathMover.

        Returns
        -------
        list[_TrajectoryPropagator]
            List of forward and (optionally) backward propagators. In this order.
        """

    async def get_or_generate_sp_fw(self, *, instep: MCstep, stepnum: int,
                                    workdir: str,
                                    model: "RCModelAsyncMixin",
                                    ) -> tuple[Trajectory, bool]:
        """
        Generate or retrieve existing forward shooting point.

        Parameters
        ----------
        instep : MCstep
            The input MC step.
        stepnum : int
            The step number of the MC step to perform.
        workdir : str
            The working directory to which the shooting point will be written or
            from which it will be retrieved.
        model : RCModelAsyncMixin
            The reaction coordinate model to use to select the shooting point.

        Returns
        -------
        tuple[Trajectory, bool]
            ``fw_sp, file_exists``: The shooting point and whether it existed.
        """
        fw_sp_name = os.path.join(
                        workdir,
                        (f"{self.forward_deffnm}{self.shooting_point_suffix}"
                         + f".{self.md_engine_spec.full_precision_traj_type}")
                                  )
        if file_exists := os.path.isfile(fw_sp_name):
            # fw sp exists already, just return it and indicate that we did not
            # generate it
            fw_sp = Trajectory(trajectory_files=fw_sp_name,
                               structure_file=instep.path.structure_file,
                               )
        else:
            fw_sp = await self.sp_selector.pick(
                                    outfile=fw_sp_name,
                                    frame_extractor=self._frame_extractor_fw,
                                    step_num=stepnum,
                                    # make it possible to pick SPs without an in-path
                                    # for schemes that support it, e.g., shooting from EQ SPs
                                    trajectory=instep.path if instep is not None else None,
                                    model=model,
                                    )
        return (fw_sp, file_exists)

    async def get_or_generate_sp_bw(self, workdir: str, fw_sp: Trajectory,
                                    ) -> tuple[Trajectory, bool]:
        """
        Generate or retrieve existing backward shooting point.

        Parameters
        ----------
        workdir : str
            The working directory to which the shooting point will be written or
            from which it will be retrieved.
        fw_sp : Trajectory
            The forward shooting point from which the backward shooting point will
            be generated.

        Returns
        -------
        tuple[Trajectory, bool]
            ``bw_sp, file_exists``: The shooting point and whether it existed.

        Raises
        ------
        RuntimeError
            If no backward frame extractor is defined, i.e.,
            if ``self._frame_extractor_bw is None``.
        """
        if self._frame_extractor_bw is None:
            raise RuntimeError("Backward frame extractor must be set to extract"
                               " to extract a backward shooting point."
                               )
        bw_sp_name = os.path.join(
                        workdir,
                        (f"{self.backward_deffnm}{self.shooting_point_suffix}"
                         + f".{self.md_engine_spec.full_precision_traj_type}")
                                  )
        if file_exists := os.path.isfile(bw_sp_name):
            bw_sp = Trajectory(trajectory_files=bw_sp_name,
                               structure_file=fw_sp.structure_file,
                               )
        else:
            bw_sp = await self._frame_extractor_bw.extract_async(
                                                outfile=bw_sp_name,
                                                traj_in=fw_sp,
                                                idx=0,
                                                )
        return (bw_sp, file_exists)

    async def get_or_generate_sps(self, *, instep: MCstep, stepnum: int, workdir: str,
                                  model: "RCModelAsyncMixin",
                                  continuation: bool,
                                  ) -> tuple[Trajectory, Trajectory] | Trajectory:
        """
        Generate or retrieve shooting points for both directions.

        Returns a tuple (fw_sp, bw_sp) if both forward and backward frame extractor
        are defined, otherwise, if only the forward frame extractor is defined
        only the forward shooting point is returned.

        Warn if ``continuation`` is ``True`` but the shooting point(s) do not exist.

        Parameters
        ----------
        instep : MCstep
            The input MC step.
        stepnum : int
            The step number of the MC step to perform.
        workdir : str
            The working directory to which the shooting point will be written or
            from which it will be retrieved.
        model : RCModelAsyncMixin
            The reaction coordinate model to use to select the shooting point.
        continuation : bool
            Whether this MC step is a continuation of a previous interrupted trial.

        Returns
        -------
        tuple[Trajectory, Trajectory] | Trajectory
            Either a tuple of (forward shooting point, backward shooting point)
            or the forward shooting point, depending on if frame extractors for
            both directions are defined or not.
        """
        # get or generate forward and backward shooting points
        fw_startconf, file_exists = await self.get_or_generate_sp_fw(
                                            instep=instep, stepnum=stepnum,
                                            workdir=workdir,
                                            model=model,
                                            )
        if continuation and not file_exists:
            logger.warning("Sampler %d: continuation=True but the forward SP"
                           " did not exist. Generated a (new) SP to continue.",
                           self.sampler_idx,
                           )
        if self._frame_extractor_bw is None:
            # can not generate a bw SP without a frame extractor
            return fw_startconf
        bw_startconf, file_exists = await self.get_or_generate_sp_bw(
                                            workdir=workdir, fw_sp=fw_startconf,
                                            )
        if continuation and not file_exists:
            logger.warning("Sampler %d: continuation=True but the backward SP"
                           " did not exist. Regenerated the backward SP from"
                           " forward SP to continue.",
                           self.sampler_idx,
                           )
        return fw_startconf, bw_startconf

    def __getstate__(self):
        state = super().__getstate__()
        state["_frame_extractor_fw"] = None
        state["_frame_extractor_bw"] = None
        state["_propagators"] = None
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._frame_extractor_fw, self._frame_extractor_bw = (
            self.build_frame_extractors()
        )
        self._propagators = self.build_propagators()


class RandomVelocitiesShootingPathMover(ShootingPathMover):
    """
    (Abstract) base class for :class:`ShootingPathMover` using random velocities
    at the shooting point.

    Sets the frame extractors to :class:`RandomVelocitiesFrameExtractor` for forward
    and :class:`InvertedVelocitiesFrameExtractor` for backward direction.
    """
    def __init__(self, states, md_engine_spec: MDEngineSpec, temperature: float,
                 *, sp_selector: SPSelector, **kwargs) -> None:
        """
        Parameters
        ----------
        states : list[asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper]
            State functions (stopping conditions) to use, passed to Propagator.
        md_engine_spec : MDEngineSpec
            Description/Specification of the MD engine (including parameters)
            used in the trial propagation. See :class:`MDEngineSpec` for
            what is included.
        temperature : float
            Temperature in degree K (used for velocity randomization).
        sp_selector : SPSelector, optional
            The shooting point selector to use to provide the (forward) SP.
        """
        self.temperature = temperature
        super().__init__(states=states, md_engine_spec=md_engine_spec,
                         sp_selector=sp_selector, **kwargs)

    def build_frame_extractors(self) -> "tuple[FrameExtractor, FrameExtractor]":
        """
        Initialize and return the forward and backward frame extractors.

        Returns
        -------
        tuple[FrameExtractor, FrameExtractor]
            :class:`RandomVelocitiesFrameExtractor` and :class:`InvertedVelocitiesFrameExtractor`
        """
        return (RandomVelocitiesFrameExtractor(T=self.temperature),
                # will be used on the extracted randomized fw SP
                InvertedVelocitiesFrameExtractor(),
                )


# TODO: DOCUMENT! Write a paragraph of (sphinx) documentation on TwoWayShootingPathMovers
#       for both fixed and flexible length!
class TwoWayShootingPathMover(RandomVelocitiesShootingPathMover):
    """
    TwoWayShootingPathMover

    Produce new trials by propagating two trajectories from the selected
    shooting point, one forward and one backward in time, until the given
    states have been reached. To modify the shooting point selection you can
    pass different `SPSelector` classes as `sp_selector` (see module
    `aimmd.distributed.spselectors`).
    """
    # narrow type hint to the propagator class used
    _propagators: list[ConditionalTrajectoryPropagator]

    def __init__(self, states, md_engine_spec: MDEngineSpec, temperature: float,
                 *, sp_selector: SPSelector = RCModelSPSelectorFromTraj(),
                 path_weight_func=None, **kwargs) -> None:
        """
        Initialize a TwoWayShootingPathMover.

        Parameters
        ----------
        states : list[asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper]
            State functions (stopping conditions) to use, passed to Propagator.
        md_engine_spec : MDEngineSpec
            Description/Specification of the MD engine (including parameters)
            used in the trial propagation. See :class:`MDEngineSpec` for
            what is included.
        temperature : float
            Temperature in degree K (used for velocity randomization).
        sp_selector : SPSelector, optional
            The shooting point selector to use, by default we will use the
            :class:`RCModelSPSelectorFromTraj` with its default options.
        path_weight_func : asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper
            Weight function for paths. Can be used to enhance the sampling of
            low probability transition mechanisms. Only makes sense when the
            shooting points are chosen from the last accepted path (as opposed
            to shooting from points with known equilibrium weight).
        """
        self.path_weight_func = path_weight_func
        super().__init__(states=states, md_engine_spec=md_engine_spec,
                         temperature=temperature, sp_selector=sp_selector,
                         **kwargs)

    def build_propagators(self) -> list[ConditionalTrajectoryPropagator]:
        return [ConditionalTrajectoryPropagator(
                                conditions=self.states,
                                engine_cls=self.md_engine_spec.engine_cls,
                                engine_kwargs=self.md_engine_spec.engine_kwargs,
                                walltime_per_part=self.md_engine_spec.walltime_per_part,
                                max_steps=self.md_engine_spec.max_steps,
                                                            )
                for _ in range(2)
                ]

    # TODO: improve?! e.g. add a description of the SP selector?!
    def __str__(self) -> str:
        return "TwoWayShootingPathMover"

    async def generate_move(self, instep: MCstep, stepnum: int, workdir: str, *,
                            continuation: bool, model: "RCModelAsyncMixin",
                            ) -> MCstep:
        """
        Perform flexible length two way shooting trial move.

        Parameters
        ----------
        instep : MCstep
            The input MC step.
        stepnum : int
            The step number of the current MC step to perform.
        workdir : str
            The working directory to use for trial generation and storage.
        continuation : bool
            Whether the current MC step is a continuation of an interrupted trial.
        model : RCModelAsyncMixin
            The reaction coordinate model to use during this MC step.

        Returns
        -------
        MCstep
            The newly generated MC step.
        """
        fw_startconf, bw_startconf = await self.get_or_generate_sps(
                                                    instep=instep,
                                                    stepnum=stepnum,
                                                    workdir=workdir,
                                                    model=model,
                                                    continuation=continuation,
                                                    )
        # propagate the two trial trajectories forward and backward in time
        trial_tasks = [asyncio.create_task(p.propagate(
                                                starting_configuration=sconf,
                                                workdir=workdir,
                                                deffnm=deffnm,
                                                continuation=continuation,
                                                       )
                                           )
                       for p, sconf, deffnm in zip(self._propagators,
                                                   [fw_startconf,
                                                    bw_startconf],
                                                   [self.forward_deffnm,
                                                    self.backward_deffnm],
                                                   )
                       ]
        # use wait to be able to cancel all tasks when the first exception
        # is raised
        done, pending = await asyncio.wait(trial_tasks,
                                           return_when=asyncio.FIRST_EXCEPTION)
        # check for exceptions
        for t in done:
            if (e := t.exception()) is not None:
                # cancel all tasks that might still be running
                for tt in trial_tasks:
                    tt.cancel()
                # raise the exception, we take care of retrying complete steps
                # in the PathSamplingChain
                raise e from None
        # if no task raised an exception all should done, so get the results
        assert len(pending) == 0  # but make sure everything went as we expect
        trials = []
        for t in trial_tasks:
            trials += [t.result()]
        # propagate returns (list_of_traj_parts, state_reached)
        (fw_trajs, fw_state), (bw_trajs, bw_state) = trials
        states_reached = np.array([0. for _ in range(len(self.states))])
        states_reached[fw_state] += 1
        states_reached[bw_state] += 1
        out_tra_names = [os.path.join(
                            workdir,
                            (self.forward_deffnm
                             + self.concatenated_trajectory_suffix
                             + f".{self.md_engine_spec.output_traj_type}")
                            ),
                         os.path.join(
                            workdir,
                            (self.backward_deffnm
                             + self.concatenated_trajectory_suffix
                             + f".{self.md_engine_spec.output_traj_type}")
                            ),
                         ]
        concats = await asyncio.gather(*(
                        p.cut_and_concatenate(trajs=trajs,
                                              tra_out=traj_out,
                                              # if we continue the traj might
                                              # already be there
                                              overwrite=continuation)
                        for p, trajs, traj_out in zip(self._propagators,
                                                      [fw_trajs, bw_trajs],
                                                      out_tra_names,
                                                      )
                                         )
                                       )
        # cut and concatenate returns (traj_to_state, first_state_reached)
        # but since we already know about the states we ignore them here
        (fw_traj, _), (bw_traj, _) = concats
        # NOTE: we already got our own private copy of the model at the
        # beginning of this move, so we can just continue using it
        if model is not None:
            # If we use a model, use selecting model to predict the commitment
            # probabilities for the SP
            predicted_committors_sp = (await model(fw_startconf))[0]
        else:
            # no model, we set the predicted pB to None as we can not know it
            predicted_committors_sp = None
        # independently of if it is a TP, we will cut and concatenate to one path,
        # if it is a TP, we order it such that it goes from lower idx state
        # to the higher idx state
        if fw_state >= bw_state:
            minus_trajs, minus_state = bw_trajs, bw_state
            plus_trajs, plus_state = fw_trajs, fw_state
        else:
            # can only be the other way round
            minus_trajs, minus_state = fw_trajs, fw_state
            plus_trajs, plus_state = bw_trajs, bw_state
        logger.info("Sampler %d: Forward trial reached state with index %d, "
                    "backward trial reached state with index %d.",
                    self.sampler_idx, fw_state, bw_state,
                    )
        tra_out = os.path.join(
                    workdir,
                    f"{self.path_filename}.{self.md_engine_spec.output_traj_type}"
                    )
        path_traj = await construct_tp_from_plus_and_minus_traj_segments(
                            minus_trajs=minus_trajs, minus_state=minus_state,
                            plus_trajs=plus_trajs, plus_state=plus_state,
                            state_funcs=self.states, tra_out=tra_out,
                            # if we continue the traj might already be there
                            struct_out=None, overwrite=continuation,
                            )
        # (potentially) remove the temporary files now that we have written all
        # concatenated trajectories
        if self.remove_temporary_trajectories:
            await asyncio.gather(*(p.remove_parts(
                                        workdir=workdir,
                                        deffnm=n,
                                        file_endings_to_remove=["trajectories",
                                                                "log"])
                                   for p, n in zip(self._propagators,
                                                   [self.forward_deffnm,
                                                    self.backward_deffnm]
                                                   )
                                   )
                                 )
        # prepare the MCStep with attributes independent of accept/reject/etc
        half_finished_step = functools.partial(
                                    MCstep,
                                    mover=self,
                                    stepnum=stepnum,
                                    directory=workdir,
                                    path=path_traj,
                                    predicted_committors_sp=predicted_committors_sp,
                                    shooting_snap=fw_startconf,
                                    states_reached=states_reached,
                                    trial_trajectories=[fw_traj, bw_traj],
                                    )
        # check if we generated a TP, if not we can save some computation
        if fw_state == bw_state:
            if self.sp_selector.probability_is_ensemble_weight:
                # We have/are creating an unordered ensemble of TP with weights
                # need to 'accept' all trials, in this case with weight=0
                return half_finished_step(accepted=True, p_acc=1, weight=0,)
            # we have a 'real' Markov chain, set p_acc=0 for non-TP
            # this kicks them out of the MCStates iteration
            return half_finished_step(accepted=False, p_acc=0, weight=1,)
        # need to actually calculate acceptance probability/ ensemble weight
        if self.sp_selector.probability_is_ensemble_weight:
            # Build an unordered ensemble of transitions with weights
            # we use this branch if we know that the probability is the
            # ensemble weight of the transition trajectory (except for a
            # constant factor, which we can normalize away)
            # This is the case if we draw the shooting points from a
            # distribution that is p_{SP}(x) = p_{eq}(x) * p_{bias}(x),
            # then we can cancel the p_{eq}(x) part in the generation
            # probability for the Path in itself and get an ensemble weight
            # for each generated transition from the known p_{bias}(x)
            # [See e.g. Falkner et al; arXiv 2207.14530]
            # Note: if we dont use a model for selection, model=None
            ensemble_weight = await self.sp_selector.probability(
                                                    snapshot=fw_startconf,
                                                    trajectory=path_traj,
                                                    step_num=stepnum,
                                                    model=model,
                                                    )
            p_acc = 1.
            log_str = f"Sampler {self.sampler_idx}: Ensemble weight "
            log_str += f"for generated trial is {round(ensemble_weight, 6)}."
            accepted = True
        else:
            # "Normal" MCMC TPS below
            # we use this branch if we need to cancel part of the
            # probabilities from old and new transition trajectory in the
            # accept/reject, i.e. if we need the ordered Markov chain to
            # get the correct ensemble of transitions, this is e.g.
            # the case for "normal" TwoWayShooting (shooting from the last
            # accepted TP), where we cancel the p_{EQ}(x_{SP}) parts in the
            # acceptance probability (such that the SPSelector does not
            # need to calculate p_{EQ}(x_{SP}))
            # accept or reject?
            # Note: if we dont use a model for selection, model=None
            p_sel_old, p_sel_new = await asyncio.gather(
                                            self.sp_selector.probability(
                                                        snapshot=fw_startconf,
                                                        trajectory=instep.path,
                                                        step_num=stepnum,
                                                        model=model),
                                            self.sp_selector.probability(
                                                        snapshot=fw_startconf,
                                                        trajectory=path_traj,
                                                        step_num=stepnum,
                                                        model=model),
                                            )
            if self.path_weight_func is None:
                # no weight function so all paths have equal weight
                w_path_new = w_path_old = 1.
            else:
                w_path_new, w_path_old = await asyncio.gather(
                                            self.path_weight_func(path_traj),
                                            self.path_weight_func(instep.path),
                                            )
            # p_acc = ((p_sel_new * p_mod_sp_new_to_old * p_eq_sp_new * W_path_new)
            #          / (p_sel_old * p_mod_sp_old_to_new * p_eq_sp_old * W_path_old)
            #          )
            # but accept only depends on p_sel * W_path, because we use
            # Maxwell-Boltzmann vels, i.e. p_mod cancels with the velocity
            # part of p_eq_sp and the configuration is the same in old and
            # new, i.e. for the positions we can cancel old with new
            p_acc = (p_sel_new * w_path_new) / (p_sel_old * w_path_old)
            # The weight of the new path is the inverse of the path bias,
            # e.g. if we give it W_path_new=2 we would sample it twice as
            # often as in equilibrium and so we need to give it the
            # ensemble_weight=1/2
            ensemble_weight = 1 / w_path_new
            log_str = f"Sampler {self.sampler_idx}: Acceptance probability"
            log_str += f" for generated trial is {round(p_acc, 6)}."
            accepted = False
            if (p_acc >= 1) or (p_acc > self._rng.random()):
                accepted = True
                log_str += " Trial was accepted."
        # In both cases: log and return the MCstep
        logger.info(log_str)
        return half_finished_step(accepted=accepted, p_acc=p_acc,
                                  weight=ensemble_weight,
                                  )


class FixedLengthTwoWayShootingPathMover(RandomVelocitiesShootingPathMover):
    """
    FixedLengthTwoWayShootingPathMover

    Produce new trials by propagating two trajectories from the selected
    shooting point, one forward and one backward in time, for a fixed number of
    frames in total. We draw the number of frames for forward and backward at
    random, essentially making this move is a combination of a shooting and a
    shifting move at the same time. To modify the shooting point selection you
    can pass different `SPSelector` classes as `sp_selector` (see module
    `aimmd.distributed.spselectors`).
    """

    _propagators: list[InPartsTrajectoryPropagator]

    def __init__(self, n_steps: int, states, md_engine_spec: MDEngineSpec,
                 temperature: float, *,
                 sp_selector: SPSelector = RCModelSPSelectorFromTraj(),
                 path_weight_func=None, **kwargs) -> None:
        """
        Initialize a FixedLengthTwoWayShootingPathMover.

        Parameters
        ----------
        n_steps : int
            Number of integration steps to perform in total (over both forward
            and backward combined).
        states : list[asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper]
            State functions to use to decide if a trial is valid.
        md_engine_spec : MDEngineSpec
            Description/Specification of the MD engine (including parameters)
            used in the trial propagation. See :class:`MDEngineSpec` for
            what is included.
        temperature : float
            Temperature in degree K (used for velocity randomization).
        sp_selector : SPSelector, optional
            The shooting point selector to use, by default we will use the
            `RCModelSPSelectorFromTraj` with its default options.
        path_weight_func : asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper
            Weight function for paths. Can be used to enhance the sampling of
            low probability transition mechanisms. Only makes sense when the
            shooting points are choosen from the last accepted path (as opposed
            to shooting from points with known equilibrium weight).
        """
        self.path_weight_func = path_weight_func
        self.n_steps = n_steps
        super().__init__(states=states, md_engine_spec=md_engine_spec,
                         temperature=temperature, sp_selector=sp_selector,
                         **kwargs)
        self._nstout = nstout_from_mdconfig(
                                    mdconfig=self.md_engine_spec.engine_kwargs["mdconfig"],
                                    output_traj_type=self.md_engine_spec.output_traj_type,
                                            )
        # ensure n_steps % nstout == 0, such that we can randomly draw number
        # of frames for forward and backward parts and end up with a traj with
        # total length of n_steps
        if not self.n_steps % self._nstout == 0:
            raise ValueError(f"The nstout ({self._nstout}) from mdconfig is "
                             + "not compatible with the total number of steps "
                             + f"({self.n_steps}).")
        self._n_frames = self.n_steps // self._nstout

    def build_propagators(self) -> list[InPartsTrajectoryPropagator]:
        return [InPartsTrajectoryPropagator(
                    n_steps=self.n_steps,
                    engine_cls=self.md_engine_spec.engine_cls,
                    engine_kwargs=self.md_engine_spec.engine_kwargs,
                    walltime_per_part=self.md_engine_spec.walltime_per_part,
                    )
                for _ in range(2)
                ]

    # TODO: improve?! e.g. add a description of the SP selector?!
    def __str__(self) -> str:
        return "FixedLengthTwoWayShootingPathMover"

    async def generate_move(self, instep: MCstep, stepnum: int, workdir: str, *,
                            continuation: bool, model: "RCModelAsyncMixin",
                            ) -> MCstep:
        """
        Perform fixed length two way shooting trial move.

        Parameters
        ----------
        instep : MCstep
            The input MC step.
        stepnum : int
            The step number of the current MC step to perform.
        workdir : str
            The working directory to use for trial generation and storage.
        continuation : bool
            Whether the current MC step is a continuation of an interrupted trial.
        model : RCModelAsyncMixin
            The reaction coordinate model to use during this MC step.

        Returns
        -------
        MCstep
            The newly generated MC step.
        """
        fw_startconf, bw_startconf = await self.get_or_generate_sps(
                                                    instep=instep,
                                                    stepnum=stepnum,
                                                    workdir=workdir,
                                                    model=model,
                                                    continuation=continuation,
                                                    )
        # propagate the two trial trajectories forward and backward in time
        # first create the names for the concatenated output trajs
        out_tra_names = [os.path.join(
                            workdir,
                            f"{self.forward_deffnm}.{self.md_engine_spec.output_traj_type}"),
                         os.path.join(
                            workdir,
                            f"{self.backward_deffnm}.{self.md_engine_spec.output_traj_type}")
                         ]
        # Decide/draw how many frames we do in each direction,
        # Note that we draw in frame space to make sure we can end up only with
        # multiples of nstout for the number of steps forward and backward,
        # i.e. make sure that nsteps % nstout == 0
        nframes_fw = self._rng.integers(self._n_frames, dtype=int)
        nframes_bw = self._n_frames - nframes_fw
        nsteps_fw = nframes_fw * self._nstout
        nsteps_bw = nframes_bw * self._nstout
        # set the number of steps for the two propagators to what we decided
        self._propagators[0].n_steps = nsteps_fw
        self._propagators[1].n_steps = nsteps_bw
        # now the actual trial generation
        trial_tasks = [asyncio.create_task(p.propagate_and_concatenate(
                                                starting_configuration=sconf,
                                                workdir=workdir,
                                                deffnm=deffnm,
                                                tra_out=out_tra,
                                                overwrite=continuation,
                                                continuation=continuation,
                                                                       )
                                           )
                       for p, sconf, deffnm, out_tra in zip(
                                self._propagators,
                                [fw_startconf, bw_startconf],
                                [self.forward_deffnm, self.backward_deffnm],
                                out_tra_names,
                                                            )
                       ]
        # use wait to be able to cancel all tasks when the first exception
        # is raised
        done, pending = await asyncio.wait(trial_tasks,
                                           return_when=asyncio.FIRST_EXCEPTION)
        # check for exceptions
        for t in done:
            if (e := t.exception()) is not None:
                # cancel all tasks that might still be running
                for tt in trial_tasks:
                    tt.cancel()
                # raise the exception, we take care of retrying complete steps
                # in the PathSamplingChain
                raise e from None
        # if no task raised an exception all should done, so get the results
        assert len(pending) == 0  # but make sure everything went as we expect
        trials = []
        for t in trial_tasks:
            trials += [t.result()]
        # propagate returns a list of traj_parts for each propagator
        fw_traj, bw_traj = trials
        if self.states is not None:
            # find out which states we reached on which traj part
            states_reached = np.array([0. for _ in range(len(self.states))])
            condition_vals = await asyncio.gather(*(s(t)
                                                    for t in (fw_traj, bw_traj)
                                                    for s in self.states
                                                    )
                                                  )
            condition_vals_fw = np.array(condition_vals[:len(self.states)])
            condition_vals_bw = np.array(condition_vals[len(self.states):])
            # find out which state/condition is fullfilled at the last frame
            # TODO: do we need to check if another state has been reached before?
            #       I (hejung) think no, because for fixed length TPS we only
            #       need to make sure that the first/last frames are in states?!
            fw_state = np.where(condition_vals_fw[:, -1])
            bw_state = np.where(condition_vals_bw[:, -1])
            states_reached[fw_state] += 1
            states_reached[bw_state] += 1
            # only transitions (i.e. ending in two different states) have a
            # non-zero weight in the ensemble
            # transition_factor will be zero (no TP) or one (it is a TP)
            transition_factor = int(fw_state != bw_state)
        else:
            # dont know about the states so no states_reached and can not know
            # if it is a transition either
            states_reached = None
            transition_factor = 1
        # NOTE: we already loaded our own private copy of the model at the
        # beginning of this move, so we can just continue using it
        if model is not None:
            # If we use a model, use selecting model to predict the commitment
            # probabilities for the SP
            predicted_committors_sp = (await model(fw_startconf))[0]
        else:
            # no model, we set the predicted pB to None as we can not know it
            predicted_committors_sp = None
        # create one concatenated trajectory from the forward and backward part
        # Note that the InPartsTrajectoryPropagator returns None if n_steps=0,
        # i.e. if it did not produce a traj because all propagation is done in
        # the other propagators direction
        if fw_traj is None:
            trajs = [bw_traj]
            slices = [(-1, None, -1)]
        elif bw_traj is None:
            trajs = [fw_traj]
            slices = [(0, None, 1)]
        else:
            # the usual case, both are not None
            trajs = [bw_traj, fw_traj]
            slices = [(-1, None, -1),
                      # the fw slice starts at 1 to exclude the SP once
                      (1, None, 1)]
        tra_out = os.path.join(workdir,
                               f"{self.path_filename}.{self.md_engine_spec.output_traj_type}")
        path_traj = await TrajectoryConcatenator().concatenate_async(
                                                        trajs=trajs,
                                                        slices=slices,
                                                        tra_out=tra_out,
                                                                     )
        if self.sp_selector.probability_is_ensemble_weight:
            # Build an unordered ensemble of transitions with weights
            # we use this branch if we know that the probability is the
            # ensemble weight of the transition trajectory (except for a
            # constant factor, which we can normalize away)
            # This is the case if we draw the shooting points from a
            # distribution that is p_{SP}(x) = p_{eq}(x) * p_{bias}(x),
            # then we can cancel the p_{eq}(x) part in the generation
            # probability for the Path in itself and get an ensemble weight
            # for each generated transition from the known p_{bias}(x)
            # [See e.g. Falkner et al; arXiv 2207.14530]
            # Note: if we dont use a model for selection, model=None
            ensemble_weight = await self.sp_selector.probability(
                                                        snapshot=fw_startconf,
                                                        trajectory=path_traj,
                                                        step_num=stepnum,
                                                        model=model,
                                                                 )
            # make sure we only accept transitions
            ensemble_weight *= transition_factor
            p_acc = 1
            log_str = f"Sampler {self.sampler_idx}: Ensemble weight "
            log_str += f"for generated trial is {round(ensemble_weight, 6)}."
            accepted = True
        else:
            # "Normal" MCMC TPS below
            # we use this branch if we need to cancel part of the
            # probabilities from old and new transition trajectory in the
            # accept/reject, i.e. if we need the ordered Markov chain to
            # get the correct ensemble of transitions, this is e.g.
            # the case for "normal" TwoWayShooting (shooting from the last
            # accepted TP), where we cancel the p_{EQ}(x_{SP}) parts in the
            # acceptance probability (such that the SPSelector does not
            # need to calculate p_{EQ}(x_{SP}))
            # accept or reject?
            # Note: if we dont use a model for selection, model=None
            p_sel_old, p_sel_new = await asyncio.gather(
                                            self.sp_selector.probability(
                                                    snapshot=fw_startconf,
                                                    trajectory=instep.path,
                                                    step_num=stepnum,
                                                    model=model),
                                            self.sp_selector.probability(
                                                    snapshot=fw_startconf,
                                                    trajectory=path_traj,
                                                    step_num=stepnum,
                                                    model=model),
                                                        )
            if self.path_weight_func is None:
                # no weight function so all paths have equal weight
                w_path_new = w_path_old = 1
            else:
                w_path_new, w_path_old = await asyncio.gather(
                                            self.path_weight_func(path_traj),
                                            self.path_weight_func(instep.path),
                                                              )
            # p_acc = ((p_sel_new * p_mod_sp_new_to_old * p_eq_sp_new * W_path_new)
            #          / (p_sel_old * p_mod_sp_old_to_new * p_eq_sp_old * W_path_old)
            #          )
            # but accept only depends on p_sel * W_path,
            # because we use Maxwell-Boltzmann vels, i.e. p_mod cancels with
            # p_eq_sp velocity part and configuration is the same in old and
            # new, i.e. for the positions we can cancel old with new
            p_acc = (p_sel_new * w_path_new) / (p_sel_old * w_path_old)
            # and make sure we can only accept transitions
            p_acc *= transition_factor
            # The weight of the new path is the inverse of the path bias,
            # i.e. if we give it W_path_new=2 we would sample it twice as often
            # as in equilibrium and so we need to give it ensemble_weight=1/2
            ensemble_weight = 1 / w_path_new
            log_str = f"Sampler {self.sampler_idx}: Acceptance probability"
            log_str += f" for generated trial is {round(p_acc, 6)}."
            accepted = False
            if (p_acc >= 1) or (p_acc > self._rng.random()):
                accepted = True
                log_str += " Trial was accepted."
        # In both cases: log and return the MCstep
        logger.info(log_str)
        # (potentially) remove the temporary trajectory files
        if self.remove_temporary_trajectories:
            await asyncio.gather(*(p.remove_parts(
                                        workdir=workdir,
                                        deffnm=n,
                                        file_endings_to_remove=["trajectories",
                                                                "log"])
                                   for p, n in zip(self._propagators,
                                                   [self.forward_deffnm,
                                                    self.backward_deffnm])
                                   )
                                 )
        return MCstep(mover=self,
                      stepnum=stepnum,
                      directory=workdir,
                      predicted_committors_sp=predicted_committors_sp,
                      shooting_snap=fw_startconf,
                      states_reached=states_reached,
                      trial_trajectories=trajs,
                      path=path_traj,
                      accepted=accepted,
                      p_acc=p_acc,
                      weight=ensemble_weight,
                      )
