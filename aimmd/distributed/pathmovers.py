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
# code to move around in path-space, pathmovers generate new trial paths and
# accept/reject them, i.e. they move the MarkovChain from one MCStep to the next
import functools
import os
import abc
import asyncio
import logging
import pickle
import typing
import textwrap
import numpy as np

from asyncmd import Trajectory
from asyncmd.trajectory.convert import (RandomVelocitiesFrameExtractor,
                                        InvertedVelocitiesFrameExtractor,
                                        )
from asyncmd.trajectory.propagate import (
                            ConditionalTrajectoryPropagator,
                            InPartsTrajectoryPropagator,
                            TrajectoryConcatenator,
                            construct_TP_from_plus_and_minus_traj_segments,
                                          )
from asyncmd.utils import ensure_mdconfig_options, nstout_from_mdconfig

from ._config import _SEMAPHORES
from .spselectors import (SPSelector, RCModelSPSelector,
                          RCModelSPSelectorFromTraj, RCModelSPSelectorFromEQ)


logger = logging.getLogger(__name__)


class MCstep:
    # TODO: make this 'immutable'? i.e. expose everything as get-only-properties?
    # TODO: some of the attributes are only relevant for shooting,
    #       do we want a subclass for shooting MCsteps?

    default_savename = "mcstep_data.pckl"

    def __init__(self, mover, stepnum, directory, predicted_committors_sp=None,
                 shooting_snap=None, states_reached=None,
                 path=None, trial_trajectories=[], accepted=False, p_acc=0,
                 weight=1,
                 ):
        # TODO: should this be the obj? or an unique string identififer? or...?
        self.mover = mover  # NOTE: currently we use the obj (and require all PathMovers to be pickleable)
        self.stepnum = stepnum
        self.directory = directory
        self.predicted_committors_sp = predicted_committors_sp
        self.shooting_snap = shooting_snap
        self.states_reached = states_reached
        self.path = path
        self.trial_trajectories = trial_trajectories
        self.accepted = accepted
        self.p_acc = p_acc
        self.weight = weight

    # TODO: improve :)
    def _str_representation(self, long, width=139) -> str:
        repr_str = ""
        repr_str += f"MCStep(mover={self.mover}, stepnum={self.stepnum}, "
        repr_str += f"states_reached={self.states_reached}, "
        repr_str += f"accepted={self.accepted}, "
        repr_str += f"p_acc={self.p_acc}, "
        repr_str += f"predicted_committors_sp={self.predicted_committors_sp}, "
        repr_str += f"weight={self.weight}, "
        repr_str += f"directory={self.directory}"
        if not long:
            repr_str += ")"  # terminate here
        else:
            repr_str += ", "  # more to come below
            repr_str += f"shooting_snap={self.shooting_snap}, "
            repr_str += f"path={self.path})"
        return textwrap.fill(repr_str, width=width,
                             break_long_words=False,
                             subsequent_indent="       ",  # as long as MCStep(
                             )

    def __str__(self) -> str:
        return self._str_representation(long=False)

    def __repr__(self) -> str:
        return self._str_representation(long=True)

    # TODO/FIXME: Do we need this? we can/could just pickle the MCSteps anyway?
    #             (for now we keep it because it makes pickling a one-liner)
    def save(self, fname: typing.Optional[str] = None,
             overwrite: bool = False) -> None:
        if fname is None:
            fname = os.path.join(self.directory, self.default_savename)
        if not overwrite and os.path.exists(fname):
            # we check if it exists, because pickle/open will happily overwrite
            raise ValueError(f"{fname} exists but overwrite=False.")
        with open(fname, "wb") as pfile:
            pickle.dump(self, pfile)

    @classmethod
    def load(cls, directory: typing.Optional[str] = None,
             fname: typing.Optional[str] = None):
        if directory is None:
            directory = os.getcwd()
        if fname is None:
            fname = cls.default_savename
        with open(os.path.join(directory, fname), "rb") as pfile:
            obj = pickle.load(pfile)
        return obj


class PathMover(abc.ABC):
    """Abstract base class defining the API for all `PathMovers`."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng()  # numpy newstyle RNG, one per Mover

    # takes an (usually accepted) in-MCstep and
    # produces an out-MCstep (not necessarily accepted)
    @abc.abstractmethod
    async def move(self, instep: MCstep, stepnum: int, wdir: str,
                   continuation: bool = False, **kwargs) -> MCstep:
        # NOTE: all movers should be able to continue an interrupted step
        #       (at least if their steps are computationally expensive,
        #        otherwise it might be ok to cleanup and start a new step from
        #        scratch when given continuation=True, but then you might have
        #        to clean up files from a previous run of the same mover in the
        #        same wdir)
        raise NotImplementedError

    # NOTE: (this is used in MCStep.__str__ so it should not be too long)
    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class ModelDependentPathMover(PathMover):
    """
    This `Pathmover` uses a `SPselector` with a model to bias the selection.

    This class takes care of saving and retrieving the model state to ensure
    the same reaction coordinate model is used for the whole trial (i.e. SP
    selection and accept/reject/ensemble weight assignment).
    """

    # PathMover that takes a model at the start of the move
    # the model would e.g. be used to select the shooting point
    # here lives the code that saves the model state at the start of the step,
    # this enables us to do the accept/reject (at the end) with the initially
    # saved model
    savename_prefix = "RCModel"
    delete_cached_model = True  # wheter to delete the model after step is done

    def __init__(self, modelstore=None, sampler_idx=None) -> None:
        super().__init__()
        # NOTE: The modelstore is an aimmd.storage.RCModelRack,
        #       we set it to None by default because it will be set through
        #       PathChainSampler.__init__ for all movers it owns to the
        #       rcmodel-store associated with it,
        #       same for sampler_idx (which is used to create a unique savename
        #       for the saved models)
        self.modelstore = modelstore
        self.sampler_idx = sampler_idx

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

    def store_model(self, model, stepnum):
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to store a model.")
        model_name = self._model_name(stepnum=stepnum)
        self.modelstore[model_name] = model

    def get_model(self, stepnum):
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to load a model.")
        model_name = self._model_name(stepnum=stepnum)
        return self.modelstore[model_name]

    def delete_model(self, stepnum):
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to delete a model.")
        model_name = self._model_name(stepnum=stepnum)
        del self.modelstore[model_name]

    async def move(self, instep: MCstep, stepnum: int, wdir: str, model,
                   continuation: bool = False, **kwargs) -> MCstep:
        # this enables us to reuse the save/delete logic for every
        # modeldependant pathmover
        await self._pre_move(instep=instep, stepnum=stepnum, wdir=wdir,
                             model=model, **kwargs)
        outstep = await self._move(instep=instep, stepnum=stepnum, wdir=wdir,
                                   continuation=continuation, **kwargs)
        await self._post_move(instep=instep, stepnum=stepnum, wdir=wdir,
                              **kwargs)
        return outstep

    async def _pre_move(self, instep, stepnum, wdir, model, **kwargs):
        # NOTE: need to select (and later register) the SP with the passed model
        # (this is the 'main' model and if it knows about the SPs we can use the
        #  prediction accuracy in the close past to decide if we want to train)
        # [registering the sp with the model is done by the trainingtask, to
        #  this end we save the predicted committors for the sp with the MCStep]
        # so we store the passed model and then load it in _move to have our
        # own copy of the model that does not change during our trial move
        async with _SEMAPHORES["BRAIN_MODEL"]:
            self.store_model(model=model, stepnum=stepnum)
        # release the Semaphore, we load the stored model for accept/reject later

    async def _post_move(self, instep, stepnum, wdir, **kwargs):
        if self.delete_cached_model:
            self.delete_model(stepnum=stepnum)

    # NOTE: the actual move! to be implemented by the subclasses
    @abc.abstractmethod
    async def _move(self, instep: MCstep, stepnum: int, wdir: str,
                    **kwargs) -> MCstep:
        # Note that we dont pass the model here (you should load it with
        # self.get_model() instead in self._move)
        raise NotImplementedError


class PathMoverSansModel(PathMover):
    """
    This `Pathmover` does not use a `SPselector` with a model.

    This class exists just so we can use the same trial propagation logic when
    using SP selectors with and without model. In the case without a model
    we dont have to save the model before the step (obviously).
    """
    def __init__(self, sampler_idx=None) -> None:
        super().__init__()
        self.sampler_idx = sampler_idx

    async def move(self, instep, stepnum, wdir, continuation=False, **kwargs):
        return await self._move(instep=instep, stepnum=stepnum, wdir=wdir,
                                continuation=continuation, **kwargs)

    @abc.abstractmethod
    async def _move(self, instep, stepnum, wdir, continuation, **kwargs):
        raise NotImplementedError

    # Need to implement getstate and setstate such that our ShootingMixins work
    # with and without model...they assume that super().__getstate__ works...
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict):
        self.__dict__ = state


# TODO: DOCUMENT! Write a paragraph of (sphinx) documentation directly!
class _TwoWayShootingPathMoverMixin:
    """
    TwoWayShootingPathMover

    Produce new trials by propagating two trajectories from the selected
    shooting point, one forward and one backward in time, until the given
    states have been reached. To modify the shooting point selection you can
    pass different `SPSelector` classes as `sp_selector` (see module
    `aimmd.distributed.spselectors`).
    """

    # for TwoWay shooting moves until any state is reached
    forward_deffnm = "forward"  # engine deffnm for forward shot
    backward_deffnm = "backward"  # same for backward shot
    path_filename = "path"  # filename for produced transitions
    # trajs to state will be named e.g. $forward_deffnm$traj_to_state_suffix
    traj_to_state_suffix = "_traj_to_state"
    # remove the trajectory parts when the trial is done (also logs)
    remove_temp_trajs = True

    def __init__(self, states, engine_cls, engine_kwargs, walltime_per_part, T,
                 sp_selector: SPSelector = RCModelSPSelectorFromTraj(),
                 max_steps: typing.Optional[int] = None,
                 path_weight_func=None):
        """
        Initialize a TwoWayShootingPathMover.

        Parameters
        ----------
        states : list[asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper]
            State functions (stopping conditions) to use, passed to Propagator.
        engine_cls :
            The class of the molecular dynamics engine to use, we expect one of
            the `asyncmd` engines.
        engine_kwargs : dict
            A dictionary with keyword arguments needed to initialize the given
            molecular dynamics engine.
        walltime_per_part : float
            Simulation walltime per trajectory part in hours, directly passed
            to the Propagator.
        T : float
            Temperature in degree K (used for velocity randomization).
        sp_selector : aimmd.distributed.spselector.SPSelector, optional
            The shooting point selector to use, by default we will use the
            `aimmd.distributed.spselector.RCModelSPSelectorFromTraj` with its
             default options.
        max_steps : int or None, optional
            The maximum number of integration steps *per part* (forward or
            backward), i.e. this bounds steps(TP) <= 2*max_steps. If None we
            will use no upper length, which means trials can get "stuck".
            Note that if the maximum is reached in any direction the trial will
            be discarded and a new trial will be started from the last accepted
            MCStep.
        path_weight_func : asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper
            Weight function for paths. Can be used to enhance the sampling of
            low probability transition mechanisms. Only makes sense when the
            shooting points are choosen from the last accepted path (as opposed
            to shooting from points with known equilibrium weight).
        """
        # TODO: check that we use the same T as GMX? or maybe even directly take T from GMX (mdp)?
        # TODO: we should make properties out of everything
        #       changing anything requires recreating the propagators!
        # NOTE on modelstore:
        # we implicitly pass None here, it will be set by
        # `PathSamplingChain.__init__()` to the rcmodel-store associated with
        # the chainstore of the chain this sampler will be used with
        super().__init__() #modelstore=None, sampler_idx=None)
        self.states = states
        self.engine_cls = engine_cls
        self.engine_kwargs = engine_kwargs
        self.engine_kwargs["mdconfig"] = ensure_mdconfig_options(
                                self.engine_kwargs["mdconfig"],
                                # dont generate velocities, we do that ourself
                                genvel="no",
                                # dont apply constraints at start of simulation
                                continuation="yes",
                                                )
        self.walltime_per_part = walltime_per_part
        self.T = T
        self.sp_selector = sp_selector
        self.max_steps = max_steps
        self.path_weight_func = path_weight_func
        try:
            # see if it is set as engine_kwarg
            output_traj_type = engine_kwargs["output_traj_type"]
        except KeyError:
            # it is not, so we use the engine_class default
            output_traj_type = engine_cls.output_traj_type
        finally:
            self.output_traj_type = output_traj_type.lower()
        self._build_extracts_and_propas()

    def _build_extracts_and_propas(self):
        self.frame_extractors = {"fw": RandomVelocitiesFrameExtractor(T=self.T),
                                 # will be used on the extracted randomized fw SP
                                 "bw": InvertedVelocitiesFrameExtractor(),
                                 }
        self.propagators = [ConditionalTrajectoryPropagator(
                                    conditions=self.states,
                                    engine_cls=self.engine_cls,
                                    engine_kwargs=self.engine_kwargs,
                                    walltime_per_part=self.walltime_per_part,
                                    max_steps=self.max_steps,
                                                            )
                            for _ in range(2)
                            ]

    # TODO: improve?! e.g. add a description of the SP selector?!
    def __str__(self) -> str:
        return "TwoWayShootingPathMover"

    def __getstate__(self):
        state = super().__getstate__()
        state["frame_extractors"] = None
        state["propagators"] = None
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._build_extracts_and_propas()

    async def _move(self, instep, stepnum, wdir, continuation, **kwargs):
        if isinstance(self, ModelDependentPathMover):
            # see if we need the model and load it only if we do
            model = self.get_model(stepnum=stepnum)
        else:
            # otherwise we set the model to None, this makes it possible to use
            # the same call to all SPselectors (including the model argument)
            model = None
        fw_sp_name = os.path.join(wdir, f"{self.forward_deffnm}_SP.trr")
        # instep can be None if we shoot from an ensemble of points
        inpath = instep.path if instep is not None else None
        if continuation:
            if inpath is None:
                fw_struct_file = os.path.join(wdir, f"{self.forward_deffnm}.tpr")
            else:
                fw_struct_file = inpath.structure_file
            if os.path.isfile(fw_sp_name) and os.path.isfile(fw_struct_file):
                # if the forward SP (and struct file for no inpath case)
                # have not yet been written it should be save to assume that we
                # can just start from scratch without loosing anything
                fw_startconf = Trajectory(trajectory_files=fw_sp_name,
                                          structure_file=fw_struct_file)
            else:
                continuation = False
        # pick a SP if we are not doing a continuation
        if not continuation:
            fw_startconf = await self.sp_selector.pick(
                                    outfile=fw_sp_name,
                                    frame_extractor=self.frame_extractors["fw"],
                                    trajectory=inpath,
                                    model=model,
                                                       )
        # NOTE: we do not apply constraints as we select from a trajectory that
        #       is produced by the engine which applies the correct constraints
        #       i.e. the configurations are alreayd constrained
        bw_sp_name = os.path.join(wdir, f"{self.backward_deffnm}_SP.trr")
        if continuation:
            if os.path.isfile(bw_sp_name):
                # if the backward SP has not yet been written it should be save
                # to assume we did not yet start the trial generation for
                # either fw or bw, so we set continuation=False
                bw_startconf = Trajectory(trajectory_files=bw_sp_name,
                                          structure_file=fw_startconf.structure_file)
            else:
                continuation = False
        # create the backwards SP if we have not yet done so
        if not continuation:
            # we only invert the fw SP
            bw_startconf = await self.frame_extractors["bw"].extract_async(
                                                        outfile=bw_sp_name,
                                                        traj_in=fw_startconf,
                                                        idx=0,
                                                                           )
        # propagate the two trial trajectories forward and backward in time
        trial_tasks = [asyncio.create_task(p.propagate(
                                                starting_configuration=sconf,
                                                workdir=wdir,
                                                deffnm=deffnm,
                                                continuation=continuation,
                                                       )
                                           )
                       for p, sconf, deffnm in zip(self.propagators,
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
            if t.exception() is not None:
                # cancel all tasks that might still be running
                for tt in trial_tasks:
                    tt.cancel()
                # raise the exception, we take care of retrying complete steps
                # in the PathSamplingChain
                raise t.exception() from None
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
        out_tra_names = [os.path.join(wdir, (self.forward_deffnm
                                             + self.traj_to_state_suffix
                                             + "." + self.output_traj_type)
                                      ),
                         os.path.join(wdir, (self.backward_deffnm
                                             + self.traj_to_state_suffix
                                             + "." + self.output_traj_type)
                                      ),
                         ]
        concats = await asyncio.gather(*(
                        p.cut_and_concatenate(trajs=trajs,
                                              tra_out=traj_out,
                                              # if we continue the traj might
                                              # already be there
                                              overwrite=continuation)
                        for p, trajs, traj_out in zip(self.propagators,
                                                      [fw_trajs, bw_trajs],
                                                      out_tra_names,
                                                      )
                                         )
                                       )
        # cut and concatenate returns (traj_to_state, first_state_reached)
        # but since we already know about the states we ignore them here
        (fw_traj, _), (bw_traj, _) = concats
        # NOTE: we already loaded our own private copy of the model at the
        # beginning of this move, so we can just continue using it
        if isinstance(self, ModelDependentPathMover):
            # If we use a model, use selecting model to predict the commitment
            # probabilities for the SP
            predicted_committors_sp = (await model(fw_startconf))[0]
        else:
            # no model, we set the predicted pB to None as we can not know it
            predicted_committors_sp = None
        # check if the two trial trajectories end in different states
        if fw_state == bw_state:
            # TODO: concatenate the forward and backward parts into a path_traj
            #       even if we did not generate a transition?!
            #       This would be useful for VIE-TPS analyses of the produced
            #       ensemble, where we need the A->A and B->B paths to estimate
            #       the equilibrium distribution
            #       It would also possibly enable us to restructure the code
            #       here such that we have the same code path for transitions
            #       and not (similar to what we do for the FixedLengthTPS where
            #        we just set the weight of the new path to zero if it is
            #        not a transition), note however that we would then either
            #       need to give up ordering transitions from low to higher
            #       index state as we do here (or we will still have two code
            #        paths: one for transitions where we can do the ordering
            #        and one for the trials without a transition, which we
            #        cand only order as [backward, forward] independent of
            #        where they went)
            logger.info("Sampler %d: Both trials reached state %d.",
                        self.sampler_idx, fw_state)
            half_finished_step = functools.partial(
                                    MCstep,
                                    mover=self,
                                    stepnum=stepnum,
                                    directory=wdir,
                                    predicted_committors_sp=predicted_committors_sp,
                                    shooting_snap=fw_startconf,
                                    states_reached=states_reached,
                                    # TODO: do we want to add fw_trajs and bw_trajs?
                                    #       i.e. the traj segments (or is the concatenated enough)
                                    trial_trajectories=[fw_traj, bw_traj],
                                    )
            # (potentially) remove the temporary files
            if self.remove_temp_trajs:
                await asyncio.gather(*(p.remove_parts(
                                            workdir=wdir,
                                            deffnm=n,
                                            file_endings_to_remove=["trajectories",
                                                                    "log"])
                                       for p, n in zip(self.propagators,
                                                       [self.forward_deffnm,
                                                        self.backward_deffnm])
                                       )
                                     )
            if self.sp_selector.probability_is_ensemble_weight:
                # We have/are creating an unordered ensemble of TP with weights
                # need to 'accept' all trials, in this case with weight=0
                return half_finished_step(accepted=True, p_acc=1, weight=0,)
            else:
                # we have a 'real' Markov chain, set p_acc=0 for non-TP
                # this kicks them out of the MCStates iteration
                return half_finished_step(accepted=False, p_acc=0, weight=1,)
        else:
            # its a TP, we will cut, concatenate and order it such that it goes
            # from lower idx state to higher idx state
            if fw_state > bw_state:
                minus_trajs, minus_state = bw_trajs, bw_state
                plus_trajs, plus_state = fw_trajs, fw_state
            else:
                # can only be the other way round
                minus_trajs, minus_state = fw_trajs, fw_state
                plus_trajs, plus_state = bw_trajs, bw_state
            logger.info(f"Sampler {self.sampler_idx}: "
                        + f"Forward trial reached state {fw_state}, "
                        + f"backward trial reached state {bw_state}.")
            tra_out = os.path.join(wdir, f"{self.path_filename}.{self.output_traj_type}")
            path_traj = await construct_TP_from_plus_and_minus_traj_segments(
                            minus_trajs=minus_trajs, minus_state=minus_state,
                            plus_trajs=plus_trajs, plus_state=plus_state,
                            state_funcs=self.states, tra_out=tra_out,
                            # if we continue the traj might already be there
                            struct_out=None, overwrite=continuation,
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
                                                        model=model,
                                                                     )
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
                                                        model=model),
                                                self.sp_selector.probability(
                                                        snapshot=fw_startconf,
                                                        trajectory=path_traj,
                                                        model=model),
                                                            )
                if self.path_weight_func is None:
                    # no weight function so all paths have equal weight
                    W_path_new = W_path_old = 1
                else:
                    W_path_new, W_path_old = await asyncio.gather(
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
                p_acc = (p_sel_new * W_path_new) / (p_sel_old * W_path_old)
                # The weight of the new path is the inverse of the path bias,
                # e.g. if we give it W_path_new=2 we would sample it twice as
                # often as in equilibrium and so we need to give it the
                # ensemble_weight=1/2
                ensemble_weight = 1 / W_path_new
                log_str = f"Sampler {self.sampler_idx}: Acceptance probability"
                log_str += f" for generated trial is {round(p_acc, 6)}."
                accepted = False
                if (p_acc >= 1) or (p_acc > self._rng.random()):
                    accepted = True
                    log_str += " Trial was accepted."
            # In both cases: log and return the MCstep
            logger.info(log_str)
            # (potentially) remove the temporary files on the way too
            if self.remove_temp_trajs:
                await asyncio.gather(*(p.remove_parts(
                                            workdir=wdir,
                                            deffnm=n,
                                            file_endings_to_remove=["trajectories",
                                                                    "log"])
                                       for p, n in zip(self.propagators,
                                                       [self.forward_deffnm,
                                                        self.backward_deffnm])
                                       )
                                     )
            return MCstep(mover=self,
                          stepnum=stepnum,
                          directory=wdir,
                          predicted_committors_sp=predicted_committors_sp,
                          shooting_snap=fw_startconf,
                          states_reached=states_reached,
                          # TODO: same as above: add fw_trajs and bw_trajs?
                          trial_trajectories=[fw_traj, bw_traj],
                          path=path_traj,
                          accepted=accepted,
                          p_acc=p_acc,
                          weight=ensemble_weight,
                          )


class TwoWayShootingPathMover(_TwoWayShootingPathMoverMixin,
                              ModelDependentPathMover):
    # the TwoWayShooting class that uses a model (saves it before every step)
    __doc__ = (_TwoWayShootingPathMoverMixin.__doc__
               + ModelDependentPathMover.__doc__
               )
    pass


class TwoWayShootingPathMoverSansModel(_TwoWayShootingPathMoverMixin,
                                       PathMoverSansModel):
    # the TwoWayShooting class that does not use a model (no saving done),
    # (this will be faster if your SPselector does not use a reaction
    #  coordinate model to bias the selection)
    __doc__ = (_TwoWayShootingPathMoverMixin.__doc__
               + PathMoverSansModel.__doc__
               )
    pass


class _FixedLengthTwoWayShootingPathMoverMixin:
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

    # for TwoWay shooting moves until any state is reached
    forward_deffnm = "forward"  # engine deffnm for forward shot
    backward_deffnm = "backward"  # same for backward shot
    path_filename = "path"  # filename for produced trajectories
    # remove the trajectory parts when the trial is done (also logs)
    remove_temp_trajs = True

    def __init__(self, n_steps: int, states, engine_cls, engine_kwargs: dict,
                 walltime_per_part: float, T: float,
                 sp_selector: SPSelector = RCModelSPSelectorFromTraj(),
                 path_weight_func=None):
        """
        Initialize a FixedLengthTwoWayShootingPathMover.

        Parameters
        ----------
        n_steps : int
            Number of integration steps to perform in total (over both forward
            and backward combined).
        states : list[asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper]
            State functions to use to decide if a trial is valid.
        engine_cls :
            The class of the molecular dynamics engine to use, we expect one of
            the `asyncmd` engines.
        engine_kwargs : dict
            A dictionary with keyword arguments needed to initialize the given
            molecular dynamics engine.
        walltime_per_part : float
            Simulation walltime per trajectory part in hours, directly passed
            to the Propagator.
        T : float
            Temperature in degree K (used for velocity randomization).
        sp_selector : aimmd.distributed.spselector.SPSelector, optional
            The shooting point selector to use, by default we will use the
            `aimmd.distributed.spselector.RCModelSPSelectorFromTraj` with its
             default options.
        path_weight_func : asyncmd.trajectory.functionwrapper.TrajectoryFunctionWrapper
            Weight function for paths. Can be used to enhance the sampling of
            low probability transition mechanisms. Only makes sense when the
            shooting points are choosen from the last accepted path (as opposed
            to shooting from points with known equilibrium weight).
        """
        # TODO: check that we use the same T as GMX? or maybe even directly take T from GMX (mdp)?
        # TODO: we should make properties out of everything
        #       changing anything requires recreating the propagators!
        #       (or at least modifying them)
        # NOTE on modelstore:
        # we implicitly pass None here, it will be set by
        # `PathSamplingChain.__init__()` to the rcmodel-store associated with
        # the chainstore of the chain this sampler will be used with
        super().__init__() #modelstore=None, sampler_idx=None)
        self.n_steps = n_steps
        self.states = states
        self.engine_cls = engine_cls
        self.engine_kwargs = engine_kwargs
        self.engine_kwargs["mdconfig"] = ensure_mdconfig_options(
                                self.engine_kwargs["mdconfig"],
                                # dont generate velocities, we do that ourself
                                genvel="no",
                                # dont apply constraints at start of simulation
                                continuation="yes",
                                                )
        self.walltime_per_part = walltime_per_part
        self.T = T
        self.sp_selector = sp_selector
        self.path_weight_func = path_weight_func
        try:
            # see if it is set as engine_kwarg
            output_traj_type = engine_kwargs["output_traj_type"]
        except KeyError:
            # it is not, so we use the engine_class default
            output_traj_type = engine_cls.output_traj_type
        finally:
            self.output_traj_type = output_traj_type.lower()
        self._nstout = nstout_from_mdconfig(
                                    mdconfig=self.engine_kwargs["mdconfig"],
                                    output_traj_type=self.output_traj_type,
                                            )
        # ensure n_steps % nstout == 0, such that we can randomly draw number
        # of frames for forward and backward parts and end up with a traj with
        # total length of n_steps
        if not self.n_steps % self._nstout == 0:
            raise ValueError(f"The nstout ({self._nstout}) from mdconfig is "
                             + "not compatible with the total number of steps "
                             + f"({self.n_steps}).")
        self._n_frames = self.n_steps // self._nstout
        self._build_extracts_and_propas()

    def _build_extracts_and_propas(self):
        self.frame_extractors = {"fw": RandomVelocitiesFrameExtractor(T=self.T),
                                 # will be used on the extracted randomized fw SP
                                 "bw": InvertedVelocitiesFrameExtractor(),
                                 }
        self.propagators = [InPartsTrajectoryPropagator(
                                    n_steps=self.n_steps,
                                    engine_cls=self.engine_cls,
                                    engine_kwargs=self.engine_kwargs,
                                    walltime_per_part=self.walltime_per_part,
                                                            )
                            for _ in range(2)
                            ]

    # TODO: improve?! e.g. add a description of the SP selector?!
    def __str__(self) -> str:
        return "FixedLengthTwoWayShootingPathMover"

    def __getstate__(self):
        state = super().__getstate__()
        state["frame_extractors"] = None
        state["propagators"] = None
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self._build_extracts_and_propas()

    async def _move(self, instep, stepnum, wdir, continuation, **kwargs):
        if isinstance(self, ModelDependentPathMover):
            # see if we need the model and load it only if we do
            model = self.get_model(stepnum=stepnum)
        else:
            # otherwise we set the model to None, this makes it possible to use
            # the same call to all SPselectors (including the model argument)
            model = None
        fw_sp_name = os.path.join(wdir, f"{self.forward_deffnm}_SP.trr")
        # instep can be None if we shoot from an ensemble of points
        inpath = instep.path if instep is not None else None
        if continuation:
            if inpath is None:
                fw_struct_file = os.path.join(wdir, f"{self.forward_deffnm}.tpr")
            else:
                fw_struct_file = inpath.structure_file
            if os.path.isfile(fw_sp_name) and os.path.isfile(fw_struct_file):
                # if the forward SP (and struct file for no inpath case)
                # have not yet been written it should be save to assume that we
                # can just start from scratch without loosing anything
                fw_startconf = Trajectory(trajectory_files=fw_sp_name,
                                          structure_file=fw_struct_file)
            else:
                continuation = False
        # pick a SP if we are not doing a continuation
        if not continuation:
            fw_startconf = await self.sp_selector.pick(
                                    outfile=fw_sp_name,
                                    frame_extractor=self.frame_extractors["fw"],
                                    trajectory=inpath,
                                    model=model,
                                                       )
        # NOTE: we do not apply constraints as we select from a trajectory that
        #       is produced by the engine which applies the correct constraints
        #       i.e. the configurations are alreayd constrained
        bw_sp_name = os.path.join(wdir, f"{self.backward_deffnm}_SP.trr")
        if continuation:
            if os.path.isfile(bw_sp_name):
                # if the backward SP has not yet been written it should be save
                # to assume we did not yet start the trial generation for
                # either fw or bw, so we set continuation=False
                bw_startconf = Trajectory(trajectory_files=bw_sp_name,
                                          structure_file=fw_startconf.structure_file)
            else:
                continuation = False
        # create the backwards SP if we have not yet done so
        if not continuation:
            # we only invert the fw SP
            bw_startconf = await self.frame_extractors["bw"].extract_async(
                                                        outfile=bw_sp_name,
                                                        traj_in=fw_startconf,
                                                        idx=0,
                                                                           )
        # propagate the two trial trajectories forward and backward in time
        # first create the names for the concatenated output trajs
        out_tra_names = [os.path.join(
                            wdir,
                            f"{self.forward_deffnm}.{self.output_traj_type}"),
                         os.path.join(
                            wdir,
                            f"{self.backward_deffnm}.{self.output_traj_type}")
                         ]
        # Decide/draw how many frames we do in each direction,
        # Note that we draw in frame space to make sure we can end up only with
        # multiples of nstout for the number of steps forward and backward,
        # i.e. make sure that nsteps % nstout == 0
        nframes_fw = self._rng.integers(self._n_frames)
        nframes_bw = self._n_frames - nframes_fw
        nsteps_fw = nframes_fw * self._nstout
        nsteps_bw = nframes_bw * self._nstout
        # set the number of steps for the two propagators to what we decided
        self.propagators[0].n_steps = nsteps_fw
        self.propagators[1].n_steps = nsteps_bw
        # now the actual trial generation
        trial_tasks = [asyncio.create_task(p.propagate_and_concatenate(
                                                starting_configuration=sconf,
                                                workdir=wdir,
                                                deffnm=deffnm,
                                                tra_out=out_tra,
                                                overwrite=continuation,
                                                continuation=continuation,
                                                                       )
                                           )
                       for p, sconf, deffnm, out_tra in zip(
                                self.propagators,
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
            if t.exception() is not None:
                # cancel all tasks that might still be running
                for tt in trial_tasks:
                    tt.cancel()
                # raise the exception, we take care of retrying complete steps
                # in the PathSamplingChain
                raise t.exception() from None
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
                                                    for t in [fw_traj, bw_traj]
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
        if isinstance(self, ModelDependentPathMover):
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
        tra_out = os.path.join(wdir,
                               f"{self.path_filename}.{self.output_traj_type}")
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
                                                    model=model),
                                            self.sp_selector.probability(
                                                    snapshot=fw_startconf,
                                                    trajectory=path_traj,
                                                    model=model),
                                                        )
            if self.path_weight_func is None:
                # no weight function so all paths have equal weight
                W_path_new = W_path_old = 1
            else:
                W_path_new, W_path_old = await asyncio.gather(
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
            p_acc = (p_sel_new * W_path_new) / (p_sel_old * W_path_old)
            # and make sure we can only accept transitions
            p_acc *= transition_factor
            # The weight of the new path is the inverse of the path bias,
            # i.e. if we give it W_path_new=2 we would sample it twice as often
            # as in equilibrium and so we need to give it ensemble_weight=1/2
            ensemble_weight = 1 / W_path_new
            log_str = f"Sampler {self.sampler_idx}: Acceptance probability"
            log_str += f" for generated trial is {round(p_acc, 6)}."
            accepted = False
            if (p_acc >= 1) or (p_acc > self._rng.random()):
                accepted = True
                log_str += " Trial was accepted."
        # In both cases: log and return the MCstep
        logger.info(log_str)
        # (potentially) remove the temporary trajectory files
        if self.remove_temp_trajs:
            await asyncio.gather(*(p.remove_parts(
                                        workdir=wdir,
                                        deffnm=n,
                                        file_endings_to_remove=["trajectories",
                                                                "log"])
                                   for p, n in zip(self.propagators,
                                                   [self.forward_deffnm,
                                                    self.backward_deffnm])
                                   )
                                 )
        return MCstep(mover=self,
                      stepnum=stepnum,
                      directory=wdir,
                      predicted_committors_sp=predicted_committors_sp,
                      shooting_snap=fw_startconf,
                      states_reached=states_reached,
                      trial_trajectories=[fw_traj, bw_traj],
                      path=path_traj,
                      accepted=accepted,
                      p_acc=p_acc,
                      weight=ensemble_weight,
                      )


class FixedLengthTwoWayShootingPathMover(
                    _FixedLengthTwoWayShootingPathMoverMixin,
                    ModelDependentPathMover):
    # the FixedLengthTwoWayShooting class that uses a model
    # (saves it before every step)
    __doc__ = (_FixedLengthTwoWayShootingPathMoverMixin.__doc__
               + ModelDependentPathMover.__doc__
               )
    pass


class FixedLengthTwoWayShootingPathMoverSansModel(
                    _FixedLengthTwoWayShootingPathMoverMixin,
                    PathMoverSansModel):
    # FixedLengthTwoWayShooting class that does not use a model,
    # (this will be faster if your SPselector does not use a reaction
    #  coordinate model to bias the selection)
    __doc__ = (_FixedLengthTwoWayShootingPathMoverMixin.__doc__
               + PathMoverSansModel.__doc__
               )
    pass
