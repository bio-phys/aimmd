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
import os
import abc
import asyncio
import logging
import numpy as np

from . import _SEM_BRAIN_MODEL
from .trajectory import (RandomVelocitiesFrameExtractor,
                         InvertedVelocitiesFrameExtractor,
                         )
from .logic import (TrajectoryPropagatorUntilAnyState,
                    construct_TP_from_plus_and_minus_traj_segments,
                    )
from .gmx_utils import ensure_mdp_options


logger = logging.getLogger(__name__)


class MCstep:
    # TODO: make this 'immutable'? i.e. expose everything as get-only-properties?
    # TODO: some of the attributes are only relevant for shooting,
    #       do we want a subclass for shooting MCsteps?
    def __init__(self, mover, stepnum, directory, predicted_committors_sp=None,
                 shooting_snap=None, states_reached=None,
                 path=None, trial_trajectories=[], accepted=False, p_acc=0):
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

    # TODO!:
    #def __repr__(self):


class PathMover(abc.ABC):
    # takes an (usually accepted) in-MCstep and
    # produces an out-MCstep (not necessarily accepted)
    @abc.abstractmethod
    async def move(self, instep, stepnum, wdir, **kwargs):
        raise NotImplementedError


#TODO: DOCUMENT
class ModelDependentPathMover(PathMover):
    # PathMover that takes a model at the start of the move
    # the model would e.g. used to select the shooting point
    # here lives the code that saves the model state at the start of the step
    # this enables us to do the accept/reject (at the end) with the initially
    # saved model
    # TODO: make it possible to use without an arcd.Storage?!
    savename_prefix = "model_at_step"

    def __init__(self, modelstore=None):
        # NOTE: modelstore - aimmd.storage.RCModelRack
        #       this should enable us to use the same arcd.Storage for multiple
        #       MC chains at the same time, if we have/create multiple RCModelRacks (in Brain?)
        # NOTE : we set it to None by default because it will be set through
        #        PathSamplingChain.__init__ for all movers it owns to the
        #        rcmodel-store associated with the chainstore
        self.modelstore = modelstore
        self._rng = np.random.default_rng()  # numpy newstyle RNG, one per Mover

    # NOTE: we take care of the modelstore in storage to
    #       enable us to set the mover as MCstep attribute directly
    #       (instead of an identifying string)
    # NOTE 2: when saving a MCstep we ensure that the modelstore is the correct
    #         (as in associated with that MCchain) RCModel rack
    #         and when loading a MCstep we can set the movers.modelstore?!
    #         subclasses can then completely overwrite __getstate__
    #        and __setstate__ to create their runtime attributes
    #        (see the TwoWayShooting for an example)

    def store_model(self, model, stepnum):
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to store a model.")
        self.modelstore[f"{self.savename_prefix}{stepnum}"] = model

    def get_model(self, stepnum):
        if self.modelstore is None:
            raise RuntimeError("self.modelstore must be set to load a model.")
        return self.modelstore[f"{self.savename_prefix}{stepnum}"]

    @abc.abstractmethod
    async def move(self, instep, stepnum, wdir, model, **kwargs):
        raise NotImplementedError


# TODO: DOCUMENT
class TwoWayShootingPathMover(ModelDependentPathMover):
    # for TwoWay shooting moves until any state is reached
    forward_deffnm = "forward"  # engine deffnm for forward shot
    backward_deffnm = "backward"  # same for backward shot
    transition_filename = "transition.trr"  # filename for produced transitions

    def __init__(self, states, engine_cls, engine_kwargs, walltime_per_part, T,
                 sp_selector=None):
        """
        states - list of state functions, passed to Propagator
        descriptor_transform - coroutine function used to calculate descriptors
        engine_cls - the class of the molecular dynamcis engine to use
        engine_kwargs - a dict with keyword arguments to initialize the given
                        molecular dynamics engine
        walltime_per_part - simulation walltime per trajectory part
        T - temperature in degree K (used for velocity randomization)
        sp_selector - `aimmd.distributed.pathmovers.RCModelSPSelector` or None,
                      if None we will initialialize a selector with defaults
        """
        # NOTE: we expect state funcs to be coroutinefuncs!
        # TODO: check that we use the same T as GMX? or maybe even directly take T from GMX (mdp)?
        # TODO: we should make properties out of everything
        #       changing anything requires recreating the propagators!
        # NOTE on modelstore:
        # we implicitly pass None here, it will be set by
        # `PathSamplingChain.__init__()` to the rcmodel-store associated with
        # the chainstore of the chain this sampler will be used with
        super().__init__() #modelstore=None)
        self.states = states
        self.engine_cls = engine_cls
        self.engine_kwargs = engine_kwargs
        # TODO: we assume gmx engines here!
        #       at some point we will need to write a general function working
        #       on any MDConfig (possibly delegating to our current gmx helper
        #       functions)
        self.engine_kwargs["mdp"] = ensure_mdp_options(
                                self.engine_kwargs["mdp"],
                                # dont generate velocities, we do that ourself
                                genvel="no",
                                # dont apply constraints at start of simulation
                                continuation="yes",
                                                )
        self.walltime_per_part = walltime_per_part
        self.T = T
        if sp_selector is None:
            sp_selector = RCModelSPSelector()
        self.sp_selector = sp_selector
        self._build_extracts_and_propas()

    def _build_extracts_and_propas(self):
        self.frame_extractors = {"fw": RandomVelocitiesFrameExtractor(T=self.T),
                                 # will be used on the extracted randomized fw SP
                                 "bw": InvertedVelocitiesFrameExtractor(),
                                 }
        self.propagators = [TrajectoryPropagatorUntilAnyState(
                                    states=self.states,
                                    engine_cls=self.engine_cls,
                                    engine_kwargs=self.engine_kwargs,
                                    walltime_per_part=self.walltime_per_part
                                                              )
                            for _ in range(2)
                            ]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["frame_extractors"] = None
        state["propagators"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_extracts_and_propas()

    async def move(self, instep, stepnum, wdir, model, **kwargs):
        # NOTE/FIXME: we assume wdir is an absolute path
        #             (or at least that it is relative to cwd)
        # NOTE: need to select (and later register) the SP with the passed model
        # (this is the 'main' model and if it knows about the SPs we can use the
        #  prediction accuracy in the close past to decide if we want to train)
        async with _SEM_BRAIN_MODEL:
            self.store_model(model=model, stepnum=stepnum)
            sp_idx = await self.sp_selector.pick(instep.path, model=model)
        # release the Semaphore, we load the stored model for accept/reject later
        fw_sp_name_uc = os.path.join(wdir, f"{self.forward_deffnm}_SP_unconstrained.trr")
        fw_sp_name = os.path.join(wdir, f"{self.forward_deffnm}_SP.trr")
        fw_startconf_uc = self.frame_extractors["fw"].extract(outfile=fw_sp_name_uc,
                                                              traj_in=instep.path,
                                                              idx=sp_idx)
        # TODO: this is still a bit hacky, do we have a better solution for the
        #       constraints? do we even need them?
        constraint_engine = self.engine_cls(**self.engine_kwargs)
        fw_startconf = await constraint_engine.apply_constraints(conf_in=fw_startconf_uc,
                                                                 conf_out_name=fw_sp_name,
                                                                 wdir=wdir)
        bw_sp_name = os.path.join(wdir, f"{self.backward_deffnm}_SP.trr")
        # we only invert the fw SP
        bw_startconf = self.frame_extractors["bw"].extract(outfile=bw_sp_name,
                                                           traj_in=fw_startconf,
                                                           idx=0)
        trial_tasks = [asyncio.create_task(p.propagate(
                                                starting_configuration=sconf,
                                                workdir=wdir,
                                                deffnm=deffnm,
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
        # load the selecting model for accept/reject
        model = self.get_model(stepnum=stepnum)
        # use selecting model to predict the commitment probabilities for the SP
        predicted_committors_sp = (await model(fw_startconf))[0]
        # check if they end in different states
        if fw_state == bw_state:
            logger.info(f"Both trials reached state {fw_state}.")
            return MCstep(mover=self,
                          stepnum=stepnum,
                          directory=wdir,
                          predicted_committors_sp=predicted_committors_sp,
                          shooting_snap=fw_startconf,
                          states_reached=states_reached,
                          trial_trajectories=fw_trajs + bw_trajs,  # two lists
                          accepted=False,
                          )
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
            tra_out = os.path.join(wdir, f"{self.transition_filename}")
            path_traj = await construct_TP_from_plus_and_minus_traj_segments(
                            minus_trajs=minus_trajs, minus_state=minus_state,
                            plus_trajs=plus_trajs, plus_state=plus_state,
                            state_funcs=self.states, tra_out=tra_out,
                            struct_out=None, overwrite=False,
                                                                             )
            # accept or reject?
            p_sel_old = await self.sp_selector.probability(fw_startconf,
                                                           instep.path,
                                                           model=model)
            p_sel_new = await self.sp_selector.probability(fw_startconf,
                                                           path_traj,
                                                           model=model)
            # p_acc = ((p_sel_new * p_mod_sp_new_to_old * p_eq_sp_new)
            #          / (p_sel_old * p_mod_sp_old_to_new * p_eq_sp_old)
            #          )
            # but accept only depends on p_sel, because Maxwell-Boltzmann vels,
            # i.e. p_mod cancel with p_eq_sp velocity part
            # and configuration is the same in old and new, i.e. for positions
            # we cancel old with new
            p_acc = p_sel_new / p_sel_old
            accepted = False
            if (p_acc >= 1) or (p_acc > self._rng.random()):
                accepted = True
            return MCstep(mover=self,
                          stepnum=stepnum,
                          directory=wdir,
                          predicted_committors_sp=predicted_committors_sp,
                          shooting_snap=fw_startconf,
                          states_reached=states_reached,
                          trial_trajectories=minus_trajs + plus_trajs,
                          path=path_traj,
                          accepted=accepted,
                          p_acc=p_acc,
                          )


#TODO: DOCUMENT
class RCModelSPSelector:
    def __init__(self, scale=1., distribution="lorentzian",
                 density_adaptation=True, exclude_first_last_frame=True):
        self.distribution = distribution
        self.scale = scale
        self.density_adaptation = density_adaptation
        # whether we allow to choose first and last frame
        # if False they will also not contribute to sum_bias and accept/reject
        self.exclude_first_last_frame = exclude_first_last_frame

    @property
    def distribution(self):
        """Return the name of the shooting point selection distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, val):
        if val.lower() == 'gaussian':
            self._f_sel = self._gaussian
            self._distribution = val
        elif val.lower() == 'lorentzian':
            self._f_sel = self._lorentzian
            self._distribution = val
        else:
            raise ValueError('Distribution must be one of: '
                             + '"gaussian" or "lorentzian"')

    def _lorentzian(self, z):
        return self.scale / (self.scale**2 + z**2)

    def _gaussian(self, z):
        return np.exp(-z**2/self.scale)

    async def f(self, snapshot, trajectory, model):
        """Return the unnormalized proposal probability of a snapshot."""
        # we expect that 'snapshot' is a len 1 trajectory!
        z_sel = await model.z_sel(snapshot)
        any_nan = np.any(np.isnan(z_sel))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         + 'We used np.nan_to_num to proceed')
            z_sel = np.nan_to_num(z_sel)
        # casting to float makes errors when the np-array is not size-1,
        # i.e. we check that snapshot really was a len-1 trajectory
        ret = float(self._f_sel(z_sel))
        if self.density_adaptation:
            committor_probs = await model(snapshot)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = model.density_collector.get_correction(
                                                            committor_probs
                                                                  )
            ret *= density_fact
        if ret == 0.:
            if await self.sum_bias(trajectory) == 0.:
                logger.error("All SP weights are 0. Using equal probabilities.")
                return 1.
        return ret

    async def probability(self, snapshot, trajectory, model):
        """Return proposal probability of the snapshot for this trajectory."""
        # we expect that 'snapshot' is a len 1 trajectory!
        sum_bias = await self.sum_bias(trajectory, model)
        if sum_bias == 0.:
            logger.error("All SP weights are 0. Using equal probabilities.")
            if self.exclude_first_last_frame:
                return 1. / (len(trajectory) - 2)
            else:
                return 1. / len(trajectory)
        return (await self.f(snapshot, trajectory, model)) / sum_bias

    async def sum_bias(self, trajectory, model):
        """
        Return the partition function of proposal probabilities for trajectory.
        """
        biases = await self._biases(trajectory, model)
        return np.sum(biases)

    async def _biases(self, trajectory, model):
        z_sels = await model.z_sel(trajectory)
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         + 'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        ret = self._f_sel(z_sels)
        if self.density_adaptation:
            committor_probs = await model(trajectory)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = model.density_collector.get_correction(
                                                            committor_probs
                                                                  )
            ret *= density_fact.reshape(committor_probs.shape)
        if self.exclude_first_last_frame:
            ret = ret[1:-1]
        return ret

    async def pick(self, trajectory, model):
        """Return the index of the chosen snapshot within trajectory."""
        # NOTE: this does not register the SP with model!
        #       i.e. we do stuff different than in the ops selector
        #       For the distributed case we need to save the predicted
        #       commitment probabilities at the shooting point with the MCStep
        #       this way we can make sure that they are added to the central model
        #       in the same order as the shooting results to the trainset
        biases = await self._biases(trajectory, model)
        sum_bias = np.sum(biases)
        if sum_bias == 0.:
            logger.error('Model not able to give educated guess.\
                         Choosing based on luck.')
            # we can not give any meaningfull advice and choose at random
            if self.exclude_first_last_frame:
                # choose from [1, len(traj) - 1 )
                return np.random.randint(1, len(trajectory) - 1)
            else:
                # choose from [0, len(traj) )
                return np.random.randint(len(trajectory))

        # if self.exclude_first_last_frame == True
        # biases will be have the length of traj - 2,
        # i.e. biases already excludes the two frames
        # that means the idx we choose here is shifted by one in that case,
        # e.g. idx=0 means frame_idx=1 in the trajectory
        # (and we can not choose the last frame because biases ends before)
        rand = np.random.random() * sum_bias
        idx = 0
        prob = biases[0]
        while prob <= rand and idx < len(biases):
            idx += 1
            prob += biases[idx]
        # and return chosen idx
        if self.exclude_first_last_frame:
            idx += 1
        return idx
