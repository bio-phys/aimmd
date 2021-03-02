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
import asyncio
import concurrent.futures
import inspect
import logging
import functools
import numpy as np

from . import _SEM_MAX_PROCESS
from .trajectory import (TrajectoryConcatenator,
                         InvertedVelocitiesFrameExtractor,
                         RandomVelocitiesFrameExtractor,
                         )


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


class SaveTask(BrainTask):
    def __init__(self, storage, model, trainset, interval=100,
                 name_prefix="Central_RCModel"):
        super().__init__(interval=interval)
        self.storage = storage
        self.model = model
        self.trainset = trainset
        self.name_prefix = name_prefix

    def run(self, brain, mcstep, chain_idx):
        # this only runs when total_steps % interval == 0
        # i.e. we can just save when we run
        self.storage.save_trainset(self.trainset)
        savename = f"{self.name_prefix}_after_step{brain.total_steps}"
        self.storage.rcmodels[savename] = self.model


class TrainingTask(BrainTask):
    # add stuff to trainset + call model trainhook
    def __init__(self, model, trainset, interval=1):
        super().__init__(interval=interval)
        self.trainset = trainset
        self.model = model

    def run(self, brain, mcstep, chain_idx):
        try:
            states_reached = mcstep.states_reached
            shooting_snap = mcstep.shooting_snap
        except AttributeError:
            # wrong kind of move?!
            logger.warning("Tried to add a step that was no shooting snapshot")
        else:
            descriptors = self.model.descriptor_transform(shooting_snap)
            # descriptors is 2d but append_point expects 1d
            self.trainset.append_point(descriptors=descriptors[0],
                                       shot_results=states_reached)
        # always call the train hook
        self.model.train_hook(self.trainset)


# TODO: finish this!! (when we have played with storage a bit,
#                      to make sure the API that we use stays)
class DensityCollectionTask(BrainTask):
    # do density collection
    def __init__(self, model, first_collection=100, recreate_interval=500,
                 interval=10):
        super().__init__(interval=interval)
        self.model = model
        self.first_collection = first_collection
        self.recreate_interval = recreate_interval

    def run(self, brain, mcstep, chain_idx):
        if brain.storage is None:
            logger.warn("Density collection/adaptation is currently only "
                        + "possible for simulations with attached storage."
                        )
            return
        dc = self.model.density_collector
        if brain.total_steps - self.first_collection >= 0:
            if brain.total_steps - self.first_collection == 0:
                # first collection
                # TODO: write this function for arcd storages!
                #tps, counts = accepted_trials_from_ops_storage(
                #                                storage=sim.storage,
                #                                start=-self.first_collection,
                #                                               )
                dc.add_density_for_trajectories(model=self.model,
                                                trajectories=tps,
                                                counts=counts,
                                                )
            elif brain.total_steps % self.interval == 0:
                # add only the last interval steps
                # TODO: write this function for arcd storages!
                #tps, counts = accepted_trials_from_ops_storage(
                #                                storage=sim.storage,
                #                                start=-self.interval,
                #                                               )
                dc.add_density_for_trajectories(model=self.model,
                                                trajectories=tps,
                                                counts=counts,
                                                )
            # Note that this below is an if because reevaluation should be
            # independent of adding new TPs in the same MCStep
            if brain.total_steps % self.recreate_interval == 0:
                # reevaluation time
                dc.reevaluate_density(model=self.model)


class Brain:
    """
    The 'brain' of the simulation.

    Attributes
    ----------
        model - the committor model
        workdir - a directory
        storage - arcd.Storage
        movers - list of PathMovers
        mover_weights - None or list of floats, entries must be probabilities,
                        i.e. must sum to 1
        tasks - list of `BrainTask` objects,
                tasks will be checked if they should run in the order they are
                in the list after any one TPS sim has finished a trial,
                note that tasks will only be ran at their specified intervals

    """
    # TODO: docstring + remove obsolete notes when we are done
    # tasks should do basically what the hooks do in the ops case:
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

    def __init__(self, model, workdir, storage, movers, mover_weights, tasks=[], **kwargs):
        # TODO: we expect movers/mover_weights to be lists of lists?
        #       one for each chain, this is lazy and put the setup burden on the user!
        # TODO: descriptor_transform and states?!
        self.model = model
        self.workdir = os.path.abspath(workdir)
        self.storage = storage
        self.tasks = tasks
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
        self.total_steps = 0
        # chain-setup
        cwdirs = [os.path.join(self.workdir, f"{self.chain_directory_prefix}{i}")
                  for i in range(len(movers))]
        # make the dirs
        [os.mkdir(d) for d in cwdirs]
        #self.storage.initialize_central_memory(n_chains=len(movers))
        cstores = [c for c in self.storage.central_memory]
        self.chains = [PathSamplingChain(workdir=wdir,
                                         chainstore=cstore,
                                         movers=movs,
                                         mover_weights=mov_ws)
                       for wdir, cstore, movs, mov_ws
                       in zip(cwdirs, cstores, movers, mover_weights)
                       ]

    # TODO: do we need/want weights? better func name?
    def seed_initial_mcsteps(self, trajectories, weights):
        # should put a trajectory into every PathSampling sim
        # tranjectories should be a list of 'trajectory' objects?
        return NotImplementedError

    # TODO:! write this to save the brain and its chains!
    #        only need to take care of the movers
    #        (and the chainstores + brain.storage)
    #        then we can use the usual arcdShelfs
    def object_for_pickle(self, group, overwrite):
        # currently overwrite will always be True
        return self

    def complete_from_h5py_group(self, group):
        return self

    async def run_for_n_accepts(self, n_accepts):
        # run for n_accepts in total over all chains
        acc = 0
        chain_tasks = [asyncio.create_task(c.run_step(self.model))
                       for c in self.chains]
        while acc < n_accepts:
            done, pending = await asyncio.wait(chain_tasks,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                # iterate over all done results, because there can be multiple
                # done sometimes?
                chain_idx = chain_tasks.index(result)
                mcstep = await result
                self.total_steps += 1
                if mcstep.accepted:
                    acc += 1
                # run tasks/hooks
                self._run_tasks(mcstep=mcstep,
                                chain_idx=chain_idx,
                                )
                # remove old task from list and start next step in the chain
                # that just finished
                _ = chain_tasks.pop(chain_idx)
                chain_tasks.insert(chain_idx, asyncio.create_task(
                                    self.chains[chain_idx].run_step(self.model)
                                                       )
                                   )
        # now that we have enough accepts finish all steps that are still pending
        done, pending = asyncio.wait(chain_tasks,
                                     return_when=asyncio.ALL_COMPLETED)
        for result in done:
            chain_idx = chain_tasks.index(result)
            mcstep = await result
            self.total_steps += 1
            if mcstep.accepted:
                acc += 1
            # run tasks/hooks
            self._run_tasks(mcstep=mcstep,
                            chain_idx=chain_idx,
                            )

    async def run_for_n_steps(self, n_steps):
        # run for n_steps total in all chains combined
        chain_tasks = [asyncio.create_task(c.run_step(self.model))
                       for c in self.chains]
        n_started = len(chain_tasks)
        while n_started < n_steps:
            done, pending = await asyncio.wait(chain_tasks,
                                               return_when=asyncio.FIRST_COMPLETED)
            for result in done:
                # iterate over all done results, because there can be multiple
                # done sometimes?
                chain_idx = chain_tasks.index(result)
                mcstep = await result
                self.total_steps += 1
                # run tasks/hooks
                self._run_tasks(mcstep=mcstep,
                                chain_idx=chain_idx,
                                )
                # remove old task from list and start next step in the chain
                # that just finished
                _ = chain_tasks.pop(chain_idx)
                chain_tasks.insert(chain_idx, asyncio.create_task(
                                    self.chains[chain_idx].run_step(self.model)
                                                       )
                                   )
                n_started += 1

        # enough started, finish steps that are still pending
        done, pending = asyncio.wait(chain_tasks,
                                     return_when=asyncio.ALL_COMPLETED)
        for result in done:
            chain_idx = chain_tasks.index(result)
            mcstep = await result
            self.total_steps += 1
            # run tasks/hooks
            self._run_tasks(mcstep=mcstep,
                            chain_idx=chain_idx,
                            )

    def _run_tasks(self, mcstep, chain_idx):
        for t in self.tasks:
            if self.total_steps % t.interval == 0:
                t.run(brain=self, mcstep=mcstep,
                      chain_idx=chain_idx)


class PathSamplingChain:
    # the single TPS simulation object:
    #   - keeps track of a single markov chain

    mcstep_foldername_prefix = "mcstep_"  # name = prefix + str(stepnum)
    mcstate_name_prefix = "mcstate_"  # name for symlinks to folders with
                                      # accepted step at stepnum
    # TODO: make this saveable!? together with its brain!
    # TODO: make it possible to run post-processing 'hooks'?!
    # TODO: initialize from directory structure?

    def __init__(self, workdir, chainstore, movers, mover_weights=None):
        self.workdir = os.path.abspath(workdir)
        self.chainstore = chainstore
        self.movers = movers
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

    @current_step.setter
    def current_step(self, step):
        self._current_step = step

    @property
    def n_steps(self):
        return self._stepnum

    @property
    def n_accepts(self):
        return sum(self._accepts)

    async def run_step(self, model):
        # run one MCstep from current_step
        # takes a model for SP selection
        # first choose a mover
        instep = self.current_step
        if instep is None:
            raise ValueError("current_step must be set.")
        self._stepnum += 1
        mover = self._rng.choice(self.movers, p=self.mover_weights)
        step_dir = os.path.join(self.workdir,
                                f"{self.mcstep_foldername_prefix}"
                                + f"{self._stepnum}"
                                )
        os.mkdir(step_dir)
        outstep = await mover.move(instep=instep, stepnum=self._stepnum,
                                   wdir=step_dir, model=model)
        self.chainstore.append(outstep)
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


class MCstep:
    # TODO: make this 'immutable'? i.e. expose everything as get-only-properties?
    def __init__(self, mover, stepnum, directory,
                 shooting_snap=None, states_reached=None,
                 path=None, trial_trajectories=[], accepted=False, p_acc=0):
        self.mover = mover  # TODO: should this be the obj? or an unique string identififer? or...?
        self.stepnum = stepnum
        self.directory = directory
        # TODO: this is special to shooting, should it be in a subclass?!
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

    def __init__(self, modelstore):
        # NOTE: modelstore - arcd.storage.RCModelRack
        #       this should enable us to use the same arcd.Storage for multiple
        #       MC chains at the same time, if we have/create multiple RCModelRacks (in Brain?)
        self.modelstore = modelstore
        self._rng = np.random.default_rng()  # numpy newstyle RNG, one per Mover

    # NOTE: we take care of the modelstore in storage to
    #       enable us to set the mover as MCstep attribute directly
    #       (instead of an identifying string)
    #TODO:
    # NOTE 2: when saving a MCstep we ensure that the modelstore is the correct
    #         (as in associated with that MCchain) RCModel rack
    #         and when loading a MCstep we can set the movers.modelstore?!
    #         subclasses can then completely overwrite __getstate__
    #        and __setstate__ to create their runtime attributes
    #        (see the TwoWayShooting for an example)

    def store_model(self, model, stepnum):
        self.modelstore[f"{self.savename_prefix}{stepnum}"] = model

    def get_model(self, stepnum):
        return self.modelstore[f"{self.savename_prefix}{stepnum}"]

    @abc.abstractmethod
    async def move(self, instep, stepnum, wdir, model, **kwargs):
        raise NotImplementedError


# TODO: DOCUMENT
class TwoWayShootingPathMover(ModelDependentPathMover):
    # for TwoWay shooting moves until any state is reached
    forward_deffnm = "forward"  # trajctory deffnm forward shot
    backward_deffnm = "backward"  # same for backward shot
    transition_filename = "transition"  # filename for produced transitions

    def __init__(self, modelstore, states, engines, engine_config,
                 walltime_per_part, T):
        """
        modelstore - arcd.storage.RCModelRack
        states - list of state functions, passed to Propagator
        engines - list/iterable with two initialized engines
        engine_config - MDConfig subclass [used in prepare() method of engines]
        walltime_per_part - simulation walltime per trajectory part
        T - temperature in degree K (used for velocity randomization)
        """
        # NOTE: we expect state funcs to be coroutinefuncs!
        # TODO: we should make properties out of everything
        #       changing anything requires recreating the propagators!
        super().__init__(modelstore=modelstore)
        self.states = states
        self.engines = engines
        self.engine_config = engine_config
        try:
            # make sure we do not generate velocities with gromacs
            gen_vel = engine_config["gen-vel"]
        except KeyError:
            logger.info("Setting 'gen-vel = no' in mdp.")
            engine_config["gen-vel"] = ["no"]
        else:
            if gen_vel[0] != "no":
                logger.warning("Setting 'gen-vel = no' in mdp (was 'yes').")
                engine_config["gen-vel"] = ["no"]
        self.walltime_per_part = walltime_per_part
        self.T = T
        self._build_extracts_and_propas()

    def _build_extracts_and_propas(self):
        self.frame_extractors = {"fw": RandomVelocitiesFrameExtractor(T=self.T),
                                 # will be used on the extracted Randomized fw SP
                                 "bw": InvertedVelocitiesFrameExtractor(),
                                 }
        self.propagators = [PropagatorUntilAnyState(
                                    states=self.states, engine=e,
                                    walltime_per_part=self.walltime_per_part
                                                    )
                            for e in self.engines
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
        # NOTE: need to select and register the SP with the passed model
        # (this is the 'main' model and if it knows about the SPs we can use the
        #  prediction accuracy in the close past to decide if we want to train)
        # to ensure we simply pick the SP before the first await!
        self.store_model(model=model, stepnum=stepnum)
        selector = RCModelSPSelector(model=model)
        # this also registers the picked SP with the model
        sp_idx = selector.pick(instep.path)
        os.mkdir(wdir)
        fw_sp_name = os.path.join(wdir, f"{self.forward_deffnm}_SP.trr")
        fw_startconf = self.frame_extractors["fw"].extract(outfile=fw_sp_name,
                                                           traj_in=instep.path,
                                                           idx=sp_idx)
        bw_sp_name = os.path.join(wdir, f"{self.backward_deffnm}_SP.trr")
        # we only invert the fw SP
        bw_startconf = self.frame_extractors["bw"].extract(outfile=bw_sp_name,
                                                           traj_in=fw_sp_name,
                                                           idx=0)
        trials = await asyncio.gather(*(
                        p.propagate(starting_configuration=sconf,
                                    workdir=wdir,
                                    deffnm=deffnm,
                                    run_config=self.engine_config)
                        for p, sconf, deffnm in zip(self.propagators,
                                                    [fw_startconf,
                                                     bw_startconf],
                                                    [self.forward_deffnm,
                                                     self.backward_deffnm]
                                                    )
                                        )
                                      )
        # propagate returns (list_of_traj_parts, state_reached)
        (fw_trajs, fw_state), (bw_trajs, bw_state) = trials
        states_reached = np.array([0. for _ in range(len(self.states))])
        states_reached[fw_state] += 1
        states_reached[bw_state] += 1
        # check if they end in differnt states
        if fw_state == bw_state:
            logger.info(f"Both trials reached state {fw_state}.")
            return MCstep(mover=self,
                          stepnum=stepnum,
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
            # now find the slices to concatenate
            # minus state first
            minus_state_vals = await asyncio.gather(
                                            *(self.states[minus_state](t)
                                              for t in minus_trajs)
                                                    )
            part_lens = [len(v) for v in minus_state_vals]
            # make it into one long array
            minus_state_vals = np.concatenate(minus_state_vals, axis=0)
            # get the first frame in state
            frames_in_minus, = np.where(minus_state_vals)  # where always returns a tuple
            first_frame_in_minus = np.min(frames_in_minus)
            # TODO: I think this is overkill, i.e. we can always expect that
            #       first frame in state is in laast part?!
            #       [this could potentially make this a bit shorter and maybe
            #        even a bit more readable :)]
            # find the part
            last_part_idx = 0
            frame_sum = part_lens[0]
            while first_frame_in_minus > frame_sum:
                last_part_idx += 1
                frame_sum += part_lens[last_part_idx]
            # find the first frame in state (counting from start of last part)
            _first_frame_in_minus = (first_frame_in_minus
                                     - sum(part_lens[:last_part_idx]))  # >= 0
            # now construct the slices and trajs list (backwards!)
            # the last/first part
            slices = [(_first_frame_in_minus + 1, 0, -1)]  # negative stride!
            trajs = [minus_trajs[last_part_idx]]
            # the ones we take fully (if any) [the range looks a bit strange
            # because we dont take last_part_index but include the zero as idx]
            slices += [(-1, 0, -1) for _ in range(last_part_idx - 1, -1, -1)]
            trajs += [minus_trajs[i] for i in range(last_part_idx - 1, -1, -1)]

            # now plus trajectories, i.e. the part we put in positive stride
            plus_state_vals = await asyncio.gather(
                                            *(self.states[plus_state](t)
                                              for t in plus_trajs)
                                                    )
            part_lens = [len(v) for v in plus_state_vals]
            # make it into one long array
            plus_state_vals = np.concatenate(plus_state_vals, axis=0)
            # get the first frame in state
            frames_in_plus, = np.where(plus_state_vals)
            first_frame_in_plus = np.min(frames_in_plus)
            # find the part
            last_part_idx = 0
            frame_sum = part_lens[0]
            while first_frame_in_plus > frame_sum:
                last_part_idx += 1
                frame_sum += part_lens[last_part_idx]
            # find the first frame in state (counting from start of last part)
            _first_frame_in_plus = (first_frame_in_plus
                                    - sum(part_lens[:last_part_idx]))  # >= 0
            # construct the slices and add trajs to list (forward!)
            # [this excludes last_part_idx so far]
            slices += [(0, -1, 1) for _ in range(last_part_idx)]
            trajs += [plus_trajs[i] for i in range(last_part_idx)]
            # add last part (with the last frame as first frame in plus state)
            slices += [(0, _first_frame_in_plus + 1, 1)]
            trajs += [plus_trajs[last_part_idx]]
            # finally produce thew concatenated path
            tra_out = os.path.join(wdir, f"{self.transition_filename}.trr")
            concat = functools.partial(TrajectoryConcatenator().concatenate,
                                       trajs=trajs,
                                       slices=slices,
                                       tra_out=tra_out, top_out=None,
                                       overwrite=False)
            loop = asyncio.get_running_loop()
            async with _SEM_MAX_PROCESS:
                # run concatenator in its own python process
                with concurrent.futures.ProcessPoolExecutor(1) as pool:
                    path_traj = await loop.run_in_executor(pool, concat)
            # accept or reject?
            model = self.get_model(stepnum=stepnum)
            # NOTE: we assume that the models descriptor_transform is an arcd
            # wrapped trajectory func, i.e. that it makes sense to first await
            # the calculation in a sepearate process and then reuse the cached
            # value in the selector.probability, which calls
            # descriptor_transform in a blocking manner
            _ = await model.descriptor_transform(path_traj)
            selector.model = model
            p_sel_old = selector.probability(fw_startconf, instep.path)
            p_sel_new = selector.probability(fw_startconf, path_traj)
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
                          shooting_snap=fw_startconf,
                          states_reached=states_reached,
                          trial_trajectories=minus_trajs + plus_trajs,
                          path=path_traj,
                          accepted=accepted,
                          p_acc=p_acc,
                          )


#TODO: DOCUMENT
class RCModelSPSelector:
    def __init__(self, model, scale=1., distribution="lorentzian",
                 density_adaptation=True):
        self.model = model
        self.scale = scale
        self.density_adaptation = density_adaptation

    @property
    def distribution(self):
        """Return the name of the shooting point selection distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, val):
        if val == 'gaussian':
            self._f_sel = lambda z: self._gaussian(z)
            self._distribution = val
        elif val == 'lorentzian':
            self._f_sel = lambda z: self._lorentzian(z)
            self._distribution = val
        else:
            raise ValueError('Distribution must be one of: '
                             + '"gaussian" or "lorentzian"')

    def _lorentzian(self, z):
        return self.scale / (self.scale**2 + z**2)

    def _gaussian(self, z):
        return np.exp(-z**2/self.scale)

    def f(self, snapshot, trajectory):
        """Return the unnormalized proposal probability of a snapshot."""
        # we expect that 'snapshot' is a len 1 trajectory!
        z_sel = self.model.z_sel(snapshot)
        any_nan = np.any(np.isnan(z_sel))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         + 'We used np.nan_to_num to proceed')
            z_sel = np.nan_to_num(z_sel)
        ret = self._f_sel(z_sel)
        if self.density_adaptation:
            committor_probs = self.model(snapshot)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = self.model.density_collector.get_correction(
                                                            committor_probs
                                                                       )
            ret *= density_fact
        if ret == 0.:
            if self.sum_bias(trajectory) == 0.:
                return 1.
        return ret

    def probability(self, snapshot, trajectory):
        """Return proposal probability of the snapshot for this trajectory."""
        # we expect that 'snapshot' is a len 1 trajectory!
        sum_bias = self.sum_bias(trajectory)
        if sum_bias == 0.:
            return 1./len(trajectory)
        return self.f(snapshot, trajectory) / sum_bias

    def sum_bias(self, trajectory):
        """
        Return the partition function of proposal probabilities for trajectory.
        """
        return np.sum(self._biases(trajectory))

    def _biases(self, trajectory):
        z_sels = self.model.z_sel(trajectory)
        any_nan = np.any(np.isnan(z_sels))
        if any_nan:
            logger.error('The model predicts NaNs. '
                         + 'We used np.nan_to_num to proceed')
            z_sels = np.nan_to_num(z_sels)
        ret = self._f_sel(z_sels)
        if self.density_adaptation:
            committor_probs = self.model(trajectory)
            if any_nan:
                committor_probs = np.nan_to_num(committor_probs)
            density_fact = self.model.density_collector.get_correction(
                                                            committor_probs
                                                                       )
            ret *= density_fact.reshape((len(trajectory), self.model.n_out))
        return ret

    def pick(self, trajectory):
        """Return the index of the chosen snapshot within trajectory."""
        # NOTE: also registers the SP with model!
        #       slightly different than in the ops selector
        #       but this way we can register before writing out the SP
        #       as a separate trajectory
        biases = self._biases(trajectory)
        sum_bias = np.sum(biases)
        if sum_bias == 0.:
            logger.error('Model not able to give educated guess.\
                         Choosing based on luck.')
            # we can not give any meaningfull advice and choose at random
            return np.random.randint(len(trajectory))

        rand = np.random.random() * sum_bias
        idx = 0
        prob = biases[0]
        while prob <= rand and idx < len(biases):
            idx += 1
            prob += biases[idx]
        # we assume that model has a descriptor_transform
        # that works on arcd.trajectories [and that we cache the values :)]
        # so we just get the values for the whole traj
        qs = self.model.q(trajectory)
        ps = self.model(trajectory)
        # and get the value for the choosen SP to bypass the model.register_sp
        self.model.expected_q.append(qs[idx])
        self.model.expected_p.append(ps[idx])
        # and return chosen idx
        return idx


# TODO: DOCUMENT
# TODO: rename to TrajectoryPropagatorUntilAnyState?
class PropagatorUntilAnyState:
    # - takes an initialized engine
    # - workdir etc in propagate method
    # NOTE: we assume that every state function returns a list/ a 1d array with
    #       True/False for each frame, i.e. if we are in state at a given frame
    # NOTE: we assume non-overlapping states, i.e. a configuration can not
    #       be inside of two states at the same time, it is the users
    #       responsibility to ensure that their states are sane

    def __init__(self, states, engine, walltime_per_part):
        # states - list of wrapped trajectory funcs
        # engine - initialized mdengine
        # walltime_per_part - walltime (in h) per mdrun, i.e. traj part/segment
        self._states = None
        self._state_func_is_coroutine = None
        self.states = states
        self.engine = engine
        self.walltime_per_part = walltime_per_part

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, states):
        # I think it is save to assume each state has a .__call__() method? :)
        # so we just check if it is awaitable
        self._state_func_is_coroutine = [inspect.iscoroutinefunction(s.__call__)
                                         for s in states]
        if not all(self._state_func_is_coroutine):
            # and warn if it is not
            logger.warning("It is recommended to use coroutinefunctions for all "
                        + "states. This can easily be achieved by wrapping any"
                        + " function in a TrajectoryFunctionWrapper. All "
                        + "non-coroutine state functions will be blocking when"
                        + " applied! ([s is coroutine for s in states] = "
                        + f"{self._state_func_is_coroutine})"
                        )
        self._states = states

    async def propagate_and_concatenate(self, starting_configuration, workdir,
                                        deffnm, run_config, tra_out,
                                        overwrite=False):
        # this just chains propagate and cut_and_concatenate
        # usefull for committor simulations, for e.g. TPS one should try to
        # directly concatenate both directions to a full TP if possible
        trajs, first_state_reached = await self.propagate(
                                starting_configuration=starting_configuration,
                                workdir=workdir,
                                deffnm=deffnm,
                                run_config=run_config,
                                                          )
        # NOTE: it should not matter too much speedwise that we recalculate
        #       the state functions, they are expected to be wrapped traj-funcs
        #       i.e. the second time we should just get the values from cache
        full_traj, first_state_reached = await self.cut_and_concatenate(
                                                        trajs=trajs,
                                                        tra_out=tra_out,
                                                        overwrite=overwrite,
                                                                        )
        return full_traj, first_state_reached

    async def propagate(self, starting_configuration, workdir, deffnm, run_config):
        # NOTE: curently this just returns a list of trajs + the state reached
        #       this feels a bit uncomfortable but avoids that we concatenate
        #       everything a quadrillion times when we use the results
        # starting_configuration - Trajectory with starting configuration (or None)
        # workdir - workdir for engine
        # deffnm - trajectory name(s) for engine (+ all other output file names)
        # run_config - MDConfig instance, e.g. a MDP
        # check first if the starting configuration is in any state
        state_vals = await self._state_vals_for_traj(starting_configuration)
        if np.any(state_vals):
            logger.error("Starting configuration already inside a state.")
            # we just return the starting configuration/trajectory
            # state reached is calculated below (is the same for both branches)
            trajs = [starting_configuration]
        else:
            await self.engine.prepare(
                            starting_configuration=starting_configuration,
                            workdir=workdir,
                            deffnm=deffnm,
                            run_config=run_config,
                                      )
            any_state_reached = False
            trajs = []
            while not any_state_reached:
                traj = await self.engine.run_walltime(self.walltime_per_part)
                state_vals = await self._state_vals_for_traj(traj)
                any_state_reached = np.any(state_vals)
                trajs.append(traj)
        # state_vals are the ones for the last traj
        # here we get which states are True and at which frame
        states_reached, frame_nums = np.where(state_vals)
        # gets the frame with the lowest idx where any state is True
        min_idx = np.argmin(frame_nums)
        # and now the idx to self.states of the state that was first reached
        # NOTE: if two states are reached simultaneously at min_idx,
        #       this will find the state with the lower idx only
        first_state_reached = states_reached[min_idx]
        return trajs, first_state_reached

    async def cut_and_concatenate(self, trajs, tra_out, overwrite=False):
        # trajs is a list of trajectoryes, e.g. the return of propagate
        # tra_out and overwrite are passed directly to the Concatenator
        # NOTE: we assume that frame0 of traj0 is outside of any state
        #       and return only the subtrajectory from frame0 until any state
        #       is first reached (the rest is ignored)
        # get all func values and put them into one big array
        states_vals = await asyncio.gather(
                                *(self._state_vals_for_traj(t) for t in trajs)
                                              )
        # states_vals is a list (trajs) of lists (states)
        # take state 0 (always present) to get the traj part lengths
        part_lens = [len(s[0]) for s in states_vals]  # s[0] is 1d (np)array
        states_vals = np.concatenate([np.asarray(s) for s in states_vals],
                                     axis=1)
        states_reached, frame_nums = np.where(states_vals)
        # gets the frame with the lowest idx where any state is True
        min_idx = np.argmin(frame_nums)
        first_state_reached = states_reached[min_idx]
        first_frame_in_state = frame_nums[min_idx]
        # find out in which part it is
        # TODO: is this overkill? can we assume first state occurence is in the
        #       last traj part?!
        last_part_idx = 0
        frame_sum = part_lens[0]
        while first_frame_in_state > frame_sum:
            last_part_idx += 1
            frame_sum += part_lens[last_part_idx]
        # find the first frame in state (counting from start of last part)
        _first_frame_in_state = (first_frame_in_state
                                 - sum(part_lens[:last_part_idx]))  # >= 0
        if last_part_idx > 0:
            # trajectory parts which we take fully
            slices = [(0, -1, 1) for _ in range(last_part_idx)]
        else:
            # only the first/last part
            slices = []
        # and the last part until including first_frame_in_state
        slices += [(0, _first_frame_in_state + 1, 1)]
        # we fill in all args as kwargs because there are so many
        concat = functools.partial(TrajectoryConcatenator().concatenate,
                                   trajs=trajs[:last_part_idx + 1],
                                   slices=slices,
                                   # take the topology of the traj, as it comes
                                   # from the engine directly
                                   tra_out=tra_out, top_out=None,
                                   overwrite=overwrite)
        loop = asyncio.get_running_loop()
        async with _SEM_MAX_PROCESS:
            # run concatenator in its own python process
            with concurrent.futures.ProcessPoolExecutor(1) as pool:
                full_traj = await loop.run_in_executor(pool, concat)
        return full_traj, first_state_reached

    async def _state_vals_for_traj(self, traj):
        # return a list of state_func results, one for each state func in states
        if all(self._state_func_is_coroutine):
            # easy, all coroutines
            return await asyncio.gather(*(s(traj) for s in self.states))
        elif not any(self._state_func_is_coroutine):
            # also easy (but blocking), none is coroutine
            return [s(traj) for s in self.states]
        else:
            # need to piece it together
            # first the coroutines concurrently
            coros = [s(traj) for s, s_is_coro
                     in zip(self.states, self._state_func_is_coroutine)
                     if s_is_coro
                     ]
            coro_res = asyncio.gather(*coros)
            # now either take the result from coro execution or calculate it
            all_results = []
            for s, s_is_coro in zip(self.states, self._state_func_is_coroutine):
                if s_is_coro:
                    all_results.append(coro_res.pop(0))
                else:
                    all_results.append(s(traj))
            return all_results
            # NOTE: this would be elegant, but to_thread() is py v>=3.9
            # we wrap the non-coroutines into tasks to schedule all together
            #all_states_as_coro = [
            #    s(traj) if s_is_cor else asyncio.to_thread(s, traj)
            #    for s, s_is_cor in zip(self.states, self._state_func_is_coroutine)
            #                      ]
            #return await asyncio.gather(*all_states_as_coro)
