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
import logging
from openpathsampling.beta.hooks import PathSimulatorHook
from openpathsampling import CollectiveVariable, Volume
from ..base.rcmodel import TrajectoryDensityCollector
from .utils import analyze_ops_mcstep, accepted_trials_from_ops_storage


logger = logging.getLogger(__name__)


class AimmdStorageHook(PathSimulatorHook):
    """Save model and trainset to aimmd storage in predefined intervals."""

    implemented_for = ["before_simulation", "after_step", "after_simulation"]

    def __init__(self, storage, model, trainset, interval=500,
                 name_prefix="RCModel"):
        """
        Parameters
        ----------
        storage - :class:`aimmd.Storage`, the storage to save to
        model - :class:`aimmd.RCModel`, the aimmd RCModel to save
        trainset - :class:`aimmd.TrainSet`, the trainingset to save
        interval - int (default=500), interval (in MC steps) at which to save
        name_prefix - str (default="RCModel"), models will be saved named as
                      this prefix plus the suffix "_after_step_$STEPNUMBER",
                      where "$STEPNUMBER" is replaced with the MCStep at save
        """
        # TODO?: do we want the option to have different saving intervals for
        #        trainset and model?
        self.storage = storage
        self.model = model
        self.trainset = trainset
        self.interval = interval
        self.name_prefix = name_prefix

    def before_simulation(self, sim, **kwargs):
        # TODO: load old model?! Or should we let the user do that explicitly?
        #       would result in less assumptions and more flexibility... :)
        # save ops stuff to ops storage
        # these should essentially be a no-ops if it is already in storage and
        # circumvents unhappy users that forgot to save the states/transform
        if sim.storage is not None:
            # Note: the trainset.states should all already be in ops storage,
            #       since they are also used as ops states for the TPS/TIS
            #       But as always: better be save than sorry!
            for s in self.model.states:
                if isinstance(s, Volume):
                    sim.storage.save(s)
            # save the descriptor_transform
            if isinstance(self.model.descriptor_transform, CollectiveVariable):
                sim.storage.save(self.model.descriptor_transform)

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        if step_number % self.interval == 0:
            # save the model
            save_name = (self.name_prefix
                         + "_after_step_{:d}".format(step_number)
                         )
            self.storage.rcmodels[save_name] = self.model
            # save the trainset
            self.storage.save_trainset(self.trainset)
            logger.info("Saved RCModel and TrainSet to aimmd-Storage.")

    def after_simulation(self, sim, hook_state):
        # save model
        step_number = sim.step
        save_name = self.name_prefix + "_after_step_{:d}".format(step_number)
        self.storage.rcmodels[save_name] = self.model
        # also save as 'most_recent' to make restarting easier
        self.storage.rcmodels["most_recent"] = self.model
        # save trainset
        self.storage.save_trainset(self.trainset)


class DensityCollectionHook(PathSimulatorHook):
    """
    Collect/update estimate of density of transition path in committor space.

    Note that this hook only operates on the density collector attached to the
    given model. However density collection is inherently model specific, since
    it uses the given models prediction to span the committor space in whic the
    histogramming is done. Therefore this hook takes a RCModel to initialize
    instead of a density collector.
    """

    implemented_for = ["before_simulation", "after_step"]

    def __init__(self, model, first_collection=100, interval=10,
                 recreate_interval=500, reinitialize=False):
        """
        Parameters
        ----------
        model - :class:`aimmd.RCModel`, model for which the density collection
                will be done
        first_collection - int (default=100), MC step at which to do the first
                           density collection
        interval - int (default=10), interval (in MC steps) in which to add
                   accepted trial trajectories to the density estimate
        recreate_interval - int (default=500), interval (in MC steps) in which
                            the complete estimate is redone for all stored
                            trajectories using the current models predictions
        reinitialize - bool (default=False), if True the models density
                       collector is completely recreated before the simulation
                       starts, the init arguments of the collector are copied
                       and all accepted trajectories that can be found in the
                       ops storage are added to the new density collector
                       Note: mostly useful if you want to later enable density
                             collection for a model that has already trained
        """
        self.model = model
        self.first_collection = first_collection
        self.interval = interval
        self.recreate_interval = recreate_interval
        self.reinitialize = reinitialize

    def before_simulation(self, sim, **kwargs):
        if self.reinitialize:
            if sim.storage is None:
                raise RuntimeError("Density collection/adaptation is currently"
                                   + " only possible for simulations with "
                                   + "attached ops storage."
                                   )
            n_dim = self.model.density_collector.n_dim
            bins = self.model.density_collector.bins
            cache_file = self.model.density_collector.cache_file
            new_dc = TrajectoryDensityCollector(n_dim=n_dim, bins=bins,
                                                cache_file=cache_file,
                                                )
            self.model.density_collector = new_dc
            tps, counts = accepted_trials_from_ops_storage(
                                                    storage=sim.storage,
                                                    start=0,
                                                           )
            self.model.density_collector.add_density_for_trajectories(
                                                        model=self.model,
                                                        trajectories=tps,
                                                        counts=counts,
                                                                      )

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        if sim.storage is None:
            logger.warn("Density collection/adaptation is currently only "
                        + "possible for simulations with attached ops storage."
                        )
            return
        dc = self.model.density_collector
        if step_number - self.first_collection >= 0:
            if step_number - self.first_collection == 0:
                # first collection
                tps, counts = accepted_trials_from_ops_storage(
                                                storage=sim.storage,
                                                start=-self.first_collection,
                                                               )
                dc.add_density_for_trajectories(model=self.model,
                                                trajectories=tps,
                                                counts=counts,
                                                )
            elif step_number % self.interval == 0:
                # add only the last interval steps
                tps, counts = accepted_trials_from_ops_storage(
                                                storage=sim.storage,
                                                start=-self.interval,
                                                               )
                dc.add_density_for_trajectories(model=self.model,
                                                trajectories=tps,
                                                counts=counts,
                                                )
            # Note that this below is an if because reevaluation should be
            # independent of adding new TPs in the same MCStep
            if step_number % self.recreate_interval == 0:
                # reevaluation time
                dc.reevaluate_density(model=self.model)


class TrainingHook(PathSimulatorHook):
    """
    Iteratively add shooting results to the trainset and train the model.

    Note that in reality the model 'decides' itself if it trains, this hook
    just calls the models train_hook function.
    """

    implemented_for = ["after_step"]

    def __init__(self, model, trainset, add_invalid_mcsteps=False):
        """
        Parameters
        ----------
        model - :class:`aimmd.RCModels`, the model to train
        trainset - :class:`aimmd.TrainSet`, the trainingset used to store
                    shooting results
        add_invalid_mcsteps - bool (default=False), wheter to also add shooting
                              trials that contain uncommitted trajectories to
                              the trainingset
        """
        self.model = model
        self.trainset = trainset
        # whether we add invalid MCSteps to the TrainSet
        # this is passed to TrainSet.add_ops_step() as add_invalid
        self.add_invalid_mcsteps = add_invalid_mcsteps

    def after_step(self, sim, step_number, step_info, state, results,
                   hook_state):
        """
        Will be called by OPS PathSimulator after every MCStep.

        Adds the MCSteps results to the trainset and lets the model decide if
        it 'wants' to train.
        """
        # results is the MCStep
        descriptors, shot_results = analyze_ops_mcstep(
                                        mcstep=results,
                                        descriptor_transform=self.model.descriptor_transform,
                                        states=self.model.states
                                                       )
        if sum(shot_results) == 2 or self.add_invalid_mcsteps:
            # add the SP only if both trials reached a state
            # OR if add_invalid is True
            self.trainset.append_point(descriptors=descriptors,
                                       shot_results=shot_results,
                                       )
        self.model.train_hook(self.trainset)
