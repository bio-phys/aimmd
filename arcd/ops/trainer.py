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
import logging
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class CommittorModelTrainer(ABC):
    """
    TODO
    """
    @abstractmethod
    def train(self, model, trainset, history):
        # called by TrainingHook every MCStep
        # needs to decide if we train, set the lr, etc and then train the model
        # NOTE: when implementing trainers for new models it might be a good
        # idea to split this in two, one function to decide and set the lr and
        # one training function that is always the same for this model type
        # see e.g. the KerasModelTrainers
        pass


class KerasModelTrainer(CommittorModelTrainer):
    """
    TODO
    """
    def __init__(self, epochs_per_training=1):
        self.epochs_per_training = epochs_per_training

    @abstractmethod
    def _train_prepare(self, model, trainset, history):
        # return true or false if we train,
        # if we train it should set the lr, etc
        pass

    def train(self, model, trainset, history):
        if self._train_prepare(model, trainset, history):
            self._train(model, trainset, history)
            history.training_decision.append(1)
        else:
            history.training_decision.append(0)

    def _train(self, model, trainset, history):
        # we do epochs_per_training runs over the trainset
        loss = []
        for epoch in range(self.epochs_per_training):
            loss.append([])
            for xt, yt, w in trainset:
                loss[epoch].append(model.model.train_on_batch(
                                         x=xt, y=yt, sample_weight=w)
                                   )

        history.loss_per_batch.append(loss)
        history.loss.append([sum([lb for lb in le]) / len(le)
                             for le in loss])


class KerasModelTrainerEE(KerasModelTrainer):
    """
    ExpectedEfficiency KerasModelTrainer
    """
    def __init__(self, epochs_per_training=2, opt_fact=0.2, trainset_minlen=15,
                 window_size=30):
        self.epochs_per_training = epochs_per_training
        self.opt_fact = opt_fact
        self.trainset_minlen = trainset_minlen
        self.window_size = window_size

    def _train_prepare(self, model, trainset, history):
        if len(trainset) >= self.trainset_minlen:
            # calculate expected eff
            if len(history.expected_efficiency) < self.window_size:
                p_tp_ex = history.expected_efficiency
                tp_gen = trainset.transitions
            else:
                p_tp_ex = history.expected_efficiency[-self.window_size:]
                tp_gen = trainset.transitions[-self.window_size:]
            n_tp_ex = sum(p_tp_ex)
            n_tp_tr = sum(tp_gen)
            d_fact = abs(1 - n_tp_tr / n_tp_ex)

            # TODO : log decisions!
            # with REASONS, i.e. TS min length vs. opt fact etc
            if d_fact > self.opt_fact:
                logger.info('Training: decision_fact={:.3f}, '
                            + 'expected TPs={:.2f}, generated TPs={:d}.'
                            + ''.format(d_fact, n_tp_ex, n_tp_tr))
                return True
            else:
                logger.info('Not training: decision_fact={:.3f}, '
                            + 'expected TPs={:.2f}, generated TPs={:d}.'
                            + ''.format(d_fact, n_tp_ex, n_tp_tr))
                return False
        else:
            logger.info('Not training: len(trainset) < trainset_minlen.')
            return False
