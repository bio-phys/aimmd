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
# we use this to be able to run tests when GPU in use
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import pytest
import arcd
import numpy as np
import openpathsampling as paths
from keras import optimizers


class Test_keras:
    @pytest.mark.parametrize("save_trainset", ['save_trainset', 'recreate_trainset'])
    def test_toy_sim_snn(self, tmp_path, ops_toy_sim_setup, save_trainset):
        p = tmp_path / 'Test_OPS_test_toy_sim_snn.nc'
        fname = str(p)
        setup_dict = ops_toy_sim_setup

        hidden_parms = [{'units_factor': 1,  # test units_fact key
                         'activation': 'elu',  # should be fixed to selu
                         'use_bias': True,
                         'kernel_initializer': 'lecun_normal',
                         'bias_initializer': 'lecun_normal',
                         }
                        ]
        hidden_parms += [{'units': 20,  # test units key
                          'activation': 'selu',
                          'use_bias': True,
                          'kernel_initializer': 'lecun_normal',
                          'bias_initializer': 'lecun_normal',
                          'dropout': 0.1,
                          }
                         for i in range(1, 4)]

        optim = optimizers.Adam(lr=1e-3)
        snn = arcd.keras.create_snn(setup_dict['cv_ndim'], hidden_parms,
                                    optim, len(setup_dict['states']),
                                    multi_state=False  # do binomial predictions
                                    )
        model = arcd.keras.EEKerasRCModel(snn, setup_dict['descriptor_transform'],
                                          ee_params={'lr_0': 1e-3, 'lr_min': 1e-4,
                                                     'epochs_per_train': 1,
                                                     'interval': 3,
                                                     'window': 100}
                                          )
        trainset = arcd.TrainSet(setup_dict['states'], setup_dict['descriptor_transform'])
        trainhook = arcd.ops.TrainingHook(model, trainset)
        if save_trainset == 'recreate_trainset':
            # we simply change the name under which the trainset is saved
            # this should result in the new trainhook not finding the saved data
            # and therefore testing the recreation
            trainhook.save_trainset_prefix = 'test_test_test'
        selector = arcd.ops.RCModelSelector(model, setup_dict['states'])
        tps = paths.TPSNetwork.from_states_all_to_all(setup_dict['states'])
        move_scheme = paths.MoveScheme(network=tps)
        strategy = paths.strategies.TwoWayShootingStrategy(modifier=setup_dict['modifier'],
                                                           selector=selector,
                                                           engine=setup_dict['engine'],
                                                           group="TwoWayShooting"
                                                           )
        move_scheme.append(strategy)
        move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
        move_scheme.build_move_decision_tree()
        initial_conditions = move_scheme.initial_conditions_from_trajectories(setup_dict['initial_TP'])
        storage = paths.Storage(fname, 'w', template=setup_dict['template'])
        sampler = paths.PathSampling(storage=storage, sample_set=initial_conditions, move_scheme=move_scheme)
        sampler.attach_hook(trainhook)
        # generate some steps
        sampler.run(5)
        # close the storage
        storage.sync_all()
        storage.close()
        # now do the testing
        load_storage = paths.Storage(fname, 'a')
        load_sampler = load_storage.pathsimulators[0]
        load_sampler.restart_at_step(load_storage.steps[-1])
        load_trainhook = arcd.ops.TrainingHook(None, None)
        load_sampler.attach_hook(load_trainhook)
        load_sampler.run(1)
        # check that the two trainsets are the same
        # at least except for the last step
        assert np.allclose(load_trainhook.trainset.descriptors[:-1],
                           trainset.descriptors)
        assert np.allclose(load_trainhook.trainset.shot_results[:-1],
                           trainset.shot_results)
