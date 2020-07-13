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
import pytest

tf = pytest.importorskip("tensorflow")
# we use this to be able to run tests when GPU in use
if tf.version.VERSION.startswith('2.'):
    # tell tf to use only the GPU mem it needs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('available GPUs: ', gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras import backend as K

else:
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.gpu_options.per_process_gpu_memory_fraction = 0.25
    #tf.enable_eager_execution(config=conf)\n",
    sess = tf.compat.v1.Session(config=conf)
    from tensorflow.keras import backend as K
    K.set_session(sess)

import arcd
import numpy as np
import openpathsampling as paths
from tensorflow.keras import optimizers


class Test_keras:
    @pytest.mark.old
    @pytest.mark.parametrize("n_states", ['binomial', 'multinomial'])
    def test_save_load_model(self, tmp_path, n_states):
        p = tmp_path / 'Test_load_save_model.pckl'
        fname = str(p)

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
        states = ['A', 'B']
        multi_state = False
        if n_states == 'multinomial':
            states += ['C']
            multi_state = True
        cv_ndim = 200
        # create random descriptors to predict probabilities for them
        descriptors = np.random.normal(size=(20, cv_ndim))
        if n_states == 'multinomial':
            shot_results = np.array([[1, 0, 1] for _ in range(20)])
        elif n_states == 'binomial':
            shot_results = np.array([[1, 1] for _ in range(20)])
        # a trainset for test_loss testing
        trainset = arcd.TrainSet(states, descriptors=descriptors,
                                 shot_results=shot_results)
        # model creation
        optim = optimizers.Adam(lr=1e-3)
        snn = arcd.keras.create_snn(cv_ndim, hidden_parms, optim, len(states),
                                    multi_state=multi_state
                                    )
        model = arcd.keras.EEScaleKerasRCModel(snn, descriptor_transform=None)
        # predict before
        predictions_before = model(descriptors, use_transform=False)
        test_loss_before = model.test_loss(trainset)
        # save the model and check that the loaded model predicts the same
        model.save(fname)
        state, cls = arcd.base.RCModel.load_state(fname, None)
        state = cls.fix_state(state)
        model_loaded = cls.set_state(state)
        # predict after loading
        predictions_after = model_loaded(descriptors, use_transform=False)
        test_loss_after = model_loaded.test_loss(trainset)
        assert np.allclose(predictions_before, predictions_after)
        assert np.allclose(test_loss_before, test_loss_after)

    @pytest.mark.old
    @pytest.mark.slow
    @pytest.mark.parametrize("save_trainset", ['load_trainset', 'recreate_trainset'])
    def test_toy_sim_snn(self, tmp_path, ops_toy_sim_setup, save_trainset):
        p = tmp_path / 'Test_OPS_test_toy_sim_snn.nc'
        fname = str(p)
        setup_dict = ops_toy_sim_setup

        hidden_parms = [{'units_factor': 1,  # test units_fact key
                         'activation': 'elu',  # should be changed to selu automatically
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
        model = arcd.keras.EEScaleKerasRCModel(snn, setup_dict['descriptor_transform'],
                                               ee_params={'lr_0': 1e-3,
                                                          'lr_min': 1e-4,
                                                          'epochs_per_train': 1,
                                                          'interval': 1,
                                                          'window': 100}
                                               )
        trainset = arcd.TrainSet(setup_dict['states'], setup_dict['descriptor_transform'])
        trainhook = arcd.ops.TrainingHookLegacy(model, trainset)
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
        sampler.run(10)
        # close the storage
        storage.sync_all()
        storage.close()
        # now do the testing
        load_storage = paths.Storage(fname, 'a')
        load_sampler = load_storage.pathsimulators[0]
        load_sampler.restart_at_step(load_storage.steps[-1])
        load_trainhook = arcd.ops.TrainingHookLegacy(None, None)
        load_sampler.attach_hook(load_trainhook)
        load_sampler.run(1)
        # check that the two trainsets are the same
        # at least except for the last step
        assert np.allclose(load_trainhook.trainset.descriptors[:-1],
                           trainset.descriptors)
        assert np.allclose(load_trainhook.trainset.shot_results[:-1],
                           trainset.shot_results)
