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
import pytest

tf = pytest.importorskip("tensorflow",
                         reason="No tensorflow installation found.")
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

import aimmd
import numpy as np
import openpathsampling as paths
from tensorflow.keras import optimizers


class Test_RCModel:
    @pytest.mark.parametrize("n_states", ['binomial', 'multinomial'])
    def test_store_model(self, tmp_path, n_states):
        aimmd_store = aimmd.Storage(tmp_path / 'Test_load_save_model.h5')

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
        trainset = aimmd.TrainSet(n_states=len(states), descriptors=descriptors,
                                 shot_results=shot_results)
        # model creation
        optim = optimizers.Adam(lr=1e-3)
        snn = aimmd.keras.create_snn(cv_ndim, hidden_parms, optim, len(states),
                                     multi_state=multi_state
                                     )
        model = aimmd.keras.EEScaleKerasRCModel(snn, states=states,
                                                descriptor_transform=None)
        # predict before
        predictions_before = model(descriptors, use_transform=False)
        test_loss_before = model.test_loss(trainset)
        # save the model and check that the loaded model predicts the same
        aimmd_store.rcmodels["test"] = model
        model_loaded = aimmd_store.rcmodels["test"]
        # predict after loading
        predictions_after = model_loaded(descriptors, use_transform=False)
        test_loss_after = model_loaded.test_loss(trainset)
        assert np.allclose(predictions_before, predictions_after)
        assert np.allclose(test_loss_before, test_loss_after)


    @pytest.mark.slow
    def test_toy_sim_snn(self, tmp_path, ops_toy_sim_setup):
        # NOTE: this is more a smoke test than anything else
        # we do not really check for anything except that it runs and restarts
        # without errors
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
        snn = aimmd.keras.create_snn(setup_dict['cv_ndim'], hidden_parms,
                                    optim, len(setup_dict['states']),
                                    multi_state=False  # do binomial predictions
                                    )
        model = aimmd.keras.EEScaleKerasRCModel(
                    snn,
                    states=setup_dict["states"],
                    descriptor_transform=setup_dict['descriptor_transform'],
                    ee_params={'lr_0': 1e-3,
                               'lr_min': 1e-4,
                               'epochs_per_train': 1,
                               'interval': 1,
                               'window': 100}
                                                )
        trainset = aimmd.TrainSet(len(setup_dict['states']))
        trainhook = aimmd.ops.TrainingHook(model, trainset)
        aimmd_store = aimmd.Storage(tmp_path / "test.h5")
        storehook = aimmd.ops.AimmdStorageHook(aimmd_store, model, trainset)
        selector = aimmd.ops.RCModelSelector(model, setup_dict['states'])
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
        storage = paths.Storage(tmp_path / "test.nc", 'w',
                                template=setup_dict['template'])
        sampler = paths.PathSampling(storage=storage, sample_set=initial_conditions, move_scheme=move_scheme)
        sampler.attach_hook(trainhook)
        sampler.attach_hook(storehook)
        # generate some steps
        sampler.run(10)
        # close the storage
        storage.sync_all()
        storage.close()
        aimmd_store.close()
        # now do the testing
        load_storage = paths.Storage(tmp_path / "test.nc", 'a')
        aimmd_store_load = aimmd.Storage(tmp_path / "test.h5", "a")
        load_ts = aimmd_store_load.load_trainset()
        load_model = aimmd_store_load.rcmodels["most_recent"]
        load_model = load_model.complete_from_ops_storage(load_storage)
        load_sampler = load_storage.pathsimulators[0]
        load_sampler.restart_at_step(load_storage.steps[-1])
        # check that the two trainsets are the same
        assert np.allclose(trainhook.trainset.descriptors,
                           load_ts.descriptors)
        assert np.allclose(trainhook.trainset.shot_results,
                           load_ts.shot_results)
        # try restarting
        aimmd.ops.set_rcmodel_in_all_selectors(load_model, load_sampler)
        # NOTE: we reattach the hooks from previous simulation instead of recreating
        sampler.attach_hook(trainhook)
        sampler.attach_hook(storehook)
        load_sampler.run(1)
