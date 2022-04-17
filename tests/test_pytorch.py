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
import aimmd
import numpy as np
import openpathsampling as paths


torch = pytest.importorskip("torch")


from aimmd.pytorch.networks import ResNet, SNN, FFNet, ModuleStack


class Test_Networks:
    # NOTE: the reset_parameters tests are essentially smoke tests
    #       they only test that the parameters changed, not that
    #       the new params follow the "rigth" distribution
    def test_reset_parameters_resnet(self):
        n_blocks = 4
        resnet = ResNet(n_units=20, n_blocks=n_blocks,
                        block_class=None, block_kwargs=None,
                        )
        old_params_weights = [[]for _ in range(n_blocks)]
        old_params_biases = [[]for _ in range(n_blocks)]
        for i, block in enumerate(resnet.block_list):
            for lay in block.layers:
                old_params_biases[i].append(lay.bias.detach().numpy().copy())
                old_params_weights[i].append(lay.weight.detach().numpy().copy())
        # reset params
        resnet.reset_parameters()
        # check that they changed
        for i, block in enumerate(resnet.block_list):
            for j, lay in enumerate(block.layers[:-1]):
                assert np.all(np.not_equal(lay.bias.detach().numpy(),
                                           old_params_biases[i][j]
                                           )
                              )
                assert np.all(np.not_equal(lay.weight.detach().numpy(),
                                           old_params_weights[i][j]
                                           )
                              )
            # note that the last layer of every residual block is initialized
            # to zeros so there we can not check that they changed
            # so we just check that they are zero
            assert np.all(np.equal(block.layers[-1].bias.detach().numpy(), 0.))
            assert np.all(np.equal(block.layers[-1].weight.detach().numpy(), 0.))

    def test_reset_parameters_snn(self):
        snn = SNN(n_in=20, n_hidden=[20, 10, 20])
        old_params_weights = []
        old_params_biases = []
        for lay in snn.hidden_layers:
            old_params_biases.append(lay.bias.detach().numpy().copy())
            old_params_weights.append(lay.weight.detach().numpy().copy())
        # reset and check
        snn.reset_parameters()
        for i, lay in enumerate(snn.hidden_layers):
            assert np.all(np.not_equal(lay.bias.detach().numpy(),
                                       old_params_biases[i]
                                       )
                          )
            assert np.all(np.not_equal(lay.weight.detach().numpy(),
                                       old_params_weights[i]
                                       )
                          )

    def test_reset_parameters_ffnet(self):
        ffnet = FFNet(n_in=20, n_hidden=[20, 10, 20])
        old_params_weights = []
        old_params_biases = []
        for lay in ffnet.hidden_layers:
            old_params_biases.append(lay.bias.detach().numpy().copy())
            old_params_weights.append(lay.weight.detach().numpy().copy())
        # reset and check
        ffnet.reset_parameters()
        for i, lay in enumerate(ffnet.hidden_layers):
            assert np.all(np.not_equal(lay.bias.detach().numpy(),
                                       old_params_biases[i]
                                       )
                          )
            assert np.all(np.not_equal(lay.weight.detach().numpy(),
                                       old_params_weights[i]
                                       )
                          )

    def test_reset_parameters_module_stack(self):
        ffnet = FFNet(n_in=20, n_hidden=[20, 10, 20])
        module_stack = ModuleStack(n_out=2, modules=[ffnet])
        old_params_weights = []
        old_params_biases = []
        for lay in ffnet.hidden_layers:
            old_params_biases.append(lay.bias.detach().numpy().copy())
            old_params_weights.append(lay.weight.detach().numpy().copy())
        lp_old_weight = module_stack.log_predictor.weight.detach().numpy().copy()
        lp_old_bias = module_stack.log_predictor.bias.detach().numpy().copy()
        # first reset only the log_predictor
        module_stack.reset_parameters_log_predictor()
        # check that log predictor changed and everything else stayed
        assert np.all(np.not_equal(module_stack.log_predictor.weight.detach().numpy(),
                                   lp_old_weight
                                   )
                      )
        assert np.all(np.not_equal(module_stack.log_predictor.bias.detach().numpy(),
                                   lp_old_bias
                                   )
                      )
        for i, lay in enumerate(ffnet.hidden_layers):
            assert np.all(np.equal(lay.bias.detach().numpy(),
                                   old_params_biases[i]
                                   )
                          )
            assert np.all(np.equal(lay.weight.detach().numpy(),
                                   old_params_weights[i]
                                   )
                          )
        # save the new log preditor weights
        lp_old_weight = module_stack.log_predictor.weight.detach().numpy().copy()
        lp_old_bias = module_stack.log_predictor.bias.detach().numpy().copy()
        # now reset all and check that they changed (again)
        module_stack.reset_parameters()
        # check that log predictor changed and everything else stayed
        assert np.all(np.not_equal(module_stack.log_predictor.weight.detach().numpy(),
                                   lp_old_weight
                                   )
                      )
        assert np.all(np.not_equal(module_stack.log_predictor.bias.detach().numpy(),
                                   lp_old_bias
                                   )
                      )
        for i, lay in enumerate(ffnet.hidden_layers):
            assert np.all(np.not_equal(lay.bias.detach().numpy(),
                                       old_params_biases[i]
                                       )
                          )
            assert np.all(np.not_equal(lay.weight.detach().numpy(),
                                       old_params_weights[i]
                                       )
                          )


class Test_RCModel:
    @pytest.mark.parametrize("n_states,model_type", [('binomial', 'MultiDomain'), ('multinomial', 'MultiDomain'),
                                                     ('binomial', 'EnsembleNet'), ('multinomial', 'EnsembleNet'),
                                                     ('binomial', 'SingleNet'), ('multinomial', 'SingleNet'),
                                                     ]
                             )
    def test_store_model(self, tmp_path, n_states, model_type):
        aimmd_store = aimmd.Storage(tmp_path / 'Test_load_save_model.h5')

        states = ["A", "B"]
        if n_states == 'multinomial':
            states += ["C"]
        cv_ndim = 200
        # create random descriptors to predict probabilities for them
        descriptors = np.random.normal(size=(20, cv_ndim))
        if n_states == 'multinomial':
            shot_results = np.array([[1, 0, 1] for _ in range(20)])
            n_out = 3
        elif n_states == 'binomial':
            shot_results = np.array([[1, 1] for _ in range(20)])
            n_out = 1
        # a trainset for test_loss testing
        trainset = aimmd.TrainSet(len(states), descriptors=descriptors,
                                  shot_results=shot_results)

        # model creation
        def make_1hidden_net(n_in, n_out):
            modules = [aimmd.pytorch.networks.FFNet(n_in=n_in,
                                                    n_hidden=[n_in, n_out])
                       ]
            torch_model = aimmd.pytorch.networks.ModuleStack(n_out=n_out,
                                                             modules=modules)
            return torch_model
        if model_type == 'SingleNet':
            # move model to GPU if CUDA is available
            torch_model = make_1hidden_net(cv_ndim, n_out)
            if torch.cuda.is_available():
                torch_model = torch_model.to('cuda')
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
            model = aimmd.pytorch.EEScalePytorchRCModel(torch_model, optimizer,
                                                        descriptor_transform=None,
                                                        states=states)
        elif model_type == 'MultiDomain':
            pnets = [make_1hidden_net(cv_ndim, n_out) for _ in range(3)]
            cnet = make_1hidden_net(cv_ndim, len(pnets))
            # move model(s) to GPU if CUDA is available
            if torch.cuda.is_available():
                pnets = [pn.to('cuda') for pn in pnets]
                cnet = cnet.to('cuda')
            poptimizer = torch.optim.Adam([{'params': pn.parameters()}
                                           for pn in pnets],
                                          lr=1e-3)
            coptimizer = torch.optim.Adam(cnet.parameters(), lr=1e-3)
            model = aimmd.pytorch.EEMDPytorchRCModel(pnets=pnets,
                                                     cnet=cnet,
                                                     poptimizer=poptimizer,
                                                     coptimizer=coptimizer,
                                                     states=states,
                                                     descriptor_transform=None)
        elif model_type == "EnsembleNet":
            nnets = [make_1hidden_net(cv_ndim, n_out) for _ in range(5)]
            if torch.cuda.is_available():
                nnets = [n.to('cuda') for n in nnets]
            optims = [aimmd.pytorch.HMC(n.parameters()) for n in nnets]
            model = aimmd.pytorch.EERandEnsemblePytorchRCModel(nnets=nnets,
                                                               optimizers=optims,
                                                               states=states
                                                               )

        # predict before
        predictions_before = model(descriptors, use_transform=False)
        if model_type == 'MultiDomain':
            # test all possible losses for MultiDomain networks
            losses = ['L_pred', 'L_gamma', 'L_class']
            losses += ['L_mod{:d}'.format(i) for i in range(len(pnets))]
            test_loss_before = [model.test_loss(trainset, loss=l) for l in losses]
        else:
            test_loss_before = model.test_loss(trainset)
        # save the model and check that the loaded model predicts the same
        aimmd_store.rcmodels["test"] = model
        model_loaded = aimmd_store.rcmodels["test"]
        # predict after loading
        predictions_after = model_loaded(descriptors, use_transform=False)
        if model_type == 'MultiDomain':
            losses = ['L_pred', 'L_gamma', 'L_class']
            losses += ['L_mod{:d}'.format(i) for i in range(len(pnets))]
            test_loss_after = [model.test_loss(trainset, loss=l) for l in losses]
        else:
            test_loss_after = model.test_loss(trainset)

        assert np.allclose(predictions_before, predictions_after)
        assert np.allclose(test_loss_before, test_loss_after)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ['EESingleDomain', 'EEMultiDomain'])
    def test_toy_sim_eepytorch(self, tmp_path, ops_toy_sim_setup, model_type):
        # NOTE: this is only a smoke test.
        # We only test if we can run + restart without errors
        setup_dict = ops_toy_sim_setup

        # model creation
        def make_1hidden_net(n_in, n_out):
            modules = [aimmd.pytorch.networks.FFNet(n_in=n_in,
                                                    n_hidden=[n_in, n_out])
                       ]
            torch_model = aimmd.pytorch.networks.ModuleStack(n_out=n_out,
                                                             modules=modules)
            return torch_model
        if model_type == 'EESingleDomain':
            torch_model = make_1hidden_net(setup_dict['cv_ndim'], 1)
            # move model to GPU if CUDA is available
            if torch.cuda.is_available():
                torch_model = torch_model.to('cuda')
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
            model = aimmd.pytorch.EEScalePytorchRCModel(
                                        nnet=torch_model,
                                        optimizer=optimizer,
                                        states=setup_dict['states'],
                                        descriptor_transform=setup_dict['descriptor_transform']
                                                       )
        elif model_type == 'EEMultiDomain':
            pnets = [make_1hidden_net(setup_dict['cv_ndim'], 1)
                     for _ in range(3)]
            cnet = make_1hidden_net(setup_dict['cv_ndim'], len(pnets))
            # move model(s) to GPU if CUDA is available
            if torch.cuda.is_available():
                pnets = [pn.to('cuda') for pn in pnets]
                cnet = cnet.to('cuda')
            poptimizer = torch.optim.Adam([{'params': pn.parameters()}
                                           for pn in pnets],
                                          lr=1e-3)
            coptimizer = torch.optim.Adam(cnet.parameters(), lr=1e-3)
            model = aimmd.pytorch.EEMDPytorchRCModel(pnets=pnets,
                                                     cnet=cnet,
                                                     poptimizer=poptimizer,
                                                     coptimizer=coptimizer,
                                                     states=setup_dict["states"],
                                                     descriptor_transform=setup_dict['descriptor_transform']
                                                     )
        # create trainset and trainhook
        trainset = aimmd.TrainSet(len(setup_dict['states']))
        trainhook = aimmd.ops.TrainingHook(model, trainset)
        aimmd_storage = aimmd.Storage(tmp_path / "test.h5")
        storehook = aimmd.ops.AimmdStorageHook(aimmd_storage, model, trainset)
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
        storage = paths.Storage(tmp_path / "test.nc", 'w', template=setup_dict['template'])
        sampler = paths.PathSampling(storage=storage, sample_set=initial_conditions, move_scheme=move_scheme)
        sampler.attach_hook(trainhook)
        sampler.attach_hook(storehook)
        # generate some steps
        sampler.run(5)
        # close the storage(s)
        storage.sync_all()
        storage.close()
        aimmd_storage.close()
        # now do the testing
        load_storage = paths.Storage(tmp_path / "test.nc", 'a')
        load_sampler = load_storage.pathsimulators[0]
        load_sampler.restart_at_step(load_storage.steps[-1])
        aimmd_store_load = aimmd.Storage(tmp_path / "test.h5", "a")
        load_ts = aimmd_store_load.load_trainset()
        load_model = aimmd_store_load.rcmodels["most_recent"]
        load_model = load_model.complete_from_ops_storage(load_storage)
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
