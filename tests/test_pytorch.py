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
import arcd
import numpy as np
import openpathsampling as paths


torch = pytest.importorskip("torch")


class Test_RCModel:
    @pytest.mark.parametrize("n_states,model_type", [('binomial', 'MultiDomain'), ('multinomial', 'MultiDomain'),
                                                     ('binomial', 'EnsembleNet'), ('multinomial', 'EnsembleNet'),
                                                     ('binomial', 'SingleNet'), ('multinomial', 'SingleNet'),
                                                     ]
                             )
    def test_store_model(self, tmp_path, n_states, model_type):
        arcd_store = arcd.Storage(tmp_path / 'Test_load_save_model.h5')

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
        trainset = arcd.TrainSet(len(states), descriptors=descriptors,
                                 shot_results=shot_results)
        # model creation
        def make_1hidden_net(n_in, n_out):
            modules = [arcd.pytorch.networks.FFNet(n_in=n_in,
                                                   n_hidden=[n_in, n_out])
                       ]
            torch_model = arcd.pytorch.networks.ModuleStack(n_out=n_out,
                                                            modules=modules)
            return torch_model
        if model_type == 'SingleNet':
            # move model to GPU if CUDA is available
            torch_model = make_1hidden_net(cv_ndim, n_out)
            if torch.cuda.is_available():
                torch_model = torch_model.to('cuda')
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
            model = arcd.pytorch.EEScalePytorchRCModel(torch_model, optimizer,
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
            model = arcd.pytorch.EEMDPytorchRCModel(pnets=pnets,
                                                    cnet=cnet,
                                                    poptimizer=poptimizer,
                                                    coptimizer=coptimizer,
                                                    states=states,
                                                    descriptor_transform=None)
        elif model_type == "EnsembleNet":
            nnets = [make_1hidden_net(cv_ndim, n_out) for _ in range(5)]
            if torch.cuda.is_available():
                nnets = [n.to('cuda') for n in nnets]
            optims = [arcd.pytorch.HMC(n.parameters()) for n in nnets]
            model = arcd.pytorch.EERandEnsemblePytorchRCModel(nnets=nnets,
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
        arcd_store.rcmodels["test"] = model
        model_loaded = arcd_store.rcmodels["test"]
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
    @pytest.mark.parametrize( "model_type", ['EESingleDomain', 'EEMultiDomain',])
    def test_toy_sim_eepytorch(self, tmp_path, ops_toy_sim_setup, model_type):
        # NOTE: this is only a smoke test.
        # We only test if we can run + restart without errors
        setup_dict = ops_toy_sim_setup

        # model creation
        def make_1hidden_net(n_in, n_out):
            modules = [arcd.pytorch.networks.FFNet(n_in=n_in,
                                                   n_hidden=[n_in, n_out])
                       ]
            torch_model = arcd.pytorch.networks.ModuleStack(n_out=n_out,
                                                            modules=modules)
            return torch_model
        if model_type == 'EESingleDomain':
            torch_model = make_1hidden_net(setup_dict['cv_ndim'], 1)
            # move model to GPU if CUDA is available
            if torch.cuda.is_available():
                torch_model = torch_model.to('cuda')
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
            model = arcd.pytorch.EEScalePytorchRCModel(
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
            model = arcd.pytorch.EEMDPytorchRCModel(pnets=pnets,
                                                    cnet=cnet,
                                                    poptimizer=poptimizer,
                                                    coptimizer=coptimizer,
                                                    states=setup_dict["states"],
                                                    descriptor_transform=setup_dict['descriptor_transform']
                                                    )
        # create trainset and trainhook
        trainset = arcd.TrainSet(len(setup_dict['states']))
        trainhook = arcd.ops.TrainingHook(model, trainset)
        arcd_storage = arcd.Storage(tmp_path / "test.h5")
        storehook = arcd.ops.ArcdStorageHook(arcd_storage, model, trainset)
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
        storage = paths.Storage(tmp_path / "test.nc", 'w', template=setup_dict['template'])
        sampler = paths.PathSampling(storage=storage, sample_set=initial_conditions, move_scheme=move_scheme)
        sampler.attach_hook(trainhook)
        sampler.attach_hook(storehook)
        # generate some steps
        sampler.run(5)
        # close the storage(s)
        storage.sync_all()
        storage.close()
        arcd_storage.close()
        # now do the testing
        load_storage = paths.Storage(tmp_path / "test.nc", 'a')
        load_sampler = load_storage.pathsimulators[0]
        load_sampler.restart_at_step(load_storage.steps[-1])
        arcd_store_load = arcd.Storage(tmp_path / "test.h5", "a")
        load_ts = arcd_store_load.load_trainset()
        load_model = arcd_store_load.rcmodels["most_recent"]
        load_model = load_model.complete_from_ops_storage(load_storage)
        # check that the two trainsets are the same
        assert np.allclose(trainhook.trainset.descriptors,
                           load_ts.descriptors)
        assert np.allclose(trainhook.trainset.shot_results,
                           load_ts.shot_results)
        # try restarting
        arcd.ops.set_rcmodel_in_all_selectors(load_model, load_sampler)
        # NOTE: we reattach the hooks from previous simulation instead of recreating
        sampler.attach_hook(trainhook)
        sampler.attach_hook(storehook)
        load_sampler.run(1)