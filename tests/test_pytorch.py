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


class Test_pytorch:
    @pytest.mark.parametrize("n_states,model_type", [('binomial', 'MultiDomain'), ('multinomial', 'MultiDomain'),
                                                     ('binomial', 'EnsembleNet'), ('multinomial', 'EnsembleNet'),
                                                     ('binomial', 'SingleNet'), ('multinomial', 'SingleNet'),
                                                     ]
                             )
    def test_store_model(self, tmp_path, n_states, model_type):
        arcd_store = arcd.Storage(tmp_path / 'Test_load_save_model.h5')

        states = ['A', 'B']
        if n_states == 'multinomial':
            states += ['C']
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
        trainset = arcd.TrainSet(states, descriptors=descriptors,
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
                                                  descriptor_transform=None)
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
                                                    descriptor_transform=None)
        elif model_type == "EnsembleNet":
            nnets = [make_1hidden_net(cv_ndim, n_out) for _ in range(5)]
            if torch.cuda.is_available():
                nnets = [n.to('cuda') for n in nnets]
            optims = [arcd.pytorch.HMC(n.parameters()) for n in nnets]
            model = arcd.pytorch.EERandEnsemblePytorchRCModel(nnets=nnets,
                                                              optimizers=optims,
                                                              )

        # predict before
        predictions_before = model(descriptors, use_transform=False)
        if model_type == 'EEMultiDomain':
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
        if model_type == 'EEMultiDomain':
            losses = ['L_pred', 'L_gamma', 'L_class']
            losses += ['L_mod{:d}'.format(i) for i in range(len(pnets))]
            test_loss_after = [model.test_loss(trainset, loss=l) for l in losses]
        else:
            test_loss_after = model.test_loss(trainset)

        assert np.allclose(predictions_before, predictions_after)
        assert np.allclose(test_loss_before, test_loss_after)


    @pytest.mark.old
    @pytest.mark.parametrize("n_states,model_type", [('binomial', 'EEMultiDomain'), ('multinomial', 'EEMultiDomain'),
                                                     ('binomial', 'EESingleDomain'), ('multinomial', 'EESingleDomain')]
                             )
    def test_save_load_model(self, tmp_path, n_states, model_type):
        p = tmp_path / 'Test_load_save_model.pckl'
        fname = str(p)

        states = ['A', 'B']
        if n_states == 'multinomial':
            states += ['C']
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
        trainset = arcd.TrainSet(states, descriptors=descriptors,
                                 shot_results=shot_results)
        # model creation
        def make_1hidden_net(n_in, n_out):
            modules = [arcd.pytorch.networks.FFNet(n_in=n_in,
                                                   n_hidden=[n_in, n_out])
                       ]
            torch_model = arcd.pytorch.networks.ModuleStack(n_out=n_out,
                                                            modules=modules)
            return torch_model
        if model_type == 'EESingleDomain':
            # move model to GPU if CUDA is available
            torch_model = make_1hidden_net(cv_ndim, n_out)
            if torch.cuda.is_available():
                torch_model = torch_model.to('cuda')
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
            model = arcd.pytorch.EEScalePytorchRCModel(torch_model, optimizer,
                                                  descriptor_transform=None)
        elif model_type == 'EEMultiDomain':
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
                                                    descriptor_transform=None)

        # predict before
        predictions_before = model(descriptors, use_transform=False)
        if model_type == 'EEMultiDomain':
            # test all possible losses for MultiDomain networks
            losses = ['L_pred', 'L_gamma', 'L_class']
            losses += ['L_mod{:d}'.format(i) for i in range(len(pnets))]
            test_loss_before = [model.test_loss(trainset, loss=l) for l in losses]
        elif model_type == 'EESingleDomain':
            test_loss_before = model.test_loss(trainset)
        # save the model and check that the loaded model predicts the same
        model.save(fname)
        state, cls = arcd.base.RCModel.load_state(fname, None)
        state = cls.fix_state(state)
        model_loaded = cls.set_state(state)
        # predict after loading
        predictions_after = model_loaded(descriptors, use_transform=False)
        if model_type == 'EEMultiDomain':
            losses = ['L_pred', 'L_gamma', 'L_class']
            losses += ['L_mod{:d}'.format(i) for i in range(len(pnets))]
            test_loss_after = [model.test_loss(trainset, loss=l) for l in losses]
        elif model_type == 'EESingleDomain':
            test_loss_after = model.test_loss(trainset)

        assert np.allclose(predictions_before, predictions_after)
        assert np.allclose(test_loss_before, test_loss_after)

    @pytest.mark.old
    @pytest.mark.slow
    @pytest.mark.parametrize("save_trainset, model_type",
                             [('load_trainset', 'EESingleDomain'),
                              ('recreate_trainset', 'EESingleDomain'),
                              ('load_trainset', 'EEMultiDomain'),
                              ('recreate_trainset', 'EEMultiDomain'),
                              ])
    def test_toy_sim_eepytorch(self, tmp_path, ops_toy_sim_setup,
                               save_trainset, model_type):
        p = tmp_path / 'Test_OPS_test_toy_sim_eepytorch.nc'
        fname = str(p)
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
            model = arcd.pytorch.EEScalePytorchRCModel(torch_model, optimizer,
                                                  descriptor_transform=setup_dict['descriptor_transform'])
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
                                                    descriptor_transform=setup_dict['descriptor_transform'])
        # create trainset and trainhook
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
        sampler.run(5)
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