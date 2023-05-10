from posteriordb import PosteriorDatabase
import os
import json
import bridgestan as bs

import numpy as np
import pandas as pd

from stein_thinning.kernel import make_imq, make_centkgm
from stein_thinning.stein import ksd
from stein_thinning.thinning import thin
from stein_pi_is.target import PiTargetIMQ, PiTargetCentKGM
from stein_pi_is.mcmc import mala_adapt
from stein_pi_is.util import flat, comp_wksd, mkdir, nearestPD, renorm_w
from stein_pi_is.progress_bar import disable_progress_bar

import wasserstein

rng = np.random.default_rng(1234)
disable_progress_bar()


def store_wksd(
        model_name,
        nits=100_000,
        dbpath="../posteriordb/posterior_database",
        s=3.0,
        iteration_list=[10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000],
        repeat_times=10
):

    # Model Preparation
    ## Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Load Dataset
    posterior = my_pdb.posterior(model_name)
    stan = posterior.model.stan_code_file_path()
    data = json.dumps(posterior.data.values())
    model = bs.StanModel.from_stan_file(stan, data)

    ## Gold Standard
    gs_list = posterior.reference_draws()
    df = pd.DataFrame(gs_list)
    gs_constrain = np.zeros((sum(flat(posterior.information['dimensions'].values())),\
                        posterior.reference_draws_info()['diagnostics']['ndraws']))
    for i in range(len(df.keys())):
        gs_s = []
        for j in range(len(df[df.keys()[i]])):
            gs_s += df[df.keys()[i]][j]
        gs_constrain[i] = gs_s
    gs_constrain = gs_constrain.T
    gs = np.zeros((gs_constrain.shape[0], len(model.param_unconstrain(gs_constrain[0].astype(np.float64)))))
    for i in range(gs_constrain.shape[0]):
        gs[i] = model.param_unconstrain(gs_constrain[i].astype(np.float64))
    # P Target and Q Target
    ## Extract log-P-pdf and its gradient
    log_p = model.log_density
    grad_log_p = lambda x: model.log_density_gradient(x)[1]
    hess_log_p = lambda x: model.log_density_hessian(x)[2]

    ## Generate Q Target
    x_unconstrain_map = np.mean(gs, axis=0)
    dim = len(x_unconstrain_map)
    linv = nearestPD(-hess_log_p(x_unconstrain_map))

    stein_q_imq = PiTargetIMQ(log_p, grad_log_p, hess_log_p, linv)
    log_q_imq = stein_q_imq.log_q
    grad_log_q_imq = stein_q_imq.grad_log_q

    stein_q_centkgm = PiTargetCentKGM(log_p, grad_log_p, hess_log_p, linv, s, x_unconstrain_map)
    log_q_centkgm = stein_q_centkgm.log_q
    grad_log_q_centkgm = stein_q_centkgm.grad_log_q

    ## MALA With pre-conditioning
    ### Parameters

    alpha = 10 * [0.3]
    epoch = 9 * [1_000] + [nits]

    _, _, x_p_epoch, _, _, _ = mala_adapt(log_p, grad_log_p, x_unconstrain_map, 0.1, np.eye(dim), alpha, epoch)

    _, _, x_q_imq_epoch, _, _, _ = mala_adapt(log_q_imq, grad_log_q_imq, x_unconstrain_map, 0.1, np.eye(dim), alpha, epoch)

    _, _, x_q_centkgm_epoch, _, _, _ = mala_adapt(log_q_centkgm, grad_log_q_centkgm, x_unconstrain_map, 0.1, np.eye(dim), alpha, epoch)

    x_p_unconstrain = np.array(x_p_epoch[-1], dtype=np.float64)
    grad_x_p_unconstrain = np.array([grad_log_p(i) for i in x_p_unconstrain])

    x_q_imq_unconstrain = np.array(x_q_imq_epoch[-1], dtype=np.float64)
    grad_x_q_imq_unconstrain = np.array([grad_log_p(i) for i in x_q_imq_unconstrain])

    x_q_centkgm_unconstrain = np.array(x_q_centkgm_epoch[-1], dtype=np.float64)
    grad_x_q_centkgm_unconstrain = np.array([grad_log_p(i) for i in x_q_centkgm_unconstrain])

    ### Create Folder
    data_save_path = f"Data/{model_name}"
    mkdir(data_save_path)


    # Thinning
    ## Kernel Selection
    vfk0_imq = make_imq(x_p_unconstrain, grad_x_p_unconstrain, pre=linv)
    vfk0_centkgm = make_centkgm(x_p_unconstrain, grad_x_p_unconstrain, x_map=x_unconstrain_map.reshape(1,-1), pre=linv, s=s)

    ## Store
    ### KSD MALA
    res_ksd_p_imq_origin = np.zeros((repeat_times, len(iteration_list)))
    res_ksd_p_centkgm_origin = np.zeros((repeat_times, len(iteration_list)))

    ### KSD P
    res_ksd_p_imq_weight = np.zeros((repeat_times, len(iteration_list)))
    res_ksd_p_centkgm_weight = np.zeros((repeat_times, len(iteration_list)))

    ### KSD Q
    res_ksd_q_imq_weight = np.zeros((repeat_times, len(iteration_list)))
    res_ksd_q_centkgm_weight = np.zeros((repeat_times, len(iteration_list)))

    ### WS
    res_p_imq_unique = []
    res_p_imq_unique_weight = []

    res_p_centkgm_unique = []
    res_p_centkgm_unique_weight = []

    res_q_imq_unique = []
    res_q_imq_unique_weight = []

    res_q_centkgm_unique = []
    res_q_centkgm_unique_weight = []

    ## Thinning Method Selection
    for i in range(repeat_times):
        start_position = rng.integers(0, nits-np.max(iteration_list))
        for j_index, j in enumerate(iteration_list):
            ### P
            x_p_unconstrain_cutting = x_p_unconstrain[start_position:start_position+j,:]
            grad_x_p_unconstrain_cutting = grad_x_p_unconstrain[start_position:start_position+j,:]
            ### IMQ
            x_q_imq_unconstrain_cutting = x_q_imq_unconstrain[start_position:start_position+j,:]
            grad_x_q_imq_unconstrain_cutting = grad_x_q_imq_unconstrain[start_position:start_position+j,:]
            ### KGM
            x_q_centkgm_unconstrain_cutting = x_q_centkgm_unconstrain[start_position:start_position+j,:]
            grad_x_q_centkgm_unconstrain_cutting = grad_x_q_centkgm_unconstrain[start_position:start_position+j,:]

            ### Weighted KSD Calculation
            #### MALA
            res_ksd_p_imq_origin[i, j_index] = ksd(x_p_unconstrain_cutting, grad_x_p_unconstrain_cutting, vfk0_imq)[-1]
            res_ksd_p_centkgm_origin[i, j_index] = ksd(x_p_unconstrain_cutting, grad_x_p_unconstrain_cutting, vfk0_centkgm)[-1]
            #### P
            res_ksd_p_imq_weight[i, j_index], p_imq_unique, p_imq_unique_weight = comp_wksd(x_p_unconstrain_cutting, grad_x_p_unconstrain_cutting, vfk0_imq)
            res_ksd_p_centkgm_weight[i, j_index], p_centkgm_unique, p_centkgm_unique_weight = comp_wksd(x_p_unconstrain_cutting, grad_x_p_unconstrain_cutting, vfk0_centkgm)
            #### Q
            res_ksd_q_imq_weight[i, j_index], q_imq_unique, q_imq_unique_weight = comp_wksd(x_q_imq_unconstrain_cutting, grad_x_q_imq_unconstrain_cutting, vfk0_imq)
            res_ksd_q_centkgm_weight[i, j_index], q_centkgm_unique, q_centkgm_unique_weight = comp_wksd(x_q_centkgm_unconstrain_cutting, grad_x_q_centkgm_unconstrain_cutting, vfk0_centkgm)

            ### Store
            #### MALA
            res_p_imq_unique.append(p_imq_unique)
            res_p_centkgm_unique.append(p_centkgm_unique)
            #### P
            res_p_imq_unique_weight.append(p_imq_unique_weight)
            res_p_centkgm_unique_weight.append(p_centkgm_unique_weight)
            #### Q
            res_q_imq_unique.append(q_imq_unique)
            res_q_centkgm_unique.append(q_centkgm_unique)
            res_q_imq_unique_weight.append(q_imq_unique_weight)
            res_q_centkgm_unique_weight.append(q_centkgm_unique_weight)

    ## Save
    ### KSD MALA
    np.save(f"{data_save_path}/res_ksd_p_imq_origin.npy", res_ksd_p_imq_origin)
    np.save(f"{data_save_path}/res_ksd_p_centkgm_origin.npy", res_ksd_p_centkgm_origin)

    ### KSD P
    np.save(f"{data_save_path}/res_ksd_p_imq_weight.npy", res_ksd_p_imq_weight)
    np.save(f"{data_save_path}/res_ksd_p_centkgm_weight.npy", res_ksd_p_centkgm_weight)

    ### KSD Q
    np.save(f"{data_save_path}/res_ksd_q_imq_weight.npy", res_ksd_q_imq_weight)
    np.save(f"{data_save_path}/res_ksd_q_centkgm_weight.npy", res_ksd_q_centkgm_weight)

    ### Wasserstein Needed
    np.savez(f"{data_save_path}/res_p_imq_unique.npz", res_p_imq_unique)
    np.savez(f"{data_save_path}/res_p_imq_unique_weight.npz", res_p_imq_unique_weight)

    np.savez(f"{data_save_path}/res_p_centkgm_unique.npz", res_p_centkgm_unique)
    np.savez(f"{data_save_path}/res_p_centkgm_unique_weight.npz", res_p_centkgm_unique_weight)

    np.savez(f"{data_save_path}/res_q_imq_unique.npz", res_q_imq_unique)
    np.savez(f"{data_save_path}/res_q_imq_unique_weight.npz", res_q_imq_unique_weight)

    np.savez(f"{data_save_path}/res_q_centkgm_unique.npz", res_q_centkgm_unique)
    np.savez(f"{data_save_path}/res_q_centkgm_unique_weight.npz", res_q_centkgm_unique_weight)

def output_gs_name(dbpath="../posteriordb/posterior_database"):
    # Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    # Extract the Names of All Models
    pos = my_pdb.posterior_names()

    # Reordering Models in Ascending Dimensional Order
    d = {}
    for i in pos:
        try:
            d[i] = sum(my_pdb.posterior(i).information['dimensions'].values())
        except TypeError:
            d[i] = sum(flat(my_pdb.posterior(i).information['dimensions'].values()))
    df = pd.DataFrame.from_dict(d, orient='index', columns=['dimensions'])
    df.sort_values(by=['dimensions'], ascending=True, inplace=True)

    # Determining Whether the Model has a Gold Standard
    no_gs = []
    for i in pos:
        posterior = my_pdb.posterior(i)
        try:
            gs = posterior.reference_draws()
        except AssertionError:
            no_gs.append(i)

    # Models with a Gold Standard
    gs_models = list(set(pos).difference(set(no_gs)))

    return gs_models

def store_wwasd(
        model_name,
        dbpath="../posteriordb/posterior_database",
        iteration_list=[10, 30, 50, 100, 300, 500, 1_000, 3_000],
        repeat_times=10
):
    # Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Store
    ### Wasserstein MALA
    res_wass_p_origin = np.zeros((repeat_times, len(iteration_list)))

    ### Wasserstein P
    res_wass_p_imq_weight = np.zeros((repeat_times, len(iteration_list)))
    res_wass_p_centkgm_weight = np.zeros((repeat_times, len(iteration_list)))

    ### Wasserstein Q
    res_wass_q_imq_weight = np.zeros((repeat_times, len(iteration_list)))
    res_wass_q_centkgm_weight = np.zeros((repeat_times, len(iteration_list)))

    ### Create Folder
    data_save_path = f"Wass/{model_name}"
    mkdir(data_save_path)
 
    for rep in range(repeat_times):
        print(rep)
        for iter_idx in range(len(iteration_list)):
            # Load Dataset
            posterior = my_pdb.posterior(model_name)
            stan = posterior.model.stan_code_file_path()
            data = json.dumps(posterior.data.values())
            model = bs.StanModel.from_stan_file(stan, data)

            ## Gold Standard
            gs_list = posterior.reference_draws()
            df = pd.DataFrame(gs_list)
            gs_constrain = np.zeros((sum(flat(posterior.information['dimensions'].values())),\
                                posterior.reference_draws_info()['diagnostics']['ndraws']))
            for i in range(len(df.keys())):
                gs_s = []
                for j in range(len(df[df.keys()[i]])):
                    gs_s += df[df.keys()[i]][j]
                gs_constrain[i] = gs_s
            gs_constrain = gs_constrain.T
            gs = np.zeros((gs_constrain.shape[0], len(model.param_unconstrain(gs_constrain[0].astype(np.float64)))))
            for i in range(gs_constrain.shape[0]):
                gs[i] = model.param_unconstrain(gs_constrain[i].astype(np.float64))

            # MALA IMQ KGM
            p_imq_unique = np.load(f"Data/{model_name}/res_p_imq_unique.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx]
            p_imq_unique_weight = renorm_w(np.load(f"Data/{model_name}/res_p_imq_unique_weight.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx])
            p_centkgm_unique = np.load(f"Data/{model_name}/res_p_centkgm_unique.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx]
            p_centkgm_unique_weight = renorm_w(np.load(f"Data/{model_name}/res_p_centkgm_unique_weight.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx])

            q_imq_unique = np.load(f"Data/{model_name}/res_q_imq_unique.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx]
            q_imq_unique_weight = renorm_w(np.load(f"Data/{model_name}/res_q_imq_unique_weight.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx])
            q_centkgm_unique = np.load(f"Data/{model_name}/res_q_centkgm_unique.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx]
            q_centkgm_unique_weight = renorm_w(np.load(f"Data/{model_name}/res_q_centkgm_unique_weight.npz", allow_pickle=True)['arr_0'][rep*len(iteration_list)+iter_idx])

            # WS
            emd = wasserstein.EMD(n_iter_max=10_000_000)
            gs_weights = np.repeat(1/gs.shape[0], gs.shape[0])
            p_origin_weights = np.repeat(1/p_imq_unique.shape[0], p_imq_unique.shape[0])

            res_wass_p_origin[rep, iter_idx] = emd(p_origin_weights, p_imq_unique, gs_weights, gs)
            res_wass_p_imq_weight[rep, iter_idx] = emd(p_imq_unique_weight, p_imq_unique, gs_weights, gs)
            res_wass_p_centkgm_weight[rep, iter_idx] = emd(p_centkgm_unique_weight, p_centkgm_unique, gs_weights, gs)

            res_wass_q_imq_weight[rep, iter_idx] = emd(q_imq_unique_weight, q_imq_unique, gs_weights, gs)
            res_wass_q_centkgm_weight[rep, iter_idx] = emd(q_centkgm_unique_weight, q_centkgm_unique, gs_weights, gs)

    # Save
    ### WS MALA
    np.save(f"{data_save_path}/res_wass_p_origin.npy", res_wass_p_origin)

    ### WS P
    np.save(f"{data_save_path}/res_wass_p_imq_weight.npy", res_wass_p_imq_weight)
    np.save(f"{data_save_path}/res_wass_p_centkgm_weight.npy", res_wass_p_centkgm_weight)

    ### WS Q
    np.save(f"{data_save_path}/res_wass_q_imq_weight.npy", res_wass_q_imq_weight)
    np.save(f"{data_save_path}/res_wass_q_centkgm_weight.npy", res_wass_q_centkgm_weight)

def store_thinning(
        model_name,
        nits=100_000,
        dbpath="../posteriordb/posterior_database",
        s=3.0,
        iteration_length = 3000,
        repeat_times=10
):
    # Model Preparation
    ## Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Load Dataset
    posterior = my_pdb.posterior(model_name)
    stan = posterior.model.stan_code_file_path()
    data = json.dumps(posterior.data.values())
    model = bs.StanModel.from_stan_file(stan, data)

    ## Gold Standard
    gs_list = posterior.reference_draws()
    df = pd.DataFrame(gs_list)
    gs_constrain = np.zeros((sum(flat(posterior.information['dimensions'].values())),\
                        posterior.reference_draws_info()['diagnostics']['ndraws']))
    for i in range(len(df.keys())):
        gs_s = []
        for j in range(len(df[df.keys()[i]])):
            gs_s += df[df.keys()[i]][j]
        gs_constrain[i] = gs_s
    gs_constrain = gs_constrain.T
    gs = np.zeros_like(gs_constrain)
    for i in range(gs_constrain.shape[0]):
        gs[i] = model.param_unconstrain(gs_constrain[i].astype(np.float64))
    # P Target and Q Target
    ## Extract log-P-pdf and its gradient
    log_p = model.log_density
    grad_log_p = lambda x: model.log_density_gradient(x)[1]
    hess_log_p = lambda x: model.log_density_hessian(x)[2]

    ## Generate Q Target
    x_unconstrain_map = np.mean(gs, axis=0)
    dim = len(x_unconstrain_map)
    linv = nearestPD(-hess_log_p(x_unconstrain_map))

    stein_q_imq = PiTargetIMQ(log_p, grad_log_p, hess_log_p, linv)
    log_q_imq = stein_q_imq.log_q
    grad_log_q_imq = stein_q_imq.grad_log_q

    stein_q_centkgm = PiTargetCentKGM(log_p, grad_log_p, hess_log_p, linv, s, x_unconstrain_map)
    log_q_centkgm = stein_q_centkgm.log_q
    grad_log_q_centkgm = stein_q_centkgm.grad_log_q

    ## MALA With pre-conditioning
    ### Parameters

    alpha = 10 * [1]
    epoch = 9 * [1_000] + [nits]

    _, _, x_p_epoch, _, _, _ = mala_adapt(log_p, grad_log_p, x_unconstrain_map, 0.1, np.eye(dim), alpha, epoch)

    _, _, x_q_imq_epoch, _, _, _ = mala_adapt(log_q_imq, grad_log_q_imq, x_unconstrain_map, 0.1, np.eye(dim), alpha, epoch)

    _, _, x_q_centkgm_epoch, _, _, _ = mala_adapt(log_q_centkgm, grad_log_q_centkgm, x_unconstrain_map, 0.1, np.eye(dim), alpha, epoch)

    x_p_unconstrain = np.array(x_p_epoch[-1], dtype=np.float64)
    grad_x_p_unconstrain = np.array([grad_log_p(i) for i in x_p_unconstrain])

    x_q_imq_unconstrain = np.array(x_q_imq_epoch[-1], dtype=np.float64)
    grad_x_q_imq_unconstrain = np.array([grad_log_p(i) for i in x_q_imq_unconstrain])

    x_q_centkgm_unconstrain = np.array(x_q_centkgm_epoch[-1], dtype=np.float64)
    grad_x_q_centkgm_unconstrain = np.array([grad_log_p(i) for i in x_q_centkgm_unconstrain])

    ### Create Folder
    data_save_path = f"Thin_Conver/{model_name}"
    mkdir(data_save_path)


    # Thinning
    ## Kernel Selection
    vfk0_imq = make_imq(x_p_unconstrain, grad_x_p_unconstrain, pre=linv)
    vfk0_centkgm = make_centkgm(x_p_unconstrain, grad_x_p_unconstrain, x_map=x_unconstrain_map.reshape(1,-1), pre=linv, s=s)

    ## Store
    ### KSD IMQ
    res_ksd_q_imq_thinning = np.zeros((repeat_times, iteration_length))
    res_ksd_q_centkgm_thinning = np.zeros((repeat_times, iteration_length))

    ## Thinning Method Selection
    for i in range(repeat_times):
        start_position = rng.integers(0, nits-iteration_length)
        for j in range(iteration_length):
            ### IMQ
            x_q_imq_unconstrain_cutting = x_q_imq_unconstrain[start_position:start_position+j+1,:]
            grad_x_q_imq_unconstrain_cutting = grad_x_q_imq_unconstrain[start_position:start_position+j+1,:]
            ### KGM
            x_q_centkgm_unconstrain_cutting = x_q_centkgm_unconstrain[start_position:start_position+j+1,:]
            grad_x_q_centkgm_unconstrain_cutting = grad_x_q_centkgm_unconstrain[start_position:start_position+j+1,:]

            idx_q_imq = thin(x_q_imq_unconstrain_cutting,\
                        grad_x_q_imq_unconstrain_cutting,\
                        j+1,\
                        pre=linv,\
                        stnd=False,\
                        kern='imq'
                    )
            idx_q_centkgm = thin(x_q_centkgm_unconstrain_cutting,\
                            grad_x_q_centkgm_unconstrain_cutting,\
                            j+1,\
                            pre=linv,\
                            stnd=False,
                            kern='centkgm',
                            xmp=x_unconstrain_map
                        )

            x_q_imq_thinning_unconstrain_cutting = x_q_imq_unconstrain_cutting[idx_q_imq]
            grad_x_q_imq_thinning_unconstrain_cutting = grad_x_q_imq_unconstrain_cutting[idx_q_imq]

            x_q_centkgm_thinning_unconstrain_cutting = x_q_centkgm_unconstrain_cutting[idx_q_centkgm]
            grad_x_q_centkgm_thinning_unconstrain_cutting = grad_x_q_centkgm_unconstrain_cutting[idx_q_centkgm]

            ks_q_imq_thinning = ksd(x_q_imq_thinning_unconstrain_cutting, grad_x_q_imq_thinning_unconstrain_cutting, vfk0_imq)
            ks_q_centkgm_thinning = ksd(x_q_centkgm_thinning_unconstrain_cutting, grad_x_q_centkgm_thinning_unconstrain_cutting, vfk0_centkgm)

            ### Store
            res_ksd_q_imq_thinning[i, j] = ks_q_imq_thinning[-1]
            res_ksd_q_centkgm_thinning[i, j] = ks_q_centkgm_thinning[-1]

    ## Save
    ### KSD MALA
    np.save(f"{data_save_path}/res_ksd_q_imq_thinning.npy", res_ksd_q_imq_thinning)
    np.save(f"{data_save_path}/res_ksd_q_centkgm_thinning.npy", res_ksd_q_centkgm_thinning)
