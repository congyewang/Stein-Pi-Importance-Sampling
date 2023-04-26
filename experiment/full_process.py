from posteriordb import PosteriorDatabase
import os
import json
import bridgestan as bs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stein_thinning.stein import ksd
from stein_thinning.thinning import thin
from stein_thinning.kernel import make_imq, make_centkgm

from stein_pi_thinning.target import PiTargetIMQ, PiTargetCentKGM
from stein_pi_thinning.mcmc import mala_adapt
from stein_pi_thinning.util import flat, comp_wksd, mkdir
from stein_pi_thinning.progress_bar import disable_progress_bar, tqdm

import wasserstein

rng = np.random.default_rng(1234)
# disable_progress_bar()


def main(
        nits = 100_000,\
        model_name="garch-garch11",\
        dbpath="posteriordb/posterior_database",\
        s=3.0,\
        fixed_ratio=0.1,\
        iteration_list=[10, 20, 50, 100, 200, 500, 1000, 2000],\
        repeat_times=10,
        thinning_method="weight"
    ):
    # Model Preparation
    ## Load DataBase Locally
    pdb_path = os.path.join(dbpath)
    my_pdb = PosteriorDatabase(pdb_path)

    ## Load Dataset
    posterior = my_pdb.posterior("garch-garch11")
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
    linv = -hess_log_p(x_unconstrain_map)

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

    _, _, x_p_epoch, _, _, nacc_p = mala_adapt(log_p, grad_log_p, x_unconstrain_map, 1, np.eye(dim), alpha, epoch)
    # print('acc_p =', np.mean(nacc_p[-1]))
    assert np.mean(nacc_p[-1]) > 0.2, "Acceptance rate is too low"

    _, _, x_q_imq_epoch, _, _, nacc_q = mala_adapt(log_q_imq, grad_log_q_imq, x_unconstrain_map, 1, np.eye(dim), alpha, epoch)
    # print('acc_q =', np.mean(nacc_q[-1]))
    assert np.mean(nacc_q[-1]) > 0.2, "Acceptance rate is too low"

    _, _, x_q_centkgm_epoch, _, _, nacc_q = mala_adapt(log_q_centkgm, grad_log_q_centkgm, x_unconstrain_map, 1, np.eye(dim), alpha, epoch)
    # print('acc_q =', np.mean(nacc_q[-1]))
    assert np.mean(nacc_q[-1]) > 0.2, "Acceptance rate is too low"

    x_p_unconstrain = np.array(x_p_epoch[-1], dtype=np.float64)
    grad_x_p_unconstrain = np.array([grad_log_p(i) for i in x_p_unconstrain])

    x_q_imq_unconstrain = np.array(x_q_imq_epoch[-1], dtype=np.float64)
    grad_x_q_imq_unconstrain = np.array([grad_log_p(i) for i in x_q_imq_unconstrain])

    x_q_centkgm_unconstrain = np.array(x_q_centkgm_epoch[-1], dtype=np.float64)
    grad_x_q_centkgm_unconstrain = np.array([grad_log_p(i) for i in x_q_centkgm_unconstrain])

    ### Create Folder
    save_path = f"Pic/{model_name}"
    mkdir(save_path)
    ### Plotting P and Q
    for i in range(dim):
        plt.cla()
        sns.kdeplot(x_p_unconstrain[:,i].flatten(), label='$P$', color="black")
        sns.kdeplot(x_q_imq_unconstrain[:,i].flatten(), label='$\Pi$ (Langevin)', color="#7e2f8e")
        sns.kdeplot(x_q_centkgm_unconstrain[:,i].flatten(), label='$\Pi$ (KGM3)', color="#4dbeee")

        plt.legend()
        plt.savefig(f"{save_path}/{model_name}_param_{i}.pdf", dpi=600)

    # Thinning
    ## Kernel Selection
    vfk0_imq = make_imq(x_p_unconstrain, grad_x_p_unconstrain, pre=linv)
    vfk0_centkgm = make_centkgm(x_p_unconstrain, grad_x_p_unconstrain, x_map=x_unconstrain_map.reshape(1,-1), pre=linv, s=s)

    ## Store
    ### KSD IMQ
    res_ksd_p_imq = []
    res_ksd_q_imq = []

    res_ksd_p_imq_thinning = []
    res_ksd_q_imq_thinning = []
    ### KSD CentKGM
    res_ksd_p_centkgm = []
    res_ksd_q_centkgm = []

    res_ksd_p_centkgm_thinning = []
    res_ksd_q_centkgm_thinning = []

    ### Wasserstein Distance
    emd = wasserstein.EMD(n_iter_max=10_000_000)
    gs_weights = np.repeat(1/gs.shape[0], gs.shape[0])

    ## WS IMQ
    res_wass_p_imq_thinning = []
    res_wass_q_imq_thinning = []

    ## WS CentKGM
    res_wass_p_centkgm_thinning = []
    res_wass_q_centkgm_thinning = []

    ## Thinning Method Selection
    for i in tqdm(iteration_list):
        thinning_number = int(fixed_ratio * i)
        ### P
        x_p_unconstrain_cutting = x_p_unconstrain[0:i,:]
        grad_x_p_unconstrain_cutting = grad_x_p_unconstrain[0:i,:]
        ### IMQ
        x_q_imq_unconstrain_cutting = x_q_imq_unconstrain[0:i,:]
        grad_x_q_imq_unconstrain_cutting = grad_x_q_imq_unconstrain[0:i,:]
        ### KGM
        x_q_centkgm_unconstrain_cutting = x_q_centkgm_unconstrain[0:i,:]
        grad_x_q_centkgm_unconstrain_cutting = grad_x_q_centkgm_unconstrain[0:i,:]

        ### KSD Calculation
        #### P
        ks_p_imq = ksd(x_p_unconstrain_cutting, grad_x_p_unconstrain_cutting, vfk0_imq)
        ks_p_centkgm = ksd(x_p_unconstrain_cutting, grad_x_p_unconstrain_cutting, vfk0_centkgm)
        #### IMQ
        ks_q_imq = ksd(x_q_imq_unconstrain_cutting, grad_x_q_imq_unconstrain_cutting, vfk0_imq)
        #### KGM
        ks_q_centkgm = ksd(x_q_centkgm_unconstrain_cutting, grad_x_q_centkgm_unconstrain_cutting, vfk0_centkgm)

        #### Store P
        ##### IMQ
        res_ksd_p_imq.append(ks_p_imq[-1])
        ##### KGM
        res_ksd_p_centkgm.append(ks_p_centkgm[-1])
        #### Store Q
        ##### IMQ
        res_ksd_q_imq.append(ks_q_imq[-1])
        ##### KGM
        res_ksd_q_centkgm.append(ks_q_centkgm[-1])

        if thinning_method == "weight":
            sum_wksd_p_imq = 0.0
            sum_wksd_p_centkgm = 0.0
            sum_wksd_q_imq = 0.0
            sum_wksd_q_centkgm = 0.0

            for j in range(repeat_times):
                x_p_thinning_unconstrain_cutting = rng.choice(x_p_unconstrain_cutting, size=thinning_number, replace=False, shuffle=False)
                grad_x_p_thinning_unconstrain_cutting = np.array([grad_log_p(i) for i in x_p_thinning_unconstrain_cutting])

                x_q_imq_thinning_unconstrain_cutting = rng.choice(x_q_imq_unconstrain_cutting, size=thinning_number, replace=False, shuffle=False)
                grad_x_q_imq_thinning_unconstrain_cutting = np.array([grad_log_p(i) for i in x_q_imq_thinning_unconstrain_cutting])

                x_q_centkgm_thinning_unconstrain_cutting = rng.choice(x_q_centkgm_unconstrain_cutting, size=thinning_number, replace=False, shuffle=False)
                grad_x_q_centkgm_thinning_unconstrain_cutting = np.array([grad_log_p(i) for i in x_q_centkgm_thinning_unconstrain_cutting])

                sum_wksd_p_imq += comp_wksd(x_p_thinning_unconstrain_cutting, grad_x_p_thinning_unconstrain_cutting, vfk0_imq)
                sum_wksd_p_centkgm += comp_wksd(x_p_thinning_unconstrain_cutting, grad_x_p_thinning_unconstrain_cutting, vfk0_centkgm)
                sum_wksd_q_imq += comp_wksd(x_q_imq_thinning_unconstrain_cutting, grad_x_q_imq_thinning_unconstrain_cutting, vfk0_imq)
                sum_wksd_q_centkgm += comp_wksd(x_q_centkgm_thinning_unconstrain_cutting, grad_x_q_centkgm_thinning_unconstrain_cutting, vfk0_centkgm)

            #Store the average
            res_ksd_p_imq_thinning.append(sum_wksd_p_imq/repeat_times)
            res_ksd_p_centkgm_thinning.append(sum_wksd_p_centkgm/repeat_times)
            res_ksd_q_imq_thinning.append(sum_wksd_q_imq/repeat_times)
            res_ksd_q_centkgm_thinning.append(sum_wksd_q_centkgm/repeat_times)

        elif thinning_method == "thin":
            idx_p_imq = thin(x_p_unconstrain_cutting,\
                        grad_x_p_unconstrain_cutting,\
                        thinning_number,\
                        pre=linv,\
                        stnd=False,\
                        kern='imq'
                    )
            idx_q_imq = thin(x_q_imq_unconstrain_cutting,\
                        grad_x_q_imq_unconstrain_cutting,\
                        thinning_number,\
                        pre=linv,\
                        stnd=False,\
                        kern='imq'
                    )
            idx_p_centkgm = thin(x_p_unconstrain_cutting,\
                            grad_x_p_unconstrain_cutting,\
                            thinning_number,\
                            pre=linv,\
                            stnd=False,
                            kern='centkgm',
                            xmp=x_unconstrain_map
                        )
            idx_q_centkgm = thin(x_q_centkgm_unconstrain_cutting,\
                            grad_x_q_centkgm_unconstrain_cutting,\
                            thinning_number,\
                            pre=linv,\
                            stnd=False,
                            kern='centkgm',
                            xmp=x_unconstrain_map
                        )
            x_p_imq_thinning_unconstrain_cutting = x_p_unconstrain_cutting[idx_p_imq]
            grad_x_p_imq_thinning_unconstrain_cutting = grad_x_p_unconstrain_cutting[idx_p_imq]

            x_p_centkgm_thinning_unconstrain_cutting = x_p_unconstrain_cutting[idx_p_centkgm]
            grad_x_p_centkgm_thinning_unconstrain_cutting = grad_x_p_unconstrain_cutting[idx_p_centkgm]

            x_q_imq_thinning_unconstrain_cutting = x_q_imq_unconstrain_cutting[idx_q_imq]
            grad_x_q_imq_thinning_unconstrain_cutting = grad_x_q_imq_unconstrain_cutting[idx_q_imq]

            x_q_centkgm_thinning_unconstrain_cutting = x_q_centkgm_unconstrain_cutting[idx_q_centkgm]
            grad_x_q_centkgm_thinning_unconstrain_cutting = grad_x_q_centkgm_unconstrain_cutting[idx_q_centkgm]

            ks_p_imq_thinning = ksd(x_p_imq_thinning_unconstrain_cutting, grad_x_p_imq_thinning_unconstrain_cutting, vfk0_imq)
            ks_p_centkgm_thinning = ksd(x_p_centkgm_thinning_unconstrain_cutting, grad_x_p_centkgm_thinning_unconstrain_cutting, vfk0_centkgm)
            ks_q_imq_thinning = ksd(x_q_imq_thinning_unconstrain_cutting, grad_x_q_imq_thinning_unconstrain_cutting, vfk0_imq)
            ks_q_centkgm_thinning = ksd(x_q_centkgm_thinning_unconstrain_cutting, grad_x_q_centkgm_thinning_unconstrain_cutting, vfk0_centkgm)

            res_ksd_p_imq_thinning.append(ks_p_imq_thinning[-1])
            res_ksd_p_centkgm_thinning.append(ks_p_centkgm_thinning[-1])
            res_ksd_q_imq_thinning.append(ks_q_imq_thinning[-1])
            res_ksd_q_centkgm_thinning.append(ks_q_centkgm_thinning[-1])

            # Wasserstein Distance
            thinning_weights = np.repeat(1/thinning_number, thinning_number)

            wass_p_imq_thinning = emd(thinning_weights, x_p_imq_thinning_unconstrain_cutting, gs_weights, gs)
            wass_p_centkgm_thinning = emd(thinning_weights, x_p_centkgm_thinning_unconstrain_cutting, gs_weights, gs)
            wass_q_imq_thinning = emd(thinning_weights, x_q_imq_thinning_unconstrain_cutting, gs_weights, gs)
            wass_q_centkgm_thinning = emd(thinning_weights, x_q_centkgm_thinning_unconstrain_cutting, gs_weights, gs)

            res_wass_p_imq_thinning.append(wass_p_imq_thinning)
            res_wass_p_centkgm_thinning.append(wass_p_centkgm_thinning)
            res_wass_q_imq_thinning.append(wass_q_imq_thinning)
            res_wass_q_centkgm_thinning.append(wass_q_centkgm_thinning)

    ## Plot KSD Curve
    plt.cla()

    plt.loglog(iteration_list, res_ksd_p_imq_thinning, color="#7e2f8e", linestyle="-", label="P (Langevin)")
    plt.loglog(iteration_list, res_ksd_q_imq_thinning, color="#7e2f8e", linestyle="--", label="$\Pi$ (Langevin)")

    plt.loglog(iteration_list, res_ksd_p_centkgm_thinning, color="#4dbeee", linestyle="-", label="P (KGM3)")
    plt.loglog(iteration_list, res_ksd_q_centkgm_thinning, color="#4dbeee", linestyle="--", label="$\Pi$ (KGM3)")

    plt.xlabel('$n$')
    plt.ylabel(r'$\bf{E}$[$KSD$]')
    plt.legend()
    plt.savefig(f"{save_path}/{model_name}_KSDCurve_{thinning_method}.pdf", dpi=600)

    if thinning_method == "thin":
        ## Plot Wasserstein Distance Curve
        plt.cla()

        plt.loglog(iteration_list, res_wass_p_imq_thinning, color="#7e2f8e", linestyle="-", label="P (Langevin)")
        plt.loglog(iteration_list, res_wass_q_imq_thinning, color="#7e2f8e", linestyle="--", label="$\Pi$ (Langevin)")

        plt.loglog(iteration_list, res_wass_p_centkgm_thinning, color="#4dbeee", linestyle="-", label="P (KGM3)")
        plt.loglog(iteration_list, res_wass_q_centkgm_thinning, color="#4dbeee", linestyle="--", label="$\Pi$ (KGM3)")

        plt.xlabel('$n$')
        plt.ylabel(r"$\bf{E}$[$WD$]")
        plt.legend()
        plt.savefig(f"{save_path}/{model_name}_WDCurve_{thinning_method}.pdf", dpi=600)


if __name__ == "__main__":
    main(thinning_method="thin")
