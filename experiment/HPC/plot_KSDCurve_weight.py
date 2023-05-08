import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from posteriordb import PosteriorDatabase
from stein_pi_thinning.util import flat, mkdir, get_non_empty_subdirectories

plt.rcParams['text.usetex'] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
plt.rcParams["axes.formatter.use_mathtext"] = True

# Load DataBase Locally
pdb_path = os.path.join("posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)

# Extract the Names of All Models
pos = my_pdb.posterior_names()

# Reordering Models in Ascending Dimensional Order
d = {}
n = 0
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
df_gs = df.loc[gs_models].reset_index(inplace=False)
df_gs.sort_values(by=['dimensions', 'index'], ascending=True, inplace=True)


model_list = get_non_empty_subdirectories('Data')
repeat_times = 10
iteration_list = [10, 30, 50, 100, 300, 500, 1_000, 3_000]


df_plot = df_gs[df_gs["index"].isin(model_list)]

mkdir(f"Pic")

for i in df_plot["index"].tolist():
    res_ksd_p_imq_origin = np.load(f"Data/{i}/res_ksd_p_imq_origin.npy")
    res_ksd_p_centkgm_origin = np.load(f"Data/{i}/res_ksd_p_centkgm_origin.npy")

    res_ksd_p_imq_weight = np.load(f"Data/{i}/res_ksd_p_imq_weight.npy")
    res_ksd_p_centkgm_weight = np.load(f"Data/{i}/res_ksd_p_centkgm_weight.npy")

    res_ksd_q_imq_weight = np.load(f"Data/{i}/res_ksd_q_imq_weight.npy")
    res_ksd_q_centkgm_weight = np.load(f"Data/{i}/res_ksd_q_centkgm_weight.npy")

    mean_ksd_p_imq_origin = np.mean(res_ksd_p_imq_origin, axis=0)
    std_error_ksd_p_imq_origin = repeat_times ** (-1/2) * np.std(res_ksd_p_imq_origin, axis=0)
    mean_ksd_p_centkgm_origin = np.mean(res_ksd_p_centkgm_origin, axis=0)
    std_error_ksd_p_centkgm_origin = repeat_times ** (-1/2) * np.std(mean_ksd_p_centkgm_origin, axis=0)

    mean_ksd_p_imq_weight = np.mean(res_ksd_p_imq_weight, axis=0)
    std_error_ksd_p_imq_weight = repeat_times ** (-1/2) * np.std(res_ksd_p_imq_weight, axis=0)
    mean_ksd_q_imq_weight = np.mean(res_ksd_q_imq_weight, axis=0)
    std_error_ksd_q_imq_weight = repeat_times ** (-1/2) * np.std(res_ksd_q_imq_weight, axis=0)

    mean_ksd_p_centkgm_weight = np.mean(res_ksd_p_centkgm_weight, axis=0)
    std_error_ksd_p_centkgm_weight = repeat_times ** (-1/2) * np.std(res_ksd_p_centkgm_weight, axis=0)
    mean_ksd_q_centkgm_weight = np.mean(res_ksd_q_centkgm_weight, axis=0)
    std_error_ksd_q_centkgm_weight = repeat_times ** (-1/2) * np.std(res_ksd_q_centkgm_weight, axis=0)

    plt.cla()

    plt.errorbar(iteration_list, mean_ksd_p_imq_origin, yerr=std_error_ksd_p_imq_origin, color="#7e2f8e", linestyle="dotted", capsize=4, label="$MALA$ (Langevin)")
    # plt.errorbar(iteration_list, mean_ksd_p_centkgm_origin, yerr=std_error_ksd_p_centkgm_origin, color="#4dbeee", linestyle="dotted", capsize=4, label="$MALA$ (KGM3)")
    plt.plot(iteration_list, mean_ksd_p_centkgm_origin, color="#4dbeee", linestyle="dotted", label="$MALA$ (KGM3)")

    plt.errorbar(iteration_list, mean_ksd_p_imq_weight, yerr=std_error_ksd_p_imq_weight, color="#7e2f8e", linestyle="-", capsize=4, label="$P$ (Langevin)")
    plt.errorbar(iteration_list, mean_ksd_q_imq_weight, yerr=std_error_ksd_q_imq_weight, color="#7e2f8e", linestyle="--", capsize=4, label="$\\Pi$ (Langevin)")

    plt.errorbar(iteration_list, mean_ksd_p_centkgm_weight, yerr=std_error_ksd_p_centkgm_weight, color="#4dbeee", linestyle="-", capsize=4, label="$P$ (KGM3)")
    plt.errorbar(iteration_list, mean_ksd_q_centkgm_weight, yerr=std_error_ksd_q_centkgm_weight, color="#4dbeee", linestyle="--", capsize=4, label="$\\Pi$ (KGM3)")

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$n$', fontsize=17)
    plt.ylabel(r'$\mathbb{E}[\mathrm{KSD}]$', fontsize=17)

    plt.xlim((np.min(iteration_list), np.max(iteration_list)))

    plt.title("$\mathrm{{{0}}} ({1}D)$".format(
        i.replace("_", "\_"),
        int(df_plot[df_plot['index'] == i]['dimensions'])), fontsize=17, fontname="cmr10")
    plt.savefig(f"Pic/{i}_KSDCurve_weight.pdf")
