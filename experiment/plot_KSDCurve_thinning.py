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


df_plot = df_gs[df_gs["index"].isin(model_list)]

mkdir(f"Pic/Thinning/")

for i in model_list:
    res_ksd_q_imq_thinning = np.load(f"Thin_Conver/{i}/res_ksd_q_imq_thinning.npy")
    res_ksd_q_centkgm_thinning = np.load(f"Thin_Conver/{i}/res_ksd_q_centkgm_thinning.npy")

    iteration_list = np.array(range(1, res_ksd_q_centkgm_thinning.shape[1]+1))

    mean_ksd_q_imq_thinning = np.mean(res_ksd_q_imq_thinning, axis=0)
    std_error_ksd_q_imq_thinning = iteration_list ** (-1/2) * np.std(res_ksd_q_imq_thinning, axis=0)
    lower_bound_ksd_q_imq_thinning = mean_ksd_q_imq_thinning - std_error_ksd_q_imq_thinning
    upper_bound_ksd_q_imq_thinning = mean_ksd_q_imq_thinning + std_error_ksd_q_imq_thinning

    mean_ksd_q_centkgm_thinning = np.mean(res_ksd_q_centkgm_thinning, axis=0)
    std_error_ksd_q_centkgm_thinning = iteration_list ** (-1/2) * np.std(mean_ksd_q_centkgm_thinning, axis=0)
    lower_bound_ksd_q_centkgm_thinning = mean_ksd_q_centkgm_thinning - std_error_ksd_q_centkgm_thinning
    upper_bound_ksd_q_centkgm_thinning = mean_ksd_q_centkgm_thinning + std_error_ksd_q_centkgm_thinning

    plt.cla()

    plt.plot(iteration_list, mean_ksd_q_imq_thinning, color="#7e2f8e")
    plt.fill_between(iteration_list, lower_bound_ksd_q_imq_thinning, upper_bound_ksd_q_imq_thinning, color="#7e2f8e", alpha=0.3)

    plt.plot(iteration_list, mean_ksd_q_centkgm_thinning, color="#4dbeee")
    plt.fill_between(iteration_list, lower_bound_ksd_q_centkgm_thinning, upper_bound_ksd_q_centkgm_thinning, color="#4dbeee", alpha=0.3)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$n$', fontsize=17)
    plt.ylabel(r'$\mathbb{E}[\mathrm{KSD}]$', fontsize=17)

    plt.xlim((np.min(iteration_list), np.max(iteration_list)))

    plt.title("$\mathrm{{{0}}} ({1}D)$".format(
        i.replace("_", "\_"),
        int(df_plot[df_plot['index'] == i]['dimensions'])), fontsize=17, fontname="cmr10")
    plt.savefig(f"Pic/Thinning/{i}_KSDCurve_thinning.pdf")
