import os
import numpy as np
import pandas as pd
from posteriordb import PosteriorDatabase
from stein_pi_thinning.util import flat, get_non_empty_subdirectories


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

df_form = df_gs[df_gs["index"].isin(model_list)]

for i in df_form["index"].tolist():
    res_ksd_p_imq_origin = np.load(f"Data/{i}/res_ksd_p_imq_origin.npy")
    res_ksd_p_centkgm_origin = np.load(f"Data/{i}/res_ksd_p_centkgm_origin.npy")

    res_ksd_p_imq_weight = np.load(f"Data/{i}/res_ksd_p_imq_weight.npy")
    res_ksd_p_centkgm_weight = np.load(f"Data/{i}/res_ksd_p_centkgm_weight.npy")

    res_ksd_q_imq_weight = np.load(f"Data/{i}/res_ksd_q_imq_weight.npy")
    res_ksd_q_centkgm_weight = np.load(f"Data/{i}/res_ksd_q_centkgm_weight.npy")

    mean_ksd_p_imq_origin = np.mean(res_ksd_p_imq_origin, axis=0)
    mean_ksd_p_centkgm_origin = np.mean(res_ksd_p_centkgm_origin, axis=0)

    mean_ksd_p_imq_weight = np.mean(res_ksd_p_imq_weight, axis=0)
    mean_ksd_q_imq_weight = np.mean(res_ksd_q_imq_weight, axis=0)

    mean_ksd_p_centkgm_weight = np.mean(res_ksd_p_centkgm_weight, axis=0)
    mean_ksd_q_centkgm_weight = np.mean(res_ksd_q_centkgm_weight, axis=0)

    f2_real = mean_ksd_p_imq_origin[-1]
    f3_real = mean_ksd_p_imq_weight[-1]
    f4_real = mean_ksd_q_imq_weight[-1]
    f5_real = mean_ksd_p_centkgm_origin[-1]
    f6_real = mean_ksd_p_centkgm_weight[-1]
    f7_real = mean_ksd_q_centkgm_weight[-1]

    f0 = i.replace('_', '\\_')
    f1 = int(df_form[df_form['index'] == i]['dimensions'])
    f2 = float("{:.3g}".format(f2_real))
    f3 = float("{:.3g}".format(f3_real))
    f4 = float("{:.3g}".format(f4_real))
    f5 = float("{:.3g}".format(f5_real))
    f6 = float("{:.3g}".format(f6_real))
    f7 = float("{:.3g}".format(f7_real))

    if f3_real < f4_real:
        fa = "\\textbf{{{0}}} & {1}".format(f3, f4)
    else:
        fa = "{0} & \\textbf{{{1}}}".format(f3, f4)

    if f6_real < f7_real:
        fb = "\\textbf{{{0}}} & {1}".format(f6, f7)
    else:
        fb = "{0} & \\textbf{{{1}}}".format(f6, f7)

    with open("posteriordb_form.tex", "a+") as f:
        f.write(f"{f0} & {f1} & {f2} & {fa} & {f5} & {fb} \\\\\n"
        )
