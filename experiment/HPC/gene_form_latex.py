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


for i_index, i in enumerate(df_form["index"].tolist()):
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

    with open("posteriordb_form.tex", "a+") as f:
        f.write(f"{i} & {int(df_form[df_form['index'] == i]['dimensions'])} & {mean_ksd_p_imq_origin[-1]} & {mean_ksd_p_imq_weight[-1]} & {mean_ksd_q_imq_weight[-1]} & {mean_ksd_p_centkgm_origin[-1]} & {mean_ksd_p_centkgm_weight[-1]} & {mean_ksd_q_centkgm_weight[-1]} \\\\\n")
