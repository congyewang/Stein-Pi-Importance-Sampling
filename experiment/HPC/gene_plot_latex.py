from posteriordb import PosteriorDatabase
import os
import pandas as pd


# Load DataBase Locally
pdb_path = os.path.join("posteriordb/posterior_database")
my_pdb = PosteriorDatabase(pdb_path)


# Extract the Names of All Models
pos = my_pdb.posterior_names()


# Expand Nested List
def flat(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


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
gs_models = set(pos).difference(set(no_gs))


df_gs = df.loc[gs_models].reset_index(inplace=False)
df_gs.sort_values(by=['dimensions', 'index'], ascending=True, inplace=True)
df_gs


no_weight = ["diamonds-diamonds",
"earnings-logearn_height",
"earnings-logearn_height_male",
"earnings-logearn_interaction",
"mcycle_gp-accel_gp",
"sblrc-blr",
"earnings-log10earn_height",
"one_comp_mm_elim_abs-one_comp_mm_elim_abs",
"gp_pois_regr-gp_pois_regr",
"kilpisjarvi_mod-kilpisjarvi"]

df_plot = df_gs[~(df_gs["index"].isin(no_weight))]


for i in range(df_plot.shape[0]):
    if i == 0:
        with open("test.tex", "a+") as f:
            f.write("""
\\begin{figure}[htpb]
    \\centering
""")
    elif i % 3 == 0:
        with open("test.tex", "a+") as f:
            f.write("""
\\end{figure}

\\clearpage

\\begin{figure}[htpb]
    \\centering
""")

    temp = """
\\begin{{subfigure}}[htpb]{{0.49\\textwidth}}
    \\includegraphics[width = 1\\textwidth]{{figures/full_results/{0}/{0}_KSDCurve_weight.pdf}}
    \\caption{{Optimal Weighted KSD}}
    \\label{{fig: {0} wksd}}
\\end{{subfigure}}
\\begin{{subfigure}}[htpb]{{0.49\\textwidth}}
    \\includegraphics[width = 1\\textwidth]{{figures/full_results/{0}/{0}_KSDCurve_thinning.pdf}}
    \\caption{{Thinning KSD}}
    \\label{{fig: {0} thin}}
\\end{{subfigure}}
\\caption{{{1} ({2}D)}}
\\label{{fig: {0}}}
""".format(df_plot.iloc[i]["index"], df_plot.iloc[i]["index"].replace("_", "\\_"), df_plot.iloc[i]["dimensions"])
    with open("test.tex", "a+") as f:
        f.write(temp)

with open("test.tex", "a+") as f:
            f.write("\n\\end{figure}\n")
