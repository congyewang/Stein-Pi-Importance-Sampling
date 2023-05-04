import os
import pandas as pd
from jinja2 import Template
from posteriordb import PosteriorDatabase
from stein_pi_thinning.util import flat

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

# Read template files
with open('Templates/temp_plot.tex') as file:
    template_content = file.read()

# Creating a Jinja2 template object
template = Template(template_content)

# Defining the data
index_list = list(range(df_plot.shape[0]))

# Rendering templates
rendered_tex = template.render(index_list=index_list, df_plot=df_plot)

# Save LaTeX code
with open("test.tex", "w") as f:
    f.write(rendered_tex)
