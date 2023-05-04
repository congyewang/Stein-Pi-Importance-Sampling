import os
import pandas as pd
import jinja2
from jinja2 import Environment, FileSystemLoader
from posteriordb import PosteriorDatabase
from stein_pi_thinning.util import flat, get_non_empty_subdirectories

jinja2.Environment()

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
df_plot = df_gs[df_gs["index"].isin(model_list)]

# Read template files
with open('temp_plot.tex') as file:
    template_content = file.read()

# Creating Jinja2 environment
env = Environment(loader=FileSystemLoader('.'), trim_blocks=True, lstrip_blocks=True)

# Loading template
template = env.get_template('temp_plot.tex')

# Defining the data
index_list = list(range(df_plot.shape[0]))

# Rendering templates
rendered_tex = template.render(index_list=index_list, df_plot=df_plot)

# Save LaTeX code
with open("posteriordb_plots.tex", "w") as f:
    f.write(rendered_tex)
