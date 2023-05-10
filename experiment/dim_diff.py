from jax import numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from stein_pi_is.util import generate_dim_diff_pi
from stein_pi_is.progress_bar import disable_progress_bar

disable_progress_bar()
plt.rcParams['text.usetex'] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
plt.rcParams["axes.formatter.use_mathtext"] = True


def main(kernel="kgm", nits=10_000_000, title=""):
    sns.kdeplot(norm.ppf(jnp.linspace(0, 1, nits)), label=f"$P$")
    for i in [1, 2, 10]:
        x_q = generate_dim_diff_pi(i, kernel=kernel, nits=nits)
        sns.kdeplot(x_q[:,0], label=f"$\Pi \quad(d={i})$")
    plt.xlim(-3, 3)
    plt.xlabel(f"$x_1$", fontsize=17)
    plt.legend(facecolor='white', edgecolor='black', fancybox=False)
    plt.title(title, fontsize=17)
    plt.show()


if __name__ == "__main__":
    main(kernel="imq", title=r"$\mathrm{Langevin-Stein}$ $\mathrm{Kernel}$")
    main(kernel="kgm", title=r"$\mathrm{KGM3-Stein}$ $\mathrm{Kernel}$")
