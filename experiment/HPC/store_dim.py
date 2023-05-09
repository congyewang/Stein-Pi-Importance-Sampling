from stein_pi_thinning.util import generate_dim_diff_pi
from stein_pi_thinning.progress_bar import disable_progress_bar
from jax import numpy as jnp

disable_progress_bar()


def main(kernel="kgm", nits=1_000_000):
    for i in [1, 2, 10]:
        x_q = generate_dim_diff_pi(i, kernel=kernel, nits=nits)
        jnp.save(f"dim_diff_pi_{kernel}_{i}.npy", x_q)


if __name__ == "__main__":
    main(kernel="kgm")
    main(kernel="imq")
