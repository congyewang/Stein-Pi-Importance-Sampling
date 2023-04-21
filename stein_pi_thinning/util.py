"""
Small helper functions.
"""


import numpy as np

from jax import numpy as jnp
from jax import jit
from jax.scipy.stats import norm, multivariate_normal

from mcmclib.metropolis import mala_adapt
from stein_pi_thinning.target import PiTargetAuto


generator = np.random.default_rng(seed=1234)


def flat(nested_list):
    """
    Expand nested list
    """
    res = []
    for i in nested_list:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def vkgm(x, sx, linv, s):
    """
    Vectorised KGM kernel function.

    Args:
        x (np.ndarry): n x d array where each row is a d-dimensional sample point.
        sx (np.ndarry): n x d array where each row is a gradient of the log target.
        linv (np.ndarry): n x n array semi-positive definition matrix.
        s (int): Control Parameter.

    Returns:
        (np.ndarry): KGM kernel results.
    """
    s = int(s)

    c0_imq = np.diag((1 + x@linv@x.T)**(s-1))
    c1_imq = ((s - 1)*(1 + np.diag(x@linv@x.T))**(s-2)).reshape(-1,1) * (linv @ x.T).T
    c2_imq = np.diag((1 + x@linv@x.T)**(s-1) * (((s-1)**2 * x@linv@linv@x.T)/((1 + x@linv@x.T)**2) + np.trace(linv)))

    # Linear
    c0_lin = 1.0
    c1_lin = 0.0
    c2_lin = np.diag(((1 + x@linv@x.T)**(-1)) * ((-1) * (x@linv@linv@x.T) / (1 + x@linv@x.T) + np.trace(linv)))

    # KGM
    c0_kgm = c0_imq + c0_lin
    c1_kgm = c1_imq + c1_lin
    c2_kgm = c2_imq + c2_lin

    return c2_kgm + np.diag(2*c1_kgm@sx.T) + c0_kgm*np.diag(sx@sx.T)

def generate_dim_diff_pi(dim, kernel="imq", nits = 100_000):
    """
    Using Standard Normal Distribution to Generate Pi Target Distribution in Different Dimensions

    Args:
        dim (int): Dimension
        kernel (str, optional): Kernel Selection. Defaults to "imq".
        nits (int, optional): MCMC Iteration Times. Defaults to 100_000.

    Raises:
        ValueError: Only 'imq' or 'kgm' base kernel function.

    Returns:
        np.ndarray: The Samples of Pi Target Distribution.
    """
    mu = jnp.repeat(0.0, dim)
    sigma = jnp.eye(dim)

    if dim == 1:
        log_p = jit(lambda x: norm.logpdf(x, mu, np.square(sigma)))
    else:
        log_p = jit(lambda x: multivariate_normal.logpdf(x, mu, sigma))

    if kernel == "imq":
        linv = jnp.linalg.inv(sigma)
        imq_kernel = lambda x, y: (1 + (x - y)@linv@(x - y))**(-0.5)

        pi_target = PiTargetAuto(log_p=log_p, base_kernel=imq_kernel)

    elif kernel == "kgm":
        s = 3.0
        beta = 0.5
        x_map = mu
        linv = jnp.linalg.inv(sigma)

        centkgm_kernel = lambda x, y: (1 + (x-y)@linv@(x-y).T )**(-beta) +\
                    (1 + (x-x_map)@linv@(y-x_map).T)/( (1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2) )

        pi_target = PiTargetAuto(log_p=log_p, base_kernel=centkgm_kernel)
    else:
        raise ValueError("Only 'imq' or 'kgm'")

    log_q = jit(lambda x: pi_target.log_q(x))
    grad_log_q = jit(lambda x: pi_target.grad_log_q(x))

    # MALA
    # Parameters
    pre_nits = 1_000 # the number of preconditional iterations

    alpha = 10 * [1]
    epoch = 9 * [pre_nits] + [nits]

    _, _, x_q_epoch, _, _, _ = mala_adapt(log_q, grad_log_q, mu, 1, sigma, alpha, epoch)

    x_q = np.array(x_q_epoch[-1], dtype=np.float64)

    return x_q
