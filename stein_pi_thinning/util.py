"""
Small helper functions.
"""


import os
import numpy as np

from jax import numpy as jnp
from jax import jit
from jax.scipy.stats import norm, multivariate_normal

import quadprog
import cvxopt

from stein_thinning.stein import kmat

from stein_pi_thinning.mcmc import mala_adapt
from stein_pi_thinning.target import PiTargetAuto, PiTargetIMQ, PiTargetCentKGM


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

def vcentkgm(x, y, sx, sy, linv, s, x_map):
    vkappa = np.diag((1 + np.diag((x-y)@linv@(x-y).T ))**(-0.5) +\
                (1 + (x-x_map)@linv@(y-x_map).T)/( (1+np.diag((x-x_map)@linv@(x-x_map).T))**(s/2) * (1+np.diag((y-x_map)@linv@(y-x_map).T))**(s/2) ))

    vdxkappa = (linv @ (x-y).T * -(1 + np.diag((x-y)@linv@(x-y).T) )**(-1.5)) +\
                (linv@(y-x_map).T - np.matmul(linv@(x-x_map).T, np.diag(np.diag(s*(1+(x-x_map)@linv@(y-x_map).T) * np.diag(1+(x-x_map)@linv@(x-x_map).T)**(-1)))))/(((1+np.diag((x-x_map)@linv@(x-x_map).T))**(s/2) * (1+np.diag((y-x_map)@linv@(y-x_map).T))**(s/2)))

    vdykappa = (linv @ (x-y).T * ((1 + np.diag((x-y)@linv@(x-y).T) )**(-1.5))) +\
                (linv@(x-x_map).T - np.matmul(linv@(y-x_map).T, np.diag(np.diag(s*(1+(x-x_map)@linv@(y-x_map).T)*(1+np.diag((y-x_map)@linv@(y-x_map).T))**(-1)))))/(((1+np.diag((x-x_map)@linv@(x-x_map).T))**(s/2) * (1+np.diag((y-x_map)@linv@(y-x_map).T))**(s/2)))

    vdxdykappa = (-3*(1+np.diag((x-y)@linv@(x-y).T))**(-2.5)) * np.diag((x-y)@linv@linv@(x-y).T) + np.trace(linv)*((1+np.diag((x-y)@linv@(x-y).T))**(-1.5))\
                + (
                    np.trace(linv)\
                    - s * ((1 + np.diag((x-x_map)@linv@(x-x_map).T))**(-1)) * np.diag((x-x_map)@linv@linv@(x-x_map).T)\
                    - s * ((1 + np.diag((y-x_map)@linv@(y-x_map).T))**(-1)) * np.diag((y-x_map)@linv@linv@(y-x_map).T)\
                    + s**2 * (1 + np.diag((x-x_map)@linv@(y-x_map).T)) * ((1 + np.diag((x-x_map)@linv@(x-x_map).T))**(-1)) * (1 + np.diag((y-x_map)@linv@(y-x_map).T))**(-1)\
                    * np.diag((x-x_map)@linv@linv@(y-x_map).T)
                    )\
                / ((1 + np.diag((x-x_map)@linv@(x-x_map).T))**(s/2) * (1 + np.diag((y-x_map)@linv@(y-x_map).T))**(s/2))

    vc = (1 + np.diag((x-x_map)@linv@(x-x_map).T))**((s-1)/2)\
                * (1 + np.diag((y-x_map)@linv@(y-x_map).T))**((s-1)/2)\
                * vkappa

    vdxc = ((1 + np.diag((x-x_map)@linv@(x-x_map).T))**((s-1)/2))\
                * ((1 + np.diag((y-x_map)@linv@(y-x_map).T))**((s-1)/2))\
                * (
                    ((s-1) * linv@(x-x_map).T * vkappa) / np.diag(1 + (x-x_map)@linv@(x-x_map).T)\
                    + vdxkappa
                )

    vdyc = ((1 + np.diag((x-x_map)@linv@(x-x_map).T))**((s-1)/2))\
                * (1 + np.diag((y-x_map)@linv@(y-x_map).T))**((s-1)/2)\
                * (
                    ((s-1) * linv@(y-x_map).T) * vkappa / (1 + np.diag((y-x_map)@linv@(y-x_map).T))\
                    + vdykappa
                )

    vdxdyc = np.diag((1+np.diag((x-x_map)@linv@(x-x_map).T))**((s-1)/2)\
                * (1+np.diag((y-x_map)@linv@(y-x_map).T))**((s-1)/2)\
                * (
                    (s-1)**2 * vkappa * np.diag((x-x_map)@linv@linv@(y-x_map).T / ((1+(x-x_map)@linv@(x-x_map).T)*(1+(y-x_map)@linv@(y-x_map).T)))\
                    + (s-1)*(y-x_map)@linv@vdxkappa / (1+(y-x_map)@linv@(y-x_map).T)\
                    + (s-1)*(x-x_map)@linv@vdykappa / (1+(x-x_map)@linv@(x-x_map).T)\
                    + vdxdykappa
                ))

    vkp = vdxdyc + np.diag(vdxc.T@sy.T) + np.diag(vdyc.T@sx.T) + vc * np.diag(sx@sy.T)

    return vkp

def quadprog_solve_qp(P, q, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def comp_wksd(x, s, vfk0, solver_type="cvxopt", output_info=False):
    """
    Computing Weighted Kernel Stein Discrepancy

    Args:
    x    - n x d array where each row is a d-dimensional sample point.
    s    - n x d array where each row is a gradient of the log target.
    vfk0 - vectorised Stein kernel function.

    Returns:
    float containing the weighted Kernel Stein Discrepancy.
    """
    # remove duplicates
    x, idx = np.unique(x, axis=0, return_index=True)
    s = s[idx]
    # dimensions
    n = x.shape[0]

    # Stein kernel matrix
    K = kmat(x=x, s=s, vfk0=vfk0)

    if isPD(K):
        P = K
    else:
        P = nearestPD(K)
    q = np.zeros(n)
    G = np.diag([-1.0]*n)
    h = np.ones(n)
    A = np.ones((1,n))
    b = 1.0

    if solver_type == "cvxopt":
        cvxopt.solvers.options['show_progress'] = output_info
        w = cvxopt_solve_qp(P, q, G, h, A, b)
    elif solver_type == "quadprog":
        w = quadprog_solve_qp(P, q, G, h, A, b)
    else:
        raise ValueError("Only 'cvxopt' or 'quadprog'")

    wksd = np.sqrt(w @ K @ w)
    return wksd

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
        x_map = mu
        linv = jnp.linalg.inv(sigma)

        centkgm_kernel = lambda x, y: (1 + (x-x_map)@linv@(x-x_map))**((s-1)/2) * (1 + (y-x_map)@linv@(y-x_map))**((s-1)/2)\
            * ((1 + (x-y)@linv@(x-y).T )**(-0.5) +\
            (1 + (x-x_map)@linv@(y-x_map).T)/( (1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2) ))

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

def generate_dim_diff_pi_manual(dim, kernel="imq", nits = 100_000):
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
    if dim == 1:
        log_p = lambda x: -0.5*np.log(2*np.pi) - x**2/2
        grad_log_p = lambda x: -x.reshape(-1,1)
    else:
        log_p = lambda x: (-dim/2) * np.log(2*np.pi) - 0.5*x@x.T
        grad_log_p = lambda x: -x
    hess_log_p = lambda x: -np.eye(dim)

    linv = np.eye(dim)

    if kernel == "imq":
        linv = np.eye(dim)

        pi_target = PiTargetIMQ(log_p=log_p, grad_log_p=grad_log_p, hess_log_p=hess_log_p, linv=linv)

    elif kernel == "kgm":
        pi_target = PiTargetCentKGM(log_p=log_p, grad_log_p=grad_log_p, hess_log_p=hess_log_p, linv=linv, s=3.0, x_map=np.repeat(0.0, dim))
    else:
        raise ValueError("Only 'imq' or 'kgm'")

    log_q = pi_target.log_q
    grad_log_q = pi_target.grad_log_q

    # MALA
    # Parameters
    pre_nits = 1_000 # the number of preconditional iterations

    alpha = 10 * [1]
    epoch = 9 * [pre_nits] + [nits]

    _, _, x_q_epoch, _, _, _ = mala_adapt(log_q, grad_log_q, np.repeat(0.0, dim), 1, np.eye(dim), alpha, epoch)

    x_q = np.array(x_q_epoch[-1], dtype=np.float64)

    return x_q

def mkdir(path):
    """
    Determine if a folder exists, if not then create a new folder.

    Args:
        path (str): The path of the folder.
    """
    path = path.strip() # remove first space
    isExists = os.path.exists(path) # determine if a path exists
    if not isExists:
        os.makedirs(path)
