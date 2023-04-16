from progress_bar import tqdm
import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.stats.correlation_tools import cov_nearest


def mala(log_pdf, grad_log_pdf, theta_start, eps, premat, iteration):
    """
    Sample from a target distribution using the Metropolis-adjusted Langevin
    algorithm.

    Args:
        log_pdf (func): handle to the log-density function of the target.
        grad_log_pdf (func): handle to the gradient function of the log target.
        theta_start (np.ndarray): vector of the starting values of the Markov chain.
        eps (float): step-size parameter.
        premat (np.ndarray): preconditioning matrix.
        iteration (int): number of MCMC iterations.

    Returns:
        x (np.ndarray) matrix of generated points.
        g (np.ndarray) matrix of gradients of the log target at X.
        p (np.ndarray) vector of log-density values of the target at X.
        a (np.ndarray) binary vector indicating whether a move is accepted.
    """

    # Initialise the chain
    d = len(theta_start)
    x = np.empty((iteration, d))
    g = np.empty((iteration, d))
    p = np.empty(iteration)
    a = np.zeros(iteration, dtype=bool)
    x[0] = theta_start
    g[0] = grad_log_pdf(theta_start)
    p[0] = log_pdf(theta_start)

    # For each MCMC iteration
    for i in tqdm(range(1, iteration)):
        # Langevin proposal
        hh = eps ** 2
        mx = x[i - 1] + hh / 2 * np.dot(premat, g[i - 1])
        s = hh * premat
        y = np.random.multivariate_normal(mx, s)

        # Log acceptance probability
        py = log_pdf(y)
        gy = grad_log_pdf(y)
        my = y + hh / 2 * np.dot(premat, gy)
        qx = multivariate_normal.logpdf(x[i - 1], my, s)
        qy = multivariate_normal.logpdf(y, mx, s)
        acc_pr = (py + qx) - (p[i - 1] + qy)

        # Accept with probability acc_pr
        if acc_pr >= 0 or np.log(np.random.uniform()) < acc_pr:
            x[i] = y
            g[i] = gy
            p[i] = py
            a[i] = True
        else:
            x[i] = x[i - 1]
            g[i] = g[i - 1]
            p[i] = p[i - 1]

    return (x, g, p, a)

def mala_adapt(log_pdf, grad_log_pdf, theta_start, eps0, premat0, alpha, epoch):
    """
    Sample from a target distribution using an adaptive version of the
    Metropolis-adjusted Langevin algorithm.

    Args:
        log_pdf (func): handle to the log-density function of the target.
        grad_log_pdf (func): handle to the gradient function of the log target.
        theta_start (np.ndarray): vector of the starting values of the Markov chain.
        eps0 (float): initial step-size parameter.
        premat0 (np.ndarray): initial preconditioning matrix.
        alpha (list[float]): adaptive schedule.
        epoch (list[int]): length of each tuning epoch.

    Returns:
        eps (float): tuned step-size.
        premat (np.ndarray): tuned preconditioning matrix.
        x (list[np.ndarray]): list of matrices of generated points.
        g (list[np.ndarray]): list of matrices of gradients of the log target at X.
        p (list[np.ndarray]): list of vectors of log-density values of the target at X.
        a (list[np.ndarray]): list of binary vectors indicating whether a move is accepted.
    """

    n_ep = len(epoch)
    x = n_ep * [None]
    g = n_ep * [None]
    p = n_ep * [None]
    a = n_ep * [None]

    # First epoch
    eps = eps0
    premat = premat0
    x[0], g[0], p[0], a[0] = mala(log_pdf, grad_log_pdf, theta_start, eps, premat, epoch[0])

    for i in range(1, n_ep):
        # Adapt preconditioning matrix
        premat = alpha[i] * premat + (1 - alpha[i]) * np.cov(x[i - 1].T)
        premat = cov_nearest(premat)

        # Tune step-size
        ar = np.mean(a[i - 1])
        eps = eps * np.exp(ar - 0.57)

        # Next epoch
        theta_start_new = x[i - 1][-1]
        x[i], g[i], p[i], a[i] = mala(log_pdf, grad_log_pdf, theta_start_new, eps, premat, epoch[i])

    return (eps, premat, x, g, p, a)