"""
Small helper functions.
"""


import numpy as np


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
