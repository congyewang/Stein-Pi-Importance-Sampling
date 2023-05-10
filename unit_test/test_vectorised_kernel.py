from stein_pi_is.target import PiTargetKGM
from stein_pi_is.util import vkgm, vcentkgm

import numpy as np
from scipy.stats import wishart

from jax import numpy as jnp
from jax import jacfwd, jit

precision_shreshold = 1e-4


def test_vkgm():
    dim = np.random.randint(low=2, high=30)

    log_p = lambda x: -np.dot(x, x)/2
    grad_log_p = lambda x: -x
    hess_log_p = lambda x: -1

    linv = wishart.rvs(dim+10, np.eye(dim), 1)
    s = np.random.randint(low=3, high=10)

    x = np.random.normal(size=(10, dim))
    sx = grad_log_p(x)

    std_check = PiTargetKGM(log_p, grad_log_p, hess_log_p, linv, s)
    res_check = np.array([std_check.stein_kernel(i) for i in x])

    assert (np.abs(res_check - vkgm(x, sx, linv, s)) < precision_shreshold).all

def test_vcentkgm():
    dim = np.random.randint(low=2, high=10)

    linv = jnp.array(wishart.rvs(dim+10, np.eye(dim), 1))
    s = np.random.randint(low=3, high=10)
    x_map = jnp.repeat(0.0, dim)

    num = np.random.randint(low=2, high=50)

    xv = jnp.array(np.random.normal(size=(num, dim)))
    yv = jnp.array(np.random.normal(size=(num, dim)))
    sxv = jnp.array(np.random.normal(size=(num, dim)))
    syv = jnp.array(np.random.normal(size=(num, dim)))

    for i in range(num):
        x = xv[i]
        y = yv[i]
        sx = sxv[i]
        sy = syv[i]

        base_kernel = jit(lambda x, y: (1 + (x-x_map)@linv@(x-x_map))**((s-1)/2) * (1 + (y-x_map)@linv@(y-x_map))**((s-1)/2)\
            * ((1 + (x-y)@linv@(x-y).T )**(-0.5) +\
                    (1 + (x-x_map)@linv@(y-x_map).T)/( (1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2) )))
        dx_k = jit(jacfwd(base_kernel, argnums=0))
        dy_k = jit(jacfwd(base_kernel, argnums=1))
        dxdy_k = jit(jacfwd(dy_k, argnums=0))
        kp_auto = jit(lambda x, y, sx, sy: jnp.trace(dxdy_k(x, y))\
                                + dx_k(x, y) @ sy\
                                + dy_k(x, y) @ sx\
                                + base_kernel(x, y) * sx @ sy)

        assert jnp.abs((kp_auto(x, y, sx, sy) - vcentkgm(xv, yv, sxv, syv, linv, s, x_map)[i])/kp_auto(x, y, sx, sy)) < precision_shreshold
