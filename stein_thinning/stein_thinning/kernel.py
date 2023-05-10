"""Kernel functions."""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.spatial.distance import pdist
from stein_thinning.util import isfloat

from jax import jit, jacfwd, vmap
from jax import numpy as jnp


def vfk0_imq(a, b, sa, sb, linv):
    amb = a.T - b.T
    qf = 1 + np.sum(np.dot(linv, amb) * amb, axis=0)
    t1 = -3 * np.sum(np.dot(np.dot(linv, linv), amb) * amb, axis=0) / (qf ** 2.5)
    t2 = (np.trace(linv) + np.sum(np.dot(linv, sa.T - sb.T) * amb, axis=0)) / (qf ** 1.5)
    t3 = np.sum(sa.T * sb.T, axis=0) / (qf ** 0.5)
    return t1 + t2 + t3

def vfk0_kgm(x, y, sx, sy, linv, s):
    base_kernel = lambda x, y: (1 + x@linv@x)**((s-1)/2) *\
        (1 + y@linv@y)**((s-1)/2) *\
        (1 + (x-y)@linv@(x-y))**(-0.5) +\
        (1 + x@linv@y )/( jnp.sqrt(1+x@linv@x) * jnp.sqrt(1+y@linv@y) )
    dx_k = jit(jacfwd(base_kernel, argnums=0))
    dy_k = jit(jacfwd(base_kernel, argnums=1))
    dxdy_k = jit(jacfwd(dy_k, argnums=0))
    kp = jit(lambda x, y, sx, sy: jnp.trace(dxdy_k(x, y))\
                    + dx_k(x, y) @ sy\
                    + dy_k(x, y) @ sx\
                    + base_kernel(x, y) * sx @ sy)
    return jit(vmap(kp))(x, y, sx, sy)

def vfk0_centkgm(x, y, sx, sy, linv, s, x_map):
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

def make_precon(smp, scr, pre='id'):
    # Sample size and dimension
    sz, dm = smp.shape

    # Squared pairwise median
    def med2(m):
        if sz > m:
            sub = smp[np.linspace(0, sz - 1, m, dtype=int)]
        else:
            sub = smp
        return np.median(pdist(sub)) ** 2

    # Select preconditioner
    m = 1000
    if type(pre) == str:
        if pre == 'id':
            linv = np.identity(dm)
        elif pre == 'med':
            m2 = med2(m)
            if m2 == 0:
                raise Exception('Too few unique samples in smp.')
            linv = inv(m2 * np.identity(dm))
        elif pre == 'sclmed':
            m2 = med2(m)
            if m2 == 0:
                raise Exception('Too few unique samples in smp.')
            linv = inv(m2 / np.log(np.minimum(m, sz)) * np.identity(dm))
        elif pre == 'smpcov':
            c = np.cov(smp, rowvar=False)
            if not all(eig(c)[0] > 0):
                raise Exception('Too few unique samples in smp.')
            linv = inv(c)
        elif isfloat(pre):
            linv = inv(float(pre) * np.identity(dm))
        else:
            raise ValueError('Incorrect preconditioner string.')
    elif type(pre) == np.ndarray and pre.shape == (dm, dm):
        if not all(eig(pre)[0] > 0):
            raise Exception('Preconditioner is not positive definite.')
        linv = pre
    else:
        raise ValueError('Incorrect preconditioner type.')
    return linv

def make_imq(smp, scr, pre='id'):
    linv = make_precon(smp, scr, pre)
    def vfk0(a, b, sa, sb):
        return vfk0_imq(a, b, sa, sb, linv)
    return vfk0

def make_kgm(smp, scr, pre='id', s=3):
    linv = make_precon(smp, scr, pre)
    def vfk0(a, b, sa, sb):
        return vfk0_kgm(a, b, sa, sb, linv, s)
    return vfk0

def make_centkgm(smp, scr, x_map, pre='id', s=3):
    linv = make_precon(smp, scr, pre)
    def vfk0(a, b, sa, sb):
        return vfk0_centkgm(a, b, sa, sb, linv, s, x_map)
    return vfk0
