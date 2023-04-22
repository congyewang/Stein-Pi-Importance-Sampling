from jax import numpy as jnp
from jax import jacfwd
from scipy.stats import wishart
import numpy as np

generator = np.random.default_rng(seed=1234)
precision_shreshold = 1e-5

dim = generator.integers(2, 10)
beta = 0.5
s = 3.0

x_map = jnp.array(generator.normal(size=dim))
linv = jnp.linalg.inv(wishart.rvs(dim + 10, jnp.eye(dim)))

x = jnp.array(generator.normal(size=dim))
y = jnp.array(generator.normal(size=dim))


class CentralKGM:
    def kappa(self, x, y):
        return (1 + (x-y)@linv@(x-y).T )**(-beta) +\
            (1 + (x-x_map)@linv@(y-x_map).T)/( (1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2) )

    def dxkappa(self, x, y):
        return -2*beta*(1 + (x-y)@linv@(x-y).T )**(-beta-1) * linv @ (x-y).T +\
            (linv@(y-x_map) - s*(1 + (x-x_map)@linv@(y-x_map))*linv@(x-x_map).T*(1+(x-x_map)@linv@(x-x_map).T)**(-1))/((1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2))

    def dykappa(self, x, y):
        return 2*beta*(1 + (x-y)@linv@(x-y).T )**(-beta-1) * linv @ (x-y).T +\
            (linv@(x-x_map) - s*(1 + (x-x_map)@linv@(y-x_map))*linv@(y-x_map).T*(1+(y-x_map)@linv@(y-x_map).T)**(-1))/((1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2))

    def dxdykappa(self, x, y):
        return -4*beta*(beta+1)*(1+(x-y)@linv@(x-y).T)**(-beta-2) * (x-y)@linv@linv@(x-y).T + 2*beta*jnp.trace(linv)*(1+(x-y)@linv@(x-y).T)**(-beta-1)+\
            (
                jnp.trace(linv)\
                - s*(1 + (x-x_map)@linv@(x-x_map).T)**(-1) * (x-x_map)@linv@linv@(x-x_map).T\
                - s*(1 + (y-x_map)@linv@(y-x_map).T)**(-1) * (y-x_map)@linv@(y-x_map).T\
                + s**2*(1 + (x-x_map)@linv@(y-x_map))*(1 + (x-x_map)@linv@(x-x_map))**(-1) * (1 + (y-x_map)@linv@(y-x_map))**(-1)\
                * ((x-x_map)@linv@linv@(y-x_map)) )/( (1 + (x-x_map)@linv@(x-x_map))**(s/2) * (1 + (y-x_map)@linv@(y-x_map))**(s/2)
                                        )

    def c(self, x, y):
        return (1 + (x-x_map)@linv@(x-x_map))**((s-1)/2) * (1 + (y-x_map)@linv@(y-x_map))**((s-1)/2) * self.kappa(x, y)

    def c0(self, x):
        return 1 + (1 + (x-x_map)@linv@(x-x_map.T))**(s-1)

    def c1(self, x):
        return (s-1)*(1 + (x-x_map)@linv@(x-x_map.T))**(s-2) * linv@(x-x_map).T

    def c2(self, x):
        return ( ((s-1)**2*(1 + (x-x_map)@linv@(x-x_map).T)**(s-1) - 1)*(x-x_map)@linv@linv@(x-x_map).T )/( (1 + (x-x_map)@linv@(x-x_map).T)**2 ) + (jnp.trace(linv)*( 1 + 2*beta *(1 + (x-x_map)@linv@(x-x_map).T)**s ) )/( (1 + (x-x_map)@linv@(x-x_map).T) )

    def grad_c0(self, x):
        return 2*(s-1)*(1 + (x-x_map)@linv@(x-x_map).T)**(s-2) * linv@(x-x_map).T

    def grad_c1(self, x):
        return 2*(s-1)*(s-2)*(1 + (x-x_map)@linv@(x-x_map).T)**(s-3) * jnp.outer(linv@(x-x_map).T, linv@(x-x_map).T)\
            + (s-1)*(1 + (x-x_map)@linv@(x-x_map))**(s-2) * linv

    def grad_c2(self, x):
        return 2*(s-1)**2*(s-3)*(1 + (x-x_map)@linv@(x-x_map).T)**(s-4) * ((x-x_map)@linv@linv@(x-x_map).T) * linv@(x-x_map)\
            + 2 * (s-1)**2 * (1 + (x-x_map)@linv@(x-x_map).T)**(s-3)* linv@linv@(x-x_map)\
            + 4 * beta * jnp.trace(linv) * (s-1) * (1 + (x-x_map)@linv@(x-x_map).T)**(s-2) * linv@(x-x_map)\
            - 2 * (1 + (x-x_map)@linv@(x-x_map).T)**(-2) * ( linv@linv@(x-x_map).T + jnp.trace(linv)* linv@(x-x_map) )\
            + 4 * (1 + (x-x_map)@linv@(x-x_map).T)**(-3) * ((x-x_map)@linv@linv@(x-x_map).T) * linv@(x-x_map)

class TestCentralKGM(CentralKGM):
    # Test the Gradient of kappa
    def test_dxkappa(self):
        assert jnp.abs(jnp.mean(jacfwd(self.kappa, argnums=0)(x, y) - self.dxkappa(x, y))) < precision_shreshold

    def test_dykappa(self):
        assert jnp.abs(jnp.mean(jacfwd(self.kappa, argnums=1)(x, y) - self.dykappa(x, y))) < precision_shreshold

    def test_dxdykappa(self):
        assert jnp.trace(jacfwd(jacfwd(self.kappa, argnums=1), argnums=0)(x, y)) - self.dxdykappa(x, y)  < precision_shreshold


    def test_c1(self):
        assert jnp.abs(jnp.mean(jacfwd(self.c, argnums=0)(x, x) - self.c1(x))) < precision_shreshold

    def test_c2(self):
        assert jnp.trace(jacfwd(jacfwd(self.c, argnums=1), argnums=0)(x, x)) - self.c2(x) < precision_shreshold

    # Test the Gradient of c0, c1, and c2
    def test_grad_c0(self):
        assert jnp.abs(jnp.mean((jacfwd(self.c0)(x) - self.grad_c0(x)) / self.grad_c0(x))) < precision_shreshold
    
    def test_grad_c1(self):
        assert jnp.abs(jnp.mean((jacfwd(self.c1)(x) - self.grad_c1(x)) / self.grad_c1(x))) < precision_shreshold

    def test_grad_c2(self):
        assert jnp.abs(jnp.mean((jacfwd(self.c2)(x) - self.grad_c2(x)) / self.grad_c2(x))) < precision_shreshold
