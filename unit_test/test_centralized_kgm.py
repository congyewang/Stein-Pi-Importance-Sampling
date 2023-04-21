import numpy as np
from scipy.stats import wishart

from jax import numpy as jnp
from jax import jacrev, jacfwd


dim = np.random.randint(2, 30)
beta = 0.5
s = np.random.randint(2, 10)

x_map = jnp.array(np.random.normal(size=dim))
linv = jnp.array(wishart.rvs(dim + 10, scale=np.eye(dim), size=1))

x = jnp.array(np.random.normal(size=dim))
y = jnp.array(np.random.normal(size=dim))


class TestCentralKGM:
    def k(self, x, y):
        """
        Base Kernel Function

        Args:
            x (jnp.ndarray): Input
            y (jnp.ndarray): Input

        Returns:
            jnp.float32: Base Kernel Result
        """
        return (1 + (x-y)@linv@(x-y).T )**(-beta) +\
            (1 + (x-x_map)@linv@(y-x_map).T)/( (1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2) )

    def dxk(self, x, y):
        return -2*beta*(1 + (x-y)@linv@(x-y).T )**(-beta-1) * linv @ (x-y).T +\
            (linv@(y-x_map) - s*(1 + (x-x_map)@linv@(y-x_map))*linv@(x-x_map).T*(1+(x-x_map)@linv@(x-x_map).T)**(-1))/((1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2))

    def dyk(self, x, y):
        return 2*beta*(1 + (x-y)@linv@(x-y).T )**(-beta-1) * linv @ (x-y).T +\
            (linv@(x-x_map) - s*(1 + (x-x_map)@linv@(y-x_map))*linv@(y-x_map).T*(1+(y-x_map)@linv@(y-x_map).T)**(-1))/((1+(x-x_map)@linv@(x-x_map).T)**(s/2) * (1+(y-x_map)@linv@(y-x_map).T)**(s/2))

    def dxdyk(self, x, y):
        return -4*beta*(beta+1)*(1+(x-y)@linv@(x-y).T)**(-beta-2) * (x-y)@linv@linv@(x-y).T + 2*beta*jnp.trace(linv)*(1+(x-y)@linv@(x-y).T)**(-beta-1)+\
            (
                jnp.trace(linv)\
                - s*(1 + (x-x_map)@linv@(x-x_map).T)**(-1) * (x-x_map)@linv@linv@(x-x_map).T\
                - s*(1 + (y-x_map)@linv@(y-x_map).T)**(-1) * (y-x_map)@linv@(y-x_map).T\
                + s**2*(1 + (x-x_map)@linv@(y-x_map))*(1 + (x-x_map)@linv@(x-x_map))**(-1) * (1 + (y-x_map)@linv@(y-x_map))**(-1)\
                * ((x-x_map)@linv@linv@(y-x_map)) )/( (1 + (x-x_map)@linv@(x-x_map))**(s/2) * (1 + (y-x_map)@linv@(y-x_map))**(s/2)
                                        )

    def test_dxk(self):
        assert jnp.sum(jacrev(self.k, argnums=0)(x, y) - self.dxk(x, y)) < 1e-05

    def test_dyk(self):
        assert jnp.sum(jacrev(self.k, argnums=1)(x, y) - self.dyk(x, y)) < 1e-05

    def test_dxdyk(self):
        assert jnp.sum(jnp.trace(jacfwd(jacrev(self.k, argnums=1), argnums=0)(x, y)) - self.dxdyk(x, y)) < 1e-05
