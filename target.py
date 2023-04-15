from abc import ABCMeta, abstractmethod
import numpy as np
from jax import numpy as jnp
from jax import jit, jacfwd


class QTargetInterface(metaclass=ABCMeta):
    """
    Interface to Q-invariant Target Distribution, which can only be inherited, prohibites directly instantiated.

    Args:
        metaclass (class): Interface Class, which defaults to ABCMeta.
    """
    def __init__(self, log_p) -> None:
        """Destructor Function for QTargetInterface class.

        Args:
            log_p (func): Probability Density Function in Logarithmic form.
        """
        self.log_p = log_p

    @abstractmethod
    def grad_log_p(self, x):
        """
        Abstract Method for Gradient of Logarithmic Probability Density Function.

        Raises:
            NotImplementedError: Gradient of Log PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def hess_log_p(self, x):
        """
        Abstract Method for Hessian of Logarithmic Probability Density Function.

        Raises:
            NotImplementedError: Hessian of Log PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    def reproducing_kernel(self):
        """
        Abstract Method for Reproducing Kernel Function.

        Reproducing Kernel Function, which can yields a Stein kernel.
            Refactor this method for calculation of Stein kernel in the auto-differential Child class.
        """
        raise NotImplementedError

    @abstractmethod
    def stein_kernel(self, x):
        """
        Abstract Method for Stein Kernel Function.

        Raises:
            NotImplementedError: Stein Kernel must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_stein_kernel(self, x):
        """
        Abstract Method for Gradient of Stein Kernel Function.

        Raises:
            NotImplementedError: Gradient of Stein Kernel must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def log_q(self, x):
        """
        Abstract Method for Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Raises:
            NotImplementedError: Log Q-invariant PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_log_q(self, x):
        """
        Abstract Method for Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Raises:
            NotImplementedError: Log Q-invariant PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

class QTargetAuto(QTargetInterface):
    """
    Using Jax to Auto-differential generate Q-invariant Target Distribution.

    Args:
        QTargetInterface (class): Interface to Q-invariant Target Distribution.
    """
    def __init__(self, log_p, reproducing_kernel) -> None:
        """
        Destructor Function for QTargetAuto class.

        Args:
            log_p (func): Probability Density Function in Logarithmic form.
            reproducing_kernel (func): Reproducing Kernel Function.
        """
        super().__init__(log_p)
        self.reproducing_kernel = reproducing_kernel

    def grad_log_p(self, x):
        """
        Gradient of Logarithmic Probability Density Function.

        Returns:
            func: Gradient of Logarithmic Probability Density Function.
        """
        return jit(jacfwd(self.log_p))(x)

    def hess_log_p(self, x):
        """
        Hessian of Logarithmic Probability Density Function.

        Returns:
            function: Hessian of Logarithmic Probability Density Function.
        """
        return jit(jacfwd(self.grad_log_p))(x)

    def reproducing_kernel(self):
        """
        Reproducing Kernel Function.

        Reproducing Kernel Function, which can yields a Stein kernel.
            Refactor this method for calculation of Stein kernel in the auto-differential Child class.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Reproducing Kernel Function.
        """
        return self.reproducing_kernel

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and a specific reproducing kernel Function to yield Stein kernel.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Stein kernel function.
        """
        dx_k = jit(jacfwd(self.reproducing_kernel, argnums=0))
        dy_k = jit(jacfwd(self.reproducing_kernel, argnums=1))
        dxdy_k = jit(jacfwd(dy_k, argnums=0))
        kp = jit(lambda x, y: jnp.trace(dxdy_k(x, y))\
                        + dx_k(x, y) @ self.grad_log_p(y)\
                        + dy_k(x, y) @ self.grad_log_p(x)\
                        + self.reproducing_kernel(x, y) * self.grad_log_p(x) @ self.grad_log_p(y))
        return kp(x, x)

    def grad_stein_kernel(self, x):
        """
        Gradient of Stein Kernel Function.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Stein Kernel Function.
        """
        return jit(jacfwd(lambda x: self.stein_kernel(x)))(x)

    def log_q(self, x):
        """
        Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            float: The result of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return self.log_p(x) + 0.5 * jnp.log(self.stein_kernel(x))

    def grad_log_q(self, x):
        """
        Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return jit(jacfwd(lambda x: self.log_q(x)))(x)

class QTargetIMQ(QTargetInterface):
    """
    Using Inverse Multi-Quadric Kernel as the reproducing kernel to generate Q-invariant Target Distribution.

    Args:
        QTargetInterface (class): Interface to Q-invariant Target Distribution.
    """
    def __init__(self, log_p, grad_log_p, hess_log_p, linv) -> None:
        """
        Destructor Function for QTargetIMQ class.

        Args:
            log_p (function): Probability Density Function in Logarithmic form.
            grad_log_p (function): Gradient of Logarithmic Probability Density Function.
            hess_log_p (function): Hessian of Logarithmic Probability Density Function.
            linv (np.ndarray): Inverse of the Covariance Matrix.
        """
        super().__init__(log_p)
        self.grad_log_p = grad_log_p
        self.hess_log_p = hess_log_p
        self.linv = linv

    def grad_log_p(self, x):
        """
        Gradient of Logarithmic Probability Density Function.

        Returns:
            func: Gradient of Logarithmic Probability Density Function.
        """
        return self.grad_log_p(x)

    def hess_log_p(self, x):
        """
        Hessian of Logarithmic Probability Density Function.

        Returns:
            func: Hessian of Logarithmic Probability Density Function.
        """
        return self.hess_log_p(x)

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and inverse multi-quadric reproducing kernel Function to yield Stein kernel.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Stein kernel function.
        """
        return np.trace(self.linv) + self.grad_log_p(x) @ self.grad_log_p(x)

    def grad_stein_kernel(self, x):
        """
        Gradient of Stein Kernel Function.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Stein Kernel Function.
        """
        return 2 * self.grad_log_p(x) * self.hess_log_p(x)

    def log_q(self, x):
        """
        Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            float: The result of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return self.log_p(x) + 0.5 * np.log(self.stein_kernel(x))

    def grad_log_q(self, x):
        """
        Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return (self.hess_log_p(x) @ self.grad_log_p(x))/self.stein_kernel(x) + self.grad_log_p(x)

class QTargetKGM(QTargetInterface):
    """
    Using Kanagawa-Gretton-Mackey Kernel as the reproducing kernel to generate Q-invariant Target Distribution.

    Args:
        QTargetInterface (class): Interface to Q-invariant Target Distribution.
    """
    def __init__(self, log_p, grad_log_p, hess_log_p, linv, s) -> None:
        """
        Destructor Function for QTargetKGM class.

        Args:
            log_p (function): Probability Density Function in Logarithmic form.
            grad_log_p (function): Gradient of Logarithmic Probability Density Function.
            hess_log_p (function): Hessian of Logarithmic Probability Density Function.
            linv (np.ndarray): Inverse of the Covariance Matrix.
            s (float): Control Parameter.
        """
        super().__init__(log_p)
        self.grad_log_p = grad_log_p
        self.hess_log_p = hess_log_p
        self.linv = linv
        self.s = s

    def grad_log_p(self, x):
        """
        Gradient of Logarithmic Probability Density Function.

        Returns:
            func: Gradient of Logarithmic Probability Density Function.
        """
        return self.grad_log_p(x)

    def hess_log_p(self, x):
        """
        Hessian of Logarithmic Probability Density Function.

        Returns:
            func: Hessian of Logarithmic Probability Density Function.
        """
        return self.hess_log_p(x)

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and kernel gradient matching reproducing kernel Function to yield Stein kernel.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Stein kernel function.
        """
        # IMQ
        c0_imq = lambda x, s: (1 + x@self.linv@x)**(s-1)
        c1_imq = lambda x, s: (s - 1)*(1 + x@self.linv@x)**(s-2) * (self.linv @ x)
        c2_imq = lambda x, s: (1 + x@self.linv@x)**(s-1) * (((s-1)**2 * x@self.linv@self.linv@x)/((1 + x@self.linv@x)**2) + np.trace(self.linv))

        # Linear
        c0_lin = lambda x, s: (1 + x@self.linv@x)**s
        c1_lin = lambda x, s: s * ((1 + x@self.linv@x)**(s-1)) * (self.linv@x)
        c2_lin = lambda x, s: ((1 + x@self.linv@x)**(s-1)) * ((s**2 - 1) * (x@self.linv@self.linv@x) / (1 + x@self.linv@x) + np.trace(self.linv))

        # KGM
        c0_kgm = lambda x, s: c0_imq(x, s) + c0_lin(x, 0.0)
        c1_kgm = lambda x, s: c1_imq(x, s) + c1_lin(x, 0.0)
        c2_kgm = lambda x, s: c2_imq(x, s) + c2_lin(x, 0.0)

        return c2_kgm(x, self.s) + 2*c1_kgm(x, self.s)@self.grad_log_p(x) + c0_kgm(x, self.s)*self.grad_log_p(x)@self.grad_log_p(x)

    def grad_stein_kernel(self, x):
        """
        Gradient of Stein Kernel Function.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Stein Kernel Function.
        """
        # IMQ
        c0_imq = lambda x, s: (1 + x@self.linv@x)**(s-1)
        c1_imq = lambda x, s: (s - 1)*(1 + x@self.linv@x)**(s-2) * (self.linv @ x)

        grad_c0_imq = lambda x, s: 2 * (s-1) * (1+x@self.linv@x)**(s-2) * (self.linv@x)
        grad_c1_imq = lambda x, s: (s-1) * ((1 + x@self.linv@x)**(s-2)) * ((2*(s-2)*np.outer(self.linv@x, self.linv@x))/(1+x@self.linv@x) + self.linv)
        grad_c2_imq = lambda x, s: 2 * (s-1) * ((1 + x@self.linv@x)**(s-2)) * (
            (s**2 - 4*s + 3) * (x@self.linv@self.linv@x) * (self.linv@x) / ((1 + x@self.linv@x)**2) +\
            (s - 1) * (self.linv@self.linv@x) / (1 + x@self.linv@x) +\
            np.trace(self.linv)*(self.linv@x)
        )

        # Linear
        c0_lin = lambda x, s: (1 + x@self.linv@x)**s
        c1_lin = lambda x, s: s * ((1 + x@self.linv@x)**(s-1)) * (self.linv@x)

        grad_c0_lin = lambda x, s: 2 * s * ((1+x@self.linv@x)**(s-1)) * (self.linv@x)
        grad_c1_lin = lambda x, s: s * ((1+x@self.linv@x)**(s-1)) * (2*(s-1)* np.outer(self.linv@x, self.linv@x) / (1 + x@self.linv@x) + self.linv)
        grad_c2_lin = lambda x, s: 2 * (s - 1) * ((1+x@self.linv@x)**(s-2)) * (
            (s + 1) * (s - 2) * (x@self.linv@self.linv@x) * (self.linv@x) / (1 + x@self.linv@x) +\
            (s + 1) * (self.linv@self.linv@x)+\
            np.trace(self.linv)*(self.linv@x)
        )

        # KGM
        c0_kgm = lambda x, s: c0_imq(x, s) + c0_lin(x, 0.0)
        c1_kgm = lambda x, s: c1_imq(x, s) + c1_lin(x, 0.0)

        grad_c0_kgm = lambda x, s: grad_c0_imq(x, s) + grad_c0_lin(x, 0.0)
        grad_c1_kgm = lambda x, s: grad_c1_imq(x, s) + grad_c1_lin(x, 0.0)
        grad_c2_kgm = lambda x, s: grad_c2_imq(x, s) + grad_c2_lin(x, 0.0)

        return grad_c2_kgm(x, self.s) + 2 * grad_c1_kgm(x, self.s)@self.grad_log_p(x) +\
                2 * self.hess_log_p(x) @ c1_kgm(x, self.s) + grad_c0_kgm(x, self.s) * (self.grad_log_p(x)@self.grad_log_p(x)) +\
                2 * c0_kgm(x, self.s) * self.hess_log_p(x)@self.grad_log_p(x)

    def log_q(self, x):
        """
        Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            float: The result of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return self.log_p(x) + 0.5 * np.log(self.stein_kernel(x))

    def grad_log_q(self, x):
        """
        Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return 0.5 * self.grad_stein_kernel(x)/self.stein_kernel(x) + self.grad_log_p(x)
