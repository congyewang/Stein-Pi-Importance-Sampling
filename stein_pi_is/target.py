from abc import ABCMeta, abstractmethod
import numpy as np
from jax import numpy as jnp
from jax import jit, jacfwd


class PiTargetInterface(metaclass=ABCMeta):
    """
    Interface to Pi-invariant Target Distribution, which can only be inherited, prohibites directly instantiated.

    Args:
        metaclass (class): Interface Class, which defaults to ABCMeta.
    """
    def __init__(self, log_p) -> None:
        """Destructor Function for PiTargetInterface class.

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

    def base_kernel(self):
        """
        Abstract Method for Base Kernel Function.

        Base Kernel Function, which can yields a Stein kernel.
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
        Abstract Method for Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Raises:
            NotImplementedError: Log Pi-invariant PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_log_q(self, x):
        """
        Abstract Method for Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Raises:
            NotImplementedError: Log Pi-invariant PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

class PiTargetAuto(PiTargetInterface):
    """
    Using Jax to Auto-differential generate Pi-invariant Target Distribution.

    Args:
        PiTargetInterface (class): Interface to Pi-invariant Target Distribution.
    """
    def __init__(self, log_p, base_kernel) -> None:
        """
        Destructor Function for PiTargetAuto class.

        Args:
            log_p (func): Probability Density Function in Logarithmic form.
            base_kernel (func): base Kernel Function.
        """
        super().__init__(log_p)
        self.base_kernel = base_kernel

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

    def base_kernel(self):
        """
        Base Kernel Function.

        Base Kernel Function, which can yields a Stein kernel.
            Refactor this method for calculation of Stein kernel in the auto-differential Child class.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Base Kernel Function.
        """
        return self.base_kernel

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and a specific base kernel Function to yield Stein kernel.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Stein kernel function.
        """
        dx_k = jit(jacfwd(self.base_kernel, argnums=0))
        dy_k = jit(jacfwd(self.base_kernel, argnums=1))
        dxdy_k = jit(jacfwd(dy_k, argnums=0))
        kp = jit(lambda x, y: jnp.trace(dxdy_k(x, y))\
                        + dx_k(x, y) @ self.grad_log_p(y)\
                        + dy_k(x, y) @ self.grad_log_p(x)\
                        + self.base_kernel(x, y) * self.grad_log_p(x) @ self.grad_log_p(y))
        return kp(x, x)

    def grad_stein_kernel(self, x):
        """
        Gradient of Stein Kernel Function.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Stein Kernel Function.
        """
        return jit(jacfwd(self.stein_kernel))(x)

    def log_q(self, x):
        """
        Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            float: The result of Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return self.log_p(x) + 0.5 * jnp.log(self.stein_kernel(x))

    def grad_log_q(self, x):
        """
        Gradient of Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return jit(jacfwd(self.log_q))(x)

class PiTargetManual(PiTargetInterface):
    """
    Using numpy to generate Pi-invariant Target Distribution manually.

    Args:
        PiTargetInterface (class): Interface to Pi-invariant Target Distribution.
    """
    def __init__(self, log_p, grad_log_p, hess_log_p, linv) -> None:
        """
        Destructor Function for PiTargetIMQ class.

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
        Stein Kernel Function.
        """
        return 1

    def grad_stein_kernel(self, x):
        """
        Gradient of Stein Kernel Function.
        """
        return 1

    def log_q(self, x):
        """
        Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            float: The result of Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return self.log_p(x) + 0.5 * np.log(self.stein_kernel(x))

    def grad_log_q(self, x):
        """
        Gradient of Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Pi-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return 0.5 * self.grad_stein_kernel(x)/self.stein_kernel(x) + self.grad_log_p(x)

class PiTargetIMQ(PiTargetManual):
    """
    Using Inverse Multi-Quadric Kernel as the base kernel to generate Pi-invariant Target Distribution.

    Args:
        PiTargetInterface (class): Interface to Pi-invariant Target Distribution.
    """
    def __init__(self, log_p, grad_log_p, hess_log_p, linv) -> None:
        """
        Destructor Function for PiTargetIMQ class.

        Args:
            log_p (function): Probability Density Function in Logarithmic form.
            grad_log_p (function): Gradient of Logarithmic Probability Density Function.
            hess_log_p (function): Hessian of Logarithmic Probability Density Function.
            linv (np.ndarray): Inverse of the Covariance Matrix.
        """
        super().__init__(log_p, grad_log_p, hess_log_p, linv)

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and inverse multi-quadric base kernel Function to yield Stein kernel.

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
        return 2 * self.grad_log_p(x) @ self.hess_log_p(x)

class PiTargetKGM(PiTargetManual):
    """
    Using Kanagawa-Gretton-Mackey Kernel as the base kernel to generate Pi-invariant Target Distribution.

    Args:
        PiTargetInterface (class): Interface to Pi-invariant Target Distribution.
    """
    def __init__(self, log_p, grad_log_p, hess_log_p, linv, s) -> None:
        """
        Destructor Function for PiTargetKGM class.

        Args:
            log_p (function): Probability Density Function in Logarithmic form.
            grad_log_p (function): Gradient of Logarithmic Probability Density Function.
            hess_log_p (function): Hessian of Logarithmic Probability Density Function.
            linv (np.ndarray): Inverse of the Covariance Matrix.
            s (int): Control Parameter.
        """
        super().__init__(log_p, grad_log_p, hess_log_p, linv)
        self.s = s

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and kernel gradient matching base kernel Function to yield Stein kernel.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Stein kernel function.
        """
        # IMQ
        c0_imq = (1 + x@self.linv@x)**(self.s-1)
        c1_imq = (self.s - 1)*(1 + x@self.linv@x)**(self.s-2) * (self.linv @ x)
        c2_imq = (1 + x@self.linv@x)**(self.s-1) * (((self.s-1)**2 * x@self.linv@self.linv@x)/((1 + x@self.linv@x)**2) + np.trace(self.linv))

        # Linear
        c0_lin = 1.0
        c1_lin = 0.0
        c2_lin = ((1 + x@self.linv@x)**(-1)) * ((-1) * (x@self.linv@self.linv@x) / (1 + x@self.linv@x) + np.trace(self.linv))

        # KGM
        c0_kgm = c0_imq + c0_lin
        c1_kgm = c1_imq + c1_lin
        c2_kgm = c2_imq + c2_lin

        return c2_kgm + 2*c1_kgm@self.grad_log_p(x) + c0_kgm*self.grad_log_p(x)@self.grad_log_p(x)

    def grad_stein_kernel(self, x):
        """
        Gradient of Stein Kernel Function.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Stein Kernel Function.
        """
        # IMQ
        c0_imq = (1 + x@self.linv@x)**(self.s-1)
        c1_imq = (self.s - 1)*(1 + x@self.linv@x)**(self.s-2) * (self.linv @ x)

        grad_c0_imq = 2 * (self.s-1) * (1+x@self.linv@x)**(self.s-2) * (self.linv@x)
        grad_c1_imq = (self.s-1) * ((1 + x@self.linv@x)**(self.s-2)) * ((2*(self.s-2)*np.outer(self.linv@x, self.linv@x))/(1+x@self.linv@x) + self.linv)
        grad_c2_imq = 2 * (self.s-1) * ((1 + x@self.linv@x)**(self.s-2)) * (
            (self.s**2 - 4*self.s + 3) * (x@self.linv@self.linv@x) * (self.linv@x) / ((1 + x@self.linv@x)**2) +\
            (self.s - 1) * (self.linv@self.linv@x) / (1 + x@self.linv@x) +\
            np.trace(self.linv)*(self.linv@x)
        )

        # Linear
        c0_lin = 1.0
        c1_lin = 0.0

        grad_c0_lin = 0.0
        grad_c1_lin = 0.0
        grad_c2_lin = 2 * (-1) * ((1+x@self.linv@x)**(-2)) * (
            (-2) * (x@self.linv@self.linv@x) * (self.linv@x) / (1 + x@self.linv@x) +\
            (self.linv@self.linv@x) +\
            np.trace(self.linv)*(self.linv@x)
        )

        # KGM
        c0_kgm = c0_imq + c0_lin
        c1_kgm = c1_imq + c1_lin

        grad_c0_kgm = grad_c0_imq + grad_c0_lin
        grad_c1_kgm = grad_c1_imq + grad_c1_lin
        grad_c2_kgm = grad_c2_imq + grad_c2_lin

        return grad_c2_kgm + 2 * grad_c1_kgm@self.grad_log_p(x) +\
                2 * self.hess_log_p(x) @ c1_kgm + grad_c0_kgm * (self.grad_log_p(x)@self.grad_log_p(x)) +\
                2 * c0_kgm * self.hess_log_p(x)@self.grad_log_p(x)

class PiTargetCentKGM(PiTargetManual):
    """
    Using Centralized Kanagawa-Gretton-Mackey Kernel as the base kernel to generate Pi-invariant Target Distribution.

    Args:
        PiTargetInterface (class): Interface to Pi-invariant Target Distribution.
    """
    def __init__(self, log_p, grad_log_p, hess_log_p, linv, s, x_map) -> None:
        """
        Destructor Function for PiTargetKGM class.

        Args:
            log_p (function): Probability Density Function in Logarithmic form.
            grad_log_p (function): Gradient of Logarithmic Probability Density Function.
            hess_log_p (function): Hessian of Logarithmic Probability Density Function.
            linv (np.ndarray): Inverse of the Covariance Matrix.
            s (int): Control Parameter.
            x_map (np.ndarray||None): The Map of Probability Density Function in Logarithmic form.
        """
        super().__init__(log_p, grad_log_p, hess_log_p, linv)
        self.s = s
        self.x_map = x_map

    def stein_kernel(self, x):
        c0 = 1 + (1 + (x-self.x_map)@self.linv@(x-self.x_map.T))**(self.s-1)
        c1 = (self.s-1)*(1 + (x-self.x_map)@self.linv@(x-self.x_map.T))**(self.s-2) * self.linv@(x-self.x_map).T
        c2 = ( ((self.s-1)**2*(1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(self.s-1) - 1)*(x-self.x_map)@self.linv@self.linv@(x-self.x_map).T )/( (1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**2 ) + (np.trace(self.linv)*( 1 + (1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**self.s ) )/( (1 + (x-self.x_map)@self.linv@(x-self.x_map).T) )

        return c2 + 2*c1@self.grad_log_p(x) + c0*self.grad_log_p(x)@self.grad_log_p(x)

    def grad_stein_kernel(self, x):
        c0 = 1 + (1 + (x-self.x_map)@self.linv@(x-self.x_map.T))**(self.s-1)
        c1 = (self.s-1)*(1 + (x-self.x_map)@self.linv@(x-self.x_map.T))**(self.s-2) * self.linv@(x-self.x_map).T

        grad_c0 = 2*(self.s-1)*(1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(self.s-2) * self.linv@(x-self.x_map).T

        grad_c1 = 2*(self.s-1)*(self.s-2)*(1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(self.s-3) * np.outer(self.linv@(x-self.x_map).T, self.linv@(x-self.x_map).T)\
            + (self.s-1)*(1 + (x-self.x_map)@self.linv@(x-self.x_map))**(self.s-2) * self.linv

        grad_c2 = 2*(self.s-1)**2*(self.s-3)*(1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(self.s-4) * ((x-self.x_map)@self.linv@self.linv@(x-self.x_map).T) * self.linv@(x-self.x_map).T\
            + 2 * (self.s-1)**2 * (1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(self.s-3)* self.linv@self.linv@(x-self.x_map).T\
            + 2 * np.trace(self.linv) * (self.s-1) * (1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(self.s-2) * self.linv@(x-self.x_map)\
            - 2 * (1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(-2) * ( self.linv@self.linv@(x-self.x_map).T + np.trace(self.linv)* self.linv@(x-self.x_map) )\
            + 4 * (1 + (x-self.x_map)@self.linv@(x-self.x_map).T)**(-3) * ((x-self.x_map)@self.linv@self.linv@(x-self.x_map).T) * self.linv@(x-self.x_map)

        return grad_c2 + 2 * grad_c1@self.grad_log_p(x) +\
                2 * self.hess_log_p(x) @ c1 + grad_c0 * (self.grad_log_p(x)@self.grad_log_p(x)) +\
                2 * c0 * self.hess_log_p(x)@self.grad_log_p(x)
