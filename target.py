from abc import ABCMeta, abstractmethod
import numpy as np


class QTargetInterface(metaclass=ABCMeta):
    """
    Interface to Q-invariant Target Distribution, which can only be inherited, prohibites directly instantiated.

    Args:
        metaclass (class): Interface Class, which defaults to ABCMeta.
    """
    def __init__(self, log_pi) -> None:
        """Destructor Function for QTargetInterface class.

        Args:
            log_pi (function): Probability Density Function in Logarithmic form.
        """
        self.log_pi = log_pi

    @abstractmethod
    def grad_log_pi(self):
        """
        Abstract Method for Gradient of Logarithmic Probability Density Function.

        Raises:
            NotImplementedError: Gradient of Log PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def hess_log_pi(self):
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
    def stein_kernel(self):
        """
        Abstract Method for Stein Kernel Function.

        Raises:
            NotImplementedError: Stein Kernel must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def log_q(self):
        """
        Abstract Method for Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Raises:
            NotImplementedError: Log Q-invariant PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

    @abstractmethod
    def grad_log_q(self):
        """
        Abstract Method for Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Raises:
            NotImplementedError: Log Q-invariant PDF must be overridden in the class in order for the Child class to be instantiated,
                otherwise no instantiated object can be created.
        """
        raise NotImplementedError

class QTargetIMQ(QTargetInterface):
    """
    Using Inverse Multi-Quadric Kernel as the reproducing kernel to generate Q-invariant Target Distribution.

    Args:
        QTargetInterface (class): Interface to Q-invariant Target Distribution.
    """
    def __init__(self, log_pi, grad_log_pi, hess_log_pi, linv) -> None:
        """
        Destructor Function for QTargetIMQ class.

        Args:
            log_pi (function): Probability Density Function in Logarithmic form.
            grad_log_pi (function): Gradient of Logarithmic Probability Density Function.
            hess_log_pi (function): Hessian of Logarithmic Probability Density Function.
            linv (np.ndarray): Inverse of the Covariance Matrix.
        """
        super().__init__(log_pi)
        self.grad_log_pi = grad_log_pi
        self.hess_log_pi = hess_log_pi
        self.linv = linv

    def grad_log_pi(self):
        """
        Gradient of Logarithmic Probability Density Function.

        Returns:
            function: Gradient of Logarithmic Probability Density Function.
        """
        return self.grad_log_pi

    def hess_log_pi(self):
        """
        Hessian of Logarithmic Probability Density Function.

        Returns:
            function: Hessian of Logarithmic Probability Density Function.
        """
        return self.hess_log_pi

    def stein_kernel(self, x):
        """
        Using the combination of Langevin-Stein operator and inverse multi-quadric reproducing kernel Function to yield Stein kernel.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Stein kernel function.
        """
        return np.trace(self.linv) + self.grad_log_pi(x) @ self.grad_log_pi(x)

    def log_q(self, x):
        """
        Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            float: The result of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return self.log_pi(x) + 0.5 * np.log(self.stein_kernel(x))

    def grad_log_q(self, x):
        """
        Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.

        Args:
            x (np.ndarray): Input.

        Returns:
            np.ndarray: The result of Gradient of Q-invariant Target Distribution as Probability Density Function in Logarithmic form.
        """
        return (self.hess_log_pi(x) @ self.grad_log_pi(x))/self.stein_kernel(x) + self.grad_log_pi(x)
