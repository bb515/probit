import numpy as np
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    Base class for samplers.

    All samplers must define an init method, which may or may not inherit Sampler as a parent class using `super()`.
    All samplers that inherit Sampler define a number of methods that return the samples.
    """

    @abstractmethod
    def __init__(self):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete sampler.

        :arg :class:`numpy.ndarray` init: The initial location of the sampler in parameter space.
        :arg :class:`numpy.ndarray` steps: The number of steps in the sampler.

        :returns: A :class:`Sampler` object
        """

    @abstractmethod
    def sampler_initialise(self, init, steps, first_step):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.
        """
        if type(steps) is not int or np.intc:
            raise TypeError(
                "Type of steps is not supported "
                "(expected {} or {}, got {})".format(
                    int, np.intc, type(steps)))
        if type(first_step) is not int or np.intc:
            raise TypeError(
                "Type of steps is not supported "
                "(expected {} or {}, got {})".format(
                    int, np.intc, type(steps)))

    @abstractmethod
    def sample(self):
        """
        Return the samples

        This method should be implemented in every concrete sampler.
        """

    @abstractmethod
    def predict(self):
        """
        Return the samples

        This method should be implemented in every concrete sampler.
        """

class Gibbs_GP(Sampler):
    """
    A Gibbs sampler for GP priors. Inherits the sampler ABC
    """
    def __init__(self, *args, **kwargs, kernel=None):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.kernel = kernel
        if not (isinstance(kernel, Kernel)):
            raise InvalidKernel(kernel)



class InvalidKernel(Exception):
    """An invalid kernel has been passed to `Sampler`"""

    def __init__(self, kernel):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`Sampler` as the kernel
            argument.
        :rtype: :class:`InvalidKernel`
        """
        message = (
            f"{kernel} is not an instance of"
            "probit.kernels.Kernel"
        )

        super().__init__(message)
