from typing import Any, Optional
import numpy as np
import tensorflow as tf
from gpflow.base import Parameter, TensorType
from gpflow.utilities import positive
from gpflow.utilities.ops import difference_matrix, square_distance
from gpflow.kernels import SquaredExponential


class SquaredExponential(SquaredExponential):
    """
    Extension of gpflow.kernels.SquaredExponential for probit.
    """
    def __init__(
        self,
        theta_hyperparameters=None,
        theta_hyperhyperparameters=None,
        variance_hyperparameters=None,
        **kwargs: Any
    ) -> None:
        """
        Create an :class:`Kernel` object.

        This method should be implemented in every concrete kernel. Initiating
        a kernel should be a very cheap operation.
 
        :arg variance_hyperparameters:
        :type variance_hyperparameters: float or :class:`numpy.ndarray` or None
        :arg theta_hyperparameters:
        :type theta_hyperparameters: float or :class:`numpy.ndarray` or None
        :arg theta_hyperhyperparameters: The (K, ) array or float or None (location/ scale)
            hyper-hyper-parameters that define theta_hyperparameters prior. Not to be confused
            with `Sigma`, which is a covariance matrix. Default None.
        :type theta_hyperhyperparameters: float or :class:`numpy.ndarray` or None

        :returns: A :class:`Kernel` object
        """
        super().__init__(**kwargs)
        if theta_hyperparameters is not None:
            self.theta_hyperparameters = theta_hyperparameters
            if theta_hyperhyperparameters is not None:
                self.theta_hyperhyperparameters = theta_hyperhyperparameters
        else:
            self.theta_hyperparameters = None
            self.theta_hyperhyperparameters = None
        if variance_hyperparameters is not None:
            self.variance_hyperparameters = variance_hyperparameters
        else:
            self.variance_hyperparameters = None
