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
        varphi_hyperparameters=None,
        varphi_hyperhyperparameters=None,
        variance_hyperparameters=None,
        **kwargs: Any
    ) -> None:
        """
        Create an :class:`Kernel` object.

        This method should be implemented in every concrete kernel. Initiating
        a kernel should be a very cheap operation.
 
        :arg variance_hyperparameters:
        :type variance_hyperparameters: float or :class:`numpy.ndarray` or None
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: float or :class:`numpy.ndarray` or None
        :arg varphi_hyperhyperparameters: The (K, ) array or float or None (location/ scale)
            hyper-hyper-parameters that define varphi_hyperparameters prior. Not to be confused
            with `Sigma`, which is a covariance matrix. Default None.
        :type varphi_hyperhyperparameters: float or :class:`numpy.ndarray` or None

        :returns: A :class:`Kernel` object
        """
        super().__init__(**kwargs)
        if varphi_hyperparameters is not None:
            self.varphi_hyperparameters = varphi_hyperparameters
            if varphi_hyperhyperparameters is not None:
                self.varphi_hyperhyperparameters = varphi_hyperhyperparameters
        else:
            self.varphi_hyperparameters = None
            self.varphi_hyperhyperparameters = None
        if variance_hyperparameters is not None:
            self.variance_hyperparameters = variance_hyperparameters
        else:
            self.variance_hyperparameters = None
