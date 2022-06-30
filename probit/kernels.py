import os
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=6
import enum
import lab as B
import numpy as np
from scipy.spatial import distance_matrix, distance
from abc import ABC, abstractmethod
# from mlkernels import EQ


class Kernel(ABC):
    """
    Base class for kernels.

    TODO: cythonise these functions - or numba. Or replace kernels with existing python GP kernel code. (dont reinvent the wheel)
    TODO: are self.L and self.M really needed?

    All kernels must define an init method, which may or may not inherit Kernel
    as a parent class using `super()`. All kernels that inherit Kernel define a
    number of methods that return the kernel value, a vector of kernel values
    (a vector of covariances), or a covariance matrix of kernel values.
    """

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.

        This method should be implemented in every concrete kernel.
        """

    @abstractmethod
    def __init__(
            self, variance=1.0, variance_hyperparameters=None,
            varphi=None, varphi_hyperparameters=None,
            varphi_hyperhyperparameters=None):
        """
        Create an :class:`Kernel` object.

        This method should be implemented in every concrete kernel. Initiating
        a kernel should be a very cheap operation.
 
        :arg float variance: The kernel variance hyperparameters as a numpy array.
            Default 1.0.
        :arg variance_hyperparameters:
        :type variance_hyperparameters: float or :class:`numpy.ndarray` or None
        :arg varphi:
        :type varphi: float or :class:`numpy.ndarray` or None
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: float or :class:`numpy.ndarray` or None
        :arg varphi_hyperhyperparameters: The (K, ) array or float or None (location/ scale)
            hyper-hyper-parameters that define varphi_hyperparameters prior. Not to be confused
            with `Sigma`, which is a covariance matrix. Default None.
        :type varphi_hyperhyperparameters: float or :class:`numpy.ndarray` or None

        :returns: A :class:`Kernel` object
        """
        variance = np.float64(variance)
        self.variance = variance
        if varphi is not None:
            self.varphi, self.L, self.M = self._initialise_varphi(
                varphi)
        else:
            raise ValueError(
                "Kernel hyperparameters `varphi` must be provided "
                "(got {})".format(None))
        if varphi_hyperparameters is not None:
            self.varphi_hyperparameters = self._initialise_hyperparameter(
                self.varphi, varphi_hyperparameters)
            if varphi_hyperhyperparameters is not None:
                self.varphi_hyperhyperparameters = self._initialise_hyperparameter(
                    self.varphi_hyperparameters, varphi_hyperhyperparameters)
            else:
                self.varphi_hyperhyperparameters = None
        else:
            self.varphi_hyperparameters = None
            self.varphi_hyperhyperparameters = None
        if variance_hyperparameters is not None:
            self.variance_hyperparameters = self._initialise_hyperparameter(
                self.variance, variance_hyperparameters)
        else:
            self.variance_hyperparameters = None

    @abstractmethod
    def kernel(self):
        """
        Return the kernel value given two input vectors.

        This method should be implemented in every concrete kernel.
        """

    @abstractmethod
    def kernel_vector(self):
        """
        Return the kernel vector given an input matrix and input vector.

        This method should be implemented in every concrete kernel.
        """

    @abstractmethod
    def kernel_matrix(self):
        """
        Return the Gram matrix given two input matrices.

        This method should be implemented in every concrete kernel.
        """

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrices for varphi samples, varphis, as an array
        of numpy arrays.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :arg varphis: (n_samples,) array of hyperparameter samples.
        :type varphis: class:`numpy.ndarray`
        :return: Cs_samples (n_samples, N1, N2) array of n_samples * (N1, N2)
            Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, N1, N2))
        for i, varphi in enumerate(varphis):
            self.update_hyperparameter(varphi=varphi)
            Cs_samples[i, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

    def _initialise_varphi(self, varphi=None):
        """
        Initialise as Matern type kernel
        (loosely defined here as a kernel with a length scale) 
        """
        if varphi is not None:
            if ((type(varphi) is list) or
                    (type(varphi) is np.ndarray)):
                if np.shape(varphi) == (1,):
                    # e.g. [[1]]
                    L = 1
                    M = 1
                elif np.shape(varphi) == ():
                    # e.g. [1]
                    L = 1
                    M = 1
                elif np.shape(varphi[0]) == (1,):
                    # e.g. [[1],[2],[3]]
                    L = np.shape(varphi)[0]
                    M = 1
                elif np.shape(varphi[0]) == ():
                    # e.g. [1, 2, 3]
                    L = 1
                    M = np.shape(varphi)[0]
                else:
                    # e.g. [[1, 2], [3, 4], [5, 6]]
                    L = np.shape(varphi)[0]
                    M = np.shape(varphi)[1]
            elif ((type(varphi) is float) or
                    (type(varphi) is np.float64)):
                # e.g. 1
                L = 1
                M = 1
            else:
                raise TypeError(
                    "Type of varphi is not supported "
                    "(expected {} or {}, got {})".format(
                        float, np.ndarray, type(varphi)))
        return varphi, L, M

    def _initialise_hyperparameter(self, parameter, hyperparameter):
        # if hyperparameter is not None:
        #     if type(hyperparameter) != type(parameter):
        #         raise TypeError(
        #             "The type of the kernel parameter {},"
        #             " should equal the type of the kernel hyperparameter {},"
        #             " varphi (got {}, expected {})".format(
        #                 parameter, hyperparameter,
        #                 type(hyperparameter), type(parameter)))
        return hyperparameter

    def update_hyperparameter(
            self, varphi=None, varphi_hyperparameters=None,
            variance=None, variance_hyperparameters=None,
            varphi_hyperhyperparameters=None):
        if varphi is not None:
            self.varphi, self.L, self.M = self._initialise_varphi(
                varphi)
        if varphi_hyperparameters is not None:
            self.varphi_hyperparameters = self._initialise_hyperparameter(
                self.varphi, varphi_hyperparameters)
        if variance is not None:
            # Update variance
            self.variance = variance
        if variance_hyperparameters is not None:
            self.variance_hyperparameters = self._initialise_hyperparameter(
                self.variance, variance_hyperparameters)
        if varphi_hyperhyperparameters is not None:
            # Update varphi hyperhyperparameter
            self.varphi_hyperhyperparameters = varphi_hyperhyperparameters

    def distance_mat(self, X1, X2):
        """
        Return a distance matrix using scipy spatial.

        This is an attempt using cdist, which should be faster.

        :arg X1: The (N1, D) input array for the distance matrix.
        :type X1: :class:`numpy.ndarray`
        :arg X2: The (N2, D) input array for the distance matrix.
        :type X2: :class:`numpy.ndarray` or float
        :return: Euclidean distance matrix (N1, N2).
        :rtype: :class:`numpy.ndarray`.
        """
        # cdist automatically handles the case that D = 1
        return distance.cdist(X1, X2, metric='euclidean')


class Linear(Kernel):
    r"""
    A linear kernel class.

    .. math::
        K(x_i, x_j) = s * x_{i}^{T} x_{j} + c,

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`s` is the
    regularising constant (or scale) and :math:`c` is the intercept
    regularisor.
    """

    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "Linear"

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Linear` kernel object.

        :arg varphi:
        :type varphi: :class:`numpy.ndarray` or float
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float

        :returns: An :class:`Linear` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the shared and single kernel for each class 
        # (i.e. non general) and single lengthscale across
        # all data dims (i.e. non-ARD) is assumed.
        if varphi is not None:
            varphi, self.L = self._initialise_varphi(varphi)
            assert self.L == 2  # 2 hyperparameters for the linear kernel
        else:
            raise ValueError(
                "Hyperparameters `varphi = [constant_variance, c]` must be "
                "provided for the Linear kernel class (got {})".format(None))
        self.constant_variance = varphi[0]
        self.offset = varphi[1]
        if self.constant_variance is None:
            self.constant_variance = 0.0
        elif not ((type(self.constant_variance) is float) or
                (type(self.constant_variance) is np.float64)):
            if type(self.constant_variance) is np.ndarray:
                if not np.shape(self.constant_variance) is ():
                    raise TypeError(
                    "Shape of constant_variance not supported "
                    "(expected float, got {} array)".format(
                        np.shape(self.constant_variance)))
            else:
                raise TypeError(
                    "Type of constant_variance type "
                    "(expected float, got {})".format(
                        type(self.constant_variance)))
        if self.offset is None:
            self.offset= 0.0
        elif not (
            (type(self.c) is float) or
            (type(self.offset) is np.float64) or
            (type(self.offset) is list) or 
            (type(self.offset) is np.ndarray)):
            raise TypeError(
                "Type of c not supported "
                "(expected array or float, got {})".format(
                    type(self.offset)))
        self.num_hyperparameters = np.size(
            self.constant_variance) + np.size(self.offset)

    def _initialise_varphi(self, varphi):
        """
        Initialise as Matern type kernel
        (loosely defined here as a kernel with a length scale) 
        """
        if type(varphi) is list:
            L = len(varphi)
        elif type(varphi) is np.ndarray:
            L = np.shape(varphi)[0]
        elif ((type(varphi) is float) or
                (type(varphi) is np.float64)):
            L = 1
            varphi = np.float64(varphi)
        else:
            raise TypeError(
                "Type of varphi is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(varphi)))
        return varphi, L

    def update_hyperparameter(
        self, varphi=None, variance=None, sigma=None, tau=None):
        if varphi is not None:
            self.varphi, self.L = self._initialise_varphi(varphi)
        if variance is not None:
            # Update variance
            self.variance = variance
        if sigma is not None:
            # Update sigma
            self.sigma = sigma
        if tau is not None:
            # Update tau
            self.tau = tau
        if bool(self.sigma) != bool(self.tau):
            raise TypeError(
                "If a sigma hyperhyperparameter is provided, then a tau "
                "hyperhyperparameter must be provided"
                " (expected {}, got {})".format(np.ndarray, type(tau))
            )

    @property
    def _ARD(self):
        return False

    @property
    def _stationary(self):
        return False

    @property
    def _Matern(self):
        return False

    @property
    def _general(self):
        return False

    def kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return (
            self.variance * (X_j - self.c).T @ (X_i - self.c)
            + self.constant_variance)
 
    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: the (N,) covariance vector.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * np.einsum(
            'ij, j -> i', X - self.c, x_new[0]
            - self.c) + self.constant_variance

    def kernel_diagonal(self, X1, X2):
        """
        Get Gram diagonal efficiently using scipy's distance matrix function.
        TODO: test

        :arg X1: (N, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N,) Gram diagonal.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * np.einsum(
            'ij, ij -> i', X1 - self.c, X2 - self.c) + self.constant_variance

    def kernel_prior_diagonal(self, X):
        """
        Get the kernel vector given an input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: the (N,) covariance vector.
        :rtype: class:`numpy.ndarray`
        """
        return self.kernel_diagonal(X, X)

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using numpy's einsum function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * np.einsum(
            'ik, jk -> ij', X1 - self.c, X2 - self.c) + self.constant_variance

    def kernel_partial_derivative_constant_variance(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
            a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the
            covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO check this.
        return np.ones((len(X1), len(X2)))

    def kernel_partial_derivative_c(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
            a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the
            covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO this is checked on scratch.py, but worth checking that the order
        # of X1 and X2 is expected
        if np.shape(self.c) == ():
            return  self.variance * (
                X1[:, np.newaxis, :]
                + X2[np.newaxis, :, :]
                + 2 * self.c)
        else:
            return self.variance * (
                X1[:, np.newaxis, :]
                + X2[np.newaxis, :, :]
                + 2 * self.c[np.newaxis, np.newaxis, :])

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to hyperparameters as
            a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (L, N1, N2) L Partial derivatives of gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        # return [
        #     self.kernel_partial_derivative_constant_variance(X1, X2),
        #     self.kernel_partial_derivative_c(X1, X2)]
        return self.kernel_partial_derivative_constant_variance(X1, X2)

    def kernel_partial_derivative_scale(self, X1, X2):
        """
        Get Gram matrix efficiently using numpy's einsum function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return np.einsum(
            'ik, jk -> ij', X1 - self.c, X2 - self.c)


class LabEQ(Kernel):
    r"""
    Uses BLab by WesselB, which is an generic interface for linear algebra
    backends.

    An isometric radial basis function (a.k.a. exponentiated quadratic,
    a.k.a squared exponential) kernel class.

    .. math::
        K(x_i, x_j) = s * \exp{- \varphi * \norm{x_{i} - x_{j}}^2},

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi` is
    the single, shared lengthscale and hyperparameter, :math:`s` is the variance.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "LabEQ"

    def __init__(
            self, *args, **kwargs):
        """
        Create an :class:`SEIso` kernel object.

        :arg varphi: The kernel lengthscale hyperparameter.
        :type varphi: float or NoneType
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float

        :returns: An :class:`SEIso` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the shared and single kernel for each class
        # (i.e. non general) and single lengthscale across
        # all data dims (i.e. non ARD) is assumed.
        if self.L != 1:
            raise ValueError(
                "L wrong for simple kernel (expected {}, got {})".format(
                    1, self.L))
        if self.M != 1:
            raise ValueError(
                "M wrong for non-ARD kernel (expected {}, got {})".format(
                    1, self.M))
        if self.variance is None:
            raise ValueError(
                "You must supply a variance for the simple kernel "
                "(expected {} type, got {})".format(float, self.variance))
        self.num_hyperparameters = np.size(self.varphi)

    @property
    def _ARD(self):
        return False

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return True

    @property
    def _general(self):
        return False

    def kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return (
            self.variance * B.exp(
                -self.varphi * B.ew_dists2(
                    X_i.reshape(1, -1), X_j.reshape(1, -1)))[0, 0])

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: the (N,) covariance vector.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * B.exp(-self.varphi * B.ew_dists2(
            X, x_new.reshape(1, -1)))

    def kernel_prior_diagonal(self, X):
        return self.variance * np.ones(np.shape(X)[0])

    def kernel_diagonal(self, X1, X2):
        """
        # TODO duplicate code and probably not correct
        Get Gram diagonal efficiently using scipy's distance matrix function.

        :arg X1: (N, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N,) Gram diagonal.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * B.exp(-self.varphi * B.ew_dists2(X1, X2))

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using MLKernels.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * B.exp(-self.varphi * B.pw_dists2(X1, X2))

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
        a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the
            covariance matrix.
        :rtype: class:`numpy.ndarray`
        """
        distance_mat_2 = B.pw_dists2(X1, X2)
        return -B.multiply(
            distance_mat_2, self.variance * B.exp(-self.varphi * distance_mat_2))

    def kernel_partial_derivative_variance(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return B.exp(-self.varphi * B.pw_dists2(X1, X2))


class LabSharpenedCosine(Kernel):
    r"""
    Uses BLab by WesselB, which is an generic interface for linear algebra
    backends.

    An isometric radial basis function (a.k.a. exponentiated quadratic,
    a.k.a squared exponential) kernel class.

    .. math::
        K(x_i, x_j) = s * \exp{- \varphi * \norm{x_{i} - x_{j}}^2},

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi` is
    the single, shared lengthscale and hyperparameter, :math:`s` is the variance.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "LabSharpenedCosine"

    def __init__(self,  *args, **kwargs):
        """
        Create an :class:`SEIso` kernel object.

        :arg varphi: The kernel lengthscale hyperparameter.
        :type varphi: float or NoneType
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float

        :returns: An :class:`SEIso` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the shared and single kernel for each class
        # (i.e. non general) and single lengthscale across
        # all data dims (i.e. non ARD) is assumed.
        if self.L != 1:
            raise ValueError(
                "L wrong for sharpened cosine kernel (expected {}, got {})".format(
                    1, self.L))
        if self.M != 2:
            raise ValueError(
                "M wrong for sharpened cosine kernel (expected {}, got {})".format(
                    2, self.M))
        if self.variance is None:
            raise ValueError(
                "You must supply a variance for the sharpened cosine kernel "
                "(expected {} type, got {})".format(float, self.variance))
        self.num_hyperparameters = np.size(self.varphi)

    @property
    def _ARD(self):
        return False

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return False

    @property
    def _general(self):
        return False

    def kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        # TODO: may need to reshape
        return self.variance * B.matmul(X_i, X_j, tr_a=True)**self.varphi[1]
        # return self.variance * (np.dot(X_i, X_j) / (
        #     (np.linalg.norm(X_i) + self.varphi[0])
        #     * (np.linalg.norm(X_j) + self.varphi[0])))**self.varphi[1]
        # return (self.variance * B.matmul(B.transpose(X_i), X_j))
        # return (self.variance * B.exp(-self.varphi * B.ew_dists2(X_i.reshape(1, -1), X_j.reshape(1, -1))))[0, 0]

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: the (N,) covariance vector.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * B.einsum('k, ik -> i', x_new, X)**self.varphi[1]
        # return self.variance * (np.einsum('k, ik -> i', x_new, X) / (
        #     (np.linalg.norm(x_new) + self.varphi[0]) * (np.linalg.norm(X, axis=1) + self.varphi[0])))**self.varphi[1]

    def kernel_prior_diagonal(self, X):
        return self.variance * np.ones(np.shape(X)[0])

    def kernel_diagonal(self, X1, X2):
        """
        Get Gram diagonal efficiently using scipy's distance matrix function.

        :arg X1: (N, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N,) Gram diagonal.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * B.einsum('ik, ik -> i', X1, X2)**self.varphi[1]
        # return self.variance * np.einsum('ik, ik -> i', X1, X2)**self.varphi[1]
        # return self.variance * (np.einsum('ik, ik -> i', X1, X2) / (
        #     (np.linalg.norm(X1, axis=1) + self.varphi[0])
        #     * (np.linalg.norm(X2, axis=1) + self.varphi[0])))**self.varphi[1]
        # return self.variance * B.exp(-self.varphi * B.ew_dists2(X1, X2))

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using MLKernels.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance * B.matmul(X1, X2, tr_b=True)**self.varphi[1]
        #return self.variance * B.outer(X1, X2)**self.varphi[1]
        #return self.variance * np.einsum('ik, jk -> ij', X1, X2)**self.varphi[1]
        # return self.variance * (np.einsum('ik, jk -> ij', X1, X2) / (
        #     np.outer(
        #         (np.linalg.norm(X1, axis=1) + self.varphi[0]),
        #         (np.linalg.norm(X2, axis=1) + self.varphi[1])
        #         )))**self.varphi[1]
        #return self.variance * B.exp(-self.varphi * B.pw_dists2(X1, X2))

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
        a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the
            covariance matrix.
        :rtype: class:`numpy.ndarray`
        """
        return np.zeros((len(X1), len(X1)))

    def kernel_partial_derivative_variance(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return B.einsum('ik, jk -> ij', X1, X2)**self.varphi[1]
        # return (np.einsum('ik, jk -> ij', X1, X2) / (
        #     np.outer(
        #         (np.linalg.norm(X1, axis=1) + self.varphi[0]),
        #         (np.linalg.norm(X2, axis=1) + self.varphi[1])
        #         )))**self.varphi[1]


class SEIso(Kernel):
    r"""
    An isometric radial basis function (a.k.a. exponentiated quadratic,
    a.k.a squared exponential) kernel class.

    .. math::
        K(x_i, x_j) = s * \exp{- \varphi * \norm{x_{i} - x_{j}}^2},

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi` is
    the single, shared lengthscale and
    hyperparameter, :math:`s` is the variance.

    Note that previous implementations used distance

    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SEIso"

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEIso` kernel object.

        :arg varphi: The kernel lengthscale hyperparameter.
        :type varphi: float or NoneType
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float

        :returns: An :class:`SEIso` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the shared and single kernel for each class
        # (i.e. non general) and single lengthscale across
        # all data dims (i.e. non ARD) is assumed.
        if self.L != 1:
            raise ValueError(
                "L wrong for simple kernel (expected {}, got {})".format(
                    1, self.L))
        if self.M != 1:
            raise ValueError(
                "M wrong for non-ARD kernel (expected {}, got {})".format(
                    1, self.M))
        if self.variance is None:
            raise ValueError(
                "You must supply a variance for the simple kernel "
                "(expected {} type, got {})".format(float, self.variance))
        self.num_hyperparameters = np.size(self.varphi)

    @property
    def _ARD(self):
        return False

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return True

    @property
    def _general(self):
        return False

    def kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return self.variance * np.exp(
            -1. * self.varphi * distance.sqeuclidean(X_i, X_j))

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: the (N,) covariance vector.
        :rtype: class:`numpy.ndarray`
        """
        X_new = np.tile(x_new, (np.shape(X)[0], 1))
        return self.kernel_diagonal(X_new, X)

    def kernel_prior_diagonal(self, X):
        return self.variance * np.ones(np.shape(X)[0])

    def kernel_diagonal(self, X1, X2):
        """
        Get Gram diagonal efficiently using scipy's distance matrix function.

        :arg X1: (N, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N,) Gram diagonal.
        :rtype: class:`numpy.ndarray`
        """
        X = X1 - X2
        return self.variance * np.exp(-1. * self.varphi * np.einsum(
            'ij, ij -> i', X, X))

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        distance_mat = self.distance_mat(X1, X2)
        return self.variance * np.exp(-1. * self.varphi * distance_mat**2)

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
        a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the
            covariance matrix.
        :rtype: class:`numpy.ndarray`
        """
        distance_mat_2 = self.distance_mat(X1, X2)**2
        C = np.multiply(self.variance, np.exp(-1. * self.varphi * distance_mat_2))
        partial_C = -np.multiply(distance_mat_2, C)
        # D = np.shape(X1)[1]
        # # The general covariance function has a different length scale for each dimension.
        # # TODO check this. I am not sure why the dimension parameter need be included here.
        # partial_C = (-1./D) * np.multiply(distance_mat_2, C)  # TODO: reason for this dimension?
        # return partial_C
        return partial_C

    def kernel_partial_derivative_variance(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        # On a GPU, want to avoid storing the distance matrix, so recalculate
        return np.exp(-1. * self.varphi * self.distance_mat(X1, X2)**2)


class SumPolynomialSEIso(Kernel):
    r"""
    Uses BLab by WesselB, which is an generic interface for linear algebra
    backends.

    An isometric radial basis function (a.k.a. exponentiated quadratic,
    a.k.a squared exponential) kernel class.

    .. math::
        K(x_i, x_j) = s * \exp{- \varphi * \norm{x_{i} - x_{j}}^2},

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi` is
    the single, shared lengthscale and hyperparameter, :math:`s` is the variance.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SumPolynomialSEIso"

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEIso` kernel object.

        :arg varphi: The kernel lengthscale hyperparameter.
        :type varphi: float or NoneType
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float

        :returns: An :class:`SEIso` object
        """
        super().__init__(*args, **kwargs)
        if self.L != 1:
            raise ValueError(
                "L wrong for sharpened cosine kernel (expected {}, got {})".format(
                    1, self.L))
        if self.M != 2:
            raise ValueError(
                "M wrong for sharpened cosine kernel (expected {}, got {})".format(
                    2, self.M))
        if self.variance is None:
            raise ValueError(
                "You must supply a variance for the sharpened cosine kernel "
                "(expected {} type, got {})".format(float, self.variance))
        self.num_hyperparameters = np.size(self.varphi)

    @property
    def _ARD(self):
        return False

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return True

    @property
    def _general(self):
        return False

    def kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        # TODO: may need to reshape
        return (
            self.variance[0] * B.matmul(X_i, X_j, tr_a=True)**self.varphi[0]
            + self.variance[1] * B.exp(
                -self.varphi[1] * B.ew_dists2(
                    X_i.reshape(1, -1), X_j.reshape(1, -1)))[0, 0])

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: the (N,) covariance vector.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance[0] * B.einsum('k, ik -> i', x_new, X)**self.varphi[0] + self.variance[0] * B.exp(-self.varphi[1] * B.ew_dists2(
            X, x_new.reshape(1, -1)))

    def kernel_prior_diagonal(self, X):
        return self.variance[0] * np.ones(np.shape(X)[0]) + self.variance[1] * np.ones(np.shape(X)[0])

    def kernel_diagonal(self, X1, X2):
        """
        Get Gram diagonal efficiently using scipy's distance matrix function.

        :arg X1: (N, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N,) Gram diagonal.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance[0] * B.einsum('ik, ik -> i', X1, X2)**self.varphi[0] + self.variance[1] * B.exp(-self.varphi[1] * B.ew_dists2(X1, X2))

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using MLKernels.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return self.variance[0] * B.matmul(X1, X2, tr_b=True)**self.varphi[0] + self.variance[1] * B.exp(-self.varphi[1] * B.pw_dists2(X1, X2))

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
        a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the
            covariance matrix.
        :rtype: class:`numpy.ndarray`
        """
        return np.zeros((len(X1), len(X1)))

    def kernel_partial_derivative_variance(self, X1, X2):
        """
        # TODO: needs checking/implementing

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return np.array([B.einsum('ik, jk -> ij', X1, X2)**self.varphi[0],  B.exp(-self.varphi[1] * B.pw_dists2(X1, X2))])


class SSSEARDMultinomial(Kernel):
    r"""
    An automatic relevance detection (ARD) and multinomial radial basis function (a.k.a. exponentiated quadratic,
        a.k.a squared exponential (SE)) kernel class.
    .. math::
        K_{k}(x_i, x_j) = s * \exp{- \sum_{d=1}^{D}(\varphi_{kd} * (x_{id} - x_{jd})^2)},

    where :math:`K_{k}(\cdot, \cdot)` is the kernel function for the :math:`k`th class, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi_{kd}` is the lengthscale for the
    :math:`k`th class and :math:`d`th dimension, and :math:`s` is the variance.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SSSEARDMultinomial"

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEARDMultinomial` kernel object.

        :returns: An :class:`SEARDMultinomial` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the general and ARD setting are assumed.
        if self.L <= 1:
            raise ValueError(
                "L wrong for simple kernel (expected {}, got {})".format(
                "more than 1", self.L))
        if self.M <= 1:
            raise ValueError(
                "M wrong for non-ARD kernel (expected {}, got {})".format(
                    "more than 1", self.M))
        if self.variance is None:
            raise ValueError(
                "You must supply a variance for the general kernel "
                "(expected {} type, got {})".format(float, self.variance))
        # In the ARD case (see 2005 paper)
        self.D = self.M
        # In the general setting with one (D, ) hyperparameter for each class
        # case (see 2005 paper bottom of page 4)
        self.K = self.L
        self.num_hyperparameters = np.size(self.varphi)

    @property
    def _ARD(self):
        return True

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return True

    @property
    def _general(self):
        return False

    def kernel(self, k, X_i, X_j):  # TODO: How does this extra argument effect the code? Probably extra outer loops
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j), the class (k) and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return self.variance * np.exp(-1. * np.sum([self.varphi[k, d] * np.power(
            (X_i[d] - X_j[d]), 2) for d in range(self.D)]))

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: (K, N) array of K (N,) covariance vectors.
        :rtype: class:`numpy.ndarray`
        """
        K = self.K  # length of the classes
        N = np.shape(X)[0]
        Cs = np.empty((K, N))
        # The general covariance function has a different lengthscale for each dimension.
        for k in range(K):
            for i in range(N):
                Cs[k, i] = self.kernel(k, X[i], x_new[0])
        return Cs

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        TODO: This is incredibly inefficient. Need to cythonize.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.
        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (K, N1, N2) array of K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs = np.empty((self.K, N1, N2))
        # The general covariance function has a different length scale for each dimension.
        for k in range(self.K):
            # for each x_i
            for i in range(N1):
                for j in range(N2):
                    Cs[k, i, j] = self.kernel(k, X1[i], X2[j])
        return Cs

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_Cs: A (D, K, N1, N2) array of K (N1, N2) partial derivatives of covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        Cs = self.kernel_matrix(X1, X2)  # (K, N1, N2)
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        partial_Cs = np.empty((self.D, self.K, N1, N2))
        # The general covariance function has a different length scale for each dimension.
        for d in range(self.D):
            X1d = X1[:, d].reshape(-1, 1)
            X2d = X2[:, d].reshape(-1, 1)
            # TODO: check this
            partial_Cs[d, :, :, :] = (-1. / self.D) * np.multiply(pow(distance_matrix(X1d, X2d), 2), Cs)
        return partial_Cs


class SEARDMultinomial(Kernel):
    r"""
    An automatic relevance detection (ARD) and multinomial radial basis function (a.k.a. exponentiated quadratic,
        a.k.a squared exponential (SE)) kernel class.
    .. math::
        K_{k}(x_i, x_j) = s * \exp{- \sum_{d=1}^{D}(\varphi_{kd} * (x_{id} - x_{jd})^2)},

    where :math:`K_{k}(\cdot, \cdot)` is the kernel function for the :math:`k`th class, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi_{kd}` is the lengthscale for the
    :math:`k`th class and :math:`d`th dimension, and :math:`s` is the variance.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SEARDMultinomial"

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEARD` kernel object.

        :arg varphi: The kernel lengthscale hyperparameters as an (L, M) numpy array. Note that
            L=K in the most general case, but it is more common to have a single and shared GP kernel over all classes,
            in which case L=1. If set to `None`, then implies the kernel is not a Matern kernel (loosely defined as a
            kernel with a length scale parameter). Default `None`.
        :type varphi: :class:`numpy.ndarray` or float
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float
        :returns: An :class:`SEARD` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the general and ARD setting are assumed.
        if self.L <= 1:
            raise ValueError(
                "L wrong for general kernel (expected {}, got {})".format(
                "more than 1", self.L))
        if self.M <= 1:
            raise ValueError(
                "M wrong for ARD kernel (expected {}, got {})".format(
                    "more than 1", self.M))
        if self.variance is None:
            raise ValueError(
                "You must supply a variance for the general kernel "
                "(expected {} type, got {})".format(float, self.variance))
        # In the ARD case (see 2005 paper).
        self.D = self.M
        # In the general setting with one (D, ) hyperparameter for each class case (see 2005 paper bottom of page 4).
        self.K = self.L
        self.num_hyperparameters = np.size(self.varphi)

    @property
    def _ARD(self):
        return True

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return True

    @property
    def _general(self):
        return True

    def _kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        the class (k) and hyper-parameters.

        For internal use only, since the multplication by the variance (s) has
        been factored out for compute speed.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return np.exp(
            -1. * np.sum(
                np.multiply(self.varphi, np.power(X_i - X_j, 2)),axis=1))

    def kernel(self, k, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j),
        the class (k) and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return self.variance * np.exp(
            -1. * np.power(
                distance.minkowski(X_i, X_j, w=self.varphi[k, :]), 2))

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: (K, N) array of K (N,) covariance vectors.
        :rtype: class:`numpy.ndarray`
        """
        # TODO: this has been tested (scratch 4), but best to make sure it works
        N = np.shape(X)[0]
        Cs = np.empty((self.K, N))
        x = x_new[0]
        for i in range(N):
            Cs[:, i] = self._kernel(X[i], x)
        return self.variance * Cs

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        This is a one of calculation that can't be factorised in the most
        general case, so we don't mind that is has a quadruple nested for loop.
        In less general cases, then scipy.spatial.distance_matrix(x, x) could
        be used.
        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (K, N1, N2) array of K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO: this has been tested (scratch 4), but best to make sure it works
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs = np.empty((self.K, N1, N2))
        for k in range(self.K):
            Cs[k, :, :] = np.exp(
                -1. * np.power(
                    distance.cdist(X1, X2,
                    'minkowski', p=2, w=self.varphi[k, :]), 2))
        return self.variance * Cs

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as
        a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_Cs: A (D, K, N1, N2) array of K (N1, N2) partial
            derivatives of covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        Cs = self.kernel_matrix(X1, X2)  # (K, N1, N2)
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        partial_Cs = np.empty((self.D, self.K, N1, N2))
        # The general covariance function has a different
        # lengthscale for each dimension.
        for d in range(self.D):
            X1d = X1[:, d].reshape(-1, 1)
            X2d = X2[:, d].reshape(-1, 1)
            # TODO: check this
            partial_Cs[d, :, :, :] = (-1. / self.D) * np.multiply(
                pow(distance_matrix(X1d, X2d), 2), Cs)
        return partial_Cs

    def kernel_partial_derivative_variance(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        This is a one of calculation that can't be factorised in the most
        general case, so we don't mind that is has a quadruple nested for loop.
        In less general cases, then scipy.spatial.distance_matrix(x, x) could
        be used.
        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (K, N1, N2) array of K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO: this has been tested (scratch 4), but best to make sure it works
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs = np.empty((self.K, N1, N2))
        for k in range(self.K):
            Cs[k, :, :] = np.exp(-1. * np.power(
                distance.cdist(
                    X1, X2, 'minkowski', p=2, w=self.varphi[k, :]), 2))
        return Cs


class SEARD(Kernel):
    r"""
    An automatic relevance detection (ARD) for a ordinal regression (single
    and shared across all classes) radial basis function (a.k.a. exponentiated
    quadratic, a.k.a squared exponential (SE)) kernel class.
    .. math::
        K(x_i, x_j) = s * \exp{- \sum_{d=1}^{D}(\varphi_{d}
            * (x_{id} - x_{jd})^2)},

    where :math:`K(\cdot, \cdot)` is the kernel function for all the
    :math:`K` classes, :math:`x_{i}` is the data point, :math:`x_{j}` is
    another data point, :math:`\varphi_{d}` is the lengthscale for the
    :math:`d`th dimension, shared across all classes, and :math:`s` is the
    variance.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SEARD"

    def __init__(self, varphi, varphi_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`SEARD` kernel object.

        :arg varphi: The kernel lengthscale hyperparameters as an (L, M) numpy
            array. Note that L=K in the most general case, but it is more
            common to have a single and shared GP kernel over all classes,
            in which case L=1. If set to `None`, then implies the kernel is
            not a Matern kernel (loosely defined as a
            kernel with a lengthscale parameter). Default `None`.
        :type varphi: :class:`numpy.ndarray` or float
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters: :class:`numpy.ndarray` or float

        :returns: An :class:`SEARD` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the ARD setting is assumed. This is not a
        # general_kernel, since the covariance function
        # is shared across classes. 
        if self.L > 1:
            raise ValueError(
                "L wrong for simple kernel (expected {}, got {})".format(
                "1", self.L))
        if self.M <= 1:
            raise ValueError(
                "M wrong for ARD kernel (expected {}, got {})".format(
                    "more than 1", self.M))
        if self.variance is None:
            raise ValueError(
                "`variance` hyperparameter must be provided for the SEIso "
                "kernel class (expected {}, got {})".format(
                    float, type(self.variance)))
        # In the ARD case (see 2005 paper).
        self.D = self.M
        # In the ordinal setting with a single and shared (D, )
        # hyperparameter for each class case.
        self.num_hyperparameters = np.size(varphi)

    @property
    def _ARD(self):
        return True

    @property
    def _stationary(self):
        return True

    @property
    def _Matern(self):
        return True

    @property
    def _general(self):
        return False

    def _kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j), the class (k) and hyper-parameters.

        For internal use only, since the multplication by the variance (s) has been factored out for compute speed.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return np.exp(-1. * np.sum(np.multiply(self.varphi, np.power(X_i - X_j, 2)), axis=1))

    def kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j), and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return self.variance * np.exp(-1. * np.power(distance.minkowski(X_i, X_j, w=self.varphi), 2))

    def kernel_vector(self, x_new, X):
        """
        Get the kernel vector given an input vector (x_new) input matrix (X).

        :arg x_new: The new (1, D) point drawn from the data space.
        :type x_new: :class:`numpy.ndarray`
        :arg X: (N, D) data matrix.
        :type X: class:`numpy.ndarray`
        :return: (K, N) array of K (N,) covariance vectors.
        :rtype: class:`numpy.ndarray`
        """
        # TODO: this has been tested (scratch 4), but best to make sure it works
        N = np.shape(X)[0]
        Cs = np.empty((self.K, N))
        x = x_new[0]
        for i in range(N):
            Cs[:, i] = self._kernel(X[i], x)
        return self.variance * Cs

    def kernel_matrix(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.
        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (K, N1, N2) array of K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO: this has been tested (scratch 4), but best to make sure it works
        return self.variance * np.exp(-1. * np.power(distance.cdist(X1, X2, 'minkowski', p=2, w=self.varphi), 2))

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_Cs: A (D, K, N1, N2) array of K (N1, N2) partial derivatives of covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        C = self.kernel_matrix(X1, X2)  # (N1, N2)
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        partial_C = np.empty((self.D, N1, N2))
        # The general covariance function has a different length scale for each dimension.
        for d in range(self.D):
            X1d = X1[:, d].reshape(-1, 1)
            X2d = X2[:, d].reshape(-1, 1)
            # TODO: check this - unconfident in it.
            partial_C[d, :, :] = (-1. / self.D) * np.multiply(pow(distance_matrix(X1d, X2d), 2), C)
        return partial_C

    def kernel_partial_derivative_variance(self, X1, X2):
        """
        Get Gram matrix efficiently using scipy's distance matrix function.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.
        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (K, N1, N2) array of K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO: this has been tested (scratch 4), but best to make sure it works
        return np.exp(-1. * np.power(distance.cdist(X1, X2, 'minkowski', p=2, w=self.varphi), 2))


class InvalidKernel(Exception):
    """An invalid kernel has been passed to `Approximator` or `Sampler`"""

    def __init__(self, kernel):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`Approximator` or `Sampler`
            as the kernel argument.
        :rtype: :class:`InvalidKernel`
        """
        message = (
            f"{kernel} is not an instance of"
            "probit.kernels.Kernel"
        )

        super().__init__(message)


class KernelLoader(enum.Enum):
    """Factory enum to load kernels.
    """
    linear = Linear
    lab_eq = LabEQ
    lab_sharpened_cosine = LabSharpenedCosine
    se_iso = SEIso
    sum_polynomial_se_iso = SumPolynomialSEIso
    se_ard = SEARD


def load_kernel(
    kernel_string,
    **kwargs):
    """
    Returns a brand new instance of the classifier manager for training.
    Observe that this instance is of no use until it has been trained.
    Input:
        kernel_string (str):    type of model to be loaded. Our interface can currently provide
                                trainable instances for: 'keras'
        model_metadata (str):   absolute path to the file where the model's metadata is going to be
                                saved. This metadata file will contain all the information required
                                to re-load the model later.
        model_kwargs (kwargs):  hyperparameters required to initialise the classification model. For
                                details look at the desired model's constructor.
    Output:
        classifier (ClassifierManager): an instance of a classifier with a standard interface to
                                        be used in our pipeline.
    Raises:
        ValueError: if the classifier type provided is not supported by the interface.
    """
    return KernelLoader[kernel_string].value(
        **kwargs)
