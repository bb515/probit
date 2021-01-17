import numpy as np
from scipy.spatial import distance_matrix, distance
from scipy.stats import expon
from abc import ABC, abstractmethod


class Kernel(ABC):
    """
    Base class for kernels. TODO: cythonise these functions.

    All kernels must define an init method, which may or may not inherit Kernel as a parent class using `super()`.
    All kernels that inherit Kernel define a number of methods that return the kernel value, a vector of kernel values
    (a vector of covariances), or a covariance matrix of kernel values.
    """

    @abstractmethod
    def __init__(self, varphi, s, sigma=None, tau=None):
        """
        Create an :class:`Kernel` object.

        This method should be implemented in every concrete kernel.

        :arg :class:`numpy.ndarray` varphi: The kernel lengthscale hyperparameters as an (N, M) numpy array.
        :arg :class: float s: The kernel scale hyperparameters as a numpy array.
        :arg sigma: The (K, ) array or float or None (location/ scale) hyper-hyper-parameters that define psi prior.
            Not to be confused with `Sigma`, which is a covariance matrix. Default None.
        :type sigma: float or None or :class:`numpy.ndarray`
        :arg tau: The (K, ) array or float or None (location/ scale) hyper-hyper-parameters that define psi prior.
            Default None.
        :type tau: float or None or :class:`numpy.ndarray`.

        :returns: A :class:`Kernel` object
        """
        if ((type(varphi) is list) or
                (type(varphi) is np.ndarray)):
            if np.shape(varphi) == (1,):
                # e.g. [[1]]
                N = 1
                M = 1
            elif np.shape(varphi) == ():
                # e.g. [1]
                N = 1
                M = 1
            elif np.shape(varphi[0]) == (1,):
                # e.g. [[1],[2],[3]]
                N = np.shape(varphi)[0]
                M = 1
            elif np.shape(varphi[0]) == ():
                # e.g. [1, 2, 3]
                N = 1
                M = np.shape(varphi)[0]
            else:
                # e.g. [[1, 2], [3, 4], [5, 6]]
                N = np.shape(varphi)[0]
                M = np.shape(varphi)[1]
        elif ((type(varphi) is float) or
                  (type(varphi) is np.float64)):
            # e.g. 1
            N = 1
            M = 1
            varphi = np.float64(varphi)
            s = np.float64(s)
        else:
            raise TypeError(
                "Type of varphi is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(varphi)))
        self.N = N
        self.M = M
        self.varphi = varphi
        self.s = s
        if sigma is not None:
            if tau is not None:
                self.sigma = sigma
                self.tau = tau
            else:
                raise TypeError(
                    "If a sigma hyperhyperparameter is provided, then a tau hyperhyperparameter must be provided"
                    " (expected {}, got {})".format(np.ndarray, type(tau))
                )
        else:
            self.sigma = None
            self.tau = None

    @abstractmethod
    def kernel(self):
        """
        Return the kernel given two input vectors

        This method should be implemented in every concrete integrator.
        """

    # @abstractmethod
    # def kernel_vector(self):
    #     """
    #     Return the kernel vector given an input matrix and input vectors
    #
    #     This method should be implemented in every concrete integrator.
    #     """

    @abstractmethod
    def kernel_matrix(self):
        """
        Return the kernel given two input vectors

        This method should be implemented in every concrete integrator.
        """


class SEIso(Kernel):
    """
    An isometric kernel class. Inherits the Kernel ABC
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`IsoBinomial` kernel object.

        :returns: An :class:`IsoBinomial` object
        """
        super().__init__(*args, **kwargs)
        self.general_kernel = False
        if self.N != 1:
            raise ValueError('K wrong for binary kernel (expected {}, got {})'.format(1, self.K))
        if self.M != 1:
            raise ValueError('M wrong for binary kernel (expected {}, got {})'.format(1, self.M))
        if self.s is None:
            raise TypeError(
                'You must supply an s for the simple kernel (expected {}, got {})'.format(float, type(self.s)))

    def kernel(self, X_i, X_j):
        """Get the ij'th element of C, given the X_i and X_j, indices and hyper-parameters."""
        return self.s * np.exp(-1. * self.varphi * distance.sqeuclidean(X_i, X_j))

    def kernel_matrix(self, X1, X2):
        """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function.

        :param X1: are the datum.
        :param X2: usually the same as X1 are the datum.
        """
        distance_mat = distance_matrix(X1, X2)
        return np.multiply(self.s, np.exp(-1. * self.varphi * pow(distance_mat, 2)))

    def kernel_vector_matrix(self, x_new, X):
        """
        :param X: are the data.
        :param x_new: is the new object drawn from the feature space.
        :param varphi: is the length scale common to all dimensions and classes.
        :param s: is the vertical scale common to all classes.
        :return: the C_new vector.
        """
        N = np.shape(X)[0]
        X_new = np.tile(x_new, (N, 1))
        # This is probably horribly inefficient
        D = distance.cdist(X_new, X)[0]
        return np.multiply(self.s, np.exp(-1. * self.varphi * pow(D, 2)))

    def kernel_matrix_matrix(self, X_new, X):
        """

        :param X_new: vector of new objects drawn from feature space
        :param X: are the data.
        :return: the C_news vectors, one C_new for each object.
        """

    def kernel_matrices(self, X1, X2, varphis):
        """
        Generate Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :param X1: (N1, D) dimensional numpy.ndarray which holds the feature vectors.
        :param X2: (N2, D) dimensional numpy.ndarray which holds the feature vectors.
        :param varphis: (n_samples, ) dimensional numpy.ndarray which holds the
            covariance hyperparameters.
        :returns Cs: A (n_samples, N1, N2) array of n_samples * K (N1, N2) covariance matrices.
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples


class SEARDMultinomial(Kernel):
    """
    A square exponential (SE) automatic relevance detection (ARD) multinomial kernel class. Inherits the Kernel ABC.
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`EulerCL` integrator object.

        :returns: An :class:`EulerCL` object
        """
        super().__init__(*args, **kwargs)
        self.general_kernel = True
        if self.N <= 1:
            raise ValueError('K wrong for simple kernel (expected {}, got {})'.format('more than 1', self.K))
        if self.M <= 1:
            raise ValueError('M wrong for simple kernel (expected {}, got {})'.format('more than 1', self.M))
        if self.s is None:
            raise TypeError('You must supply an s for the general kernel (expected {}, got {})'.format(
                float, type(self.s)))
        # In the ARD case (see 2005 paper)
        self.D = self.M
        # In the one (D, ) hyperparameter for each class case (see 2005 paper)
        self.K = self.N

    def kernel(self, k, X_i, X_j):  # TODO: How does this extra argument effect the code? Probably extra outer loops
        """Get the ij'th element of C_k, given the X_i and X_j, k."""
        return self.s * np.exp(-1. * np.sum([self.varphi[k, d] * np.power(
            (X_i[d] - X_j[d]), 2) for d in range(self.D)]))

    def kernel_matrix(self, X1, X2):
        """
        Generate Gaussian kernel matrices as a numpy array.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :param X1: (N1, D) dimensional numpy.ndarray which holds the feature vectors.
        :param X2: (N2, D) dimensional numpy.ndarray which holds the feature vectors.
        :param varphi: (K, D) dimensional numpy.ndarray which holds the
            covariance hyperparameters.
        :returns Cs: A (K, N1, N2) array of K (N1, N2) covariance matrices.
        """
        Cs = []
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        # The general covariance function has a different length scale for each dimension.
        for k in range(self.K):
            # for each x_i
            C = -1. * np.ones((N1, N2))
            for i in range(N1):
                for j in range(N2):
                    C[i, j] = self.kernel(k, X1[i], X2[j])
            Cs.append(C)
        return np.array(Cs)

    def kernel_vector_matrix(self, x_new, X):
        """
        Generate Gaussian kernel matrices as a numpy array.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :param x_new: (1, D) dimensional numpy.ndarray of the new feature vector.
        :param X: (N, D) dimensional numpy.ndarray which holds the data feature vectors.
        :param varphi: (K, D) dimensional numpy.ndarray which holds the
            covariance hyperparameters.
        :returns Cs: A (K, N, N) array of K (N, N) covariance matrices.
        """
        Cs_new = []
        K = self.K  # length of the classes
        N = np.shape(X)[0]
        # The general covariance function has a different length scale for each dimension.
        for k in range(K):
            C_new = -1. * np.ones(N)
            for i in range(N):
                C_new[i] = self.kernel(k, X[i], x_new[0])
            Cs_new.append(C_new)
        return np.array(Cs_new)

    def kernel_matrices(self, X1, X2, varphis):
        """
        Generate Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :param X1: (N1, D) dimensional numpy.ndarray which holds the feature vectors.
        :param X2: (N2, D) dimensional numpy.ndarray which holds the feature vectors.
        :param varphis: (n_samples, K, D) dimensional numpy.ndarray which holds the
            covariance hyperparameters.
        :returns Cs: A (n_samples, K, N1, N2) array of n_samples * K (N1, N2) covariance matrices.
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, self.K, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples


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
