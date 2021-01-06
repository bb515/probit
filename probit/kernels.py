import numpy as np
from scipy.spatial import distance_matrix, distance
from abc import ABC, abstractmethod


class Kernel(ABC):
    """
    Base class for kernels. TODO: cythonise these functions.

    All kernels must define an init method, which may or may not inherit Kernel as a parent class using `super()`.
    All kernels that inherit Kernel define a number of methods that return the kernel value, a vector of kernel values
    (a vector of covariances), or a covariance matrix of kernel values.
    """

    @abstractmethod
    def __init__(self, varphi, s=None):
        """
        Create an :class:`Kernel` object.

        This method should be implemented in every concrete kernel.

        :arg :class:`numpy.ndarray` varphi: The kernel lengthscale hyperparameters as an (K, D) numpy array.
        :arg :class:`numpy.ndarray` or None s: The kernel scale hyperparameters as an (K, D) numpy array.

        :returns: A :class:`Kernel` object
        """
        if ((type(varphi) is list) or
                (type(varphi) is np.ndarray)):
            if s:
                if np.shape(varphi) != np.shape(s):
                    raise ValueError(
                        "The shape of varphi must be equal to the shape "
                        "of s (expected {}, got {})".format(
                            np.shape(varphi),
                            np.shape(s)))
            else:
                if np.shape(varphi) == (1,):
                    # e.g. [[1]]
                    K = 1
                    D = 1
                elif np.shape(varphi) == ():
                    # e.g. [1]
                    K = 1
                    D = 1
                elif np.shape(varphi[0]) == (1,):
                    # e.g. [[1],[2],[3]]
                    K = np.shape(varphi)[0]
                    D = 1
                elif np.shape(varphi[0]) == ():
                    # e.g. [1, 2, 3]
                    K = 1
                    D = np.shape(varphi)[0]
                else:
                    # e.g. [[1, 2], [3, 4], [5, 6]]
                    K = np.shape(varphi)[0]
                    D = np.shape(varphi)[1]
        elif ((type(varphi) is float) or
                  (type(varphi) is np.float64)):
            # e.g. 1
            K = 1
            D = 1
            varphi = np.float64(varphi)
            s = np.float64(s)
        else:
            raise TypeError(
                "Type of varphi is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(bond_stiffness)))
        self.K = K
        self.D = D
        self.varphi = varphi
        self.s = s


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


class Binary(Kernel):
    """
    A binary kernel class. Inherits the Kernel ABC
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`EulerCL` integrator object.

        :returns: An :class:`EulerCL` object
        """
        super().__init__(*args, **kwargs)
        if self.K != 1:
            raise ValueError('K wrong for simple kernel (expected {}, got {})'.format(1, K))
        if self.D != 1:
            raise ValueError('D wrong for simple kernel (expected {}, got {})'.format(1, D))
        if self.s == None:
            raise TypeError('You must supply an s for the simple kernel (expected {}, got {})'.format(float, type(s)))

    def kernel(self, X_i, X_j):
        """Get the ij'th element of C, given the X_i and X_j, indices and hyper-parameters."""
        sum = 0
        for d in range(self.D):
            sum += pow((X_i[d] - X_j[d]), 2)
        sum *= self.varphi
        C_ij = np.exp(-1. * sum)
        return C_ij

    def kernel_matrix(self, X):
        """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function.

        :param X: are the datum.
        """
        distance_mat = distance_matrix(X, X)
        return np.multiply(self.s, np.exp(-1. * self.varphi * pow(distance_mat, 2)))

    def kernel_vector_matrix(self, x_new, X):
        """
        :param X: are the objects drawn from feature space.
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


class Multivariate(Kernel):
    """
    A multivariate kernel class. Inherits the Kernel ABC
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`EulerCL` integrator object.

        :returns: An :class:`EulerCL` object
        """
        super().__init__(*args, **kwargs)
        if self.K <= 1:
            raise ValueError('K wrong for simple kernel (expected {}, got {})'.format('more than 1', K))
        if self.D <= 1:
            raise ValueError('D wrong for simple kernel (expected {}, got {})'.format('more than 1', D))
        if self.s != None:
            raise TypeError('You must not supply an s for the general kernel (expected {}, got {})'.format(
                None, type(s)))

    def kernel(self, k, X_i, X_j):
        """Get the ij'th element of C, given the X_i and X_j, indices and hyper-parameters."""
        sum = 0
        for d in range(self.D):
            sum += self.varphi[k, d] * pow((X_i[d] - X_j[d]), 2)
        C_ij = np.exp(-1. * sum)
        return C_ij


    def kernel_matrix(self, X):
        """Generate Gaussian kernel matrices as a numpy array.

        This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
        quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

        e.g.
        for k in range(K):
            Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

        :param X: (N, D) dimensional numpy.ndarray which holds the feature vectors.
        :param varphi: (K, D) dimensional numpy.ndarray which holds the
            covariance hyperparameters.
        :returns Cs: A (K, N, N) array of K (N, N) covariance matrices.
        """
        Cs = []
        K = self.K
        N = np.shape(X)[0]
        # The general covariance function has a different length scale for each dimension.
        for k in range(K):
            # for each x_i
            C = -1.* np.ones((N, N))
            for i in range(N):
                for j in range(N):
                    C[i, j] = self.kernel(k, X[i], X[j])
            Cs.append(C)

        return Cs


    def kernel_vector_matrix(self, x_new, X):
        """Generate Gaussian kernel matrices as a numpy array.

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
        return Cs_new


