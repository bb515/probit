import numpy as np
from scipy.spatial import distance_matrix, distance
from abc import ABC, abstractmethod


class Kernel(ABC):
    """
    Base class for kernels. TODO: cythonise these functions - or make C bindings.

    All kernels must define an init method, which may or may not inherit Kernel as a parent class using `super()`.
    All kernels that inherit Kernel define a number of methods that return the kernel value, a vector of kernel values
    (a vector of covariances), or a covariance matrix of kernel values.
    """

    @abstractmethod
    def __init__(self, varphi=None, scale=1.0, sigma=None, tau=None):
        """
        Create an :class:`Kernel` object.

        This method should be implemented in every concrete kernel. Initiating a kernel should be a very cheap
        operation.

        :arg varphi: The kernel lengthscale hyperparameters as an (L, M) numpy array. Note that
            L=K in the most general case, but it is more common to have a single and shared GP kernel over all classes,
            in which case L=1. If set to `None`, then implies the kernel is not a Matern kernel (loosely defined as a
            kernel with a length scale parameter). Default `None`.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float scale: The kernel scale hyperparameters as a numpy array. Default 1.0.
        :arg sigma: The (K, ) array or float or None (location/ scale) hyper-hyper-parameters that define psi prior.
            Not to be confused with `Sigma`, which is a covariance matrix. Default None.
        :type sigma: float or :class:`numpy.ndarray` or None
        :arg tau: The (K, ) array or float or None (location/ scale) hyper-hyper-parameters that define psi prior.
            Default None.
        :type tau: float or :class:`numpy.ndarray` or None

        :returns: A :class:`Kernel` object
        """
        # Boolean field that if `True` then the kernel has Automatic-relavance-detection (ARD) capabilities, if `False`,
        # then there is a shared and single lengthscale across all data dimensions. Default `False`.
        self.ARD_kernel = False
        # Boolean field that if `True` then the kernel has a unique kernel for each and every class, if `False` then
        # there is a single and shared kernel for each class. Default `False`.
        self.general_kernel = False
        # Initialise as Matern type kernel (loosely defined here as a kernel with a length scale)
        self.Matern_kernel = True
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
            varphi = np.float64(varphi)
        elif varphi is None:
            # this is not a Gaussian kernel
            self.Matern_kernel = False
        else:
            raise TypeError(
                "Type of varphi is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(varphi)))
        if self.Matern_kernel is True:
            self.L = L
            self.M = M
        self.varphi = varphi
        scale = np.float64(scale)
        self.scale = scale
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

    @abstractmethod
    def kernel_matrices(self):
        """
        Return samples of the Gram matrix given two input matrices, and random samples of the hyperparameters.

        This method should be implemented in every concrete kernel.
        """

    def hyperparameter_update(self, varphi=None, scale=None, sigma=None, tau=None):
        if varphi is not None:
            # Update
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
                varphi = np.float64(varphi)
            else:
                raise TypeError(
                    "Type of varphi is not supported "
                    "(expected {} or {}, got {})".format(
                        float, np.ndarray, type(varphi)))
            self.L = L
            self.M = M
            self.varphi = varphi
        if scale is not None:
            # Update scale
            self.scale = scale
        if sigma is not None:
            # Update sigma
            self.sigma = sigma
        if tau is not None:
            # Update tau
            self.tau = tau
        if bool(self.sigma) != bool(self.tau):
            raise TypeError(
                "If a sigma hyperhyperparameter is provided, then a tau hyperhyperparameter must be provided"
                " (expected {}, got {})".format(np.ndarray, type(tau))
            )
        return 0

    def distance_mat(self, X1, X2):
        """
        Return a distance matrix using scipy spatial.

        :arg X1: The (N1, D) input array for the distance matrix.
        :type X1: :class:`numpy.ndarray`
        :arg X2: The (N2, D) input array for the distance matrix.
        :type X2: :class:`numpy.ndarray` or float
        :return: Euclidean distance matrix (N1, N2).
        :rtype: :class:`numpy.ndarray`.
        """
        # Case that D = 1
        if len(np.shape(X1)) == 1:
            X1 = X1.reshape(-1, 1)
            X2 = X2.reshape(-1, 1)
        return distance_matrix(X1, X2)


class Linear(Kernel):
    r"""
    A linear kernel class.

    .. math::
        K(x_i, x_j) = s * x_{i}^{T} x_{j} + c,

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`s` is the scale and :math:`c` is the intercept.
    """
    def __init__(self, intercept=0.0, *args, **kwargs):
        """
        Create an :class:`Linear` kernel object.

        :returns: An :class:`Linear` object
        """
        super().__init__(*args, **kwargs)
        if self.Matern_kernel is True:
            raise ValueError('Lengthscale was supplied, but this is a Linear kernel. (expected {}, got {})'.format(
                None, self.varphi
            ))
        # For this kernel, the shared and single kernel for each class (i.e. non general) and single lengthscale across
        # all data dims (i.e. non-ARD) is assumed.
        self.ARD_kernel = False
        self.general_kernel = False
        self.intercept = intercept

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
        return self.scale * X_j.T @ X_i + self.intercept

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
        return self.scale * np.einsum('ij, j -> i', X, x_new[0]) + self.intercept

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
        return self.scale * np.einsum('ik, jk -> ij', X1, X2) + self.intercept

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :arg varphis: (n_samples, K, D) array of hyperparameter samples.
        :type varphis: class:`numpy.ndarray`
        :return: Cs_samples (n_samples, K, N1, N2) array of n_samples * K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO check this.
        return np.einsum('ik, jk -> ij', X1, X2)

    def kernel_partial_derivative_s(self, X1, X2):
        """
        Get Gram matrix efficiently using numpy's einsum function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return np.einsum('ik, jk -> ij', X1, X2)



class Polynomial(Kernel):
    r"""
    A polynomial kernel class.

    .. math::
        K(x_i, x_j) = s * (x_{i}^{T} x_{j} + c)^d,

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`d` is the order, :math:`s` is the scale and
    :math:`c` is the intercept.
    """
    def __init__(self, intercept=0.0, order=2.0, *args, **kwargs):
        """
        Create an :class:`Polynomial` kernel object.

        :arg float intercept: Intercept of the polynomial kernel. When intercept=0.0, the kernel is called homogeneous.
            Default 0.0.
        :arg float order: Order of the polynomial kernel. When order=2, the kernel is a quadratic kernel. Default 2.
        :returns: An :class:`Polynomial` object
        """
        super().__init__(*args, **kwargs)
        if self.Matern_kernel is True:
            raise ValueError('Lengthscale was supplied, but this is a Polynomial kernel. (expected {}, got {})'.format(
                None, self.varphi
            ))
        # For this kernel, the shared and single kernel for each class (i.e. non general) and single lengthscale across
        # all data dims (i.e. non-ARD) is assumed.
        self.ARD_kernel = False
        self.general_kernel = False
        self.intercept = intercept
        self.order = order

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
        return self.scale * pow((X_i.T @ X_j + self.intercept), self.order)

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
        return self.scale * pow(np.einsum('ij, j -> i', X, x_new[0]) + self.intercept, self.order)

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
        return self.scale * pow(np.einsum('ik, jk -> ij', X1, X2) + self.intercept, self.order)

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :arg varphis: (n_samples, K, D) array of hyperparameter samples.
        :type varphis: class:`numpy.ndarray`
        :return: Cs_samples (n_samples, K, N1, N2) array of n_samples * K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the covariance matrices.
        :rtype: class:`numpy.ndarray`
        """
        # TODO check this.
        return pow(np.einsum('ik, jk -> ij', X1, X2) + self.intercept, self.order)

    def kernel_partial_derivative_s(self, X1, X2):
        """
        Get Gram matrix efficiently using numpy's einsum function.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :return: (N1, N2) Gram matrix.
        :rtype: class:`numpy.ndarray`
        """
        return pow(np.einsum('ik, jk -> ij', X1, X2) + self.intercept, self.order)


class SEIso(Kernel):
    r"""
    An isometric radial basis function (a.k.a. exponentiated quadratic, a.k.a squared exponential) kernel class.

    .. math::
        K(x_i, x_j) = s * \exp{- \varphi * \norm{x_{i} - x_{j}}^2},

    where :math:`K(\cdot, \cdot)` is the kernel function, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi` is the single, shared lengthscale and
    hyperparameter, :math:`s` is the scale.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEIso` kernel object.

        :returns: An :class:`SEIso` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the shared and single kernel for each class (i.e. non general) and single lengthscale across
        # all data dims (i.e. non ARD) is assumed.
        self.ARD_kernel = False
        self.general_kernel = False
        if self.L != 1:
            raise ValueError('L wrong for simple kernel (expected {}, got {})'.format(1, self.L))
        if self.M != 1:
            raise ValueError('M wrong for non-ARD kernel (expected {}, got {})'.format(1, self.M))
        if self.scale is None:
            raise TypeError(
                'You must supply a scale for the simple kernel (expected {}, got {})'.format(float, type(self.scale)))

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
        return self.scale * np.exp(-1. * self.varphi * distance.sqeuclidean(X_i, X_j))

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
        N = np.shape(X)[0]
        X_new = np.tile(x_new, (N, 1))
        # This is probably horribly inefficient
        D = distance.cdist(X_new, X)[0]
        return np.multiply(self.scale, np.exp(-1. * self.varphi * D**2))

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
        return np.multiply(self.scale, np.exp(-1. * self.varphi * distance_mat**2))

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :arg varphis: (n_samples,) array of hyperparameter samples.
        :type varphis: class:`numpy.ndarray`
        :return: Cs_samples (n_samples, N1, N2) array of n_samples * (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

    def kernel_partial_derivative_varphi(self, X1, X2):
        """
        Get partial derivative with respect to lengthscale hyperparameters as a numpy array.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :returns partial_C: A (N1, N2) array of the partial derivative of the covariance matrix.
        :rtype: class:`numpy.ndarray`
        """
        D = np.shape(X1)[1]
        C = self.kernel_matrix(X1, X2)  # (N1, N2)
        # The general covariance function has a different length scale for each dimension.
        # TODO check this.
        partial_C = (-1./D) * np.multiply(pow(distance_matrix(X1, X2), 2), C)  # TODO: reason for this dimension?
        return partial_C

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
        return np.exp(-1. * self.varphi * distance_mat**2)


class SSSEARDMultinomial(Kernel):
    r"""
    An automatic relevance detection (ARD) and multinomial radial basis function (a.k.a. exponentiated quadratic,
        a.k.a squared exponential (SE)) kernel class.
    .. math::
        K_{k}(x_i, x_j) = s * \exp{- \sum_{d=1}^{D}(\varphi_{kd} * (x_{id} - x_{jd})^2)},

    where :math:`K_{k}(\cdot, \cdot)` is the kernel function for the :math:`k`th class, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi_{kd}` is the lengthscale for the
    :math:`k`th class and :math:`d`th dimension, and :math:`s` is the scale.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEARDMultinomial` kernel object.

        :returns: An :class:`SEARDMultinomial` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the general and ARD setting are assumed.
        self.ARD_kernel = True
        self.general_kernel = True
        if self.L <= 1:
            raise ValueError('L wrong for simple kernel (expected {}, got {})'.format(
                'more than 1', self.L))
        if self.M <= 1:
            raise ValueError('M wrong for non-ARD kernel (expected {}, got {})'.format('more than 1', self.M))
        if self.scale is None:
            raise TypeError('You must supply a scale for the general kernel (expected {}, got {})'.format(
                float, type(self.scale)))
        # In the ARD case (see 2005 paper)
        self.D = self.M
        # In the general setting with one (D, ) hyperparameter for each class case (see 2005 paper bottom of page 4)
        self.K = self.L

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
        return self.scale * np.exp(-1. * np.sum([self.varphi[k, d] * np.power(
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
        # The general covariance function has a different length scale for each dimension.
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

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrix samples for corresponding varphi samples, `varphis`.

        :arg X1: (N1, D) dimensional numpy.ndarray which holds the feature vectors.
        :arg X2: (N2, D) dimensional numpy.ndarray which holds the feature vectors.
        :arg varphis: (n_samples, K, D) dimensional numpy.ndarray which holds the
            covariance hyperparameters.
        :return Cs_samples: (n_samples, K, N1, N2) array of n_samples * K (N1, N2) covariance matrices.
        :rtype: :class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, self.K, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

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
    :math:`k`th class and :math:`d`th dimension, and :math:`s` is the scale.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEARD` kernel object.

        :returns: An :class:`SEARD` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the general and ARD setting are assumed.
        self.ARD_kernel = True
        self.general_kernel = True
        if self.L <= 1:
            raise ValueError('L wrong for general kernel (expected {}, got {})'.format(
                'more than 1', self.L))
        if self.M <= 1:
            raise ValueError('M wrong for ARD kernel (expected {}, got {})'.format('more than 1', self.M))
        if self.scale is None:
            raise TypeError('You must supply a scale for the general kernel (expected {}, got {})'.format(
                float, type(self.scale)))
        # In the ARD case (see 2005 paper).
        self.D = self.M
        # In the general setting with one (D, ) hyperparameter for each class case (see 2005 paper bottom of page 4).
        self.K = self.L

    def _kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j), the class (k) and hyper-parameters.

        For internal use only, since the multplication by the scale (s) has been factored out for compute speed.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return np.exp(-1. * np.sum(np.multiply(self.varphi, np.power(X_i - X_j, 2)), axis=1))

    def kernel(self, k, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j), the class (k) and hyper-parameters.

        :arg X_i: (D, ) data point.
        :type X_i: :class:`numpy.ndarray`
        :arg X_j: (D, ) data point.
        :type X_j: :class:`numpy.ndarray`
        :returns: ij'th element of the Gram matrix.
        :rtype: float
        """
        return self.scale * np.exp(-1. * np.power(distance.minkowski(X_i, X_j, w=self.varphi[k, :]), 2))

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
        return self.scale * Cs

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
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs = np.empty((self.K, N1, N2))
        for k in range(self.K):
            Cs[k, :, :] = np.exp(-1. * np.power(distance.cdist(X1, X2, 'minkowski', p=2, w=self.varphi[k, :]), 2))
        return self.scale * Cs

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :arg varphis: (n_samples, K, D) array of hyperparameter samples.
        :type varphis: class:`numpy.ndarray`
        :return: Cs_samples (n_samples, K, N1, N2) array of n_samples * K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, self.K, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

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

    def kernel_partial_derivative_s(self, X1, X2):
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
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs = np.empty((self.K, N1, N2))
        for k in range(self.K):
            Cs[k, :, :] = np.exp(-1. * np.power(distance.cdist(X1, X2, 'minkowski', p=2, w=self.varphi[k, :]), 2))
        return Cs


class SEARD(Kernel):
    r"""
    An automatic relevance detection (ARD) for a ordinal regression (singlr and shared across all classes) radial basis
        function (a.k.a. exponentiated quadratic, a.k.a squared exponential (SE)) kernel class.
    .. math::
        K(x_i, x_j) = s * \exp{- \sum_{d=1}^{D}(\varphi_{d} * (x_{id} - x_{jd})^2)},

    where :math:`K(\cdot, \cdot)` is the kernel function for all the :math:`K` classes, :math:`x_{i}` is
    the data point, :math:`x_{j}` is another data point, :math:`\varphi_{d}` is the lengthscale for the
    :math:`d`th dimension, shared across all classes, and :math:`s` is the scale.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`SEARD` kernel object.

        :returns: An :class:`SEARD` object
        """
        super().__init__(*args, **kwargs)
        # For this kernel, the ARD setting is assumed. This is not a general_kernel, since the covariance function
        # is shared across classes. 
        self.ARD_kernel = True
        self.general_kernel = False
        if self.L > 1:
            raise ValueError('L wrong for simple kernel (expected {}, got {})'.format(
                '1', self.L))
        if self.M <= 1:
            raise ValueError('M wrong for ARD kernel (expected {}, got {})'.format('more than 1', self.M))
        if self.scale is None:
            raise TypeError('You must supply a scale for the general kernel (expected {}, got {})'.format(
                float, type(self.scale)))
        # In the ARD case (see 2005 paper).
        self.D = self.M
        # In the ordinal setting with a single and shared (D, ) hyperparameter for each class case.

    def _kernel(self, X_i, X_j):
        """
        Get the ij'th element of the Gram matrix, given the data (X_i and X_j), the class (k) and hyper-parameters.

        For internal use only, since the multplication by the scale (s) has been factored out for compute speed.

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
        return self.scale * np.exp(-1. * np.power(distance.minkowski(X_i, X_j, w=self.varphi), 2))

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
        return self.scale * Cs

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
        return self.scale * np.exp(-1. * np.power(distance.cdist(X1, X2, 'minkowski', p=2, w=self.varphi), 2))

    def kernel_matrices(self, X1, X2, varphis):
        """
        Get Gaussian kernel matrices for varphi samples, varphis, as an array of numpy arrays.

        :arg X1: (N1, D) data matrix.
        :type X1: class:`numpy.ndarray`
        :arg X2: (N2, D) data matrix. Can be the same as X1.
        :type X2: class:`numpy.ndarray`
        :arg varphis: (n_samples, K, D) array of hyperparameter samples.
        :type varphis: class:`numpy.ndarray`
        :return: Cs_samples (n_samples, K, N1, N2) array of n_samples * K (N1, N2) Gram matrices.
        :rtype: class:`numpy.ndarray`
        """
        n_samples = np.shape(varphis)[0]
        N1 = np.shape(X1)[0]
        N2 = np.shape(X2)[0]
        Cs_samples = np.empty((n_samples, N1, N2))
        for i, varphi in enumerate(varphis):
            self.varphi = varphi
            Cs_samples[i, :, :] = self.kernel_matrix(X1, X2)
        return Cs_samples

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

     def kernel_partial_derivative_s(self, X1, X2):
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
