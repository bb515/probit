"""Multinomial Probit regression using Gibbs sampling with GP priors."""
import pathlib

from .kernels import Kernel
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from tqdm import trange


# from .utilities import log_heaviside_probit_likelihood


class GibbsClassifier(object):
    """
    A Gibbs classifier.

    This class allows users to define a classification problem, get predictions
    using a exact Bayesian inference.

    For this a Gaussian Process :class:`probit.kernels.Kernel` is required.

    The :meth:`GibbsClassifier.sample` can be used to take samples from the posterior.
    The :meth:`GibbsClassifier.predict` can be  used to make predictions.
    """

    def __init__(self, X, t, sampler=None, kernel=None, binomial=False, write_path=None):
        """
        Create a :class:`GibbsClassifier` object.
        :arg X: The data vector.
        :type X: :class:`numpy.ndarray`
        :param t: The target vector.
        :type t: :class:`numpy.ndarray`
        :arg kernel: The kernel to use, see
            :mod:`probit.kernels` for options.
        :arg bool binomial: If implementation is binomial (True) or multinomial (False).
            Default: False.
        :arg str write_path: Write path for outputs.
        """
        # If no write path was provided, assign it as None so that arrays are not written
        # otherwise, ensure write_path is a Path object
        if write_path is None:
            self.write_path = None
        else:
            self.write_path = pathlib.Path(write_path)

        self.D = np.shape(X)[1]
        self.N = np.shape(X)[0]
        self.X = X  # (N, D)
        self.X_T = X.T
        self.t = t

        if kernel is None:
            # This is linear in the parameters regression
            self.kernel = None
            self.cov = np.linalg.inv(self.X_T @ self.X)  # From Mark's lecture notes
        elif not (isinstance(kernel, Kernel)):
            raise InvalidKernel(kernel)
        else:
            # This is GP priors
            self.kernel = kernel
            self.I = np.eye(self.N)  # Q: this eye is different to np.eye(K). Why use of both?
            self.C = kernel.kernel_matrix(X)  # Q: do varphi and s change in gibbs or VB?
            self.sigma = np.linalg.inv(self.I + self.C)
            self.cov = self.C @ self.sigma

        if np.all(np.mod(t, 1) == 0):
            t = t.astype(int)
        else:
            raise ValueError("t must contain only integer values (got {})".format(t))
        if np.all(t >= 0):
            self.K = int(np.max(t) + 1)  # the number of classes
        else:
            raise ValueError("t must contain only positive integer values (got {})").format(t)

        # Binomial probit regression
        if binomial:
            if t.all() in [0, 1]:
                self.binomial = 1
            else:
                raise ValueError("In the binomial case, t must contain only 1s and/or 0s (got {})".format(t))

    # def predict_gibbs(varphi, s, sigma, X_test, X_train, Y_samples, scalar=None):
    #     if not scalar:
    #         predictive_multinomial_distributions = vector_predict_gibbs(varphi, s, sigma, X_test, X_train, Y_samples)
    #     else:
    #         N_test = np.shape(X_test)[0]
    #         predictive_multinomial_distributions = []
    #         for i in range(N_test):
    #             predictive_multinomial_distributions.append(
    #                 scalar_predict_gibbs(varphi, s, sigma, X_test[i], X_train, Y_samples))
    #     return predictive_multinomial_distributions

    def sample(self, init, steps, first_step=1):
        """Return the Gibbs sampler depending on the implementation."""
        if self.kernel is not None:
            return self.sample_gp(init, steps, first_step=first_step)
        else:
            return self.sample_no_gp(init, steps, first_step=first_step)

    def predict(self):

    def _sample_gp_initiate(self, M_0):
        """Initialise variables for the sample method."""
        K = np.shape(M_0)[1]
        if K != self.K:
            raise ValueError("Shape of axis 0 of M_0 must equal K (the number of classes)"
                             " (expected {}, got {})".format(
                self.K, K))
        I = np.eye(self.K)  # (K, K)
        return I, M_0

    def sample_gp(self, M_0, steps, first_step=1):
        """
        Sampling occurs in blocks over the parameters: Y (auxilliaries) and M.

        :param M_0: (N, K) numpy.ndarray of the initial location of the sampler.
        :type M_0: :class:`np.ndarray`.
        :param n: the number of iterations.
        :param varphi: (K, D) np.array for
        :return: Gibbs samples. The acceptance rate for Gibbs is 1.
        """
        M_samples = []
        Y_samples = []
        I, M = self._sample_gp_initiate(M_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty Y (N, K) matrix to collect y_n samples over
            Y = -1. * np.ones(self.N, self.K)
            for n, m in enumerate(M):  # i in range N
                # Class index, k, is the target class
                k_true = self.t[n]
                # Initiate yi at 0
                y_n = np.zeros(self.K)
                y_n[k_true] = -1.0  # TODO: this is a bit hacky
                # Sample from the cone of the truncated multivariate Gaussian
                # TODO: find out a way to compare yi[k_true] to
                #  all the other values apart from itself, as that is the theory
                while y_n[k_true] < np.max(y_n):
                    # sample Y jointly
                    y_n = multivariate_normal.rvs(mean=m, cov=I)
                # Add sample to the Y vector
                Y[n, :] = y_n[:]
            # Calculate statistics, then sample other conditional
            # Empty M_T (K, N) matrix to collect m_k samples over
            M_T = -1. * np.ones(self.K, self.N)
            for k in range(self.K):
                mean = self.cov @ Y.T[k]
                m_k = multivariate_normal.rvs(mean=mean, cov=self.cov)
                # Add sample to the M vector
                M_T.append(m_k)
                M_T[k, :] = m_k
            M = M_T.T
            M_samples.append(M)
            Y_samples.append(Y)
        return M_samples, Y_samples

    def _sample_no_gp_initiate(self, beta_0):
        """Initialise variables for the sample method."""
        K_plus_1 = np.shape(beta_0)[0]
        if K_plus_1 != self.K + 1:
            raise ValueError("Shape of axis 0 of beta_0 must equal K + 1 (the number of classes plus one)"
                             " (expected {}, got {})".format(
                self.K + 1, K_plus_1))
        I = np.eye(self.K)  # (K, K)
        return I, beta_0

    def sample_no_gp(self, beta_0, steps, first_step=1):
        """
        Take n Gibbs samples.

        :arg beta_0: The initial location of the sampler in parameter space (K + 1, ) ndarray.
        :type beta_0: :class:`numpy.ndarray`
        :arg int n: Number of samples.
        :return: Array of n samples and acceptance rate.
        :rtype: :class:`numpy.ndarray`
        """
        I, beta = self._sample_no_gp_initiate(beta_0)

        beta_samples = []
        Y_samples = []
        for _ in trange(first_step, first_step + steps,
                        desc="Regression Sampler Progress", unit="samples"):
            # Empty Y vector to collect Y_i samples over
            Y = []
            for i, x in enumerate(self.X):
                # Sample from truncated Gaussian depending on t
                if self.t[i] == 1:
                    yi = 0
                    while yi <= 0:
                        yi = norm.rvs(loc=np.dot(beta, x), scale=1)
                else:
                    yi = 0
                    while yi >= 0:
                        yi = norm.rvs(loc=np.dot(beta, x), scale=1)
                # Add sample to the Y vector
                Y.append(yi)

            # Calculate statistics, then sample other conditional
            mean = self.cov @ self.X_T @ np.array(Y)
            beta = multivariate_normal.rvs(mean=mean, cov=self.cov)
            beta_samples.append(beta)
            Y_samples.append(Y)
        beta_samples = np.array(beta_samples, dtype=np.float64)
        Y_samples = np.array(Y_samples, dtype=np.float64)
        return beta_samples, Y_samples

    def _predict_initiate(self, X_new):
        """Initialise variables for prediction."""
        # Deal with the shape of X_new.
        if ((type(X_new) is list) or
                (type(X_new) is np.ndarray)):
            if np.shape(X_new) == (1,):  # e.g. [[1]]
                if self.D != 1:
                    raise ValueError("x_new must be the same dimension as x (expected {}, got {})".format(self.D, 1))
            elif np.shape(X_new) == ():  # e.g. [1]
                if self.D != 1:
                    raise ValueError("x_new must be the same dimension as x (expected {}, got {})".format(self.D, 1))
                X_new = np.array([X_new])
            elif np.shape(X_new[0]) == (1,):  # e.g. [[1],[2],[3]]
                if self.D != 1:
                    raise ValueError("x_new must be the same dimension as x (expected {}, got {})".format(self.D, 1))
            elif np.shape(X_new[0]) == ():  # e.g. [1, 2, 3]
                if self.D == 1:
                    X_new = np.array(X_new).reshape((np.shape(X_new)[0], 1))
                elif self.D == np.shape(X_new)[0]:
                    X_new = np.array([X_new])
                else:
                    raise ValueError("x_new must be the same dimension as x (expected {}, got {})".format(
                        self.D, np.shape(X_new)[0]))
            else:
                # e.g. [[1, 2], [3, 4], [5, 6]]
                D = np.shape(X_new)[1]
                if self.D != D:
                    raise ValueError("x_new must be the same dimension as x (expected {}, got {})".format(self.D, D))
        elif (type(X_new) is float) or (type(X_new) is np.float64):
            # e.g. 1
            if self.D != 1:
                raise ValueError("x_new must be the same dimension as x (expected {}, got {})".format(self.D, 1))
            X_new = np.array([[X_new]])
        else:
            raise TypeError(
                "Type of X_new is not supported "
                "(expected {} or {}, got {})".format(
                    float, np.ndarray, type(X_new)))
        return X_new

    def predict(self, beta_samples, x_new):
        """Make gibbs prediction class of x_new given the beta posterior samples."""
        if self.binomial:  # binomial case
            f = [norm.cdf(np.dot(beta, x_new)) for beta in beta_samples]
            return sum(f) / len(beta_samples)
        else:  # multivariate case
            return None


class InvalidKernel(Exception):
    """An invalid kernel has been passed to `GibbsClassifier`"""

    def __init__(self, kernel):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`GibbsClassifier` as the kernel
            argument.
        :rtype: :class:`InvalidKernel`
        """
        message = (
            f"{kernel} is not an instance of"
            "probit.kernels.Kernel"
        )

        super().__init__(message)
