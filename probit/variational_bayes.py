from .kernels import Kernel
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal, expon
from .utilities import (
    sample_U, function_u3, function_u2, sample_varphi, samples_varphi)


class VariationBayesClassifier(object):
    """
    A Variational Bayes classifier.

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference.

    For this a Gaussian Process :class:`probit.kernels.Kernel` is required.

    The :meth:`VariationalBayesClassifier.` can be used to converge to the posterior.
    The :meth:`VariationalBayes.predict` can be  used to make predictions.
    The :meth: `VariationalBayes.log_likelihood` can be used to get the log likelihood of VB approximation
        given the hyperparameters.
    """

    def __init__(self, kernel, X, t, binomial=0, write_path=None):
        """
        Create a :class:`GibbsClassifier` object.
        :arg kernel: The kernel to use, see
            :mod:`probit.kernels` for options .
        :arg X: The data vector.
        :type X: :class:`numpy.ndarray`
        :param t: The target vector.
        :type t: :class:`numpy.ndarray`
        """
        if not isinstance(kernel, Kernel):
            raise InvalidKernel(kernel)
        else:
            self.kernel = kernel

        # If no write path was provided, assign it as None so that arrays are not written
        # otherwise, ensure write_path is a Path object
        if write_path is None:
            self.write_path = None
        else:
            self.write_path = pathlib.Path(write_path)

        self.X = X
        self.X_T = X.T
        # self.cov = np.linalg.inv(self.X_T @ self.X)
        self.N = np.shape(X)[0]
        self.I = np.eye(N)
        self.mean_0 = np.zeros(N)

        if np.all(np.mod(t, 1) == 0):
            t = t.astype(int)
        else:
            raise ValueError("t must contain only integer values (got {})".format(t))

        # Binomial probit regression
        if binomial:
            if np.all(t in [0, 1]):
                self.binomial = 1
            else:
                raise ValueError("In the binomial case, t must contain only 1s and/or 0s (got {})".format(t))

    def ws_general_unnormalised(self, psi_tilde, n_samples, M_tilde):
        """
        Return the w values of the sample.
        TODO
        :arg psi_tilde: Posterior mean estimate of psi
        :arg M_tilde: ?
        """
        # Draw from varphi
        varphi_samples = samples_varphi(psi_tilde)
        for varphi_sample in varphi_samples:
            self.kernel.varphi = varphi_sample
            Cs = self.kernel.kernel_matrix(self.X)
            M_tilde = m_tilde[K]
            for i, C in enumerate(Cs):
                normal_pdf = multivariate_normal.pdf(M_tilde[i], mean=self.mean_0, cov=C)
        return normal_pdf

    def varphi_tilde(self, varphi_samples):
        """Return the posterior mean estimate of varphi via importance sampling."""
        sum = 0
        # TODO: define n_samples in class scope
        n_samples = np.shape(varphi_samples)[0]
        for i in range(n_samples):
            varphi_sample = varphi_samples[i]
            sum += varphi_sample * w(varphi_sample)
        return sum

    def M_tilde(self, kernel, Y_tilde, varphi_tilde, X):
        """Return the posterior mean estimate of M."""
        # Update the varphi with new values
        self.kernel.varphi = varphi_tilde
        # calculate updated C and sigma
        C = self.kernel.kernel_matrix(self.X)
        I = np.eye(self.N)
        sigma = np.linalg.inv(I + C)
        return C @ sigma @ Y_tilde

    def Y_tilde(self, M_tilde):
        """Calculate y_tilde elements as defined on page 9 of the paper."""
        N, K = np.shape(Y_tilde) # TODO I mean M_tilde?
        Y_tilde = -1. * np.ones((N, K))
        # TODO: I can vectorise expectation_p_m with tensors later.
        for i in range(N):
            m_tilde_n = M_tilde[i, :]
            t_n = np.argmax(m_tilde_n)
            expectation_3 = expectation_p_m(function_u3, m_tilde_n, t_n, n_samples=1000)
            expectation_2 = expectation_p_m(function_u2, m_tilde_n, t_n, n_samples=1000)
            # Equation 5
            y_tilde_n = m_tilde_n - np.divide(expectation_2, expectation_3)
            # Equation 6 follows
            # This part of the differences sum must be 0, since sum is over j \neq i
            y_tilde_n[t_n] = m_tilde_n[t_n]
            diff_sum = np.sum(y_tilde_n - m_tilde_n)
            y_tilde_n[t_n] = m_tilde_n[t_n] - diff_sum
            Y_tilde[i, :] = y_tilde_n
        return Y_tilde

    def expectation_p_m(function, m_n, t_n, n_samples):
        """
        m is an (K, ) np.ndarray filled with m_k^{new, s} where s is the sample, and k is the class indicator.

        function is a numpy function. e,g, function_u1(difference, U) which is the numerator of eq () and
        function_u2(difference, U) which is the denominator of eq ()
        """
        # Factored out calculations
        difference = matrix_of_differences(m_n)
        t_n = np.argmax(m_n)
        vector_difference = difference[:, t_n]
        K = len(m_n)
        # Take samples
        samples = []
        for i in range(n_samples):
            U = sample_U(K, different_across_classes=1)
            function_eval = function(difference, vector_difference, U, t_n, K)
            samples.append(function_eval)

        distribution_over_classes = 1 / n_samples * np.sum(samples, axis=0)
        return(distribution_over_classes)

    def expn_u_M_tilde(self, M_tilde, n_samples):
        """
        Return a sample estimate of a function of the r.v. u ~ N(0, 1)

        :param function: a function of u
        """

        def function(u, M_tilde):
            norm.pdf(u, loc=M)

        sum = 0
        for i in range(n_samples):
            sum += function(u, M_tilde)
        return sum / n_samples

    def M(self, sigma, Y_tilde):
        """Q(M) where Y_tilde is the expectation of Y with respect to the posterior component over Y."""
        M_tilde = sigma @ Y_tilde
        # The approximate posterior component Q(M) is a product of normal distributions over the classes
        # But how do I translate this function into code? Must need to take an expectation wrt to it
        return M_tilde

    def expn_M(self, sigma, m_tilde, n_samples):
        # Draw samples from the random variable, M ~ Q(M)
        # Use M to calculate f(M)
        # Take monte carlo estimate
        K = np.shape(m_tilde)[0]
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
