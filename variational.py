"""Variational approximation."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
from utilities import sample_U, function_u3, function_u2


def Y_tilde(M_tilde)
    """Calculate y_tilde elements as defined on page 9 of the paper."""
    N, K = np.shape(Y_tilde)
    Y_tilde = -1. * np.ones((N, K))
    N = np.shape(Y_tilde)[0]
    # TODO: I can vectorise expectation_p_m with tensors later.
    for i in range(N):
        m_tilde_n = M_tilde[i, :]
        tn = np.argmax(m_tilde_n)
        expectation_3 = expectation_p_m(function_u3, m_tilde_n, t_n, n_samples=1000)
        expectation_2 = expectation_p_m(function_u2, m_tilde_n, t_n, n_samples=1000)
        # Equation 5
        y_tilde_n = m_n - np.divide(expectation_2, expectation_3)
        # Equation 6 follows
        y_tilde_n[]
        Y_tilde[i, :] = y_tilde_n


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


def expn_u_M_tilde(M_tilde, n_samples):
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



def M(sigma, Y_tilde):
    """Q(M) where Y_tilde is the expectation of Y with respect to the posterior component over Y."""
    M_tilde = sigma @ Y_tilde
    # The approximate posterior component Q(M) is a product of normal distributions over the classes
    # But how do I translate this function into code? Must need to take an expectation wrt to it
    return M_tilde


def expn_M(sigma, m_tilde, n_samples):
    # Draw samples from the random variable, M ~ Q(M)
    # Use M to calculate f(M)
    # Take monte carlo estimate
    K = np.shape(m_tilde)[0]
    return None


def Y_tilde(M_tilde):
    """
    This calculates the expectations of Y wrt Q(Y) and is equations 5 and 6
    :param M_tilde: (N, K)
    :return Y_tilde: (N, K)
    """

    None



