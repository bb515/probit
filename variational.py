"""Variational approximation."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal


def M(sigma, Y_tilde):
    """Q(M) where Y_tilde is the expectation of Y with respect to the posterior component over Y."""
    M_tilde = sigma @ Y_tilde
    # The approximate posterior component Q(M) is a product of normal distributions over the classes
    # But how do I translate this function into code? Must need to take an expectation wrt to it
    return None


def expn_M(sigma, m_tilde, n_samples):
    # Draw samples from the random variable, M ~ Q(M)
    # Use M to calculate f(M)
    # Take monte carlo estimate
    K = np.shape(m_tilde)[0]
    return None


def expn_u_M_tilde(function, M_tilde, n_samples):
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


def Y_tilde(M_tilde):
    """
    This calculates the expectations of Y wrt Q(Y) and is equations 5 and 6
    :param M_tilde: (N, K)
    :return Y_tilde: (N, K)
    """

    None



