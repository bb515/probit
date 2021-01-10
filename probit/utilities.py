"""Utility functions."""
import numpy as np
from scipy.stats import norm, expon


def log_heaviside_probit_likelihood(u, t, G):
    """The log(p(t|u)) when t=1 indicates inclass and t=0 indicates outclass."""
    v_sample = np.dot(G, u)
    ones = np.ones(len(v_sample))
    phi = norm.cdf(v_sample)
    one_minus_phi = np.subtract(ones, phi)
    log_one_minus_phi = np.log(one_minus_phi)
    log_phi = np.log(phi)
    log_likelihood = (np.dot(t, log_phi)
                      + np.dot(np.subtract(ones, t), log_one_minus_phi))
    return log_likelihood


def sample_varphis(psi, n_samples, general=True):
    """
    Take n_samples of varphi, given the hyper-hyperparameter psi.

    psi is a rate parameter since, with an uninformative prior (sigma=tau=0), then the posterior mean of Q(psi) is
    psi_tilde = 1. / varphi_tilde. Therefore, by taking the expected value of the prior on varphi ~ Exp(psi_tilde),
    we expect to obtain varphi_tilde = 1. / psi_tilde. We get this if psi_tilde is a rate.

    :arg psi: float (Array) of hyper-hyperparameter(s)
    :type psi: :class:`np.ndarray`
    :arg int n_samples: The number of samples for the importance sample.
    """
    scale = 1. / psi
    if general:
        size = (n_samples, np.shape(psi)[0], np.shape(psi)[1])
    else:
        size = n_samples
    return expon.rvs(scale=scale, size=size)


def sample_U(K, different_across_classes=None):  # TODO: Has been superseded by sample_Us
    if not different_across_classes:
        # Will this induce an unwanted link between the y_nk across k? What effect will it have?
        u = norm.rvs(0, 1, 1)
        U = np.multiply(u, np.ones((K, K)))
    else:
        # This might be a better option as the sample u across classes shouldn't be correlated, does it matter though?
        u = norm.rvs(0, 1, K)
        U = np.tile(u, (K, 1))
        # Needs to be the same across rows, as we sum over the rows
        U = U.T
    return U


def sample_Us(K, n_samples, different_across_classes=None):
    if not different_across_classes:
        # Will this induce an unwanted link between the y_nk across k? What effect will it have?
        us = norm.rvs(0, 1, (n_samples, 1, 1))
        ones = np.ones((n_samples, K, K))
        Us = np.multiply(us, ones)
    else:
        # This might be a better option as the sample u across classes shouldn't be correlated, does it matter though?
        us = norm.rvs(0, 1, (n_samples, K, 1))
        # Needs to be the same across rows, as we sum over the rows
        Us = np.tile(us, (1, 1, K))
    return Us


def matrix_of_differences(m_n, K):  # TODO: superseded by matrix_of_differencess
    """
    Get a matrix of differences of the vector m.

    :arg m: is an (K, 1) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
    :type m: :class:`numpy.ndarray`
    """
    # Find matrix of coefficients
    Lambda = np.tile(m_n, (K, 1))
    Lambda_T = Lambda.T
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    difference = Lambda_T - Lambda
    return difference


def matrix_of_differencess(M, K, N_test):
    """
    Get an array of matrix of differences of the vectors m_ns.

    :arg M: An (N_test, K) array filled with e.g. m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object. Or in the VB implementation, this function  is used for M_tilde (N, K).
    :type M: :class:`numpy.ndarray`
    """
    # Find the matrix of coefficients
    M = M.reshape((N_test, K, 1))
    # Lambdas is an (n_test, K, K) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambda_Ts = np.tile(M, (1, 1, K))
    Lambdas = Lambda_Ts.transpose((0, 2, 1))
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    return np.subtract(Lambda_Ts, Lambdas)  # (N_test, K, K)
