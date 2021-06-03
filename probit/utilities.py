"""Utility functions."""
import numpy as np
from scipy.stats import norm, expon


def vectorised_multiclass_unnormalised_log_multivariate_normal_pdf(Ms, mean=None, covs=None):
    """
    Evaluate a vector of log multivariate normal pdf in a numerically stable way.

    :arg Ms: (N, K)
    :arg int n_samples: n_samples
    :arg covs: (n_samples, K, N, N)
    :return: (n_samples, K)
    """
    if covs is None:
        raise ValueError("Must provide sample covariance matrices ({} was provided)".format(covs))
    if mean is not None:
        Ms = Ms - mean
    Ls = np.linalg.cholesky(covs)
    L_invs = np.linalg.inv(Ls)
    K_invs = L_invs.transpose((0, 1, 3, 2)) @ L_invs  # (n_samples, K, N, N)
    half_log_det_covs = np.trace(np.log(Ls), axis1=2, axis2=3)
    # A stacked ms.T @ K_inv @ ms
    out = np.einsum('ijkl, lj->ijk', K_invs, Ms)
    out = np.einsum('ijk, kj->ij', out, Ms)
    return -1. * half_log_det_covs - 0.5 * out  # ((n_samples, K, N, N) @ (N, )) @ (N,)


def vectorised_unnormalised_log_multivariate_normal_pdf(ms, mean=None, covs=None):
    """Evaluate a vector of log multivariate normal pdf in a numerically stable way."""
    if covs is None:
        raise ValueError("Must provide sample covariance matrices ({} was provided)".format(covs))
    if mean is not None:
        ms = ms - mean
    Ls = np.linalg.cholesky(covs)
    L_invs = np.linalg.inv(Ls)
    K_invs = L_invs.transpose((0, 2, 1)) @ L_invs
    half_log_det_covs = np.trace(np.log(Ls), axis1=1, axis2=2)
    return -1. * half_log_det_covs - 0.5 * (K_invs @ ms) @ ms


def unnormalised_log_multivariate_normal_pdf(x, mean=None, cov=None):
    """Evaluate the multivariate normal pdf in a numerically stable way."""
    if cov is None:
        cov = np.eye(len(x))
    if mean is not None:
        x = x - mean
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)
    K_inv = L_inv.T @ L_inv
    half_log_det_cov = np.trace(np.log(L))
    return -1. * half_log_det_cov - 0.5 * x.T @ K_inv @ x


def unnormalised_multivariate_normal_pdf(x, mean=None, cov=None):
    """Evaluate the multivariate normal pdf in a numerically stable way."""
    if cov is None:
        cov = np.eye(len(x))
    if mean is None:
        #return np.exp(-0.5 * np.log(np.linalg.det(cov)) - 0.5 * np.dot(x, np.linalg.solve(cov, x)))
        L = np.linalg.cholesky(cov)
        L_inv = np.linalg.inv(L)
        K_inv = L_inv.T @ L_inv
        return np.exp(-0.5 * np.log(np.linalg.det(cov)) - 0.5 * x @ K_inv @ x)
    elif mean is not None:
        return np.exp(-0.5 * np.log(np.linalg.det(cov)) - 0.5 * (x - mean).T * np.linalg.solve(cov, (x - mean)))


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


def sample_varphis(psi, n_samples):
    """
    Take n_samples of varphi, given the hyper-hyperparameter psi.

    psi is a rate parameter since, with an uninformative prior (sigma=tau=0), then the posterior mean of Q(psi) is
    psi_tilde = 1. / varphi_tilde. Therefore, by taking the expected value of the prior on varphi ~ Exp(psi_tilde),
    we expect to obtain varphi_tilde = 1. / psi_tilde. We get this if psi_tilde is a rate.

    :arg psi: float (Array) of hyper-hyperparameter(s)
    :type psi: :class:`np.ndarray`
    :arg int n_samples: The number of samples for the importance sample.
    """
    # scale = psi
    scale = 1. / psi
    shape = np.shape(psi)
    if shape == ():
        size = (n_samples,)
    else:
        size = (n_samples, shape[0], shape[1])
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


def sample_Us_vector(K, n_samples, N_test, different_across_classes=None):
    if not different_across_classes:
        # Will this induce an unwanted link between the y_nk across k? What effect will it have?
        us = norm.rvs(0, 1, (n_samples, 1, 1, 1))
        ones = np.ones((n_samples, N_test, K, K))
        Us = np.multiply(us, ones)
    else:
        # TODO: test this.
        # This might be a better option as the sample u across classes shouldn't be correlated, does it matter though?
        us = norm.rvs(0, 1, (n_samples, 1, K, 1))
        # Needs to be the same across rows, as we sum over the rows
        Us = np.tile(us, (1, N_test, 1, K))
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


def matrix_of_VB_differences(m_n, K, t_n):
    """
    Get a matrix of differences of the vector m.

    Superseded by matrix_of_VB_differencess, but is still needed in the non-vectorised case for Sparse GP regression.

    :arg m: is an (K, 1) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
    :type m: :class:`numpy.ndarray`
    """
    # Find matrix of coefficients
    Lambda = np.tile(m_n, (K, 1))
    Lambda_t_n = np.tile(m_n[t_n], (K, K))
    # Should get the same rows out
    difference = Lambda_t_n - Lambda
    return difference


def matrix_of_differencess(M, K, N_test):
    """
    Get an array of matrix of differences of the vectors m_ns.

    :arg M: An (N_test, K) array filled with e.g. m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object. Or in the VB implementation, this function  is used for M_tilde (N, K).
    :type M: :class:`numpy.ndarray`
    :arg K: The number of classes.
    :arg N_test: The number of test objects.
    """
    M = M.reshape((N_test, K, 1))
    # Lambdas is an (N_test, K, K) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambda_Ts = np.tile(M, (1, 1, K))
    Lambdas = Lambda_Ts.transpose((0, 2, 1))
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    return np.subtract(Lambda_Ts, Lambdas)  # (N_test, K, K)


def matrix_of_VB_differencess(M, K, N_test, t, grid):
    """
    Get an array of matrix of differences of the vectors m_ns.

    :arg M: An (N_test, K) array filled with e.g. m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object. Or in the VB implementation, this function  is used for M_tilde (N, K).
    :type M: :class:`numpy.ndarray`
    :arg K: The number of classes.
    :arg N_test: The number of test objects.
    """
    # Lambdas is an (N_test, K, K) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambda_t_ns = np.tile(M[grid, t], (1, K * K))
    Lambda_t_ns = Lambda_t_ns.reshape((N_test, K, K))
    Lambdas = np.tile(M, (1, 1, K))
    Lambdas = Lambdas.reshape((N_test, K, K))
    # Matrix of differences for VB, the rows are the same and contain the elements of the product of interest
    return np.subtract(Lambda_t_ns, Lambdas)  # (N_test, K, K)


def matrix_of_valuess(nu, K, N_test):
    """
    Get an array of matrix of values of the vectors M.

    :param nu: An (N_test, K) array filled with nu_tilde_k^{new_i} where  k is the class indicator
        and i is the index of the test object.
    :param K: The number of classes.
    :param N_test: The number of test objects.

    :return: (N_test, K, K) Matrix of rows of nu_tilde_k^{new_i}, where the rows(=axis 2) contain identical values.
    """
    nu = nu.reshape((N_test, K, 1))
    # Lambdas is an (N_test, K, K) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambdas_Ts = np.tile(nu, (1, 1, K))
    return Lambdas_Ts  # (N_test, K, K)

