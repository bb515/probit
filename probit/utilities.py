"""Utility functions."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal, expon


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


def samples_varphi(psi, n_samples):
    """Tensor version of sample_varphi"""
    scale = 1./psi
    varphi_samples = expon.rvs(scale=scale, size=((n_samples, np.shape(psi)[0], np.shape(psi)[1])))
    return varphi_samples


def sample_varphi(psi):
    """In the 2005 paper, psi is a rate parameter. The justification for this is that we know that with a totally
    uninformative prior (sigma=tau=0) then the posterior mean of Q(psi) is psi_tilde = 1./ varphi_tilde.
    Therefore, taking an expectation over the prior on varphi varphi ~ Exp(psi_tilde), we would expect to obtain
    varphi_tilde = 1./ psi_tilde, which we get if psi_tilde is a rate."""
    # Sample independently over the M covariance kernel hyperparameters for each class
    scale = 1. / psi
    varphi_sample = expon.rvs(scale=scale)
    return varphi_sample


def sample_U(K, different_across_classes=None):  # TODO: May have been SS
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


def matrix_of_differences(m_n, K):  # TODO: changed this, VB will need changing
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


def matrix_of_differencess(m_ns, K, N_test):
    """
    Get an array of matrix of differences of the vectors m_ns.

    :arg m_ns: An (N_test, K) array filled with m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object.
    :type m_ns: :class:`numpy.ndarray`
    """
    # Find the matrix of coefficients
    m_ns = m_ns.reshape((N_test, K, 1))
    # Lambdas is an (n_test, K, K) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambda_Ts = np.tile(m_ns, (1, 1, K))
    Lambdas = Lambda_Ts.transpose((0, 2, 1))
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    return np.subtract(Lambda_Ts, Lambdas)  # (N_test, K, K)


def function_u1(difference, U):
    """
    The multinomial probit likelihood for a new datapoint given the latent variables
    is the expectation of this function.
    """
    random_variables = np.add(U, difference)
    cum_dist = norm.cdf(random_variables, loc=0, scale=1)
    log_cum_dist = np.log(cum_dist)
    # sum is over j \neq k
    np.fill_diagonal(log_cum_dist, 0)
    # Sum across the elements of the log product of interest (rows, so axis=1)
    log_sample = np.sum(log_cum_dist, axis=1)
    function_eval = np.exp(log_sample)
    return function_eval


def function_u1_alt(difference, U, t_n):
    """
    The multinomial probit likelihood for a new datapoint given the latent variables
    is the expectation of this function. This is for the kth class. It could be vectorised over k, but I will
    leave that until code refactoring.
    """
    random_variables = np.add(U, difference)
    cum_dist = norm.cdf(random_variables, loc=0, scale=1)
    log_cum_dist = np.log(cum_dist)
    # sum is over j \neq k
    np.fill_diagonal(log_cum_dist, 0)
    # sum is over j \neq tn=i
    log_cum_dist[:, t_n] = 0
    # Sum across the elements of the log product of interest (rows, so axis=1)
    log_sample = np.sum(log_cum_dist, axis=1)
    function_eval = np.exp(log_sample)
    return function_eval


def function_u2(difference, vector_difference, U, t_n, K):
    """Function evaluation of sample U.

    This is the numerator of the rightmost term of equation (5).

    :param: difference is the matrix_of_differences(u_n)
    :param: vector difference
    :param U: is the random variable (K, K) and constant across elemnents
    :param t_n: is the np.argmax(m_n), the class chosen by the latent variable for that object n
    """
    function_eval = function_u1_alt(difference, U, t_n)
    # Take the sample U as a vector (K, )
    u = U[:, 0]
    normal_pdf = norm.pdf(u - vector_difference, loc=0, scale=1)
    # Find the elementwise product of these two vectors which returns a (K, ) array
    return np.multiply(normal_pdf, function_eval)


def function_u3(difference, vector_difference, U, t_n, K):
    """
    Function evaluation of sample U.

    This is the denominator of the rightmost term of equation (5).

    :param: difference is the matrix_of_differences(u_n)
    :param: vector difference
    :param U: is the random variable (K, K) and constant across elemnents
    :param t_n: is the np.argmax(m_n), the class chosen by the latent variable for that object n
    """
    function_eval = function_u1_alt(difference, U, t_n)
    # Take the sample U as a vector (K, )
    u = U[:, 0]
    normal_cdf = norm.cdf(u - vector_difference, 1)
    # Find the elementwise product of these two vectors which returns a (K, ) array
    return np.multiply(function_eval, normal_cdf)


def expectation_p_m(function, m_n, n_samples):
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
