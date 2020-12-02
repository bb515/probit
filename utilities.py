"""Utility functions."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal


def sample_U(K, different_across_classes=None):
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


def matrix_of_differences(m_n):
    """ Get a matrix of differences of the vector m."""
    # Find matrix of coefficients
    K  = np.shape(m_n)[0]
    I = np.eye(K)
    Lambda = np.tile(m_n, (K, 1))
    Lambda_T = Lambda.T
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    difference = Lambda_T - Lambda
    return difference


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
    normal_pdf = norm.pdf(vector_difference, 1)
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

    # Take the sample in vector form (K, )
    u = U[:, 0]
    normal_cdf = norm.cdf(u - vector_difference, 1)
    # Find the elementwise product of these two vectors which returns a (K, ) array
    return np.multiply(function_eval, normal_cdf)


# #Testing
# K = 3
# m_n = np.array([-1, 0, 1])
# difference = matrix_of_differences(m_n)
# print(difference)
# U = sample_U(K)
# print(U)
# print(function_u1(difference, U))
# t_n = np.argmax(m_n)
# print(t_n)
# print(function_u1_alt(difference, U, np.argmax(t_n)))
# vector_difference = difference[:, t_n]
# print(vector_difference)
# print(function_u2(difference, vector_difference, U, t_n, K))
# print(function_u3(difference, vector_difference, U, t_n, K))


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



