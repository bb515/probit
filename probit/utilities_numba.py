""" """
import numpy as np
from scipy.stats import norm, expon
from scipy.special import erf, ndtr, log_ndtr
from numba import njit
import numpy as np


over_sqrt_2_pi = 1. / np.sqrt(2 * np.pi)
log_over_sqrt_2_pi = np.log(over_sqrt_2_pi)


def norm_pdf(x, loc=None, scale=1.0):
    if loc is not None:
        x = x - loc
    return (1./ scale) * over_sqrt_2_pi * np.exp(- x**2 / (2.0 * scale))


def log_norm_pdf(x, loc=None, scale=1.0):
    if loc is not None:
        x = x - loc
    return -np.log(scale) + log_over_sqrt_2_pi - x**2 / (2.0 * scale)


def norm_cdf(x, loc=None, scale=1.0):
    if loc is not None:
        x = x - loc
    return ndtr(x / scale)


def log_norm_cdf(x, loc=None, scale=1.0):
    if loc is not None:
        x = x - loc
    return log_ndtr(x / scale)


@njit
def vector_norm_pdf(x, loc=None, scale=1.0):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = norm_pdf(x, loc=loc, scale=scale)
    return out

@njit
def vector_norm_cdf(x, loc=None, scale=1.0):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = norm_cdf(x, loc=loc, scale=scale)
    return out


@njit
def vector_norm_log_pdf(x, loc=None, scale=1.0):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = log_norm_pdf(x, loc=loc, scale=scale)
    return out


@njit
def vector_norm_log_cdf(x, loc=None, scale=1.0):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = log_norm_cdf(x, loc=loc, scale=scale)
    return out
