import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = "4" # export NUMBA_NUM_THREADS=6

"""Numba implementation of EP."""
from numba import njit, prange
import numpy as np
from scipy.special import erf, ndtr, log_ndtr
import numba_scipy  # Numba overloads for scipy and scipy.special


over_sqrt_2_pi = 1. / np.sqrt(2 * np.pi)
log_over_sqrt_2_pi = np.log(over_sqrt_2_pi)


@njit
def norm_pdf(x):
    return over_sqrt_2_pi * np.exp(- x**2 / 2.0 )


@njit
def norm_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


@njit
def norm_cdf(x):
    return ndtr(x)


@njit
def norm_logcdf(x):
    return log_ndtr(x)


@njit(parallel=True)
def vector_norm_pdf(x):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in prange(n):
        out[i] = norm_pdf(x[i])
    return out


@njit(parallel=True)
def vector_norm_cdf(x):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in prange(n):
        out[i] = ndtr(x[i])
    return out


@njit(parallel=True)
def vector_norm_logpdf(x):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in prange(n):
        out[i] = norm_logpdf(x[i])
    return out


@njit(parallel=True)
def vector_norm_logcdf(x):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(x)
    out = np.empty(n)
    for i in prange(n):
        out[i] = ndtr(x[i])
    return out


@njit
def return_prob(z1, z2, EPS_2):
    """Return a Gaussian probability."""
    p = ndtr(z1) - ndtr(z2)
    if p > EPS_2:
        return p
    else:
        return EPS_2


@njit(parallel=True)
def vector_return_prob(z1, z2, EPS_2):
    """Return a vector of Gaussian probability."""
    # TODO: make arguments of b, mean, sigma, noise_std, gamma_t, gamma_tplus1, EPS_2 ?
    n = len(z1)
    out = np.empty(n)
    for i in prange(n):
        out[i] = return_prob(z1[i], z2[i], EPS_2)
    return out


@njit(parallel=True)
def return_z(x, loc, scale):
    """Return z values given a mean and standard deviation."""
    return (x - loc) / scale


@njit(parallel=True)
def fromb_fft1_vector(
        b, mean, sigma, noise_std, gamma_t, gamma_tplus1, EPS_2):
    """
    TODO: This version works pretty well. as it vectorises the numpy (which numba parallelizes efficiently anyway), and uses
    a single loop of scalar scipy functions.
    Maybe uses a bit of extra code that is not needed, but probably fine.

    :arg float b: The approximate posterior mean vector.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int J: The number of ordinal classes.
    :arg gamma_t: The vector of the lower cutpoints the data.
    :type gamma_t: `nd.numpy.array`
    :arg gamma_t_plus_1: The vector of the upper cutpoints the data.
    :type gamma_t_plus_1: `nd.numpy.array`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS_2: A (squared) machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    z1 = (gamma_tplus1 - b) / noise_std
    z2 = (gamma_t - b) / noise_std
    x = (b - mean) / sigma
    n = len(b)
    out = np.empty(n)
    # TODO: in the jax implementation, this part is just substituted out
    for i in prange(n):
        out[i] = norm_pdf(x[i]) * np.log(return_prob(z1[i], z2[i], EPS_2))
    return out


def fromb_t1_vector(
        y, posterior_mean, posterior_covariance, gamma_t,
        gamma_tplus1, noise_std, EPS):
    """
    :arg posterior_mean: The approximate posterior mean vector.
    :type posterior_mean: :class:`numpy.ndarray`
    :arg float posterior_covariance: The approximate posterior marginal
        variance vector.
    :type posterior_covariance: :class:`numpy.ndarray`
    :arg int J: The number of ordinal classes.
    :arg gamma_t: The vector of the lower cutpoints the data.
    :type gamma_t: `nd.numpy.array`
    :arg gamma_t_plus_1: The vector of the upper cutpoints the data.
    :type gamma_t_plus_1: `nd.numpy.array`
    :arg float noise_std: A noise standard deviation for the likelihood.
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral vector.
    :rtype: float
    """
    EPS_2 = EPS**2
    posterior_std = np.sqrt(posterior_covariance)
    a = posterior_mean - 5.0 * posterior_std
    b = posterior_mean + 5.0 * posterior_std
    h = b - a
    y[0, :] = h * (
        fromb_fft1_vector(
            a, posterior_mean, posterior_std, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
        + fromb_fft1_vector(
            b, posterior_mean, posterior_std, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (np.any(ep>=EPS) and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p += fromb_fft1_vector(
                x, posterior_mean, posterior_std, noise_std,
                gamma_t, gamma_tplus1,
                EPS_2)
        p = (y[0, :] + h * p) / 2.0
        s = 1.0
        for k in range(m):
            s *= 4.0
            q = (s * p - y[k, :]) / (s - 1.0)
            y[k, :] = p
            p = q
        ep = np.abs(q - y[m - 1, :])
        m += 1
        y[m - 1, :] = q
        n += n
        h /= 2.0
    return q


# def fromb_fft2(
#     b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#     gamma, noise_variance, EPS):
#     """
#     :arg b: The approximate posterior mean evaluated at the datapoint.
#     :arg mean: A mean value of a pdf inside the integrand.
#     :arg sigma: A standard deviation of a pdf inside the integrand.
#     :arg t: The target value for the datapoint.
#     :arg J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :arg noise_variance: A noise variance for the likelihood.
#     :arg EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     noise_std = np.sqrt(noise_variance)
#     EPS_2 = EPS * EPS
#     func = norm.pdf(b, loc=mean, scale=sigma)
#     prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
#     if prob < EPS_2:
#         prob = EPS_2
#     func = func / prob * norm.pdf(posterior_mean, loc=gamma[t], scale=np.sqrt(
#         noise_variance + posterior_covariance))
#     if math.isnan(func):
#         raise ValueError("{} {} {} {} ({} {} {})".format(
#             t, gamma[t], prob, func, b, mean, sigma))
#     return func


# def fromb_fft2_vector(
#         b, mean, sigma, posterior_mean, posterior_covariance,
#         noise_variance, noise_std, gamma_t, gamma_tplus1,
#         EPS_2):
#     """
#     :arg b: The approximate posterior mean evaluated at the datapoint.
#     :arg mean: A mean value of a pdf inside the integrand.
#     :arg sigma: A standard deviation of a pdf inside the integrand.
#     :arg t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :arg float noise_std: A noise standard deviation for the likelihood.
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     prob = return_prob_vector(
#         b, gamma_t, gamma_tplus1, noise_std)
#     prob[prob < EPS_2] = EPS_2
#     return norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
#         posterior_mean, loc=gamma_t, scale=np.sqrt(
#         noise_variance + posterior_covariance))


# def fromb_t2(
#         posterior_mean, posterior_covariance, t, J,
#         gamma, noise_variance, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target for the datapoint
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     if t == 0:
#         return 0
#     mean = (posterior_mean * noise_variance
#             + posterior_covariance * gamma[t]) / (
#             noise_variance + posterior_covariance)
#     sigma = np.sqrt((noise_variance * posterior_covariance) / (
#             noise_variance + posterior_covariance))
#     y = np.zeros((20,))
#     a = mean - 5.0 * sigma
#     b = mean + 5.0 * sigma
#     h = b - a
#     y[0] = h * (
#         fromb_fft2(
#             a, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#         + fromb_fft2(
#             b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (ep >=EPS and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p += fromb_fft2(
#                 x, mean, sigma, posterior_mean, posterior_covariance, t,
#                 J, gamma, noise_variance, EPS)
#         p = (y[0] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k]) / (s - 1.0)
#             y[k] = p
#             p = q
#         ep = np.abs(q - y[m - 1])
#         m += 1
#         y[m - 1] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_t2_vector(
#     y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
#     gamma_t, gamma_tplus1,
#     noise_variance, noise_std, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int J: The number of ordinal classes.
#     :arg gamma_t: The vector of the lower cutpoints the data.
#     :type gamma_t: `nd.numpy.array`
#     :arg gamma_t_plus_1: The vector of the upper cutpoints the data.
#     :type gamma_t_plus_1: `nd.numpy.array`
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_std: A noise standard deviation for the likelihood.
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     EPS_2 = EPS**2
#     y[0, :] = h * (
#         fromb_fft2_vector(
#             a, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#         + fromb_fft2_vector(
#             b, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std, gamma_t, gamma_tplus1,
#             EPS_2)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (np.any(ep>=EPS) and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p += fromb_fft2_vector(
#                 x, mean, sigma, posterior_mean, posterior_covariance,
#                 noise_variance, noise_std,
#                 gamma_t, gamma_tplus1,
#                 EPS_2)
#         p = (y[0, :] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k, :]) / (s - 1.0)
#             y[k, :] = p
#             p = q
#         ep = np.abs(q - y[m - 1, :])
#         m += 1
#         y[m - 1, :] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_fft3(
#         b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#         gamma, noise_variance, EPS, numerically_stable=True):
#     """
#     :arg float b: The approximate posterior mean evaluated at the datapoint.
#     :arg float mean: A mean value of a pdf inside the integrand.
#     :arg float sigma: A standard deviation of a pdf inside the integrand.
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     EPS_2 = EPS * EPS
#     noise_std = np.sqrt(noise_variance)
#     func = norm.pdf(b, loc=mean, scale=sigma)
#     prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
#     if prob < EPS_2:
#         prob = EPS_2
#     func = func / prob * norm.pdf(
#         posterior_mean, loc=gamma[t + 1], scale=np.sqrt(
#         noise_variance + posterior_covariance))
#     if math.isnan(func):
#         raise ValueError("{} {} {} {}".format(t, gamma[t], prob, func))
#     return func


# def fromb_fft3_vector(
#         b, mean, sigma, posterior_mean, posterior_covariance,
#         noise_variance, noise_std, gamma_t, gamma_tplus1,
#         EPS_2):
#     """
#     :arg float b: The approximate posterior mean evaluated at the datapoint.
#     :arg float mean: A mean value of a pdf inside the integrand.
#     :arg float sigma: A standard deviation of a pdf inside the integrand.
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int J: The number of ordinal classes.
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     prob = return_prob_vector(
#         b, gamma_t, gamma_tplus1, noise_std)
#     prob[prob < EPS_2] = EPS_2
#     return  norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
#         posterior_mean, loc=gamma_tplus1, scale=np.sqrt(
#         noise_variance + posterior_covariance))


# def fromb_t3(
#     posterior_mean, posterior_covariance, t, J, gamma, noise_variance, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     if t == J - 1:
#         return 0
#     mean = (posterior_mean * noise_variance
#             + posterior_covariance * gamma[t + 1]) / (
#             noise_variance + posterior_covariance)
#     sigma = np.sqrt((noise_variance * posterior_covariance)
#                     / (noise_variance + posterior_covariance))
#     y = np.zeros((20,))
#     a = mean - 5.0 * sigma
#     b = mean + 5.0 * sigma
#     h = b - a
#     y[0] = h * (
#         fromb_fft3(
#             a, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#         + fromb_fft3(
#             b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (ep >=EPS and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p = p + fromb_fft3(
#                 x, mean, sigma, posterior_mean, posterior_covariance, t, J,
#                 gamma, noise_variance, EPS)
#         p = (y[0] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k]) / (s - 1.0)
#             y[k] = p
#             p = q
#         ep = np.abs(q - y[m - 1])
#         m += 1
#         y[m - 1] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_t3_vector(
#     y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
#     gamma_t, gamma_tplus1,
#     noise_std, noise_variance, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int J: The number of ordinal classes.
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     EPS_2 = EPS**2
#     y[0, :] = h * (
#         fromb_fft3_vector(
#             a, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#         + fromb_fft3_vector(
#             b, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (np.any(ep>=EPS) and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p = p + fromb_fft3_vector(
#                 x, mean, sigma, posterior_mean, posterior_covariance,
#                 noise_variance, noise_std,
#                 gamma_t, gamma_tplus1,
#                 EPS_2)
#         p = (y[0, :] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k]) / (s - 1.0)
#             y[k, :] = p
#             p = q
#         ep = np.abs(q - y[m - 1, :])
#         m += 1
#         y[m - 1, :] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_fft4(
#         b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#         gamma, noise_variance, EPS):
#     """
#     :arg float b: The approximate posterior mean evaluated at the datapoint.
#     :arg float mean: A mean value of a pdf inside the integrand.
#     :arg float sigma: A standard deviation of a pdf inside the integrand.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     EPS_2 = EPS * EPS
#     noise_std = np.sqrt(noise_variance)
#     func = norm.pdf(b, loc=mean, scale=sigma)
#     prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
#     if prob < EPS_2:
#         prob = EPS_2
#     func = func / prob * norm.pdf(
#         posterior_mean, loc=gamma[t + 1], scale=np.sqrt(
#         noise_variance + posterior_covariance)) * (gamma[t + 1] - b)
#     return func


# def fromb_fft4_vector(
#         b, mean, sigma, posterior_mean, posterior_covariance,
#         noise_std, noise_variance, gamma_t, gamma_tplus1,
#         EPS_2):
#     """
#     :arg float b: The approximate posterior mean evaluated at the datapoint.
#     :arg float mean: A mean value of a pdf inside the integrand.
#     :arg float sigma: A standard deviation of a pdf inside the integrand.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     prob = return_prob_vector(
#         b, gamma_t, gamma_tplus1, noise_std)
#     prob[prob < EPS_2] = EPS_2
#     return norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
#         posterior_mean, loc=gamma_tplus1, scale=np.sqrt(
#         noise_variance + posterior_covariance)) * (gamma_tplus1 - b)


# def fromb_t4(
#     posterior_mean, posterior_covariance, t, J, gamma, noise_variance, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     if t == J - 1:
#         return 0
#     mean = (posterior_mean * noise_variance
#             + posterior_covariance * gamma[t + 1]) / (
#             noise_variance + posterior_covariance)
#     sigma = np.sqrt((noise_variance * posterior_covariance)
#             / (noise_variance + posterior_covariance))
#     y = np.zeros((20,))
#     a = mean - 5.0 * sigma
#     b = mean + 5.0 * sigma
#     h = b - a
#     y[0] = h * (
#         fromb_fft4(
#             a, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#         + fromb_fft4(
#             b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (ep >=EPS and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p = p + fromb_fft4(
#                 x, mean, sigma, posterior_mean, posterior_covariance, t, J,
#                 gamma, noise_variance, EPS)
#         p = (y[0] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k]) / (s - 1.0)
#             y[k] = p
#             p = q
#         ep = np.abs(q - y[m - 1])
#         m += 1
#         y[m - 1] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_t4_vector(
#     y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
#     gamma_t, gamma_tplus1,
#     noise_variance, noise_std, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     EPS_2 = EPS**2
#     y[0, :] = h * (
#         fromb_fft4_vector(
#             a, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#         + fromb_fft4_vector(
#             b, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (np.any(ep>=EPS) and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p = p + fromb_fft4_vector(
#                 x, mean, sigma, posterior_mean, posterior_covariance,
#                 noise_variance, noise_std,
#                 gamma_t, gamma_tplus1,
#                 EPS_2)
#         p = (y[0, :] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k, :]) / (s - 1.0)
#             y[k, :] = p
#             p = q
#         ep = np.abs(q - y[m - 1, :])
#         m += 1
#         y[m - 1, :] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_fft5(
#     b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#     gamma, noise_variance, EPS):
#     """
#     :arg float b: The approximate posterior mean evaluated at the datapoint.
#     :arg float mean: A mean value of a pdf inside the integrand.
#     :arg float sigma: A standard deviation of a pdf inside the integrand.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     EPS_2 = EPS * EPS
#     noise_std = np.sqrt(noise_variance)
#     func = norm.pdf(b, loc=mean, scale=sigma)
#     prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
#     if prob < EPS_2:
#         prob = EPS_2
#     func = func / prob * norm.pdf(posterior_mean, loc=gamma[t], scale=np.sqrt(
#         noise_variance + posterior_covariance)) * (gamma[t] - b)
#     return func


# def fromb_fft5_vector(
#         b, mean, sigma, posterior_mean, posterior_covariance,
#         noise_variance, noise_std,
#         gamma_t, gamma_tplus1,
#         EPS_2):
#     """
#     :arg float b: The approximate posterior mean evaluated at the datapoint.
#     :arg float mean: A mean value of a pdf inside the integrand.
#     :arg float sigma: A standard deviation of a pdf inside the integrand.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral point evaluation.
#     :rtype: float
#     """
#     prob = return_prob_vector(
#         b, gamma_t, gamma_tplus1, noise_std)
#     prob[prob < EPS_2] = EPS_2
#     return norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
#         posterior_mean, loc=gamma_t, scale=np.sqrt(
#         noise_variance + posterior_covariance)) * (gamma_t - b)


# def fromb_t5(
#         posterior_mean, posterior_covariance, t, J, gamma, noise_variance,
#         EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     if t == 0:
#         return 0
#     mean = (posterior_mean * noise_variance
#             + posterior_covariance * gamma[t]) / (
#             noise_variance + posterior_covariance)
#     sigma = np.sqrt((noise_variance * posterior_covariance)
#             / (noise_variance + posterior_covariance))
#     y = np.zeros((20,))
#     a = mean - 5.0 * sigma
#     b = mean + 5.0 * sigma
#     h = b - a
#     y[0] = h * (
#         fromb_fft5(
#             a, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#         + fromb_fft5(
#             b, mean, sigma, posterior_mean, posterior_covariance, t, J,
#             gamma, noise_variance, EPS)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (ep >=EPS and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p += fromb_fft5(
#                 x, mean, sigma, posterior_mean, posterior_covariance, t, J,
#                 gamma, noise_variance, EPS)
#         p = (y[0] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k]) / (s - 1.0)
#             y[k] = p
#             p = q
#         ep = np.abs(q - y[m - 1])
#         m += 1
#         y[m - 1] = q
#         n += n
#         h /= 2.0
#     return q


# def fromb_t5_vector(
#         y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
#         gamma_t, gamma_tplus1,
#         noise_variance, noise_std, EPS):
#     """
#     :arg float posterior_mean: The approximate posterior mean evaluated at the
#         datapoint. (pdf inside the integrand)
#     :arg float posterior_covariance: The approximate posterior marginal
#         variance.
#     :arg int t: The target value for the datapoint.
#     :arg int J: The number of ordinal classes.
#     :arg gamma: The vector of cutpoints.
#     :type gamma: `numpy.ndarray`
#     :arg float noise_variance: A noise variance for the likelihood.
#     :arg float EPS: A machine tolerance to be used.
#     :return: fromberg numerical integral numerical value.
#     :rtype: float
#     """
#     EPS_2 = EPS**2
#     y[0, :] = h * (
#         fromb_fft5_vector(
#             a, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#         + fromb_fft5_vector(
#             b, mean, sigma, posterior_mean, posterior_covariance,
#             noise_variance, noise_std,
#             gamma_t, gamma_tplus1,
#             EPS_2)
#     ) / 2.0
#     m = 1
#     n = 1
#     ep = EPS + 1.0
#     while (np.any(ep>=EPS) and m <=19):
#         p = 0.0
#         for i in range(n):
#             x = a + (i + 0.5) * h
#             p += fromb_fft5_vector(
#                 x, mean, sigma, posterior_mean, posterior_covariance,
#                 noise_variance, noise_std,
#                 gamma_t, gamma_tplus1,
#                 EPS_2)
#         p = (y[0, :] + h * p) / 2.0
#         s = 1.0
#         for k in range(m):
#             s *= 4.0
#             q = (s * p - y[k, :]) / (s - 1.0)
#             y[k, :] = p
#             p = q
#         ep = np.abs(q - y[m - 1, :])
#         m += 1
#         y[m - 1, :] = q
#         n += n
#         h /= 2.0
#     return q


def compute_integrals_vector(
        gamma, t_train, posterior_variance, posterior_mean, noise_variance):
    """
    Compute the integrals required for the gradient evaluation.
    """
    # calculate gamma_t and gamma_tplus1 here
    N = len(posterior_mean)
    noise_std = np.sqrt(noise_variance) * np.sqrt(2)  # TODO
    gamma_t = gamma[t_train]
    gamma_tplus1 = gamma[t_train + 1] 
    y_0 = np.zeros((20,  N))
    return (
        fromb_t1_vector(
            y_0.copy(), posterior_mean, posterior_variance,
            gamma_t, gamma_tplus1,
            noise_std, EPS=0.001),
    )


import time


N = 10000

gamma = np.array([-np.inf, -0.2, 0.2, np.inf])

posterior_variance = np.abs(np.random.rand(N))

posterior_mean = np.random.rand(N)

noise_variance = 1.0

t_train = np.random.randint(low=0, high=3, size=N)

# print(posterior_variance)
# print(posterior_mean)
# print(noise_variance)
# print(t_train)


# The first time that it runs, it will take a while
time0 = time.time()

print(compute_integrals_vector(
    gamma, t_train, posterior_variance, posterior_mean, noise_variance))
print("\n")

time1 = time.time()
print("time={}".format(time1 - time0))


time0 = time.time()

compute_integrals_vector(
    gamma, t_train, posterior_variance, posterior_mean, noise_variance)

time1 = time.time()

print("time1={}".format(time1 - time0))


assert 0
