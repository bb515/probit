"""Numba implementation of EP."""
from numba import njit, prange
import numpy as np
from scipy.stats import norm, expon
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


@njit
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
    for i in prange(n):
        out[i] = norm_cdf(x[i])
    return out


@njit
def vector_norm_logpdf(x, loc=None, scale=1.0):
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


@njit
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
def return_prob(b, gamma_t, gamma_tplus1, noise_std):
    """Return a Gaussian probability."""
    # TODO: could experiment with parallelising additions and divisions
    # but is probably neglidgeable speedup
    return norm_cdf(
        (gamma_t - b) / noise_std) - norm_cdf((gamma_tplus1 - b) / noise_std)


@njit
def return_prob_vector(b, gamma_t, gamma_tplus1, noise_std):
    """Return a vector of Gaussian probability."""
    return vector_norm_cdf(
        (gamma_tplus1 - b) / noise_std) - vector_norm_cdf(
            (gamma_t - b) / noise_std)


@njit
def fromb_fft1(
    b, mean, sigma, gamma_t, gamma_tplus1, noise_std, EPS):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int t: The target value for the datapoint, indexed from 0.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `nd.numpy.array`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float noise_std: A noise std for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    func = norm_pdf((b - mean) / sigma)
    prob = return_prob(b, gamma_t, gamma_tplus1, noise_std)
    tol = EPS*EPS
    if prob < tol:
        prob = tol
    func = func * np.log(prob)
    if np.isnan(func):
        raise ValueError("fft1 {} {} {} {}".format(
            gamma_t, gamma_tplus1, prob, func))
    return func


@njit
def fromb_fft1_vectorv1(
        b, mean, sigma, noise_std, gamma_t, gamma_tplus1, EPS_2):
    """
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
    N = len(b)
    fft1 = np.empty(N)
    for i in prange(N):
        fft1[i] = fromb_fft1(
            b[i], mean[i], sigma[i], gamma_t[i], gamma_tplus1[i],
            noise_std, EPS_2)
    return fft1


@njit
def fromb_fft1_vectorv2(
    b, mean, sigma, noise_std, gamma_t, gamma_tplus1, EPS_2):
    """
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
    prob = return_prob_vector(
        b, gamma_t, gamma_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return vector_norm_pdf((b - mean) / sigma) * np.log(prob)


@njit
def fromb_fft1_vectorv3(
    b, mean, sigma, noise_std, gamma_t, gamma_tplus1, EPS_2):
    """
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
    prob = ndtr(
        (gamma_tplus1 - b) / noise_std) - ndtr(
            (gamma_t - b) / noise_std)
    prob[prob < EPS_2] = EPS_2
    return over_sqrt_2_pi * np.exp(
        -((b - mean) / sigma)**2 / 2.0) * np.log(prob)


def fromb_t1(
    posterior_mean, posterior_covariance, t, J, gamma, noise_variance, EPS):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint.
    :arg float posterior_covariance: The approximate posterior marginal
        variance evaluated at the datapoint.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    posterior_std = np.sqrt(posterior_covariance)
    y = np.zeros((20,))
    a = posterior_mean - 5.0 * posterior_std
    b = posterior_mean + 5.0 * posterior_std
    h = b - a
    y[0] = h * (
        fromb_fft1(
            a, posterior_mean, posterior_std, t, J, gamma, noise_variance, EPS)
        + fromb_fft1(
            b, posterior_mean, posterior_std, t, J, gamma, noise_variance, EPS)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (ep >=EPS and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p += fromb_fft1(
                x, posterior_mean, posterior_std, t, J,
                gamma, noise_variance, EPS)
        p = (y[0] + h * p) / 2.0
        s = 1.0
        for k in range(m):
            s *= 4.0
            q = (s * p - y[k]) / (s - 1.0)
            y[k] = p
            p = q
        ep = np.abs(q - y[m - 1])
        m += 1
        y[m - 1] = q
        n += n
        h /= 2.0
    return q


@njit
def fromb_t1_vector(
    y, posterior_mean, posterior_covariance, gamma_t, gamma_tplus1,
    noise_std, EPS):
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
        fromb_fft1_vectorv1(
            a, posterior_mean, posterior_std, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
        + fromb_fft1_vectorv1(
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
            p += fromb_fft1_vectorv1(
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
            self, gamma, Sigma, precision_EP, posterior_mean, noise_variance):
        """
        Compute the integrals required for the gradient evaluation.
        """
        # calculate gamma_t and gamma_tplus1 here
        noise_std = np.sqrt(noise_variance) * np.sqrt(2)  # TODO
        gamma_t = gamma[self.t_train]
        gamma_tplus1 = gamma[self.t_train + 1] 
        posterior_covariance = np.diag(Sigma)
       
        y_0 = np.zeros((20, self.N))
        return (
            fromb_t1_vector(
                y_0.copy(), posterior_mean, posterior_covariance,
                self.gamma_ts, self.gamma_tplus1s,
                noise_std, self.EPS),
        )
