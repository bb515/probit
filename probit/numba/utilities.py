"""Numba implementation of EP."""
from numba import njit, prange, set_num_threads
import numpy as np
from scipy.special import ndtr, log_ndtr
import numba_scipy  # Numba overloads for scipy and scipy.special
from scipy.stats import norm


# Make sure to limit CPU usage (here as a failsafe)
set_num_threads(6)


over_sqrt_2_pi = 1. / np.sqrt(2 * np.pi)
log_over_sqrt_2_pi = np.log(over_sqrt_2_pi)


@njit
def norm_z_pdf(z):
    return over_sqrt_2_pi * np.exp(- z**2 / 2.0 )


@njit
def norm_z_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


@njit
def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


@njit
def norm_logpdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_logpdf(z) - np.log(scale)


@njit
def norm_cdf(x):
    return ndtr(x)


@njit
def norm_logcdf(x):
    return log_ndtr(x)


@njit(parallel=True)
def vector_norm_z_pdf(z):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    n = len(z)
    out = np.empty(n)
    for i in prange(n):
        out[i] = norm_z_pdf(z[i])
    return out


@njit(parallel=True)
def vector_norm_pdf(x, loc=0.0, scale=1.0):
    """
    Return the pdf of a standard normal evaluated at multiple different values

    :arg x: the vector of values to return pdf of (n, )
    :arg loc: the scalar mean of the values to return pdf of
    :arg scale: the scalar std_dev to return the pdf of

    :rtype: numpy.ndarray
    """
    z = (x - loc) / scale
    n = len(z)
    out = np.empty(n)
    for i in prange(n):
        out[i] = norm_z_pdf(z[i]) / scale[i]
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
    if p >= EPS_2:
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
        b, mean, sigma, noise_std, gamma_t, gamma_tplus1, EPS_2, N):
    """
    This is a numba implementation that was tested to perform well.
    Performance comes from paralellising the numpy array operatios, and from
    a single loop of scalar scipy functions, which are also parallelised by
    numba. Scipy functions are only available to numba as scalar operations.

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
    out = np.empty(N)
    # TODO: in the jax implementation, this part is just substituted out
    for i in prange(N):
        out[i] = (norm_pdf(x[i]) / sigma[i]) * np.log(return_prob(z1[i], z2[i], EPS_2))
    return out


@njit(parallel=True)
def fromb_fft2_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_variance, noise_std, gamma_t, gamma_tplus1,
        EPS_2, N):
    """
    :arg b: The approximate posterior mean evaluated at the datapoint.
    :arg mean: A mean value of a pdf inside the integrand.
    :arg sigma: A standard deviation of a pdf inside the integrand.
    :arg t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :arg float noise_std: A noise standard deviation for the likelihood.
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    z1 = (gamma_tplus1 - b) / noise_std
    z2 = (gamma_t - b) / noise_std
    std = np.sqrt(noise_variance + posterior_covariance)
    x = (b - mean) / sigma
    y = (posterior_mean - gamma_t) / std
    out = np.empty(N)
    for i in prange(N):
        out[i] = (norm_pdf(x[i]) / sigma[i]) / return_prob(
            z1[i], z2[i], EPS_2) * (norm_pdf(y[i]) / std[i])
    return out


@njit(parallel=True)
def fromb_fft3_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_variance, noise_std, gamma_t, gamma_tplus1,
        EPS_2, N):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int J: The number of ordinal classes.
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    z1 = (gamma_tplus1 - b) / noise_std
    z2 = (gamma_t - b) / noise_std
    x = (b - mean) / sigma
    std = np.sqrt(noise_variance + posterior_covariance)
    y = (posterior_mean - gamma_tplus1) / std
    out = np.empty(N)
    for i in prange(N):
        out[i] = (
            norm_pdf(x[i]) / sigma[i]) / return_prob(z1[i], z2[i], EPS_2) * (
            norm_pdf(y[i]) / std[i])
    return out


@njit(parallel=True)
def fromb_fft4_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_std, noise_variance, gamma_t, gamma_tplus1,
        EPS_2, N):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    z1 = (gamma_tplus1 - b) / noise_std
    z2 = (gamma_t - b) / noise_std
    x = (b - mean) / sigma
    std = np.sqrt(noise_variance + posterior_covariance)
    y = (posterior_mean - gamma_tplus1) / std
    z = gamma_tplus1 - b
    out = np.empty(N)
    for i in prange(N):
        out[i] = (
            norm_pdf(x[i]) / sigma[i]) / return_prob(
                z1[i], z2[i], EPS_2) * (norm_pdf(y[i]) / std[i]) * z[i]
    return out


@njit(parallel=True)
def fromb_fft5_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_variance, noise_std,
        gamma_t, gamma_tplus1,
        EPS_2, N):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    z1 = (gamma_tplus1 - b) / noise_std
    z2 = (gamma_t - b) / noise_std
    x = (b - mean) / sigma
    std = np.sqrt(noise_variance + posterior_covariance)
    y = (posterior_mean - gamma_t) / std
    z = gamma_t - b
    out = np.empty(N)
    for i in prange(N):
        out[i] = (norm_pdf(x[i]) / sigma[i]) / return_prob(
            z1[i], z2[i], EPS_2) * (norm_pdf(y[i]) / std[i]) * z[i]
    return out


def fromb_t1_vector(
        y, posterior_mean, posterior_covariance, gamma_t,
        gamma_tplus1, noise_std, EPS, EPS_2, N):
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
    posterior_std = np.sqrt(posterior_covariance)
    a = posterior_mean - 5.0 * posterior_std
    b = posterior_mean + 5.0 * posterior_std
    h = b - a
    y[0, :] = h * (
        fromb_fft1_vector(
            a, posterior_mean, posterior_std, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
        + fromb_fft1_vector(
            b, posterior_mean, posterior_std, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
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
                EPS_2, N)
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


def fromb_t2_vector(
    y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
    gamma_t, gamma_tplus1,
    noise_variance, noise_std, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int J: The number of ordinal classes.
    :arg gamma_t: The vector of the lower cutpoints the data.
    :type gamma_t: `nd.numpy.array`
    :arg gamma_t_plus_1: The vector of the upper cutpoints the data.
    :type gamma_t_plus_1: `nd.numpy.array`
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_std: A noise standard deviation for the likelihood.
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    y[0, :] = h * (
        fromb_fft2_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
        + fromb_fft2_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std, gamma_t, gamma_tplus1,
            EPS_2, N)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (np.any(ep>=EPS) and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p += fromb_fft2_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
                gamma_t, gamma_tplus1,
                EPS_2, N)
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


def fromb_t3_vector(
    y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
    gamma_t, gamma_tplus1,
    noise_std, noise_variance, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int J: The number of ordinal classes.
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    y[0, :] = h * (
        fromb_fft3_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
        + fromb_fft3_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (np.any(ep>=EPS) and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p = p + fromb_fft3_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
                gamma_t, gamma_tplus1,
                EPS_2, N)
        p = (y[0, :] + h * p) / 2.0
        s = 1.0
        for k in range(m):
            s *= 4.0
            q = (s * p - y[k]) / (s - 1.0)
            y[k, :] = p
            p = q
        ep = np.abs(q - y[m - 1, :])
        m += 1
        y[m - 1, :] = q
        n += n
        h /= 2.0
    return q


def fromb_t4_vector(
    y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
    gamma_t, gamma_tplus1,
    noise_variance, noise_std, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    y[0, :] = h * (
        fromb_fft4_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
        + fromb_fft4_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (np.any(ep>=EPS) and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p = p + fromb_fft4_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
                gamma_t, gamma_tplus1,
                EPS_2, N)
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


def fromb_t5_vector(
        y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
        gamma_t, gamma_tplus1,
        noise_variance, noise_std, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    y[0, :] = h * (
        fromb_fft5_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
        + fromb_fft5_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2, N)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (np.any(ep>=EPS) and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p += fromb_fft5_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
                gamma_t, gamma_tplus1,
                EPS_2, N)
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


def sample_y(y, m, t_train, gamma, noise_std, N):
    for i in prange(N):
        # Target class index
        j_true = t_train[i]
        y_i = np.NINF  # this is a trick for the next line
        # Sample from the truncated Gaussian
        while y_i > gamma[j_true + 1] or y_i <= gamma[j_true]:
            # sample y
            y_i = m[i] + np.random.normal(loc=m[i], scale=noise_std)
        # Add sample to the Y vector
        y[i] = y_i
    return y
