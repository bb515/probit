"""Utility functions for probit."""
import numpy as np
from scipy.stats import norm, expon
from scipy.special import erf
import math
import matplotlib.pyplot as plt


# def return_prob_vector(b, gamma_t, gamma_tplus1, noise_std):
#     return norm.cdf(
#         (gamma_tplus1 - b) / noise_std
#         ) - norm.cdf(
#             (gamma_t - b) / noise_std)

def return_prob_vector(b, gamma_t, gamma_tplus1, noise_std):
    return 0.5*(erf((gamma_tplus1 - b) / noise_std)
                - erf((gamma_t - b) / noise_std))

# def return_prob_vector(
#     N, b, gamma_t, gamma_tplus1,
#     where_t_0, where_t_neither, where_t_Jminus1, noise_std,
#     numerically_stable=True):
#     """Return a vector of Gaussian probabilities."""
#     if numerically_stable:
#         phi1 = np.ones(N)
#         phi2 = np.zeros(N)
#         phi1[where_t_neither] = norm.cdf(
#             (gamma_tplus1[where_t_neither] - b[where_t_neither]) / noise_std)
#         phi2[where_t_neither] = norm.cdf(
#             (gamma_t[where_t_neither] - b[where_t_neither]) / noise_std)
#         phi1[where_t_0] = norm.cdf(
#             (gamma_tplus1[where_t_0] - b[where_t_0]) / noise_std)
#         phi2[where_t_Jminus1] = norm.cdf(
#             (gamma_t[where_t_0] - b[where_t_0]) / noise_std)
#         return phi1 - phi2
#     else:
#         return norm.cdf(
#         (gamma_tplus1 - b) / noise_std
#         ) - norm.cdf(
#             (gamma_t - b) / noise_std)


def return_prob(b, t, J, gamma, noise_std, numerically_stable=True):
    """Return a Gaussian probability."""
    if numerically_stable:
        phi1 = 1.0
        phi2 = 0.0
        if t >= 1:
            z2 = (gamma[t] - b) / noise_std
            phi2 = norm.cdf(z2)
        if t <= J - 2:
            z1 = (gamma[t + 1] - b) / noise_std
            phi1 = norm.cdf(z1)
    else:
        phi1 = norm.cdf((gamma[t] - b) / noise_std)
        phi2 = norm.cdf((gamma[t + 1] - b) / noise_std)
    return phi1 - phi2


def fromb_fft1(b, mean, sigma, t, J, gamma, noise_variance, EPS):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int t: The target value for the datapoint, indexed from 0.
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `nd.numpy.array`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    noise_std = np.sqrt(noise_variance)
    func = norm.pdf(b, loc=mean, scale=sigma)
    prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
    if prob < EPS*EPS:
        prob = EPS*EPS
    func = func * np.log(prob)
    if math.isnan(func):
        raise ValueError("fft1 {} {} {} {}".format(t, gamma[t], prob, func))
    return func


def fromb_fft1_vector(
    b, mean, sigma, noise_std, gamma_t, gamma_tplus1,
    EPS_2):
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
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    prob = return_prob_vector(
        b, gamma_t, gamma_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm.pdf(b, loc=mean, scale=sigma) * np.log(prob)


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


def fromb_fft2(
    b, mean, sigma, posterior_mean, posterior_covariance, t, J,
    gamma, noise_variance, EPS):
    """
    :arg b: The approximate posterior mean evaluated at the datapoint.
    :arg mean: A mean value of a pdf inside the integrand.
    :arg sigma: A standard deviation of a pdf inside the integrand.
    :arg t: The target value for the datapoint.
    :arg J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :arg noise_variance: A noise variance for the likelihood.
    :arg EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    noise_std = np.sqrt(noise_variance)
    EPS_2 = EPS * EPS
    func = norm.pdf(b, loc=mean, scale=sigma)
    prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
    if prob < EPS_2:
        prob = EPS_2
    func = func / prob * norm.pdf(posterior_mean, loc=gamma[t], scale=np.sqrt(
        noise_variance + posterior_covariance))
    if math.isnan(func):
        raise ValueError("{} {} {} {} ({} {} {})".format(
            t, gamma[t], prob, func, b, mean, sigma))
    return func


def fromb_fft2_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_variance, noise_std, gamma_t, gamma_tplus1,
        EPS_2):
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
    prob = return_prob_vector(
        b, gamma_t, gamma_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
        posterior_mean, loc=gamma_t, scale=np.sqrt(
        noise_variance + posterior_covariance))


def fromb_t2(
        posterior_mean, posterior_covariance, t, J,
        gamma, noise_variance, EPS):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int t: The target for the datapoint
    :arg int J: The number of ordinal classes.
    :arg gamma: The vector of cutpoints.
    :type gamma: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    if t == 0:
        return 0
    mean = (posterior_mean * noise_variance
            + posterior_covariance * gamma[t]) / (
            noise_variance + posterior_covariance)
    sigma = np.sqrt((noise_variance * posterior_covariance) / (
            noise_variance + posterior_covariance))
    y = np.zeros((20,))
    a = mean - 5.0 * sigma
    b = mean + 5.0 * sigma
    h = b - a
    y[0] = h * (
        fromb_fft2(
            a, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
        + fromb_fft2(
            b, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (ep >=EPS and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p += fromb_fft2(
                x, mean, sigma, posterior_mean, posterior_covariance, t,
                J, gamma, noise_variance, EPS)
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


def fromb_t2_vector(
    y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
    gamma_t, gamma_tplus1,
    noise_variance, noise_std, EPS):
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
    EPS_2 = EPS**2
    y[0, :] = h * (
        fromb_fft2_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
        + fromb_fft2_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std, gamma_t, gamma_tplus1,
            EPS_2)
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


def fromb_fft3(
        b, mean, sigma, posterior_mean, posterior_covariance, t, J,
        gamma, noise_variance, EPS, numerically_stable=True):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
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
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    EPS_2 = EPS * EPS
    noise_std = np.sqrt(noise_variance)
    func = norm.pdf(b, loc=mean, scale=sigma)
    prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
    if prob < EPS_2:
        prob = EPS_2
    func = func / prob * norm.pdf(
        posterior_mean, loc=gamma[t + 1], scale=np.sqrt(
        noise_variance + posterior_covariance))
    if math.isnan(func):
        raise ValueError("{} {} {} {}".format(t, gamma[t], prob, func))
    return func


def fromb_fft3_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_variance, noise_std, gamma_t, gamma_tplus1,
        EPS_2):
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
    prob = return_prob_vector(
        b, gamma_t, gamma_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return  norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
        posterior_mean, loc=gamma_tplus1, scale=np.sqrt(
        noise_variance + posterior_covariance))


def fromb_t3(
    posterior_mean, posterior_covariance, t, J, gamma, noise_variance, EPS):
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
    if t == J - 1:
        return 0
    mean = (posterior_mean * noise_variance
            + posterior_covariance * gamma[t + 1]) / (
            noise_variance + posterior_covariance)
    sigma = np.sqrt((noise_variance * posterior_covariance)
                    / (noise_variance + posterior_covariance))
    y = np.zeros((20,))
    a = mean - 5.0 * sigma
    b = mean + 5.0 * sigma
    h = b - a
    y[0] = h * (
        fromb_fft3(
            a, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
        + fromb_fft3(
            b, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (ep >=EPS and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p = p + fromb_fft3(
                x, mean, sigma, posterior_mean, posterior_covariance, t, J,
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


def fromb_t3_vector(
    y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
    gamma_t, gamma_tplus1,
    noise_std, noise_variance, EPS):
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
    EPS_2 = EPS**2
    y[0, :] = h * (
        fromb_fft3_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
        + fromb_fft3_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
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
            p = p + fromb_fft3_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
                gamma_t, gamma_tplus1,
                EPS_2)
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


def fromb_fft4(
        b, mean, sigma, posterior_mean, posterior_covariance, t, J,
        gamma, noise_variance, EPS):
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
    EPS_2 = EPS * EPS
    noise_std = np.sqrt(noise_variance)
    func = norm.pdf(b, loc=mean, scale=sigma)
    prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
    if prob < EPS_2:
        prob = EPS_2
    func = func / prob * norm.pdf(
        posterior_mean, loc=gamma[t + 1], scale=np.sqrt(
        noise_variance + posterior_covariance)) * (gamma[t + 1] - b)
    return func


def fromb_fft4_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_std, noise_variance, gamma_t, gamma_tplus1,
        EPS_2):
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
    prob = return_prob_vector(
        b, gamma_t, gamma_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
        posterior_mean, loc=gamma_tplus1, scale=np.sqrt(
        noise_variance + posterior_covariance)) * (gamma_tplus1 - b)


def fromb_t4(
    posterior_mean, posterior_covariance, t, J, gamma, noise_variance, EPS):
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
    if t == J - 1:
        return 0
    mean = (posterior_mean * noise_variance
            + posterior_covariance * gamma[t + 1]) / (
            noise_variance + posterior_covariance)
    sigma = np.sqrt((noise_variance * posterior_covariance)
            / (noise_variance + posterior_covariance))
    y = np.zeros((20,))
    a = mean - 5.0 * sigma
    b = mean + 5.0 * sigma
    h = b - a
    y[0] = h * (
        fromb_fft4(
            a, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
        + fromb_fft4(
            b, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (ep >=EPS and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p = p + fromb_fft4(
                x, mean, sigma, posterior_mean, posterior_covariance, t, J,
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


def fromb_t4_vector(
    y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
    gamma_t, gamma_tplus1,
    noise_variance, noise_std, EPS):
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
    EPS_2 = EPS**2
    y[0, :] = h * (
        fromb_fft4_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
        + fromb_fft4_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
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
            p = p + fromb_fft4_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
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


def fromb_fft5(
    b, mean, sigma, posterior_mean, posterior_covariance, t, J,
    gamma, noise_variance, EPS):
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
    EPS_2 = EPS * EPS
    noise_std = np.sqrt(noise_variance)
    func = norm.pdf(b, loc=mean, scale=sigma)
    prob = return_prob(b, t, J, gamma, noise_std, numerically_stable=True)
    if prob < EPS_2:
        prob = EPS_2
    func = func / prob * norm.pdf(posterior_mean, loc=gamma[t], scale=np.sqrt(
        noise_variance + posterior_covariance)) * (gamma[t] - b)
    return func


def fromb_fft5_vector(
        b, mean, sigma, posterior_mean, posterior_covariance,
        noise_variance, noise_std,
        gamma_t, gamma_tplus1,
        EPS_2):
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
    prob = return_prob_vector(
        b, gamma_t, gamma_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm.pdf(b, loc=mean, scale=sigma) / prob * norm.pdf(
        posterior_mean, loc=gamma_t, scale=np.sqrt(
        noise_variance + posterior_covariance)) * (gamma_t - b)


def fromb_t5(
        posterior_mean, posterior_covariance, t, J, gamma, noise_variance,
        EPS):
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
    if t == 0:
        return 0
    mean = (posterior_mean * noise_variance
            + posterior_covariance * gamma[t]) / (
            noise_variance + posterior_covariance)
    sigma = np.sqrt((noise_variance * posterior_covariance)
            / (noise_variance + posterior_covariance))
    y = np.zeros((20,))
    a = mean - 5.0 * sigma
    b = mean + 5.0 * sigma
    h = b - a
    y[0] = h * (
        fromb_fft5(
            a, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
        + fromb_fft5(
            b, mean, sigma, posterior_mean, posterior_covariance, t, J,
            gamma, noise_variance, EPS)
    ) / 2.0
    m = 1
    n = 1
    ep = EPS + 1.0
    while (ep >=EPS and m <=19):
        p = 0.0
        for i in range(n):
            x = a + (i + 0.5) * h
            p += fromb_fft5(
                x, mean, sigma, posterior_mean, posterior_covariance, t, J,
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


def fromb_t5_vector(
        y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
        gamma_t, gamma_tplus1,
        noise_variance, noise_std, EPS):
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
    EPS_2 = EPS**2
    y[0, :] = h * (
        fromb_fft5_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            gamma_t, gamma_tplus1,
            EPS_2)
        + fromb_fft5_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
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
            p += fromb_fft5_vector(
                x, mean, sigma, posterior_mean, posterior_covariance,
                noise_variance, noise_std,
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


def vectorised_multiclass_unnormalised_log_multivariate_normal_pdf(
        Ms, mean=None, covs=None):
    """
    Evaluate a vector of log multivariate normal pdf in a numerically stable
    way.

    :arg Ms: (N, J)
    :arg int n_samples: n_samples
    :arg covs: (n_samples, J, N, N)
    :return: (n_samples, J)
    """
    if covs is None:
        raise ValueError(
            "Must provide sample covariance"
            " matrices ({} was provided)".format(covs))
    if mean is not None:
        Ms = Ms - mean
    Ls = np.linalg.cholesky(covs)
    L_invs = np.linalg.inv(Ls)
    K_invs = L_invs.transpose((0, 1, 3, 2)) @ L_invs  # (n_samples, J, N, N)
    half_log_det_covs = np.trace(np.log(Ls), axis1=2, axis2=3)
    # A stacked ms.T @ J_inv @ ms
    out = np.einsum('ijkl, lj->ijk', K_invs, Ms)
    out = np.einsum('ijk, kj->ij', out, Ms)
    return -1. * half_log_det_covs - 0.5 * out  # ((n_samples, J, N, N) @ (N, )) @ (N,)


def vectorised_unnormalised_log_multivariate_normal_pdf(
        ms, mean=None, covs=None):
    """
    Evaluate a vector of log multivariate normal pdf in a
    numerically stable way.
    """
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
    J_inv = L_inv.T @ L_inv
    half_log_det_cov = np.trace(np.log(L))
    return -1. * half_log_det_cov - 0.5 * x.T @ J_inv @ x


def unnormalised_multivariate_normal_pdf(x, mean=None, cov=None):
    """Evaluate the multivariate normal pdf in a numerically stable way."""
    if cov is None:
        cov = np.eye(len(x))
    if mean is None:
        #return np.exp(-0.5 * np.log(np.linalg.det(cov)) - 0.5 * np.dot(x, np.linalg.solve(cov, x)))
        L = np.linalg.cholesky(cov)
        L_inv = np.linalg.inv(L)
        J_inv = L_inv.T @ L_inv
        return np.exp(-0.5 * np.log(np.linalg.det(cov)) - 0.5 * x @ J_inv @ x)
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


def sample_U(J, different_across_classes=None):  # TODO: Has been superseded by sample_Us
    if not different_across_classes:
        # Will this induce an unwanted link between the y_nk across k? What effect will it have?
        u = norm.rvs(0, 1, 1)
        U = np.multiply(u, np.ones((J, J)))
    else:
        # This might be a better option as the sample u across classes shouldn't be correlated, does it matter though?
        u = norm.rvs(0, 1, J)
        U = np.tile(u, (J, 1))
        # Needs to be the same across rows, as we sum over the rows
        U = U.T
    return U


def sample_Us(J, n_samples, different_across_classes=None):
    if not different_across_classes:
        # Will this induce an unwanted link between the y_nk across k? What effect will it have?
        us = norm.rvs(0, 1, (n_samples, 1, 1))
        ones = np.ones((n_samples, J, J))
        Us = np.multiply(us, ones)
    else:
        # This might be a better option as the sample u across classes shouldn't be correlated, does it matter though?
        us = norm.rvs(0, 1, (n_samples, J, 1))
        # Needs to be the same across rows, as we sum over the rows
        Us = np.tile(us, (1, 1, J))
    return Us


def sample_Us_vector(J, n_samples, N_test, different_across_classes=None):
    if not different_across_classes:
        # Will this induce an unwanted link between the y_nk across k? What effect will it have?
        us = norm.rvs(0, 1, (n_samples, 1, 1, 1))
        ones = np.ones((n_samples, N_test, J, J))
        Us = np.multiply(us, ones)
    else:
        # TODO: test this.
        # This might be a better option as the sample u across classes shouldn't be correlated, does it matter though?
        us = norm.rvs(0, 1, (n_samples, 1, J, 1))
        # Needs to be the same across rows, as we sum over the rows
        Us = np.tile(us, (1, N_test, 1, J))
    return Us


def matrix_of_differences(m_n, J):  # TODO: superseded by matrix_of_differencess
    """
    Get a matrix of differences of the vector m.

    :arg m: is an (J, 1) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
    :type m: :class:`numpy.ndarray`
    """
    # Find matrix of coefficients
    Lambda = np.tile(m_n, (J, 1))
    Lambda_T = Lambda.T
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    difference = Lambda_T - Lambda
    return difference


def matrix_of_VB_differences(m_n, J, t_n):
    """
    Get a matrix of differences of the vector m.

    Superseded by matrix_of_VB_differencess, but is still needed in the non-vectorised case for Sparse GP regression.

    :arg m: is an (J, 1) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
    :type m: :class:`numpy.ndarray`
    """
    # Find matrix of coefficients
    Lambda = np.tile(m_n, (J, 1))
    Lambda_t_n = np.tile(m_n[t_n], (J, J))
    # Should get the same rows out
    difference = Lambda_t_n - Lambda
    return difference


def matrix_of_differencess(M, J, N_test):
    """
    Get an array of matrix of differences of the vectors m_ns.

    :arg M: An (N_test, J) array filled with e.g. m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object. Or in the VB implementation, this function  is used for M_tilde (N, J).
    :type M: :class:`numpy.ndarray`
    :arg J: The number of classes.
    :arg N_test: The number of test objects.
    """
    M = M.reshape((N_test, J, 1))
    # Lambdas is an (N_test, J, J) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambda_Ts = np.tile(M, (1, 1, J))
    Lambdas = Lambda_Ts.transpose((0, 2, 1))
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    return np.subtract(Lambda_Ts, Lambdas)  # (N_test, J, J)


def matrix_of_VB_differencess(M, J, N_test, t, grid):
    """
    Get an array of matrix of differences of the vectors m_ns.

    :arg M: An (N_test, J) array filled with e.g. m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object. Or in the VB implementation, this function  is used for M_tilde (N, J).
    :type M: :class:`numpy.ndarray`
    :arg J: The number of classes.
    :arg N_test: The number of test objects.
    """
    # Lambdas is an (N_test, J, J) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambda_t_ns = np.tile(M[grid, t], (1, J * J))
    Lambda_t_ns = Lambda_t_ns.reshape((N_test, J, J))
    Lambdas = np.tile(M, (1, 1, J))
    Lambdas = Lambdas.reshape((N_test, J, J))
    # Matrix of differences for VB, the rows are the same and contain the elements of the product of interest
    return np.subtract(Lambda_t_ns, Lambdas)  # (N_test, J, J)


def matrix_of_valuess(nu, J, N_test):
    """
    Get an array of matrix of values of the vectors M.

    :param nu: An (N_test, J) array filled with nu_tilde_k^{new_i} where  k is the class indicator
        and i is the index of the test object.
    :param J: The number of classes.
    :param N_test: The number of test objects.

    :return: (N_test, J, J) Matrix of rows of nu_tilde_k^{new_i}, where the rows(=axis 2) contain identical values.
    """
    nu = nu.reshape((N_test, J, 1))
    # Lambdas is an (N_test, J, J) stack of Lambda matrices
    # Tile along the rows, as they are the elements of interest
    Lambdas_Ts = np.tile(nu, (1, 1, J))
    return Lambdas_Ts  # (N_test, J, J)
