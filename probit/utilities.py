"""Utility functions for probit."""
import numpy as np
from scipy.stats import expon, norm
from scipy.special import ndtr, log_ndtr
import warnings


over_sqrt_2_pi = 1. / np.sqrt(2 * np.pi)
log_over_sqrt_2_pi = np.log(over_sqrt_2_pi)


def norm_z_pdf(z):
    return over_sqrt_2_pi * np.exp(- z**2 / 2.0 )


def norm_z_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


def norm_logpdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_logpdf(z) - np.log(scale)


def norm_cdf(x):
    return ndtr(x)


def norm_logcdf(x):
    return log_ndtr(x)


def return_prob_vector(b, gamma_t, gamma_tplus1, noise_std):
    return ndtr((gamma_tplus1 - b) / noise_std) - ndtr((gamma_t - b) / noise_std)


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
    return norm_pdf(b, loc=mean, scale=sigma) * np.log(prob)
    #return norm.pdf(b, loc=mean, scale=sigma) * np.log(prob)


def fromb_t1_vector(
    y, posterior_mean, posterior_covariance, gamma_t, gamma_tplus1,
    noise_std, EPS, EPS_2, N):
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
    return norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=gamma_t, scale=np.sqrt(
        noise_variance + posterior_covariance))


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
    return  norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=gamma_tplus1, scale=np.sqrt(
        noise_variance + posterior_covariance))


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
    return norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=gamma_tplus1, scale=np.sqrt(
        noise_variance + posterior_covariance)) * (gamma_tplus1 - b)


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
    return norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=gamma_t, scale=np.sqrt(
        noise_variance + posterior_covariance)) * (gamma_t - b)


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


def _g(x):
    """
    Polynomial part of a series expansion for log survival function for a
    normal random variable. With the third term, for x>4, this is accurate
    to three decimal places. The third term becomes significant when sigma
    is large. 
    """
    return -1. / x**2 + 5/ (2 * x**4) - 37 / (3 *  x**6)


def _Z_tails(z1, z2):
    """
    Series expansion at infinity.

    Even for z1, z2 >= 4 this is accurate to three decimal places.
    """
    return 1/np.sqrt(2 * np.pi) * (
    1 / z1 * np.exp(-0.5 * z1**2 + _g(z1)) - 1 / z2 * np.exp(
        -0.5 * z2**2 + _g(z2)))


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    return 1 / (z * np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2 + _g(z))


def dp_tails(self, z1, z2):
    """Series expansion at infinity."""
    return (
        z1 * np.exp(-0.5 * z1**2) - z2 * np.exp(-0.5 * z2**2)) / (
            1 / z1 * np.exp(-0.5 * z1**2)* np.exp(_g(z1))
            - 1 / z2 * np.exp(-0.5 * z2**2) * np.exp(_g(z2)))


def dp_far_tails(z):
    """Prevents overflow at large z."""
    return z**2 * np.exp(-_g(z))


def p_tails(z1, z2):
    """
    Series expansion at infinity. Even for z1, z2 >= 4,
    this is accurate to three decimal places.
    """
    return (
        np.exp(-0.5 * z1**2) - np.exp(-0.5 * z2**2)) / (
            1 / z1 * np.exp(-0.5 * z1**2)* np.exp(_g(z1))
            - 1 / z2 * np.exp(-0.5 * z2**2) * np.exp(_g(z2)))


def p_far_tails(z):
    """Prevents overflow at large z."""
    return z * np.exp(-_g(z))


def truncated_norm_normalising_constant(
        gamma_ts, gamma_tplus1s, noise_std, m, EPS,
        upper_bound=None, upper_bound2=None, numerically_stable=False):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

    TODO: Make a numba version, shouldn't be difficult, but will have to be
        parallelised scalar (due to the boolean logic).
    TODO: There is no way to calculate this in the log domain (unless expansion
    approximations are used). Could investigate only using approximations here.
    :arg gamma_ts: gamma[t_train] (N, ) array of cutpoints
    :type gamma_ts: :class:`numpy.ndarray`
    :arg gamma_tplus1s: gamma[t_train + 1] (N, ) array of cutpoints
    :type gamma_ts: :class:`numpy.ndarray`
    :arg float noise_std: The noise standard deviation.
    :arg m: The mean vector.
    :type m: :class:`numpy.ndarray`
    :arg float EPS: The tolerated absolute error.
    :arg float upper_bound: The threshold of the normal z value for which
        the pdf is close enough to zero.
    :arg float upper_bound2: The threshold of the normal z value for which
        the pdf is close enough to zero. 
    :arg bool numerical_stability: If set to true, will calculate in a
        numerically stable way. If set to false,
        will calculate in a faster, but less numerically stable way.
    :returns: (
        Z,
        norm_pdf_z1s, norm_pdf_z2s,
        norm_cdf_z1s, norm_cdf_z2s,
        z1s, z2s)
    :rtype: tuple (
        :class:`numpy.ndarray`,
        :class:`numpy.ndarray`, :class:`numpy.ndarray`,
        :class:`numpy.ndarray`, :class:`numpy.ndarray`,
        :class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    # Otherwise
    z1s = (gamma_ts - m) / noise_std
    z2s = (gamma_tplus1s - m) / noise_std
    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)
    norm_cdf_z1s = norm_cdf(z1s)
    norm_cdf_z2s = norm_cdf(z2s)
    ## SS: test timings
    # norm_pdf_z1s = norm.pdf(z1s)
    # norm_pdf_z2s = norm.pdf(z2s)
    # norm_cdf_z1s = norm.cdf(z1s)
    # norm_cdf_z2s = norm.cdf(z2s)
    Z = norm_cdf_z2s - norm_cdf_z1s
    if upper_bound is not None:
        # Using series expansion approximations
        indices1 = np.where(z1s > upper_bound)
        indices2 = np.where(z2s < -upper_bound)
        #print(indices1)
        #print(indices2)
        if np.ndim(z1s) == 1:
            indices = np.union1d(indices1, indices2)
        elif np.ndim(z1s) == 2:  # m is (num_samples, N). This is a quick (but not dirty) hack.
            indices = (np.append(indices1[0], indices2[0]), np.append(indices1[1], indices2[1]))
        z1_indices = z1s[indices]
        z2_indices = z2s[indices]
        Z[indices] = _Z_tails(
            z1_indices, z2_indices)
        if upper_bound2 is not None:
            indices = np.where(z1s > upper_bound2)
            z1_indices = z1s[indices]
            Z[indices] = _Z_far_tails(
                z1_indices)
            indices = np.where(z2s < -upper_bound2)
            z2_indices = z2s[indices]
            Z[indices] = _Z_far_tails(
                -z2_indices)
    if numerically_stable is True:
        small_densities = np.where(Z < EPS)
        if np.size(small_densities) != 0:
            warnings.warn(
                "Z (normalising constants for truncated norma"
                "l random variables) must be greater than"
                " tolerance={} (got {}): SETTING to"
                " Z_ns[Z_ns<tolerance]=tolerance\nz1s={}, z2s={}".format(
                    EPS, Z, z1s, z2s))
            Z[small_densities] = EPS
    return (
        Z,
        norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, norm_cdf_z1s, norm_cdf_z2s)


def p(m, gamma_ts, gamma_tplus1s, noise_std,
        EPS, upper_bound, upper_bound2):
    """
    The rightmost term of 2021 Page Eq.(),
        correction terms that squish the function value m
        between the two cutpoints for that particle.

    :arg m: The current posterior mean estimate.
    :type m: :class:`numpy.ndarray`
    :arg gamma_ts: gamma[t_train] (N, ) array of cutpoints
    :type gamma_ts: :class:`numpy.ndarray`
    :arg gamma_tplus1s: gamma[t_train + 1] (N, ) array of cutpoints
    :type gamma_ts: :class:`numpy.ndarray`
    :arg float noise_std: The noise standard deviation.
    :arg float EPS: The tolerated absolute error.
    :arg float upper_bound: Threshold of single sided standard
        deviations that the normal cdf can be approximated to 0 or 1.
    :arg float upper_bound2: Optional threshold to be robust agains
        numerical overflow. Default `None`.

    :returns: p
    :rtype: :class:`numpy.ndarray`
    """
    (Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s,
    *_) = truncated_norm_normalising_constant(
        gamma_ts, gamma_tplus1s, noise_std, m, EPS)
    p = (norm_pdf_z1s - norm_pdf_z2s) / Z
    # Need to deal with the tails to prevent catestrophic cancellation
    indices1 = np.where(z1s > upper_bound)
    indices2 = np.where(z2s < -upper_bound)
    indices = np.union1d(indices1, indices2)
    z1_indices = z1s[indices]
    z2_indices = z2s[indices]
    p[indices] = p_tails(z1_indices, z2_indices)
    # Finally, get the far tails for the non-infinity case to prevent overflow
    if upper_bound2:
        indices = np.where(z1s > upper_bound2)
        z1_indices = z1s[indices]
        p[indices] = p_far_tails(z1_indices)
        indices = np.where(z2s < -upper_bound2)
        z2_indices = z2s[indices]
        p[indices] = p_far_tails(z2_indices)
    return p


def dp(m, gamma_ts, gamma_tplus1s, noise_std, EPS,
        upper_bound, upper_bound2=None):
    """
    The analytic derivative of :meth:`p` (p are the correction
        terms that squish the function value m
        between the two cutpoints for that particle).

    :arg m: The current posterior mean estimate.
    :type m: :class:`numpy.ndarray`
    :arg gamma_ts: gamma[t_train] (N, ) array of cutpoints
    :type gamma_ts: :class:`numpy.ndarray`
    :arg gamma_tplus1s: gamma[t_train + 1] (N, ) array of cutpoints
    :type gamma_ts: :class:`numpy.ndarray`
    :arg float noise_std: The noise standard deviation.
    :arg float EPS: The tolerated absolute error.
    :arg float upper_bound: Threshold of single sided standard
        deviations that the normal cdf can be approximated to 0 or 1.
    :arg float upper_bound2: Optional threshold to be robust agains
        numerical overflow. Default `None`.

    :returns: dp
    :rtype: :class:`numpy.ndarray`
    """
    (Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s,
    *_) = truncated_norm_normalising_constant(
        gamma_ts, gamma_tplus1s, noise_std, m, EPS)
    sigma_dp = (z1s * norm_pdf_z1s - z2s * norm_pdf_z2s) / Z
    # Need to deal with the tails to prevent catestrophic cancellation
    indices1 = np.where(z1s > upper_bound)
    indices2 = np.where(z2s < -upper_bound)
    indices = np.union1d(indices1, indices2)
    z1_indices = z1s[indices]
    z2_indices = z2s[indices]
    sigma_dp[indices] = dp_tails(z1_indices, z2_indices)
    # The derivative when (z2/z1) take a value of (+/-)infinity
    indices = np.where(z1s==-np.inf)
    sigma_dp[indices] = (- z2s[indices] * norm_pdf_z2s[indices]
        / Z[indices])
    indices = np.intersect1d(indices, indices2)
    sigma_dp[indices] = dp_far_tails(z2s[indices])
    indices = np.where(z2s==np.inf)
    sigma_dp[indices] = (z1s[indices] * norm_pdf_z1s[indices]
        / Z[indices])
    indices = np.intersect1d(indices, indices1)
    sigma_dp[indices] = dp_far_tails(z1s[indices])
    # Get the far tails for the non-infinity case to prevent overflow
    if upper_bound2 is not None:
        indices = np.where(z1s > upper_bound2)
        z1_indices = z1s[indices]
        sigma_dp[indices] = dp_far_tails(z1_indices)
        indices = np.where(z2s < -upper_bound2)
        z2_indices = z2s[indices]
        sigma_dp[indices] = dp_far_tails(z2_indices)
    return sigma_dp


def sample_y(y, m, t_train, gamma, noise_std, N):
    for i in range(N):
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
