"""Utility functions for probit."""
import numpy as np
from scipy.stats import expon
from scipy.special import ndtr, log_ndtr, erf
import warnings
import h5py


def check_cutpoints(cutpoints, J):
    """
    Check that the cutpoints are compatible with this class.

    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    """
    # Convert cutpoints to numpy array
    cutpoints = np.array(cutpoints)
    # Not including -\infty or \infty
    if np.shape(cutpoints)[0] == J - 1:
        # Append \infty
        cutpoints = np.append(cutpoints, np.inf)
        # Insert -\infty at index 0
        cutpoints = np.insert(cutpoints, 0, np.NINF)
        pass  # correct format
    # Not including one cutpoints
    elif np.shape(cutpoints)[0] == J:
        if cutpoints[-1] != np.inf:
            if cutpoints[0] != np.NINF:
                raise ValueError(
                    "Either the largest cutpoint parameter b_J is not "
                    "positive infinity, or the smallest cutpoint "
                    "parameter must b_0 is not negative infinity."
                    "(got {}, expected {})".format(
                    [cutpoints[0], cutpoints[-1]], [np.inf, np.NINF]))
            else:  #cutpoints[0] is -\infty
                cutpoints.append(np.inf)
                pass  # correct format
        else:
            cutpoints = np.insert(cutpoints, 0, np.NINF)
            pass  # correct format
    # Including all the cutpoints
    elif np.shape(cutpoints)[0] == J + 1:
        if cutpoints[0] != np.NINF:
            raise ValueError(
                "The smallest cutpoint parameter b_0 must be negative "
                "infinity (got {}, expected {})".format(
                    cutpoints[0], np.NINF))
        if cutpoints[-1] != np.inf:
            raise ValueError(
                "The largest cutpoint parameter b_J must be positive "
                "infinity (got {}, expected {})".format(
                    cutpoints[-1], np.inf))
        pass  # correct format
    else:
        raise ValueError(
            "Could not recognise cutpoints shape. "
            "(np.shape(cutpoints) was {})".format(np.shape(cutpoints)))
    assert cutpoints[0] == np.NINF
    assert cutpoints[-1] == np.inf
    assert np.shape(cutpoints)[0] == J + 1
    if not all(
            cutpoints[i] <= cutpoints[i + 1]
            for i in range(J)):
        raise CutpointValueError(cutpoints)
    return cutpoints


def write_array(write_path, dataset, array):
    """
    Write a :class: numpy.ndarray to a HDF5 file.
    :arg write_path: The path to which the HDF5 file is written.
    :type write_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :array: The array to be written to file.
    :type array: :class: numpy.ndarray
    :return: None
    :rtype: None type
    """
    with h5py.File(write_path, 'a') as hf:
        hf.create_dataset(dataset,  data=array)


def read_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.
    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                array = hf[dataset][:]
                return array
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
                raise
    except OSError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        raise



def read_scalar(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.
    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                scalar = hf[dataset][()]
                return scalar 
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
                raise
    except OSError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        raise

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


def return_prob_vector(b, cutpoints_t, cutpoints_tplus1, noise_std):
    return ndtr(
        (cutpoints_tplus1 - b) / noise_std) - ndtr(
            (cutpoints_t - b) / noise_std)


def fromb_fft1_vector(
        b, mean, sigma, noise_std, cutpoints_t, cutpoints_tplus1, EPS_2):
    """
    :arg float b: The approximate posterior mean vector.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int J: The number of ordinal classes.
    :arg cutpoints_t: The vector of the lower cutpoints the data.
    :type cutpoints_t: `nd.numpy.array`
    :arg cutpoints_t_plus_1: The vector of the upper cutpoints the data.
    :type cutpoints_t_plus_1: `nd.numpy.array`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    prob = return_prob_vector(
        b, cutpoints_t, cutpoints_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm_pdf(b, loc=mean, scale=sigma) * np.log(prob)


def fromb_t1_vector(
        y, posterior_mean, posterior_covariance, cutpoints_t, cutpoints_tplus1,
        noise_std, EPS, EPS_2, N):
    """
    :arg posterior_mean: The approximate posterior mean vector.
    :type posterior_mean: :class:`numpy.ndarray`
    :arg float posterior_covariance: The approximate posterior marginal
        variance vector.
    :type posterior_covariance: :class:`numpy.ndarray`
    :arg int J: The number of ordinal classes.
    :arg cutpoints_t: The vector of the lower cutpoints the data.
    :type cutpoints_t: `nd.numpy.array`
    :arg cutpoints_t_plus_1: The vector of the upper cutpoints the data.
    :type cutpoints_t_plus_1: `nd.numpy.array`
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
            cutpoints_t, cutpoints_tplus1,
            EPS_2)
        + fromb_fft1_vector(
            b, posterior_mean, posterior_std, noise_std,
            cutpoints_t, cutpoints_tplus1,
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
                cutpoints_t, cutpoints_tplus1,
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
        noise_variance, noise_std, cutpoints_t, cutpoints_tplus1,
        EPS_2):
    """
    :arg b: The approximate posterior mean evaluated at the datapoint.
    :arg mean: A mean value of a pdf inside the integrand.
    :arg sigma: A standard deviation of a pdf inside the integrand.
    :arg t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg cutpoints: The vector of cutpoints.
    :arg float noise_std: A noise standard deviation for the likelihood.
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    prob = return_prob_vector(
        b, cutpoints_t, cutpoints_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=cutpoints_t, scale=np.sqrt(
        noise_variance + posterior_covariance))


def fromb_t2_vector(
        y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
        cutpoints_t, cutpoints_tplus1,
        noise_variance, noise_std, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int J: The number of ordinal classes.
    :arg cutpoints_t: The vector of the lower cutpoints the data.
    :type cutpoints_t: `nd.numpy.array`
    :arg cutpoints_t_plus_1: The vector of the upper cutpoints the data.
    :type cutpoints_t_plus_1: `nd.numpy.array`
    :arg cutpoints: The vector of cutpoints.
    :type cutpoints: `numpy.ndarray`
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
            cutpoints_t, cutpoints_tplus1,
            EPS_2)
        + fromb_fft2_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std, cutpoints_t, cutpoints_tplus1,
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
                cutpoints_t, cutpoints_tplus1,
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
        noise_variance, noise_std, cutpoints_t, cutpoints_tplus1,
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
        b, cutpoints_t, cutpoints_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return  norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=cutpoints_tplus1, scale=np.sqrt(
        noise_variance + posterior_covariance))


def fromb_t3_vector(
        y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
        cutpoints_t, cutpoints_tplus1,
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
            cutpoints_t, cutpoints_tplus1,
            EPS_2)
        + fromb_fft3_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            cutpoints_t, cutpoints_tplus1,
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
                cutpoints_t, cutpoints_tplus1,
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
        noise_std, noise_variance, cutpoints_t, cutpoints_tplus1,
        EPS_2):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg cutpoints: The vector of cutpoints.
    :type cutpoints: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    prob = return_prob_vector(
        b, cutpoints_t, cutpoints_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=cutpoints_tplus1, scale=np.sqrt(
        noise_variance + posterior_covariance)) * (cutpoints_tplus1 - b)


def fromb_t4_vector(
        y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
        cutpoints_t, cutpoints_tplus1,
        noise_variance, noise_std, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg cutpoints: The vector of cutpoints.
    :type cutpoints: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    y[0, :] = h * (
        fromb_fft4_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            cutpoints_t, cutpoints_tplus1,
            EPS_2)
        + fromb_fft4_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            cutpoints_t, cutpoints_tplus1,
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
                cutpoints_t, cutpoints_tplus1,
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
        cutpoints_t, cutpoints_tplus1,
        EPS_2):
    """
    :arg float b: The approximate posterior mean evaluated at the datapoint.
    :arg float mean: A mean value of a pdf inside the integrand.
    :arg float sigma: A standard deviation of a pdf inside the integrand.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg cutpoints: The vector of cutpoints.
    :type cutpoints: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral point evaluation.
    :rtype: float
    """
    prob = return_prob_vector(
        b, cutpoints_t, cutpoints_tplus1, noise_std)
    prob[prob < EPS_2] = EPS_2
    return norm_pdf(b, loc=mean, scale=sigma) / prob * norm_pdf(
        posterior_mean, loc=cutpoints_t, scale=np.sqrt(
        noise_variance + posterior_covariance)) * (cutpoints_t - b)


def fromb_t5_vector(
        y, mean, sigma, a, b, h, posterior_mean, posterior_covariance,
        cutpoints_t, cutpoints_tplus1,
        noise_variance, noise_std, EPS, EPS_2, N):
    """
    :arg float posterior_mean: The approximate posterior mean evaluated at the
        datapoint. (pdf inside the integrand)
    :arg float posterior_covariance: The approximate posterior marginal
        variance.
    :arg int t: The target value for the datapoint.
    :arg int J: The number of ordinal classes.
    :arg cutpoints: The vector of cutpoints.
    :type cutpoints: `numpy.ndarray`
    :arg float noise_variance: A noise variance for the likelihood.
    :arg float EPS: A machine tolerance to be used.
    :return: fromberg numerical integral numerical value.
    :rtype: float
    """
    y[0, :] = h * (
        fromb_fft5_vector(
            a, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            cutpoints_t, cutpoints_tplus1,
            EPS_2)
        + fromb_fft5_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_variance, noise_std,
            cutpoints_t, cutpoints_tplus1,
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
                cutpoints_t, cutpoints_tplus1,
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


def d_trace_MKzz_dhypers(lls, lsf, z, M, Kzz):

    dKzz_dlsf = Kzz
    ls = np.exp(lls)

    # This is extracted from the R-code of Scalable EP for GP Classification by DHL and JMHL

    gr_lsf = np.sum(M * dKzz_dlsf)

    # This uses the vact that the distance is v^21^T - vv^T + 1v^2^T, where v is a vector with the l-dimension
    # of the inducing points. 

    Ml = 0.5 * M * Kzz
    Xl = z * np.outer(np.ones(z.shape[ 0 ]), 1.0 / np.sqrt(ls))
    gr_lls = np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml.T, Xl**2)) + np.dot(np.ones(Ml.shape[ 0 ]), np.dot(Ml, Xl**2)) \
    - 2.0 * np.dot(np.ones(Xl.shape[ 0 ]), (Xl * np.dot(Ml, Xl)))

    Xbar = z * np.outer(np.ones(z.shape[ 0 ]), 1.0 / ls)
    Mbar1 = - M.T * Kzz
    Mbar2 = - M * Kzz
    gr_z = (Xbar * np.outer(np.dot(np.ones(Mbar1.shape[ 0 ]) , Mbar1), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar1, Xbar)) +\
        (Xbar * np.outer(np.dot(np.ones(Mbar2.shape[ 0 ]) , Mbar2), np.ones(Xbar.shape[ 1 ])) - np.dot(Mbar2, Xbar))

    # The cost of this function is dominated by five matrix multiplications with cost M^2 * D each where D is 
    # the dimensionality of the data!!!

    return gr_lsf, gr_lls, gr_z


def ordinal_logZtilted_vector(
        cutpoints_yplus1, cutpoints_y, noise_std, m, v, alpha, deg):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
    gh_x = gh_x.reshape(1, -1)
    ts = gh_x * np.sqrt(2*v) + m
    pdfs = ndtr(
        (cutpoints_yplus1 - ts) / noise_std) - ndtr(
            (cutpoints_y - ts) / noise_std)
    r = np.log(pdfs**alpha @ gh_w / np.sqrt(np.pi))
    return r


def ordinal_dlogZtilted_dm_vector(
        cutpoints_yplus1, cutpoints_y, noise_std, m, v, alpha, deg):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(deg) 
    gh_x = gh_x.reshape(1, -1)
    eps = 1e-8
    ts = gh_x * np.sqrt(2*v) + m
    uppers = (cutpoints_yplus1 - ts) / noise_std
    lowers = (cutpoints_y - ts) / noise_std
    pdfs = ndtr(uppers) - ndtr(lowers) + eps
    # TODO: simpler way?
    Ztilted = pdfs**alpha @ gh_w / np.sqrt(np.pi)
    dZdm = pdfs**(alpha-1.0) * (-np.exp(-uppers**2/2) + np.exp(-lowers**2/2)) @ gh_w * alpha / np.pi / np.sqrt(2) / noise_std  # TODO: should it be sqrt(2) * noise_std
    return dZdm / Ztilted


def ordinal_dlogZtilted_dm2_vector(
        cutpoints_yplus1, cutpoints_y, noise_std, m, v, alpha, deg):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
    gh_x = gh_x.reshape(1, -1)
    eps = 1e-8
    ts = gh_x * np.sqrt(2*v) + m
    uppers = (cutpoints_yplus1 - ts) / noise_std
    lowers = (cutpoints_y - ts) / noise_std
    pdfs = ndtr(uppers) - ndtr(lowers) + eps
    Ztilted = pdfs**alpha @ gh_w / np.sqrt(np.pi)
    dZdv = pdfs**(alpha-1.0) * (-np.exp(-uppers**2/2) + np.exp(-lowers**2/2)) * gh_x / np.sqrt(2*v) @ gh_w * alpha / np.pi / np.sqrt(2) / noise_std
    return dZdv / Ztilted



###############################################################################

def probit_logZtilted_vector(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(noise_variance + v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2)))  # was math.erf
        eps = 1e-16
        return np.log(Z + eps)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        gh_x = gh_x.reshape(1, -1)
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(ts * y / np.sqrt(2 * noise_variance)))
        r = np.log(pdfs**alpha @ gh_w / np.sqrt(np.pi))
        return r


def probit_dlogZtilted_dm_vector(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(noise_variance + v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2)))  # was math.erf
        eps = 1e-16
        Zeps = Z + eps
        beta = 1 / Zeps / np.sqrt(noise_variance + v) * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
        return y*beta
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg) 
        gh_x = gh_x.reshape(1, -1)
        eps = 1e-8
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(ts * y / np.sqrt(2 * noise_variance))) + eps
        Ztilted = pdfs**alpha @ gh_w / np.sqrt(np.pi)
        dZdm =  pdfs**(alpha-1.0) * np.exp(-ts**2/(2* noise_variance)) * y @ gh_w * alpha / np.pi / np.sqrt(2 * noise_variance)  # TODO: / sqrt noise_variance
        return dZdm / Ztilted


def probit_dlogZtilted_dm2_vector(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(noise_variance + v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2)))  # was math.erf
        eps = 1e-16
        Zeps = Z + eps
        return - 0.5 * y * m / Zeps / (noise_variance + v)**1.5 * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        gh_x = gh_x.reshape(1, -1)
        eps = 1e-8    
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(ts * y / np.sqrt(2 * noise_variance))) + eps
        Ztilted = pdfs**alpha @ gh_w / np.sqrt(np.pi)
        dZdv = pdfs**(alpha-1.0) * np.exp(-ts**2/(2 * noise_variance)) * gh_x * y / np.sqrt(2*v) @ gh_w * alpha / np.pi / np.sqrt(2 * noise_variance)  # TODO: / sqrt noise_variance
        return dZdv / Ztilted


def probit_dlogZtilted_dsn(y_i, m_si_i, v_si_ii, alpha, deg):
    return 0


def probit_logZtilted(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(1+v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2 * noise_variance)))  # was math.erf
        eps = 1e-16
        return np.log(Z + eps)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2 * noise_variance)))
        return np.log(np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)) 


def probit_dlogZtilted_dm(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(noise_variance + v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2)))  # was math.erf
        eps = 1e-16
        Zeps = Z + eps
        beta = 1 / Zeps / np.sqrt(noise_variance + v) * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
        return y*beta
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg) 
        eps = 1e-8
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2 * noise_variance))) + eps
        Ztilted = np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)
        dZdm = np.dot(gh_w, pdfs**(alpha-1.0)*np.exp(-ts**2/2)) * y * alpha / np.pi / np.sqrt(2 * noise_variance)
        return dZdm / Ztilted + eps


def probit_dlogZtilted_dm2(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(noise_variance + v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2)))  # was math.erf
        eps = 1e-16
        Zeps = Z + eps
        return - 0.5 * y * m / Zeps / (noise_variance + v)**1.5 * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)   
        eps = 1e-8    
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2 * noise_variance))) + eps
        Ztilted = np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)
        dZdv = np.dot(gh_w, pdfs**(alpha-1.0)*np.exp(-ts**2/2) * gh_x) * y * alpha / np.pi / np.sqrt(2 * noise_variance) / np.sqrt(2*v)
        return dZdv / Ztilted + eps


def probit_dlogZtilted_dv(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / np.sqrt(noise_variance + v)
        Z = 0.5 * (1 + erf(t / np.sqrt(2)))  # was math.erf
        eps = 1e-16
        Zeps = Z + eps
        beta = 1 / Zeps / np.sqrt(noise_variance + v) * 1/np.sqrt(2*np.pi) * np.exp(-t**2.0 / 2)
        return - (beta**2 + m*y_*beta/(1+v))
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        eps = 1e-8
        ts = gh_x * np.sqrt(2*v) + m
        pdfs = 0.5 * (1 + erf(y*ts / np.sqrt(2 * noise_variance))) + eps
        Ztilted = np.dot(pdfs**alpha, gh_w) / np.sqrt(np.pi)
        dZdm = np.dot(gh_w, pdfs**(alpha-1)*np.exp(-ts**2/2)) * y * alpha / np.pi / np.sqrt(2 * noise_variance)
        dZdm2 = np.dot(gh_w, (alpha-1)*pdfs**(alpha-2)*np.exp(-ts**2)/np.sqrt(2*np.pi)  
            - pdfs**(alpha-1) * y * ts * np.exp(-ts**2/2) ) * alpha / np.pi / np.sqrt(2 * noise_variance)
        return -dZdm**2 / Ztilted**2 + dZdm2 / Ztilted + eps


def log_multivariate_normal_pdf(
        x, cov_inv, half_log_det_cov, mean=None):
    """Get the pdf of the multivariate normal distribution."""
    if mean is not None:
        x = x - mean
    return -0.5 * np.log(2 * np.pi)\
        - half_log_det_cov - 0.5 * x.T @ cov_inv @ x  # log likelihood


def log_multivariate_normal_pdf_vectorised(
        xs, cov_inv, half_log_det_cov, mean=None):
    """Get the pdf of the multivariate normal distribution."""
    if mean is not None:
        xs = xs - mean
    return -0.5 * np.log(2 * np.pi) - half_log_det_cov - 0.5 * np.einsum(
        'kj, kj -> k', np.einsum('ij, ki -> kj', cov_inv, xs), xs)
        

def sample_varphis(varphi_hyperparameter, n_samples):
    """
    Take n_samples of varphi, given the hyperparameter of varphi.

    varphi_hyperparameter is a rate parameter since, with an uninformative
    prior (sigma=tau=0), then the posterior mean of Q(psi) is
    psi_tilde = 1. / varphi_tilde. Therefore, by taking the expected value of
    the prior on varphi ~ Exp(psi_tilde),
    we expect to obtain varphi_tilde = 1. / psi_tilde. We get this if
    psi_tilde is a rate.

    :arg psi: float (Array) of hyper-hyperparameter(s)
    :type psi: :class:`np.ndarray`
    :arg int n_samples: The number of samples for the importance sample.
    """
    # scale = varphi_hyperparameter
    scale = 1. / varphi_hyperparameter
    shape = np.shape(varphi_hyperparameter)
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
        cutpoints_ts, cutpoints_tplus1s, noise_std, m, EPS,
        upper_bound=None, upper_bound2=None, numerically_stable=False):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

    TODO: Could a numba version, shouldn't be difficult, but will have to be
        parallelised scalar (due to the boolean logic).
    TODO: There is no way to calculate this in the log domain (unless expansion
    approximations are used). Could investigate only using approximations here.
    :arg cutpoints_ts: cutpoints[y_train] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
    :arg cutpoints_tplus1s: cutpoints[y_train + 1] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
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
    z1s = (cutpoints_ts - m) / noise_std
    z2s = (cutpoints_tplus1s - m) / noise_std
    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)
    norm_cdf_z1s = norm_cdf(z1s)
    norm_cdf_z2s = norm_cdf(z2s)
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


def p(m, cutpoints_ts, cutpoints_tplus1s, noise_std,
        EPS, upper_bound, upper_bound2):
    """
    The rightmost term of 2021 Page Eq.(),
        correction terms that squish the function value m
        between the two cutpoints for that particle.

    :arg m: The current posterior mean estimate.
    :type m: :class:`numpy.ndarray`
    :arg cutpoints_ts: cutpoints[y_train] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
    :arg cutpoints_tplus1s: cutpoints[y_train + 1] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
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
        cutpoints_ts, cutpoints_tplus1s, noise_std, m, EPS)
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


def dp(m, cutpoints_ts, cutpoints_tplus1s, noise_std, EPS,
        upper_bound, upper_bound2=None):
    """
    The analytic derivative of :meth:`p` (p are the correction
        terms that squish the function value m
        between the two cutpoints for that particle).

    :arg m: The current posterior mean estimate.
    :type m: :class:`numpy.ndarray`
    :arg cutpoints_ts: cutpoints[y_train] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
    :arg cutpoints_tplus1s: cutpoints[y_train + 1] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
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
        cutpoints_ts, cutpoints_tplus1s, noise_std, m, EPS)
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


def sample_g(g, f, y_train, cutpoints, noise_std, N):
    for i in range(N):
        # Target class index
        j_true = y_train[i]
        g_i = np.NINF  # this is a trick for the next line
        # Sample from the truncated Gaussian
        while g_i > cutpoints[j_true + 1] or g_i <= cutpoints[j_true]:
            # sample y
            g_i = f[i] + np.random.normal(loc=f[i], scale=noise_std)
        # Add sample to the Y vector
        g[i] = g_i
    return g


class CutpointValueError(Exception):
    """
    An invalid cutpoint argument was used to construct the classifier model.
    """

    def __init__(self, cutpoint):
        """
        Construct the exception.

        :arg cutpoint: The cutpoint parameters array.
        :type cutpoint: :class:`numpy.array` or list

        :rtype: :class:`CutpointValueError`
        """
        message = (
                "The cutpoint list or array "
                "must be in ascending order, "
                f" {cutpoint} was given."
                )

        super().__init__(message)
