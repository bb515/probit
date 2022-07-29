"""Utility functions for probit."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import ndtr, log_ndtr, erf
import warnings
import h5py


over_sqrt_2_pi = 1. / np.sqrt(2 * np.pi)
log_over_sqrt_2_pi = -0.5 * np.log(2 * np.pi)
sqrt_2 = np.sqrt(2)


def matrix_inverse(matrix, N):
    "another version"
    L_cov = cholesky(matrix, lower=True)
    L_covT_inv = solve_triangular(
        L_cov, np.eye(N), lower=True)
    cov = solve_triangular(L_cov.T, L_covT_inv, lower=False)
    return cov, L_cov


def return_prob_vector(b, cutpoints_t, cutpoints_tplus1, noise_std):
    return ndtr(
        (cutpoints_tplus1 - b) / noise_std) - ndtr(
            (cutpoints_t - b) / noise_std)


def posterior_covariance(K, cov, precision):
    return K @ cov @ np.diag(1./precision)


def check_cutpoints(cutpoints, J):
    """
    TODO: this will not be compatable with autodiff
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
        

def h(x):
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
    return over_sqrt_2_pi * (
    1 / z1 * np.exp(-0.5 * z1**2 + h(z1)) - 1 / z2 * np.exp(
        -0.5 * z2**2 + h(z2)))


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    return over_sqrt_2_pi / z * np.exp(-0.5 * z**2 + h(z))


def truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std, m,
        upper_bound=None, upper_bound2=None, tolerance=None):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

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
    if tolerance is not None:
        small_densities = np.where(Z < tolerance)
        if np.size(small_densities) != 0:
            warnings.warn(
                "Z (normalising constants for truncated norma"
                "l random variables) must be greater than"
                " tolerance={} (got {}): SETTING to"
                " Z_ns[Z_ns<tolerance]=tolerance\nz1s={}, z2s={}".format(
                    tolerance, Z, z1s, z2s))
            Z[small_densities] = tolerance
    return (
        Z,
        norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, norm_cdf_z1s, norm_cdf_z2s)


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
