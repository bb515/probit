"""Utility functions for probit."""

# Enable double precision
from jax.config import config
config.update("jax_enable_x64", True)

import jax
from functools import partial
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
from jax.scipy.special import ndtr, log_ndtr
from jax.scipy.linalg import solve_triangular
import warnings


over_sqrt_2_pi = 1. / jnp.sqrt(2 * jnp.pi)
log_over_sqrt_2_pi = jnp.log(over_sqrt_2_pi)


@partial(jax.jit, static_argnames=['N'])
def matrix_inverse(matrix, N):
    L_cov = cholesky(matrix, lower=True)
    L_covT_inv = solve_triangular(
        L_cov, jnp.eye(N), lower=True)
    cov = solve_triangular(L_cov.T, L_covT_inv, lower=False)
    return cov, L_cov


def return_prob_vector(b, cutpoints_t, cutpoints_tplus1, noise_std):
    return ndtr(
        (cutpoints_tplus1 - b) / noise_std) - ndtr(
            (cutpoints_t - b) / noise_std)


def norm_z_pdf(z):
    return over_sqrt_2_pi * jnp.exp(- z**2 / 2.0 )


def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


def norm_z_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


def norm_cdf(x):
    return ndtr(x)


def norm_logpdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_logpdf(z) - jnp.log(scale)


def norm_logcdf(x):
    return log_ndtr(x)


def _Z_tails(z1, z2):
    """
    Series expansion at infinity.

    Even for z1, z2 >= 4 this is accurate to three decimal places.
    """
    return over_sqrt_2_pi * (
    1 / z1 * jnp.exp(-0.5 * z1**2 + h(z1)) - 1 / z2 * jnp.exp(
        -0.5 * z2**2 + h(z2)))


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    return over_sqrt_2_pi / z * jnp.exp(-0.5 * z**2 + h(z))


def truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std, f,
        upper_bound=None, upper_bound2=None, tolerance=None):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

    approximations are used). Could investigate only using approximations here.
    :arg cutpoints_ts: cutpoints[y_train] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
    :arg cutpoints_tplus1s: cutpoints[y_train + 1] (N, ) array of cutpoints
    :type cutpoints_ts: :class:`numpy.ndarray`
    :arg float noise_std: The noise standard deviation.
    :arg m: The mean vector.
    :type m: :class:`numpy.ndarray`
    :arg float tolerance: The tolerated absolute error.
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
    z1s = (cutpoints_ts - f) / noise_std
    z2s = (cutpoints_tplus1s - f) / noise_std
    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)
    norm_cdf_z1s = norm_cdf(z1s)
    norm_cdf_z2s = norm_cdf(z2s)
    Z = norm_cdf_z2s - norm_cdf_z1s
    if upper_bound is not None:
        # Using series expansion approximations
        indices1 = jnp.where(z1s > upper_bound)
        indices2 = jnp.where(z2s < -upper_bound)
        if jnp.size(indices1) > 0 or jnp.size(indices2) > 0:
            print(indices1)
            print(indices2)
            if jnp.ndim(z1s) == 1:
                indices = jnp.union1d(indices1[0], indices2[0])
            elif jnp.ndim(z1s) == 2:
                # f is (num_samples, N). This is quick (but not dirty) hack.
                indices = (jnp.append(indices1[0], indices2[0]),
                    jnp.append(indices1[1], indices2[1]))
            z1_indices = z1s[indices]
            z2_indices = z2s[indices]
            Z = Z.at[indices].set(_Z_tails(
                z1_indices, z2_indices))
            if upper_bound2 is not None:
                indices = jnp.where(z1s > upper_bound2)
                z1_indices = z1s[indices]
                Z = Z.at[indices].set(_Z_far_tails(
                    z1_indices))
                indices = jnp.where(z2s < -upper_bound2)
                z2_indices = z2s[indices]
                Z = Z.at[indices].set(_Z_far_tails(
                    -z2_indices))
    if tolerance is not None:
        small_densities = jnp.where(Z < tolerance)
        if jnp.size(small_densities) != 0:
            warnings.warn(
                "Z (normalising constants for truncated norma"
                "l random variables) must be greater than"
                " tolerance={} (got {}): SETTING to"
                " Z_ns[Z_ns<tolerance]=tolerance\nz1s={}, z2s={}".format(
                    tolerance, Z, z1s, z2s))
            Z = Z.at[small_densities].set(tolerance)
    return (
        Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, norm_cdf_z1s, norm_cdf_z2s)


def posterior_covariance(K, cov, precision):
    return K @ cov @ jnp.diag(1./precision)


########################################################################
# TODO: convert to jax
########################################################################

# def log_multivariate_normal_pdf(
#         x, cov_inv, half_log_det_cov, mean=None):
#     """Get the pdf of the multivariate normal distribution."""
#     if mean is not None:
#         x = x - mean
#     return -0.5 * np.log(2 * np.pi)\
#         - half_log_det_cov - 0.5 * x.T @ cov_inv @ x  # log likelihood


# def log_multivariate_normal_pdf_vectorised(
#         xs, cov_inv, half_log_det_cov, mean=None):
#     """Get the pdf of the multivariate normal distribution."""
#     if mean is not None:
#         xs 