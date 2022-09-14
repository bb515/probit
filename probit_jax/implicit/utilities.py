"""Utility functions for probit."""
import lab as B
import jax
import jax.numpy as jnp
from jax import (vmap, grad, jit)
import warnings
from math import inf
from functools import partial

over_sqrt_2_pi = 1. / B.sqrt(2 * B.pi)
log_over_sqrt_2_pi = -0.5 * B.log(2 * B.pi)
sqrt_2 = B.sqrt(2)


def ndtr(z):
    return 0.5 * (1 + B.erf(z/sqrt_2))


def log_likelihood(
        m, cutpoints_ts, cutpoints_tplus1s, noise_std,
        upper_bound, upper_bound2, tolerance):
    """
    TODO: May be redundant - used in sampling code?
    Likelihood of ordinal regression. This is product of scalar normal cdf.

    If np.ndim(m) == 2, vectorised so that it returns (num_samples,)
    vector from (num_samples, N) samples of the posterior mean.

    Note that numerical stability has been turned off in favour of
    exactness - but experiments should be run twice with numerical
    stability turned on to see if it makes a difference.
    """
    # TODO this is the scalar version
    Z, *_ = probit_likelihood(
        cutpoints_ts, cutpoints_tplus1s,
        noise_std, m,
        upper_bound=upper_bound,
        upper_bound2=upper_bound2,  # optional
        tolerance=tolerance  # optional
        )
    if B.ndim(m) == 2:
        return B.sum(B.log(Z), axis=1)  # (num_samples,)
    elif B.ndim(m) == 1:
        return B.sum(B.log(Z))  # (1,)


@partial(jax.jit, static_argnames=['N'])  # TODO: keep this here?
def matrix_inverse(matrix, N):
    L_cov = B.cholesky(matrix)
    L_covT_inv = B.triangular_solve(L_cov, B.eye(N), lower_a=True)
    cov = B.triangular_solve(L_cov.T, L_covT_inv, lower_a=False)
    return cov, L_cov


def return_prob_vector(b, cutpoints_t, cutpoints_tplus1, noise_std):
    return ndtr((cutpoints_tplus1 - b) / noise_std) - ndtr(
        (cutpoints_t - b) / noise_std)


def posterior_covariance(K, cov, precision):
    return K @ cov @ B.diag(1./precision)


def norm_z_pdf(z):
    return over_sqrt_2_pi * B.exp(- z**2 / 2.0 )


def norm_z_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


def norm_logpdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_logpdf(z) - B.log(scale)


def norm_cdf(x):
    return ndtr(x)


def log_multivariate_normal_pdf(
        x, cov_inv, half_log_det_cov, mean=None):
    """Get the pdf of the multivariate normal distribution."""
    if mean is not None:
        x = x - mean
    # log likelihood
    return -0.5 * B.log(2 * B.pi) - half_log_det_cov - 0.5 * x.T @ cov_inv @ x 


def log_multivariate_normal_pdf_vectorised(
        xs, cov_inv, half_log_det_cov, mean=None):
    """Get the pdf of the multivariate normal distribution."""
    if mean is not None:
        xs = xs - mean
    return -0.5 * B.log(2 * B.pi) - half_log_det_cov - 0.5 * B.einsum(
        'kj, kj -> k', B.einsum('ij, ki -> kj', cov_inv, xs), xs)


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
    1 / z1 * B.exp(-0.5 * z1**2 + h(z1)) - 1 / z2 * B.exp(
        -0.5 * z2**2 + h(z2)))


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    return over_sqrt_2_pi / z * B.exp(-0.5 * z**2 + h(z))


# @partial(jit, static_argnums=[2, 3])  # TODO, which function to jit compile, the parent or the child? Parent when possible? child when not?
def probit_likelihood(
        f, y, likelihood_parameters, single_precision):
    return probit(
        likelihood_parameters[0], likelihood_parameters[1][y], likelihood_parameters[1][y + 1],
        f, single_precision)


def negative_log_probit_likelihood(
        f, y, likelihood_parameters, single_precision):
    return -jnp.log(probit_likelihood(f, y, likelihood_parameters, single_precision))


def probit(
        noise_std, cutpoints_y, cutpoints_yplus1, f,
        single_precision=True):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

    :arg float noise_std: The noise standard deviation.
    :arg cutpoints_y: cutpoints[y_train] (N, ) array of cutpoints
    :type cutpoints_y: :class:`numpy.ndarray`
    :arg cutpoints_yplus1: cutpoints[y_train + 1] (N, ) array of cutpoints
    :type cutpoints_y: :class:`numpy.ndarray`
    :arg f: The mean vector.
    :type f: :class:`numpy.ndarray`
    :arg y: data
    :type y: :class:`numpy.ndarray`
    :arg float upper_bound: The threshold of the normal z value for which
        the pdf is close enough to zero.
    :arg float upper_bound2: The threshold of the normal z value for which
        the pdf is close enough to zero. 
    :arg float tolerance: The tolerated absolute error.
    :returns: Z
    :rtype: :class:`numpy.ndarray`
    """
    if single_precision is True:
        # Single precision
        # float upper_bound: The threshold of the normal z value for which
        # float upper_bound2: The threshold of the normal z value for which
        # the pdf is close enough to zero. 
        # float tolerance: The tolerated absolute error.
        # Threshold of single sided standard deviations that
        # normal cdf can be approximated to 0 or 1
        # More than this + redundancy leads to numerical instability
        # due to catestrophic cancellation
        # Less than this leads to a poor approximation due to series
        # expansion at infinity truncation
        # Good values found between 4 and 6
        upper_bound = 4
        # More than this + redundancy leads to numerical
        # instability due to overflow
        # Less than this results in poor approximation due to
        # neglected probability mass in the tails
        # Good values found between 18 and 30
        # Try decreasing if experiencing infs or NaNs
        upper_bound2 = 18
        # Acts as a machine tolerance for small probabilities
        tolerance = 1e-4
    elif single_precision is False:
        # Double precision
        upper_bound = 6
        upper_bound2 = 30
        tolerance = 1e-8
    else:
        upper_bound = None
        upper_bound2 = None
        tolerance = None
    # Otherwise
    safe_z1s = jnp.where(cutpoints_y == -jnp.inf, 0.0, (cutpoints_y - f))
    safe_z2s = jnp.where(cutpoints_yplus1 == jnp.inf, 0.0, (cutpoints_yplus1 - f))
    norm_cdf_z1s = jnp.where(cutpoints_y == -jnp.inf, 0.0, norm_cdf(safe_z1s / noise_std))
    norm_cdf_z2s = jnp.where(cutpoints_yplus1 == jnp.inf, 1.0, norm_cdf(safe_z2s / noise_std))
    Z = norm_cdf_z2s - norm_cdf_z1s
    # TODO: probably cannot use single_precision, unless make it a static argument?
    if upper_bound is not None:
        # TODO: This doesn't seem to be working properly, needs testing
        # Using series expansion approximations
        Z = B.where(safe_z1s > upper_bound, _Z_tails(safe_z1s, safe_z2s), Z)
        Z = B.where(safe_z2s < -upper_bound, _Z_tails(safe_z1s, safe_z2s), Z)
        if upper_bound2 is not None:
            # Using one sided series expansion approximations
            Z = B.where(safe_z1s > upper_bound2, _Z_far_tails(safe_z1s), Z)
            Z = B.where(safe_z2s < -upper_bound2, _Z_far_tails(-safe_z2s), Z)
    if tolerance is not None:
        Z = B.where(Z < tolerance, tolerance, Z)
    return Z


def ordinal_predictive_distributions(
        posterior_pred_mean, posterior_pred_std, N_test,
        cutpoints, single_precision=True):
    """
    Return predictive distributions for the ordinal likelihood.
    """
    J = B.size(cutpoints) - 1
    predictive_distributions = B.ones(N_test, J)
    for j in range(J):
        Z, *_ = probit(
                posterior_pred_std,
                cutpoints[j], cutpoints[j + 1],
                posterior_pred_mean,
                single_precision)
        predictive_distributions[:, j] = Z
    return predictive_distributions


def predict_reparameterised(
        Kss, Kfs, cov, weight, cutpoints, noise_variance,
        single_precision=True):
    """
    Make posterior prediction over ordinal classes of X_test.

    :arg X_test: The new data points, array like (N_test, D).
    :arg cov: A covariance matrix used in calculation of posterior
        predictions. (\sigma^2I + K)^{-1} Array like (N, N).
    :type cov: :class:`numpy.ndarray`
    :arg weight: The approximate inverse-covariance-posterior-mean.
        .. math::
            \nu = (\mathbf{K} + \sigma^{2}\mathbf{I})^{-1} \mathbf{y}
            = \mathbf{K}^{-1} \mathbf{f}
        Array like (N,).
    :type weight: :class:`numpy.ndarray`
    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    :arg float noise_variance: The noise variance.
    :arg bool numerically_stable: Use matmul or triangular solve.
        Default `False`. 
    :return: A Monte Carlo estimate of the class probabilities.
    :rtype tuple: ((N_test, J), (N_test,), (N_test,))
    """
    N_test = B.shape(Kss)[0]
    temp = cov @ Kfs
    posterior_variance = Kss - B.einsum(
        'ij, ij -> j', Kfs, temp)
    posterior_std = B.sqrt(posterior_variance)
    posterior_pred_mean = Kfs.T @ weight
    posterior_pred_variance = posterior_variance + noise_variance
    posterior_pred_std = B.sqrt(posterior_pred_variance)
    return (
        ordinal_predictive_distributions(
                posterior_pred_mean, posterior_pred_std, N_test, cutpoints,
                single_precision),
            posterior_pred_mean, posterior_std)


def sample_g(g, f, y_train, cutpoints, noise_std, N):
    """TODO: Seems like this has to be done in numpy or numba."""
    for i in range(N):
        # Target class index
        j_true = y_train[i]
        g_i = -inf  # this is a trick for the next line
        # Sample from the truncated Gaussian
        while g_i > cutpoints[j_true + 1] or g_i <= cutpoints[j_true]:
            # sample y
            # TODO: test if this works
            g_i = f[i] + (f[i] - B.randn(1)) / noise_std
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
