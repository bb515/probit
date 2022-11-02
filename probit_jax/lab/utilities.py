"""Utility functions for probit."""
import lab as B
import jax
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


def ordinal_predictive_distributions(
    posterior_pred_mean, posterior_pred_std, N_test, cutpoints, J,
    upper_bound, upper_bound2):
    """
    Return predictive distributions for the ordinal likelihood.
    """
    predictive_distributions = B.ones(N_test, J)
    
    for j in range(J):
        Z, *_ = probit_likelihood(
                cutpoints[j], cutpoints[j + 1],
                posterior_pred_std, posterior_pred_mean,
                upper_bound=upper_bound, upper_bound2=upper_bound2)
        predictive_distributions[:, j] = Z
    return predictive_distributions


def predict_reparameterised(
        Kss, Kfs, cov, weight, cutpoints, noise_variance, J,
        upper_bound, upper_bound2):
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
                J, upper_bound, upper_bound2),
            posterior_pred_mean, posterior_std)


@partial(jax.jit, static_argnames=['N'])
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


def probit_likelihood(
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
        Z = B.where(z1s > upper_bound, _Z_tails(z1s, z2s), Z)
        Z = B.where(z2s < -upper_bound, _Z_tails(z1s, z2s), Z)
        if upper_bound2 is not None:
            # Using one sided series expansion approximations
            Z = B.where(z1s > upper_bound2, _Z_far_tails(z1s), Z)
            Z = B.where(z2s < -upper_bound2, _Z_far_tails(-z2s), Z)
    if tolerance is not None:
        small_densities = B.where(Z < tolerance, 0, 1)
        if B.sum(small_densities) != 0:
            warnings.warn(
                "Z (normalising constants for truncated norma"
                "l random variables) must be greater than"
                " tolerance={} (got {}): SETTING to"
                " Z_ns[Z_ns<tolerance]=tolerance\nz1s={}, z2s={}".format(
                    tolerance, Z, z1s, z2s))
            Z = B.where(Z < tolerance, tolerance, Z)
    return (
        Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, norm_cdf_z1s, norm_cdf_z2s)


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
