"""Utility functions for probit."""

import lab as B
import jax
import jax.numpy as jnp
from math import inf
import warnings


over_sqrt_2_pi = 1.0 / B.sqrt(2 * B.pi)
log_over_sqrt_2_pi = jnp.log(over_sqrt_2_pi)
sqrt_2 = B.sqrt(2)


BOUNDS = {"single": [1.3, 1.8, 2.3], "double": [2.3, 3.6, 4.8]}


def ndtr(z):
    return 0.5 * (1 + jax.lax.erf(z / sqrt_2))


def norm_z_pdf(z):
    return over_sqrt_2_pi * jnp.exp(-0.5 * z**2)


def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


def norm_cdf(x):
    _x = jnp.where(jnp.isinf(x), 1.0, x)
    extrema = jnp.where(x == jnp.inf, 1.0, 0.0)
    return jnp.where(jnp.isinf(x), extrema, ndtr(_x))


def h(x):
    """
    Polynomial part of a series expansion for log survival function for a
    normal random variable. With the third term, for x>4, this is accurate
    to three decimal places. The third term becomes significant when sigma
    is large.
    """
    return -1 * x**-2 + 5 / 2 * x**-4 - 37 / 3 * x**-6


def probit_likelihood(f, y, likelihood_parameters):
    return probit(
        likelihood_parameters[0],
        likelihood_parameters[1][y],
        likelihood_parameters[1][y + 1],
        f,
    )


def log_probit_likelihood(f, y, likelihood_parameters):
    return jnp.log(probit_likelihood(f, y, likelihood_parameters) + 1e-10)


def log_gaussian_likelihood(f, y, likelihood_parameters):
    return norm_logpdf(f, loc=y, scale=likelihood_parameters[0])


def norm_z_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


def norm_logpdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_logpdf(z) - jnp.log(scale)


def _Z_tails(z1, z2):
    """
    Series expansion at infinity.

    Even for z1, z2 >= 4 this is accurate to three decimal places.
    """
    tails = _Z_far_tails(z1) - _Z_far_tails(z2)
    return tails


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    return over_sqrt_2_pi / z * jnp.exp(-0.5 * z**2 + h(z))


def _safe_Z(
    f,
    y,
    likelihood_parameters,
    upper_bound=jnp.inf,
    upper_bound2=jnp.inf,
    upper_bound3=jnp.inf,
):
    """Calculate the difference in CDFs between two z-scores, where z2 >= z1.
    Use approximations to avoid catastrophic cancellation at extreme values.

    Nans are tracked through gradients. This function ensures that the functions
    are not evaluated at possible nan values."""
    cutpoints_tplus1 = jnp.asarray(likelihood_parameters[1])[y + 1]
    cutpoints_t = jnp.asarray(likelihood_parameters[1])[y]

    noise_std = likelihood_parameters[0]

    _b = jnp.where(cutpoints_tplus1 == jnp.inf, 0.0, cutpoints_tplus1)
    _a = jnp.where(cutpoints_t == -jnp.inf, 0.0, cutpoints_t)

    z2s = jnp.where(cutpoints_tplus1 == jnp.inf, jnp.inf, (_b - f) / noise_std)
    z1s = jnp.where(cutpoints_t == -jnp.inf, -jnp.inf, (_a - f) / noise_std)

    # Placeholder value used to signify that the function is *not* evalutated
    # at this point
    SAFE = 1.0

    # TODO: tidy up the below commented lines
    # _z1s = jnp.where(jnp.abs(z1s) < upper_bound, z1s, SAFE)
    # _z2s = jnp.where(jnp.abs(z2s) < upper_bound, z2s, SAFE+1)
    Z = norm_cdf(z2s) - norm_cdf(z1s)

    # Remove any zero-values of z1s and z2s to avoid divide-by-zero
    # these values aren't used - only to avoid nans (https://github.com/google/jax/issues/1052 and 8247)
    _z1s = jnp.where((upper_bound < z1s) & (z1s <= upper_bound2), z1s, SAFE)
    __z2s = jnp.where(upper_bound < z1s, z2s, SAFE)

    _z2s = jnp.where((-upper_bound2 <= z2s) & (z2s < -upper_bound), z2s, SAFE)
    __z1s = jnp.where(-upper_bound > z2s, z1s, SAFE)

    # Using series expansion approximations
    Z = jnp.where(z1s > upper_bound, _Z_tails(_z1s, __z2s), Z)
    Z = jnp.where(z2s < -upper_bound, _Z_tails(__z1s, _z2s), Z)

    _z1s = jnp.where(
        (upper_bound2 < jnp.abs(z1s)) & (jnp.abs(z1s) < upper_bound3), z1s, SAFE
    )
    _z2s = jnp.where(
        (upper_bound2 < jnp.abs(z2s)) & (jnp.abs(z2s) < upper_bound3), z2s, SAFE
    )

    # Using one sided series expansion approximations
    Z = jnp.where(z1s > upper_bound2, _Z_far_tails(_z1s), Z)
    Z = jnp.where(z2s < -upper_bound2, _Z_far_tails(-_z2s), Z)

    # Ignore Z for linear approximation
    Z = jnp.where(z1s >= upper_bound3, SAFE, Z)
    Z = jnp.where(z2s <= -upper_bound3, SAFE, Z)

    return Z, z1s, z2s


def grad_log_probit_likelihood(f, y, likelihood_parameters, single_precision=True):
    upper_bound, upper_bound2, upper_bound3 = BOUNDS[
        "single" if single_precision else "double"
    ]

    noise_std = likelihood_parameters[0]
    Z, z1s, z2s = _safe_Z(
        f, y, likelihood_parameters, upper_bound, upper_bound2, upper_bound3
    )

    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)

    # ratio is approximated well linearly
    E = (norm_pdf_z1s - norm_pdf_z2s) / Z
    E = jnp.where(z1s > upper_bound3, z1s, E)
    E = jnp.where(z2s < -upper_bound3, z2s, E)

    return E / noise_std


def hessian_log_probit_likelihood(f, y, likelihood_parameters, single_precision=True):
    upper_bound, upper_bound2, upper_bound3 = BOUNDS[
        "single" if single_precision else "double"
    ]

    noise_std = likelihood_parameters[0]
    Z, z1s, z2s = _safe_Z(
        f, y, likelihood_parameters, upper_bound, upper_bound2, upper_bound3
    )
    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)

    w = grad_log_probit_likelihood(f, y, likelihood_parameters, single_precision)

    _z1s = jnp.where((z1s == -inf) | (z1s == inf), 0.0, z1s)
    _z2s = jnp.where((z2s == -inf) | (z2s == inf), 0.0, z2s)
    V = -(w**2) + (_z1s * norm_pdf_z1s - _z2s * norm_pdf_z2s) / Z / noise_std**2

    V = jnp.where(z1s > upper_bound3, -(noise_std**-2), V)
    V = jnp.where(z2s < -upper_bound3, -(noise_std**-2), V)
    return V


def probit(noise_std, cutpoints_y, cutpoints_yplus1, f):
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
    # NOTE this is removing infs before and after an operation so that
    # the function can be automatically differentiated
    # TODO: double check that this is absolutely needed
    safe_z1s = jnp.where(cutpoints_y == -jnp.inf, 0.0, (cutpoints_y - f))
    safe_z2s = jnp.where(cutpoints_yplus1 == jnp.inf, 0.0, (cutpoints_yplus1 - f))
    norm_cdf_z1s = jnp.where(
        cutpoints_y == -jnp.inf, 0.0, norm_cdf(safe_z1s / noise_std)
    )
    norm_cdf_z2s = jnp.where(
        cutpoints_yplus1 == jnp.inf, 1.0, norm_cdf(safe_z2s / noise_std)
    )
    Z = norm_cdf_z2s - norm_cdf_z1s
    return Z


def probit_predictive_distributions(
    likelihood_parameters, posterior_mean, posterior_variance
):
    """
    Return predictive distributions for the ordinal likelihood.
    """
    N_test = posterior_mean.shape[0]
    noise_std, cutpoints = likelihood_parameters
    J = B.size(cutpoints) - 1
    predictive_distributions = B.ones(N_test, J)
    posterior_pred_std = jnp.sqrt(posterior_variance + noise_std**2)
    posterior_pred_mean = posterior_mean
    for j in range(J):
        Z = probit(
            posterior_pred_std, cutpoints[j], cutpoints[j + 1], posterior_pred_mean
        )
        predictive_distributions[:, j] = Z
    return predictive_distributions


def check_data(data):
    X_train, y_train = data
    if y_train.dtype not in [int, jnp.int32]:
        raise TypeError(
            "t must contain only integer values (got {})".format(y_train.dtype)
        )
    else:
        y_train = y_train.astype(int)
    return X_train, y_train


def check_cutpoints(cutpoints, J):
    """
    Check that the cutpoints are compatible with this class.
    # TODO: convert to B lab code

    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    """
    warnings.warn("Checking cutpoints...")
    # Convert cutpoints to numpy array
    cutpoints = jnp.array(cutpoints)
    # Not including -\infty or \infty
    if jnp.shape(cutpoints)[0] == J - 1:
        # Append \infty
        cutpoints = jnp.append(cutpoints, jnp.inf)
        # Insert -\infty at index 0
        cutpoints = jnp.insert(cutpoints, 0, jnp.NINF)
        pass  # correct format
    # Not including one cutpoints
    elif jnp.shape(cutpoints)[0] == J:
        if cutpoints[-1] != jnp.inf:
            if cutpoints[0] != jnp.NINF:
                raise ValueError(
                    "Either the largest cutpoint parameter b_J is not "
                    "positive infinity, or the smallest cutpoint "
                    "parameter must b_0 is not negative infinity."
                    "(got {}, expected {})".format(
                        [cutpoints[0], cutpoints[-1]], [jnp.inf, jnp.NINF]
                    )
                )
            else:  # cutpoints[0] is -\infty
                cutpoints.append(jnp.inf)
                pass  # correct format
        else:
            cutpoints = jnp.insert(cutpoints, 0, jnp.NINF)
            pass  # correct format
    # Including all the cutpoints
    elif jnp.shape(cutpoints)[0] == J + 1:
        if cutpoints[0] != jnp.NINF:
            raise ValueError(
                "The smallest cutpoint parameter b_0 must be negative "
                "infinity (got {}, expected {})".format(cutpoints[0], jnp.NINF)
            )
        if cutpoints[-1] != jnp.inf:
            raise ValueError(
                "The largest cutpoint parameter b_J must be positive "
                "infinity (got {}, expected {})".format(cutpoints[-1], jnp.inf)
            )
        pass  # correct format
    else:
        raise ValueError(
            "Could not recognise cutpoints shape. "
            "(jnp.shape(cutpoints) was {})".format(jnp.shape(cutpoints))
        )
    assert cutpoints[0] == jnp.NINF
    assert cutpoints[-1] == jnp.inf
    assert jnp.shape(cutpoints)[0] == J + 1
    if not all(cutpoints[i] <= cutpoints[i + 1] for i in range(J)):
        raise CutpointValueError(cutpoints)
    return cutpoints


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


class InvalidKernel(Exception):
    """An invalid kernel has been passed to `Approximator` or `Sampler`"""

    def __init__(self, kernel):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`Approximator` or `Sampler`
            as the kernel argument.
        :rtype: :class:`InvalidKernel`
        """
        message = f"{kernel} is not an instance of " "mlkernels.Kernel"

        super().__init__(message)
