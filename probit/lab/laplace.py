from lab import cholesky
from lab import triangular_solve as solve_triangular
from lab import erf
import lab as B
import jax.numpy as jnp
import lab.jax
import warnings


over_sqrt_2_pi = 1. / B.sqrt(2 * B.pi)
log_over_sqrt_2_pi = B.log(over_sqrt_2_pi)
sqrt_2 = B.sqrt(2)


def matrix_inverse(matrix, N):
    "another version"
    L_cov = cholesky(matrix)
    L_covT_inv = solve_triangular(
        L_cov, B.eye(N), lower_a=True)
    cov = solve_triangular(L_cov.T, L_covT_inv, lower_a=False)
    return cov, L_cov


def ndtr(z):
    return 0.5 * (1 + erf(z/sqrt_2))


def norm_z_pdf(z):
    return over_sqrt_2_pi * B.exp(-z**2 / 2.0)


def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


def norm_cdf(x):
    return ndtr(x)


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
    return over_sqrt_2_pi * (
    1 / z1 * B.exp(-0.5 * z1**2 + _g(z1)) - 1 / z2 * B.exp(
        -0.5 * z2**2 + _g(z2)))


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    return over_sqrt_2_pi / z * B.exp(-0.5 * z**2 + _g(z))


def truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std, m,
        upper_bound=None, upper_bound2=None, tolerance=None):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

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
    # if upper_bound is not None:
    #     # Using series expansion approximations
    #     # TODO: Is this the correct way to use where?
    #     indices1 = B.where(z1s > upper_bound)
    #     indices2 = B.where(z2s < -upper_bound)
    #     if B.ndim(z1s) == 1:
    #         indices = B.union1d(indices1, indices2)
    #     elif B.ndim(z1s) == 2:
    #         # f is (num_samples, N). This is quick (but not dirty) hack.
    #         indices = (B.append(indices1[0], indices2[0]),
    #             B.append(indices1[1], indices2[1]))
    #     z1_indices = z1s[indices]
    #     z2_indices = z2s[indices]
    #     # TODO: do I need to do set here?
    #     Z[indices] = _Z_tails(
    #         z1_indices, z2_indices)
    #     if upper_bound2 is not None:
    #         indices = B.where(z1s > upper_bound2)
    #         z1_indices = z1s[indices]
    #         Z[indices] = _Z_far_tails(
    #             z1_indices)
    #         indices = B.where(z2s < -upper_bound2)
    #         z2_indices = z2s[indices]
    #         Z[indices] = _Z_far_tails(
    #             -z2_indices)
    # if tolerance is not None:
    #     small_densities = B.where(Z < tolerance)
    #     if B.size(small_densities) != 0:
    #         warnings.warn(
    #             "Z (normalising constants for truncated norma"
    #             "l random variables) must be greater than"
    #             " tolerance={} (got {}): SETTING to"
    #             " Z_ns[Z_ns<tolerance]=tolerance\nz1s={}, z2s={}".format(
    #                 tolerance, Z, z1s, z2s))
    #         Z[small_densities] = tolerance
    return (
        Z,
        norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, norm_cdf_z1s, norm_cdf_z2s)


def update_posterior(noise_std, noise_variance, posterior_mean,
        cutpoints_ts, cutpoints_tplus1s, K, N,
        upper_bound, upper_bound2):
    """Update Laplace approximation posterior covariance in Newton step."""
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = truncated_norm_normalising_constant(
            cutpoints_ts, cutpoints_tplus1s, noise_std,
            posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2)
    weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
    # This is not for numerical stability, it is mathematically correct
    z1s = jnp.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
    z2s = jnp.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
    precision  = weight**2 + (
        z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
        ) / Z / noise_variance
    m = - K @ weight + posterior_mean
    cov, L_cov = matrix_inverse(K + B.diag(1. / precision), N)
    log_det_cov = -2 * jnp.sum(B.log(B.diag(L_cov)))
    t1 = - (cov @ m) / precision
    posterior_mean += t1
    error = B.max(B.abs(t1))
    return error, weight, precision, cov, log_det_cov, posterior_mean


def posterior_covariance(K, cov, precision):
    return K @ cov @ B.diag(1./precision)


def compute_weights(
        posterior_mean, cutpoints_ts, cutpoints_tplus1s, noise_std,
        noise_variance, upper_bound, upper_bound2, N, K):
    # Numerically stable calculation of ordinal likelihood!
    (Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s, *_) = truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound=upper_bound, upper_bound2=upper_bound2)
    w1 = norm_pdf_z1s / Z
    w2 = norm_pdf_z2s / Z
    # Could use nanprod instead, here
    z1s = jnp.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
    z2s = jnp.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
    g1 = z1s * w1
    g2 = z2s * w2
    v1 = z1s * g1
    v2 = z2s * g2
    q1 = z1s * v1
    q2 = z2s * v2
    weight = (w1 - w2) / noise_std
    precision = weight**2 + (g2 - g1) / noise_variance
    cov, L_cov = matrix_inverse(K + B.diag(1./ precision), N)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    return (weight, precision, w1, w2, g1, g2, v1, v2, q1, q2, L_cov, cov, Z,
        log_det_cov)


def objective(weight, posterior_mean, precision, L_cov, Z):
    fx = -B.sum(B.log(Z))
    fx += 0.5 * posterior_mean.T @ weight
    fx += B.sum(B.log(B.diag(L_cov)))
    fx += 0.5 * B.sum(B.log(precision))
    return fx


def objective_gradient(
        gx, intervals, w1, w2, g1, g2, v1, v2, q1, q2,
        cov, weight, precision, y_train, trainables, K, partial_K_theta,
        partial_K_variance, noise_std, noise_variance, theta, variance,
        N, J, D, ARD):
    if trainables is not None:
        # diagonal of posterior covariance
        dsigma = cov @ K
        diag = B.diag(dsigma) / precision
        # partial lambda / partial phi_b = - partial lambda / partial f (* SIGMA)
        t1 = ((w2 - w1) - 3.0 * (w2 - w1) * (g2 - g1) - 2.0 * (w2 - w1)**3 - (v2 - v1)) / noise_variance
        # Update gx
        if trainables[0]:
            # For gx[0] -- ln\sigma
            cache = ((w2 - w1) * (g2 - g1) - (w2 - w1) + (v2 - v1)) / noise_variance
            # prepare D f / D delta_l
            t2 = - dsigma @ cache / precision
            tmp = (
                - 2.0 * precision
                + (2.0 * (w2 - w1) * (v2 - v1)
                + 2.0 * (w2 - w1)**2 * (g2 - g1)
                - (g2 - g1)
                + (g2 - g1)**2
                + (q2 - q1)) / noise_variance)
            gx[0] = B.sum(g2 - g1 + 0.5 * (tmp - t2 * t1) * diag)
            gx[0] = - gx[0] / 2.0 * noise_variance
        # For gx[1] -- \b_1
        if trainables[1]:
            # For gx[1], \phi_b^1
            t2 = dsigma @ precision
            # t2 = t2 / precision
            gx[1] = B.sum(w1 - w2)
            gx[1] += 0.5 * B.sum(t1 * (1 - t2) * diag)
            gx[1] = gx[1] / noise_std
        # For gx[2] -- ln\Delta^r
        for j in range(2, J):
            # Prepare D f / D delta_l
            cache0 = -(g2 + (w2 - w1) * w2) / noise_variance
            cache1 = - (g2 - g1 + (w2 - w1)**2) / noise_variance
            if trainables[j]:
                idxj = B.where(y_train == j - 1)
                idxg = B.where(y_train > j - 1)
                idxl = B.where(y_train < j - 1)
                cache = B.zeros(N)
                cache[idxj] = cache0[idxj]
                cache[idxg] = cache1[idxg]
                t2 = dsigma @ cache
                t2 = t2 / precision
                gx[j] -= B.sum(w2[idxj])
                temp = (
                    w2[idxj]
                    - 2.0 * (w2[idxj] - w1[idxj]) * g2[idxj]
                    - 2.0 * (w2[idxj] - w1[idxj])**2 * w2[idxj]
                    - v2[idxj]
                    - (g2[idxj] - g1[idxj]) * w2[idxj]) / noise_variance
                gx[j] += 0.5 * B.sum((temp - t2[idxj] * t1[idxj]) * diag[idxj])
                gx[j] -= B.sum(w2[idxg] - w1[idxg])
                gx[j] += 0.5 * B.sum(t1[idxg] * (1.0 - t2[idxg]) * diag[idxg])
                gx[j] += 0.5 * B.sum(-t2[idxl] * t1[idxl] * diag[idxl])
                gx[j] = gx[j] * intervals[j - 2] / noise_std
        # For gx[J] -- variance
        if trainables[J]:
            dmat = partial_K_variance @ cov
            t2 = (dmat @ weight) / precision
            # VC * VC * a' * partial_K_theta * a / 2
            gx[J] = -variance * 0.5 * weight.T @ partial_K_variance @ weight  # That's wrong. not the same calculation.
            # equivalent to -= theta * 0.5 * B.trace(cov @ partial_K_theta)
            gx[J] += variance * 0.5 * B.trace(dmat)
            gx[J] *= 2.0  # since theta = kappa / 2
        # For gx[J + 1] -- theta
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    dmat = partial_K_theta[d] @ cov
                    t2 = (dmat @ weight) / precision
                    gx[J + 1 + d] -= theta[d] * 0.5 * weight.T @ partial_K_theta[d] @ weight
                    gx[J + 1 + d] += theta[d] * 0.5 * B.sum((-diag * t1 * t2) / (noise_std))
                    gx[J + 1 + d] += theta[d] * 0.5 * B.sum(B.multiply(cov, partial_K_theta[d]))
        else:
            if trainables[J + 1]:
                dmat = partial_K_theta @ cov
                t2 = (dmat @ weight) / precision
                gx[J + 1] -= theta * 0.5 * weight.T @ partial_K_theta @ weight
                gx[J + 1] += theta * 0.5 * B.sum((-diag * t1 * t2) / (noise_std))
                gx[J + 1] += theta * 0.5 * B.sum(B.multiply(cov, partial_K_theta))
    return gx
