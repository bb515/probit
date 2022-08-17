from cmath import inf
import lab.jax as B
from math import inf
from probit.lab.utilities import (
    truncated_norm_normalising_constant, matrix_inverse)


def update_posterior_LA(noise_std, noise_variance, posterior_mean,
        cutpoints_ts, cutpoints_tplus1s, K, N,
        upper_bound=None, upper_bound2=None):
    """Update Laplace approximation posterior covariance in Newton step."""
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = truncated_norm_normalising_constant(
            cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2)
    weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
    # This is not for numerical stability, it is mathematically correct
    z1s = B.where(z1s == -inf, 0.0, z1s)
    z1s = B.where(z1s == inf, 0.0, z1s)
    z2s = B.where(z2s == -inf, 0.0, z2s)
    z2s = B.where(z2s == inf, 0.0, z2s)
    precision  = weight**2 + (
        z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
        ) / Z / noise_variance
    cov, L_cov = matrix_inverse(K + B.diag(1. / precision), N)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    residual = - K @ weight + posterior_mean
    t1 = - (cov @ residual) / precision
    posterior_mean += t1
    error = B.max(B.abs(t1))
    print(error)
    return error, weight, precision, cov, log_det_cov, posterior_mean


def compute_weights_LA(
        posterior_mean, cutpoints_ts, cutpoints_tplus1s, noise_std,
        noise_variance,
        N, K, upper_bound=None, upper_bound2=None):
    # TODO: combine this with update_posterior_LA
    # Numerically stable calculation of ordinal likelihood!
    (Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s, *_) = truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound=upper_bound, upper_bound2=upper_bound2)
    w1 = norm_pdf_z1s / Z
    w2 = norm_pdf_z2s / Z
    # This is not for numerical stability, it is mathematically correct
    z1s = B.where(z1s == -inf, 0.0, z1s)
    z1s = B.where(z1s == inf, 0.0, z1s)
    z2s = B.where(z2s == -inf, 0.0, z2s)
    z2s = B.where(z2s == inf, 0.0, z2s)
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


def objective_LA(weight, posterior_mean, precision, L_cov, Z):
    return (-B.sum(B.log(Z))
        + 0.5 * posterior_mean.T @ weight
        + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * B.sum(B.log(precision)))


def objective_gradient_LA(
        gx, intervals, w1, w2, g1, g2, v1, v2, q1, q2,
        cov, weight, precision, y_train, trainables, K, partial_K_theta,
        partial_K_variance, noise_std, noise_variance, theta, variance,
        J, D, ARD):
    if trainables is not None:
        # diagonal of posterior covariance
        dsigma = cov @ K
        diag = B.diag(dsigma) / precision
        t1 = ((w2 - w1) - 3.0 * (w2 - w1) * (g2 - g1)
            - 2.0 * (w2 - w1)**3 - (v2 - v1)) / noise_variance
        # Update gx
        if trainables[0]:
            # For gx[0] -- ln\sigma
            cache = ((w2 - w1) * (g2 - g1)
                - (w2 - w1) + (v2 - v1)) / noise_variance
            # prepare D f / D delta_l
            t2 = - dsigma @ cache / precision
            tmp = (
                - 2.0 * precision
                + (2.0 * (w2 - w1) * (v2 - v1)
                + 2.0 * (w2 - w1)**2 * (g2 - g1)
                - (g2 - g1)
                + (g2 - g1)**2
                + (q2 - q1)) / noise_variance)
            gx[0] = B.sum(g2 - g1) + B.sum((tmp - t2 * t1) * diag / 2.0)
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
                cache = B.where(y_train == j - 1, cache0, 0)
                cache = B.where(y_train > j - 1, cache1, cache)
                t2 = dsigma @ cache / precision
                gx[j] -= B.sum(B.where(y_train == j - 1, w2, 0))
                gx[j] += 0.5 * B.sum(B.where(y_train == j - 1,
                    ((w2 - 2.0 * (w2 - w1) * g2
                    - 2.0 * (w2 - w1)**2 * w2
                    - v2
                    - (g2 - g1) * w2) / noise_variance - t2 * t1) * diag, 0))
                gx[j] -= B.sum(B.where(y_train > j - 1, w2 - w1, 0))
                gx[j] += 0.5 * B.sum(B.where(y_train > j - 1, t1 * (1.0 - t2) * diag, 0))
                gx[j] += 0.5 * B.sum(B.where(y_train < j - 1, -t2 * t1 * diag, 0))
                gx[j] = gx[j] * intervals[j - 2] / noise_std
        # For gx[J] -- variance
        if trainables[J]:
            dmat = partial_K_variance @ cov
            t2 = (dmat @ weight) / precision
            gx[J] = -variance * 0.5 * weight.T @ partial_K_variance @ weight
            gx[J] += variance * 0.5 * B.trace(dmat)
            gx[J] *= 2.0
        # For gx[J + 1] -- theta
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    dmat = partial_K_theta[d] @ cov
                    t2 = (dmat @ weight) / precision
                    gx[J + 1 + d] = (
                - theta[d] * 0.5 * weight.T @ partial_K_theta[d] @ weight
                + theta[d] * 0.5 * B.sum((-diag * t1 * t2) / (noise_std))
                + theta[d] * 0.5 * B.sum(B.multiply(cov, partial_K_theta[d])))
        else:
            if trainables[J + 1]:
                dmat = partial_K_theta @ cov
                t2 = (dmat @ weight) / precision
                gx[J + 1] = (
            -theta * 0.5 * weight.T @ partial_K_theta @ weight
            + theta * 0.5 * B.sum((-diag * t1 * t2) / (noise_std))
            + theta * 0.5 * B.sum(B.multiply(cov, partial_K_theta)))
    return gx
