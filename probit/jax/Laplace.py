# # Enable double precision
# from jax.config import config
# config.update("jax_enable_x64", True)

import jax.numpy as jnp
from probit.jax.utilities import (
    truncated_norm_normalising_constant, matrix_inverse)



def update_posterior_LA(noise_std, noise_variance, posterior_mean,
        cutpoints_ts, cutpoints_tplus1s, K, N,
        upper_bound=None, upper_bound2=None, tolerance=None):
    """Update Laplace approximation posterior covariance in Newton step."""
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = truncated_norm_normalising_constant(
            cutpoints_ts, cutpoints_tplus1s, noise_std,
            posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2,
            tolerance=tolerance)
    weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
    # This is not for numerical stability, it is mathematically correct
    z1s = jnp.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
    z2s = jnp.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
    precision  = weight**2 + (
        z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
        ) / Z / noise_variance
    m = - K @ weight + posterior_mean
    cov, L_cov = matrix_inverse(K + jnp.diag(1. / precision), N)
    log_det_cov = -2 * jnp.sum(jnp.log(jnp.diag(L_cov)))
    t1 = - (cov @ m) / precision
    posterior_mean += t1
    # TODO: there is a better way of doing this
    error = jnp.abs(max(t1.min(), t1.max(), key=abs))
    return error, weight, precision, cov, log_det_cov, posterior_mean


def compute_weights_LA(
        posterior_mean, cutpoints_ts, cutpoints_tplus1s, noise_std,
        noise_variance,
        N, K,
        upper_bound=None, upper_bound2=None, tolerance=None):
    # Numerically stable calculation of ordinal likelihood!
    (Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s, *_) = truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound=upper_bound, upper_bound2=upper_bound2,
        tolerance=tolerance)
    w1 = norm_pdf_z1s / Z
    w2 = norm_pdf_z2s / Z
    # This is not for numerical stability, it is mathematically correct
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
    cov, L_cov = matrix_inverse(K + jnp.diag(1./ precision), N)
    log_det_cov = -2 * jnp.sum(jnp.log(jnp.diag(L_cov)))
    return (weight, precision, w1, w2, g1, g2, v1, v2, q1, q2, L_cov, cov, Z,
        log_det_cov)


def objective_LA(weight, posterior_mean, precision, L_cov, Z):
    fx = -jnp.sum(jnp.log(Z))
    fx += 0.5 * posterior_mean.T @ weight
    fx += jnp.sum(jnp.log(jnp.diag(L_cov)))
    fx += 0.5 * jnp.sum(jnp.log(precision))
    return fx


def objective_gradient_LA(
        gx, intervals, w1, w2, g1, g2, v1, v2, q1, q2,
        cov, weight, precision, y_train, trainables, K, partial_K_theta,
        partial_K_variance, noise_std, noise_variance, theta, variance,
        N, J, D, ARD):
    if trainables is not None:
        # diagonal of posterior covariance
        dsigma = cov @ K
        diag = jnp.diag(dsigma) / precision
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
            gx[0] = jnp.sum(g2 - g1 + 0.5 * (tmp - t2 * t1) * diag)
            gx[0] = - gx[0] / 2.0 * noise_variance
        # For gx[1] -- \b_1
        if trainables[1]:
            # For gx[1], \phi_b^1
            t2 = dsigma @ precision
            # t2 = t2 / precision
            gx[1] = jnp.sum(w1 - w2)
            gx[1] += 0.5 * jnp.sum(t1 * (1 - t2) * diag)
            gx[1] = gx[1] / noise_std
        # For gx[2] -- ln\Delta^r
        for j in range(2, J):
            # Prepare D f / D delta_l
            cache0 = -(g2 + (w2 - w1) * w2) / noise_variance
            cache1 = - (g2 - g1 + (w2 - w1)**2) / noise_variance
            if trainables[j]:
                idxj = jnp.where(y_train == j - 1)
                idxg = jnp.where(y_train > j - 1)
                idxl = jnp.where(y_train < j - 1)
                cache = jnp.zeros(N)
                cache[idxj] = cache0[idxj]
                cache[idxg] = cache1[idxg]
                t2 = dsigma @ cache
                t2 = t2 / precision
                gx[j] -= jnp.sum(w2[idxj])
                temp = (
                    w2[idxj]
                    - 2.0 * (w2[idxj] - w1[idxj]) * g2[idxj]
                    - 2.0 * (w2[idxj] - w1[idxj])**2 * w2[idxj]
                    - v2[idxj]
                    - (g2[idxj] - g1[idxj]) * w2[idxj]) / noise_variance
                gx[j] += 0.5 * jnp.sum((temp - t2[idxj] * t1[idxj]) * diag[idxj])
                gx[j] -= jnp.sum(w2[idxg] - w1[idxg])
                gx[j] += 0.5 * jnp.sum(t1[idxg] * (1.0 - t2[idxg]) * diag[idxg])
                gx[j] += 0.5 * jnp.sum(-t2[idxl] * t1[idxl] * diag[idxl])
                gx[j] = gx[j] * intervals[j - 2] / noise_std
        # For gx[J] -- variance
        if trainables[J]:
            dmat = partial_K_variance @ cov
            t2 = (dmat @ weight) / precision
            # VC * VC * a' * partial_K_theta * a / 2
            gx[J] = -variance * 0.5 * weight.T @ partial_K_variance @ weight  # That's wrong. not the same calculation.
            # equivalent to -= theta * 0.5 * jnp.trace(cov @ partial_K_theta)
            gx[J] += variance * 0.5 * jnp.trace(dmat)
            gx[J] *= 2.0  # since theta = kappa / 2
        # For gx[J + 1] -- theta
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    dmat = partial_K_theta[d] @ cov
                    t2 = (dmat @ weight) / precision
                    gx[J + 1 + d] -= theta[d] * 0.5 * weight.T @ partial_K_theta[d] @ weight
                    gx[J + 1 + d] += theta[d] * 0.5 * jnp.sum((-diag * t1 * t2) / (noise_std))
                    gx[J + 1 + d] += theta[d] * 0.5 * jnp.sum(jnp.multiply(cov, partial_K_theta[d]))
        else:
            if trainables[J + 1]:
                dmat = partial_K_theta @ cov
                t2 = (dmat @ weight) / precision
                gx[J + 1] -= theta * 0.5 * weight.T @ partial_K_theta @ weight
                gx[J + 1] += theta * 0.5 * jnp.sum((-diag * t1 * t2) / (noise_std))
                gx[J + 1] += theta * 0.5 * jnp.sum(jnp.multiply(cov, partial_K_theta))
    return gx
