import numpy as np
from scipy.linalg import solve_triangular
from probit.utilities import (
    norm_z_pdf, norm_pdf, norm_cdf, matrix_inverse)
from scipy.special import ndtr
import warnings
import math


def objective_EP(
        precision_EP, posterior_mean, t1, L_cov, cov, weights,
        tolerance2):
    """
    Calculate fx, the variational lower bound of the log marginal
    likelihood at the EP equilibrium.

    .. math::
            \mathcal{F(\theta)} =,

        where :math:`F(\theta)` is the variational lower bound of the log
        marginal likelihood at the EP equilibrium,
        :math:`h`, :math:`\Pi`, :math:`K`. #TODO

    :arg precision_EP:
    :type precision_EP:
    :arg posterior_mean:
    :type posterior_mean:
    :arg t1:
    :type t1:
    :arg L_cov:
    :type L_cov:
    :arg cov:
    :type cov:
    :arg weights:
    :type weights:
    :returns: fx
    :rtype: float
    """
    # Fill possible zeros in with machine precision
    precision_EP[precision_EP == 0.0] = tolerance2
    fx = -np.sum(np.log(np.diag(L_cov)))  # log det cov
    fx -= 0.5 * posterior_mean.T @ weights
    fx -= 0.5 * np.sum(np.log(precision_EP))
    # cov = L^{-1} L^{-T}  # requires a backsolve with the identity
    # TODO: check if there is a simpler calculation that can be done
    fx -= 0.5 * np.sum(np.divide(np.diag(cov), precision_EP))
    fx += np.sum(t1)
    # Regularisation - penalise large theta (overfitting)
    # fx -= 0.1 * theta
    return -fx

def objective_gradient_EP(
        gx, intervals, theta, variance, noise_variance, cutpoints,
        t2, t3, t4, t5, y_train, cov, weights, trainables,
        partial_K_theta, partial_K_variance, J, D, ARD):
    """
    Calculate gx, the jacobian of the variational lower bound of the
    log marginal likelihood at the EP equilibrium.

    .. math::
            \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

        where :math:`F(\theta)` is the variational lower bound of the 
        log marginal likelihood at the EP equilibrium,
        :math:`\theta` is the set of hyperparameters,
        :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

    :arg intervals:
    :type intervals:
    :arg theta: The kernel hyper-parameters.
    :type theta: :class:`numpy.ndarray` or float.
    :arg theta: The kernel hyper-parameters.
    :type theta: :class:`numpy.ndarray` or float.
    :arg t2:
    :type t2:
    :arg t3:
    :type t3:
    :arg t4:
    :type t4:
    :arg t5:
    :type t5:
    :arg cov:
    :type cov:
    :arg weights:
    :type weights:
    :return: gx
    :rtype: float
    """
    if trainables is not None:
        # Update gx
        if trainables[0]:
            # TODO: Doesn't work due to numerical instability, also need to check for algebraic error
            # For gx[0] -- ln\sigma
            # gx[0] -= 1 / np.sqrt(noise_variance) * np.sum(np.multiply(cov, K))
            # gx[0] *= -0.5 * noise_variance  # This is a typo in the Chu code
            gx[0] = np.sum(t5 - t4)
            #gx[0] *= np.sqrt(noise_variance) / 2.0
        # For gx[1] -- \b_1
        if trainables[1]:
            gx[1] = np.sum(t3 - t2)
            gx[1] *= cutpoints[1]
        # For gx[2] -- ln\Delta^r
        for j in range(2, J):
            if trainables[j]:
                targets = y_train
                gx[j] = np.sum(t3[targets == j - 1])
                gx[j] -= np.sum(t2[targets == J - 1])
                # TODO: check this, since it may be an `else` condition!!!!
                gx[j] += np.sum(t3[targets > j - 1] - t2[targets > j - 1])
                gx[j] *= intervals[j - 2]
        # For gx[J] -- variance
        if trainables[J]:
            # For gx[J] -- s
            # VC * VC * a' * partial_K_theta * a / 2
            gx[J] = variance * 0.5 * weights.T @ partial_K_variance @ weights  # That's wrong. not the same calculation.
            # equivalent to -= theta * 0.5 * np.trace(cov @ partial_K_theta)
            gx[J] -= variance * 0.5 * np.sum(np.multiply(cov, partial_K_variance))
            # ad-hoc Regularisation term - penalise large theta, but Occam's term should do this already
            # gx[J] -= 0.1 * theta
            gx[J] *= 2.0  # since theta = kappa / 2
        # For gx[J + 1] -- theta
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    gx[J + 1 + d] = theta[d] * 0.5 * weights.T @ partial_K_theta[d] @ weights
                    gx[J + 1 + d] -= theta[d] * 0.5 * np.sum(np.multiply(cov, partial_K_theta[d]))
        else:
            if trainables[J + 1]:
                # elif 1:
                #     gx[J + 1] = theta * 0.5 * weights.T @ partial_K_theta @ weights
                # TODO: This needs fixing/ checking vs original code
                # VC * VC * a' * partial_K_theta * a / 2
                gx[J + 1] = theta * 0.5 * weights.T @ partial_K_theta @ weights  # That's wrong. not the same calculation.
                # equivalent to -= theta * 0.5 * np.trace(cov @ partial_K_theta)
                gx[J + 1] -= theta * 0.5 * np.sum(np.multiply(cov, partial_K_theta))
                # ad-hoc Regularisation term - penalise large theta, but Occam's term should do this already
                # gx[J] -= 0.1 * theta
    return -gx


def update_posterior_EP(indices, posterior_mean, posterior_cov,
        mean_EP, precision_EP, amplitude_EP, dlogZ_dcavity_mean, error,
        y_train, cutpoints, noise_variance, J, upper_bound, tolerance,
        tolerance2):
    for index in indices:
        if (index + 1) % 1000 == 0: print(index)
        target = y_train[index]
        posterior_variance_n = posterior_cov[index, index]
        precision_EP_n_old = precision_EP[index]
        mean_EP_n_old = mean_EP[index]
        posterior_mean_n = posterior_mean[index]
        amplitude_EP_n_old = amplitude_EP[index]
        if posterior_variance_n > 0:
            cavity_variance_n = posterior_variance_n / (
                1 - posterior_variance_n * precision_EP_n_old)
            if cavity_variance_n > 0:
                # Remove
                # Calculate the product of approximate posterior factors
                # with the current index removed. This is called the cavity
                # distribution, "a bit like leaving a hole in the dataset"
                cavity_mean_n = (posterior_mean_n
                    + cavity_variance_n * precision_EP_n_old * (
                        posterior_mean_n - mean_EP_n_old))
                # Tilt/ moment match
                (mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
                dlogZ_dcavity_mean_n, posterior_covariance_n_new,
                z1, z2, nu_n) = _include(
                    target, cavity_mean_n, cavity_variance_n,
                    cutpoints[target], cutpoints[target + 1],
                    noise_variance, upper_bound, tolerance, tolerance2, J)
                # Update EP weight (alpha)
                dlogZ_dcavity_mean[index] = dlogZ_dcavity_mean_n
                diff = precision_EP_n - precision_EP_n_old
                if (np.abs(diff) > tolerance
                        and Z_n > tolerance
                        and precision_EP_n > 0.0
                        and posterior_covariance_n_new > 0.0):
                    posterior_mean, posterior_cov = _update(
                        index, posterior_mean, posterior_cov,
                        posterior_mean_n, posterior_variance_n,
                        mean_EP_n_old, precision_EP_n_old,
                        dlogZ_dcavity_mean_n, diff)
                    # Update EP parameters
                    error += (diff**2
                            + (mean_EP_n - mean_EP_n_old)**2
                            + (amplitude_EP_n - amplitude_EP_n_old)**2)
                    precision_EP[index] = precision_EP_n
                    mean_EP[index] = mean_EP_n
                    amplitude_EP[index] = amplitude_EP_n
                else:
                    if precision_EP_n < 0.0 or posterior_covariance_n_new < 0.0:
                        print(
                            "Skip {} update z1={}, z2={}, nu={} p_new={},"
                            " p_old={}.\n".format(
                            index, z1, z2, nu_n,
                            precision_EP_n, precision_EP_n_old))
    return (posterior_mean, posterior_cov,
        mean_EP, precision_EP, amplitude_EP, dlogZ_dcavity_mean, error)


def _assert_valid_values(nu_n, variance, cavity_mean_n,
        cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1, norm_pdf_z2,
        dlogZ_dcavity_variance_n, dlogZ_dcavity_mean_n, tolerance):
    if math.isnan(dlogZ_dcavity_mean_n):
        print(
            "cavity_mean_n={} \n"
            "cavity_variance_n={} \n"
            "target={} \n"
            "z1 = {} z2 = {} \n"
            "Z_n = {} \n"
            "norm_pdf_z1 = {} \n"
            "norm_pdf_z2 = {} \n"
            "beta = {} alpha = {}".format(
                cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                norm_pdf_z1, norm_pdf_z2, dlogZ_dcavity_variance_n,
                dlogZ_dcavity_mean_n))
        raise ValueError(
            "dlogZ_dcavity_mean is nan (got {})".format(
            dlogZ_dcavity_mean_n))
    if math.isnan(dlogZ_dcavity_variance_n):
        print(
            "cavity_mean_n={} \n"
            "cavity_variance_n={} \n"
            "target={} \n"
            "z1 = {} z2 = {} \n"
            "Z_n = {} \n"
            "norm_pdf_z1 = {} \n"
            "norm_pdf_z2 = {} \n"
            "beta = {} alpha = {}".format(
                cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                norm_pdf_z1, norm_pdf_z2, dlogZ_dcavity_variance_n,
                dlogZ_dcavity_mean_n))
        raise ValueError(
            "dlogZ_dcavity_variance is nan (got {})".format(
                dlogZ_dcavity_variance_n))
    if nu_n <= 0:
        print(
            "cavity_mean_n={} \n"
            "cavity_variance_n={} \n"
            "target={} \n"
            "z1 = {} z2 = {} \n"
            "Z_n = {} \n"
            "norm_pdf_z1 = {} \n"
            "norm_pdf_z2 = {} \n"
            "beta = {} alpha = {}".format(
                cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                norm_pdf_z1, norm_pdf_z2, dlogZ_dcavity_variance_n,
                dlogZ_dcavity_mean_n))
        raise ValueError("nu_n must be positive (got {})".format(nu_n))
    if nu_n > 1.0 / variance + tolerance:
        print(
            "cavity_mean_n={} \n"
            "cavity_variance_n={} \n"
            "target={} \n"
            "z1 = {} z2 = {} \n"
            "Z_n = {} \n"
            "norm_pdf_z1 = {} \n"
            "norm_pdf_z2 = {} \n"
            "beta = {} alpha = {}".format(
                cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                norm_pdf_z1, norm_pdf_z2, dlogZ_dcavity_variance_n,
                dlogZ_dcavity_mean_n))
        raise ValueError(
            "nu_n must be less than 1.0 / (cavity_variance_n + "
            "noise_variance) = {}, got {}".format(
                1.0 / variance, nu_n))


def _include(
        target, cavity_mean_n, cavity_variance_n,
        cutpoints_t, cutpoints_tplus1, noise_variance,
        upper_bound, tolerance, tolerance2, J, numerically_stable=False):
    """
    Update the approximate posterior by incorporating the message
    p(t_i|m_i) into Q^{\i}(\bm{f}).
    Wei Chu, Zoubin Ghahramani 2005 page 20, Eq. (23)
    This includes one true-observation likelihood, and 'tilts' the
    approximation towards the true posterior. It updates the approximation
    to the true posterior by minimising a moment-matching KL divergence
    between the tilted distribution and the posterior distribution. This
    gives us an approximate posterior in the approximating family. The
    update to posterior_cov is a rank-1 update (see the outer product of
    two 1d vectors), and so it essentially constructs a piecewise low rank
    approximation to the GP posterior covariance matrix, until convergence
    (by which point it will no longer be low rank).
    :arg int target: The ordinal class index of the current site
        (the class of the datapoint that is "left out").
    :arg float cavity_mean_n: The cavity mean of the current site.
    :arg float cavity_variance_n: The cavity variance of the current site.
    :arg float cutpoints_t: The upper cutpoint parameters.
    :arg float cutpoints_tplus1: The lower cutpoint parameter.
    :arg float noise_variance: Initialisation of noise variance. If
        `None` then initialised to one, default `None`.
    :arg bool numerically_stable: Boolean variable for assert valid
        numerical values. Default `False'.
    :returns: A (10,) tuple containing cavity mean and variance, and old
        site states.
    """
    variance = cavity_variance_n + noise_variance
    std_dev = np.sqrt(variance)
    # Compute Z
    norm_cdf_z2 = 0.0
    norm_cdf_z1 = 1.0
    norm_pdf_z1 = 0.0
    norm_pdf_z2 = 0.0
    z1 = 0.0
    z2 = 0.0
    if target == 0:
        z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
        z1_abs = np.abs(z1)
        if z1_abs > upper_bound:
            z1 = np.sign(z1) * upper_bound
        Z_n = norm_cdf(z1) - norm_cdf_z2
        norm_pdf_z1 = norm_z_pdf(z1)
    elif target == J - 1:
        z2 = (cutpoints_t - cavity_mean_n) / std_dev
        z2_abs = np.abs(z2)
        if z2_abs > upper_bound:
            z2 = np.sign(z2) * upper_bound
        Z_n = norm_cdf_z1 - norm_cdf(z2)
        norm_pdf_z2 = norm_z_pdf(z2)
    else:
        z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
        z2 = (cutpoints_t - cavity_mean_n) / std_dev
        Z_n = norm_cdf(z1) - norm_cdf(z2)
        norm_pdf_z1 = norm_z_pdf(z1)
        norm_pdf_z2 = norm_z_pdf(z2)
    if Z_n < tolerance:
        if np.abs(np.exp(-0.5*z1**2 + 0.5*z2**2) - 1.0) > tolerance2:
            dlogZ_dcavity_mean_n = (z1 * np.exp(
                    -0.5*z1**2 + 0.5*z2**2) - z2**2) / (
                (
                    (np.exp(-0.5 * z1 ** 2) + 0.5 * z2 ** 2) - 1.0)
                    * variance
            )
            dlogZ_dcavity_variance_n = (
                -1.0 + (z1**2 + 0.5 * z2**2) - z2**2) / (
                (
                    (np.exp(-0.5*z1**2 + 0.5 * z2**2) - 1.0)
                    * 2.0 * variance)
            )
            dlogZ_dcavity_mean_n_2 = dlogZ_dcavity_mean_n**2
            nu_n = (
                dlogZ_dcavity_mean_n_2
                - 2.0 * dlogZ_dcavity_variance_n)
        else:
            dlogZ_dcavity_mean_n = 0.0
            dlogZ_dcavity_mean_n_2 = 0.0
            dlogZ_dcavity_variance_n = -(
                1.0 - tolerance)/(2.0 * variance)
            nu_n = (1.0 - tolerance) / variance
            warnings.warn(
                "Z_n must be greater than tolerance={} (got {}): "
                "SETTING to Z_n to approximate value\n"
                "z1={}, z2={}".format(
                    tolerance, Z_n, z1, z2))
        if nu_n >= 1.0 / variance:
            nu_n = (1.0 - tolerance) / variance
        if nu_n <= 0.0:
            nu_n = tolerance * variance
    else:
        dlogZ_dcavity_variance_n = (
            - z1 * norm_pdf_z1 + z2 * norm_pdf_z2) / (
                2.0 * variance * Z_n)  # beta
        dlogZ_dcavity_mean_n = (
            - norm_pdf_z1 + norm_pdf_z2) / (
                std_dev * Z_n)  # alpha/gamma
        dlogZ_dcavity_mean_n_2 = dlogZ_dcavity_mean_n**2
        nu_n = (dlogZ_dcavity_mean_n_2
            - 2.0 * dlogZ_dcavity_variance_n)
    # Update alphas
    if numerically_stable:
        _assert_valid_values(
            nu_n, variance, cavity_mean_n, cavity_variance_n, target,
            z1, z2, Z_n, norm_pdf_z1,
            norm_pdf_z2, dlogZ_dcavity_variance_n,
            dlogZ_dcavity_mean_n, tolerance)
    # posterior_mean_n_new = (  # Not used for anything
    #     cavity_mean_n + cavity_variance_n * dlogZ_dcavity_mean_n)
    posterior_covariance_n_new = (
        cavity_variance_n - cavity_variance_n**2 * nu_n)
    precision_EP_n = nu_n / (1.0 - cavity_variance_n * nu_n)
    mean_EP_n = cavity_mean_n + dlogZ_dcavity_mean_n / nu_n
    amplitude_EP_n = Z_n * np.sqrt(
        cavity_variance_n * precision_EP_n + 1.0) * np.exp(
            0.5 * dlogZ_dcavity_mean_n_2 / nu_n)
    return (
        mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
        dlogZ_dcavity_mean_n,
        posterior_covariance_n_new, z1, z2, nu_n)

def _update(
        index, posterior_mean, posterior_cov,
        posterior_mean_n, posterior_variance_n,
        mean_EP_n_old, precision_EP_n_old,
        dlogZ_dcavity_mean_n, diff):
    """
    Update the posterior mean and covariance.

    Projects the tilted distribution on to an approximating family.
    The update for the t_n is a rank-1 update. Constructs a low rank
    approximation to the GP posterior covariance matrix.

    :arg int index: The index of the current likelihood (the index of the
        datapoint that is "left out").
    :arg float mean_EP_n_old: The state of the individual (site) mean (N,).
    :arg posterior_cov: The current approximate posterior covariance
        (N, N).
    :type posterior_cov: :class:`numpy.ndarray`
    :arg float posterior_variance_n: The current approximate posterior
        site variance.
    :arg float posterior_mean_n: The current site approximate posterior
        mean.
    :arg float precision_EP_n_old: The state of the individual (site)
        variance (N,).
    :arg float dlogZ_dcavity_mean_n: The gradient of the log
        normalising constant with respect to the site cavity mean
        (The EP "weight").
    :arg float posterior_mean_n_new: The state of the site approximate
        posterior mean.
    :arg float posterior_covariance_n_new: The state of the site
        approximate posterior variance.
    :arg float diff: The differance between precision_EP_n and
        precision_EP_n_old.
    :returns: The updated approximate posterior mean and covariance.
    :rtype: tuple (`numpy.ndarray`, `numpy.ndarray`)
    """
    rho = diff / (1 + diff * posterior_variance_n)
    eta = (
        dlogZ_dcavity_mean_n
        + precision_EP_n_old * (posterior_mean_n - mean_EP_n_old)) / (
            1.0 - posterior_variance_n * precision_EP_n_old)
    # Update posterior mean and rank-1 covariance
    a_n = posterior_cov[:, index]
    posterior_mean += eta * a_n
    posterior_cov = posterior_cov - rho * np.outer(
        a_n, a_n) 
    return posterior_mean, posterior_cov


def approximate_evidence_EP(mean_EP, precision_EP, amplitude_EP,
        K, posterior_cov):
    """
    TODO: check and return line could be at risk of overflow
    Compute the approximate evidence at the EP solution.

    :return:
    """
    temp = np.multiply(mean_EP, precision_EP)
    B = temp.T @ posterior_cov @ temp - np.multiply(
        temp, mean_EP)
    Pi_inv = np.diag(1. / precision_EP)
    return (
        np.prod(
            amplitude_EP) * np.sqrt(np.linalg.det(Pi_inv)) * np.exp(B / 2)
            / np.sqrt(np.linalg.det(np.add(Pi_inv, K))))


def compute_weights_EP(
        precision_EP, mean_EP, dlogZ_dcavity_mean, K, tolerance, tolerance2,
        N,
        L_cov=None, cov=None, numerically_stable=False):
    """
    TODO: There may be an issue, where dlogZ_dcavity_mean is updated
    when it shouldn't be, on line 2045.

    Compute the regression weights required for the gradient evaluation,
    and check that they are in equilibrium with
    the gradients of Z wrt cavity means.

    A matrix inverse is always required to evaluate fx.

    :arg precision_EP:
    :arg mean_EP:
    :arg dlogZ_dcavity_mean:
    :arg L_cov: . Default `None`.
    :arg cov: . Default `None`.
    """
    if np.any(precision_EP == 0.0):
        # TODO: Only check for equilibrium if it has been updated in this swipe
        warnings.warn("Some sample(s) have not been updated.\n")
        precision_EP[precision_EP == 0.0] = tolerance2
    Pi_inv = np.diag(1. / precision_EP)
    if L_cov is None or cov is None:
        # TODO It is necessary to do this triangular solve to get
        # diag(cov) for the lower bound on the marginal likelihood
        # calculation. Note no tf implementation for diag(A^{-1}) yet.
        cov, L_cov = matrix_inverse(Pi_inv + K, N)
    if numerically_stable:
        # This is 3-4 times slower on CPU,
        # what about with jit compiled CPU or GPU?
        # Is this ever more stable than a matmul by the inverse?
        # TODO changed to solve_triangular of 28/07/2022 double check it works
        g = solve_triangular(L_cov, mean_EP, lower=False)
        weight = solve_triangular(L_cov.T, g, lower=True)
    else:
        weight = cov @ mean_EP
    if np.any(
        np.abs(weight - dlogZ_dcavity_mean) > np.sqrt(tolerance)):
        warnings.warn("Fatal error: the weights are not in equilibrium wit"
            "h the gradients".format(
                weight, dlogZ_dcavity_mean))
    return weight, precision_EP, L_cov, cov
    

def approximate_log_marginal_likelihood_EP(
        K, posterior_cov, precision_EP, amplitude_EP, mean_EP,
        numerical_stability):
    """
    Calculate the approximate log marginal likelihood.
    TODO: need to finish this. Probably not useful if using EP.

    :arg posterior_cov: The approximate posterior covariance.
    :type posterior_cov:
    :arg precision_EP: The state of the individual (site) variance (N,).
    :type precision_EP:
    :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
    :type amplitude EP:
    :arg mean_EP: The state of the individual (site) mean (N,).
    :type mean_EP:
    :arg bool numerical_stability: If the calculation is made in a
        numerically stable manner.
    """
    precision_matrix = np.diag(precision_EP)
    inverse_precision_matrix = 1. / precision_matrix  # Since it is a diagonal, this is the inverse.
    log_amplitude_EP = np.log(amplitude_EP)
    temp = np.multiply(mean_EP, precision_EP)
    B = temp.T @ posterior_cov @ temp\
            - temp.T @ mean_EP
    if numerical_stability is True:
        approximate_marginal_likelihood = np.add(
            log_amplitude_EP, 0.5 * np.trace(
                np.log(inverse_precision_matrix)))
        approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, B/2)
        approximate_marginal_likelihood = np.subtract(
            approximate_marginal_likelihood, 0.5 * np.trace(
                np.log(K + inverse_precision_matrix)))
        return np.sum(approximate_marginal_likelihood)
    else:
        approximate_marginal_likelihood = np.add(
            log_amplitude_EP, 0.5 * np.log(np.linalg.det(
                inverse_precision_matrix)))  # TODO: use log det C trick
        approximate_marginal_likelihood = np.add(
            approximate_marginal_likelihood, B/2
        )
        approximate_marginal_likelihood = np.add(
            approximate_marginal_likelihood, 0.5 * np.log(
                np.linalg.det(K + inverse_precision_matrix))
        )  # TODO: use log det C trick
        return np.sum(approximate_marginal_likelihood)


def compute_integrals_vector_EP(
        posterior_variance, posterior_mean, noise_variance,
        cutpoints_ts, cutpoints_tplus1s, indices_where_J_1,
        indices_where_0, N, tolerance, tolerance2):
    """
    Compute the integrals required for the gradient evaluation.
    """
    noise_std = np.sqrt(noise_variance)
    mean_ts = (posterior_mean * noise_variance
        + posterior_variance * cutpoints_ts) / (
            noise_variance + posterior_variance)
    mean_tplus1s = (posterior_mean * noise_variance
        + posterior_variance * cutpoints_tplus1s) / (
            noise_variance + posterior_variance)
    sigma = np.sqrt(
        (noise_variance * posterior_variance) / (
        noise_variance + posterior_variance))
    a_ts = mean_ts - 5.0 * sigma
    b_ts = mean_ts + 5.0 * sigma
    h_ts = b_ts - a_ts
    a_tplus1s = mean_tplus1s - 5.0 * sigma
    b_tplus1s = mean_tplus1s + 5.0 * sigma
    h_tplus1s = b_tplus1s - a_tplus1s
    y_0 = np.zeros((20, N))
    t1 = fromb_t1_vector(
            y_0.copy(), posterior_mean, posterior_variance,
            cutpoints_ts, cutpoints_tplus1s,
            noise_std, tolerance, tolerance2, N)
    t2 = fromb_t2_vector(
            y_0.copy(), mean_ts, sigma,
            a_ts, b_ts, h_ts,
            posterior_mean,
            posterior_variance,
            cutpoints_ts,
            cutpoints_tplus1s,
            noise_variance, noise_std, tolerance, tolerance2,
            N)
    t2[indices_where_0] = 0.0
    t3 = fromb_t3_vector(
            y_0.copy(), mean_tplus1s, sigma,
            a_tplus1s, b_tplus1s,
            h_tplus1s, posterior_mean,
            posterior_variance,
            cutpoints_ts,
            cutpoints_tplus1s,
            noise_variance, noise_std, tolerance, tolerance2,
            N)
    t3[indices_where_J_1] = 0.0
    t4 = fromb_t4_vector(
            y_0.copy(), mean_tplus1s, sigma,
            a_tplus1s, b_tplus1s,
            h_tplus1s, posterior_mean,
            posterior_variance,
            cutpoints_ts,
            cutpoints_tplus1s,
            noise_variance, noise_std, tolerance, tolerance2,
            N)
    t4[indices_where_J_1] = 0.0
    t5 = fromb_t5_vector(
            y_0.copy(), mean_ts, sigma,
            a_ts, b_ts, h_ts,
            posterior_mean,
            posterior_variance,
            cutpoints_ts,
            cutpoints_tplus1s,
            noise_variance, noise_std, tolerance, tolerance2,
            N)
    t5[indices_where_0] = 0.0
    # print("t4", t4)
    # print("t5", t5)
    return t1, t2, t3, t4, t5


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
            noise_std, noise_variance,
            cutpoints_t, cutpoints_tplus1,
            EPS_2)
        + fromb_fft4_vector(
            b, mean, sigma, posterior_mean, posterior_covariance,
            noise_std, noise_variance,
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
                noise_std, noise_variance,
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
