import lab as B
from math import inf
from scipy.stats import expon
from probit.lab.utilities import (
    h, truncated_norm_normalising_constant, matrix_inverse, log_over_sqrt_2_pi)


def objective_VB(
    N, posterior_mean, weight, trace_cov,
    trace_posterior_cov_div_var, Z, noise_variance,
    log_det_cov, verbose=False):
    """
    # TODO: tidy
    Calculate fx, the variational lower bound of the log marginal
    likelihood.

    .. math::
            \mathcal{F(	heta)} =,

        where :math:`F(\theta)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
        :math:`h`, :math:`\Pi`, :math:`K`. #TODO

    :arg int N: The number of datapoints.
    :arg m: The posterior mean.
    :type m: :class:`numpy.ndarray`
    :arg y: The posterior mean.
    :type y: :class:`numpy.ndarray`
    :arg K: The prior covariance.
    :type K: :class:`numpy.ndarray`
    :arg float noise_variance: The noise variance.
    :arg float log_det_cov: The log determinant of (a factor in) the
        posterior covariance.
    :arg Z: The array of normalising constants.
    :type Z: :class:`numpy.ndarray`
    :return: fx
    :rtype: float
    """
    trace_K_inv_posterior_cov = noise_variance * trace_cov
    one = - trace_posterior_cov_div_var / 2
    three = - trace_K_inv_posterior_cov / 2
    four = - posterior_mean.T @ weight / 2
    five = (N * B.log(noise_variance) + log_det_cov) / 2
    six = N / 2
    seven = B.sum(Z)
    fx = one + three + four + five + six  + seven
    if verbose:
        print("one ", one)
        print("three ", three)
        print("four ", four)  # Sometimes largest contribution
        print("five ", five)
        print("six ", six)
        print("seven ", seven)
        print('fx = {}'.format(fx))
    return -fx


def objective_gradient_VB(
        gx, intervals, cutpoints_ts, cutpoints_tplus1s, theta, variance,
        noise_variance, noise_std, y_train,
        posterior_mean, weight, cov, trace_cov, partial_K_theta, partial_K_variance,
        N, J, D, ARD,
        upper_bound, upper_bound2,
        Z, norm_pdf_z1s, norm_pdf_z2s, trainables,
        numerical_stability=True, verbose=False):
    """
    Calculate gx, the jacobian of the variational lower bound of the log
    marginal likelihood at the VB equilibrium,

    .. math::
            \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

        where :math:`F(\theta)` is the variational lower bound of the log
        marginal likelihood at the EP equilibrium,
        :math:`\theta` is the set of hyperparameters, :math:`h`,
        :math:`\Pi`, :math:`K`. #TODO

    :arg intervals: The vector of the first cutpoint and the intervals
        between cutpoints for unconstrained optimisation of the cutpoint
        parameters.
    :type intervals: :class:`numpy.ndarray`
    :arg theta: The kernel hyper-parameters.
    :type theta: :class:`numpy.ndarray` or float.
    :arg float noise_variance: The noise variance.
    :arg float noise_std:
    :arg m: The posterior mean.
    :type m: :class:`numpy.ndarray`
    :arg cov: An intermediate matrix in calculating the posterior
        covariance, posterior_cov.
    :type cov: :class:`numpy.ndarray`
    :arg posterior_cov: The posterior covariance.
    :type posterior_cov: :class:`numpy.ndarray`
    :arg K_inv: The inverse of the prior covariance.
    :type K_inv: :class:`numpy.ndarray`
    :arg Z: The array of normalising constants.
    :type Z: :class:`numpy.ndarray`
    :arg bool numerical_stability: If the function is evaluated in a
        numerically stable way, default `True`.
    :return: fx
    :rtype: float
    """
    if trainables is not None:
        # For gx[0] -- ln\sigma  # TODO: currently seems analytically incorrect
        if trainables[0]:
            one = N - noise_variance * trace_cov
            sigma_dp = dp(posterior_mean, cutpoints_ts, cutpoints_tplus1s,
                noise_std, upper_bound, upper_bound2)
            two = - (1. / noise_std) * B.sum(sigma_dp)
            if verbose:
                print("one ", one)
                print("two ", two)
                print("gx_sigma = ", one + two)
            gx[0] = one + two
        # For gx[1] -- \b_1
        if any(trainables[1:J]):  # TODO: analytic and numeric gradients do not match
            # TODO: treat these with numerical stability, or fix them
            temp_1s = B.divide(norm_pdf_z1s, Z)
            temp_2s = B.divide(norm_pdf_z2s, Z)

            gx[1] = B.sum(B.where(y_train == 0, temp_1s, 1.0))
            for j in range(2, J):
                gx[j - 1] += B.sum(B.where(y_train == j - 1, temp_2s, 0.0))
                gx[j] += B.sum(B.where(y_train == j - 1, temp_1s, 1.0))
            # gx[J] -= 0
            gx[1:J] /= noise_std
            # For gx[2:J] -- ln\Delta^r
            gx[2:J] *= intervals
            if verbose:
                print(gx[2:J])
        # For gx[J] -- s
        if trainables[J]:
            # VC * VC * a' * partial_K_theta * a / 2
            gx[J] = variance * 0.5 * weight.T @ partial_K_variance @ weight  # That's wrong. not the same calculation.
            # equivalent to -= theta * 0.5 * B.trace(cov @ partial_K_theta)
            gx[J] -= variance * 0.5 * B.sum(B.multiply(partial_K_variance, cov))
            gx[J] *= 2.0  # since theta = kappa / 2
        # For kernel parameters
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    if numerical_stability is True:
                        # Update gx[-1], the partial derivative of the lower bound
                        # wrt the lengthscale. Using matrix inversion Lemma
                        one = (theta[d] / 2) * weight.T @ partial_K_theta[d] @ weight
                        # TODO: slower but what about @jit compile CPU or GPU?
                        # D = solve_triangular(
                        #     L_cov.T, partial_K_theta, lower=True)
                        # D_inv = solve_triangular(L_cov, D, lower=False)
                        # two = - (theta / 2) * B.trace(D_inv)
                        two = - (theta[d] / 2) * B.einsum(
                            'ij, ji ->', partial_K_theta[d], cov)
                        gx[J + 1 + d] = one + two
                        if verbose:
                            print("one", one)
                            print("two", two)
                            print("gx = {}".format(gx[J + 1 + d]))
        else:
            if trainables[J + 1]:
                if numerical_stability is True:
                    # Update gx[-1], the partial derivative of the lower bound
                    # wrt the lengthscale. Using matrix inversion Lemma
                    one = (theta / 2) * weight.T @ partial_K_theta @ weight
                    # TODO: Seems to be missing a term that EP and LA have got!
                    # TODO: slower but what about @jit compile CPU or GPU?
                    # D = solve_triangular(
                    #     L_cov.T, partial_K_theta, lower=True)
                    # D_inv = solve_triangular(L_cov, D, lower=False)
                    # two = - (theta / 2) * B.trace(D_inv)
                    two = - (theta / 2) * B.einsum(
                        'ij, ji ->', partial_K_theta, cov)
                    gx[J + 1] = one + two
                    if verbose:
                        print("one", one)
                        print("two", two)
                        print("gx = {}".format(gx[J + 1]))
    return -gx


def update_posterior_mean_VB(noise_std, posterior_mean, cov,
        cutpoints_ts, cutpoints_tplus1s, K,
        upper_bound, upper_bound2):
    """Update VB approximation posterior covariance."""
    p = _p(
        posterior_mean, cutpoints_ts, cutpoints_tplus1s,
        noise_std, upper_bound, upper_bound2)
    g = _g(
        p, posterior_mean, noise_std)
    posterior_mean, weight = _posterior_mean(
        g, cov, K)
    return posterior_mean, weight


def update_posterior_covariance_VB(noise_variance, N, K):
    """Update posterior covariances."""
    cov, L_cov = matrix_inverse(noise_variance * B.eye(N) + K, N)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum(
        'ij, ij -> ', K, cov)
    return L_cov, cov, log_det_cov, trace_cov, trace_posterior_cov_div_var


def _g(p, posterior_mean, noise_std):
    """
    Calculate y elements 2021 Page Eq.().

    :arg p:
    :type p:
    :arg posterior_mean:
    :type posterior_mean:
    :arg float noise_std: The square root of the noise variance.
    """
    return B.add(posterior_mean, noise_std * p)


def _p(f, cutpoints_ts, cutpoints_tplus1s, noise_std,
        upper_bound, upper_bound2):
    """
    The rightmost term of 2021 Page Eq.(),
        correction terms that squish the function value m
        between the two cutpoints for that particle.

    :arg f: The current posterior mean estimate.
    :type f: :class:`numpy.ndarray`
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
        cutpoints_ts, cutpoints_tplus1s, noise_std, f)
    p = (norm_pdf_z1s - norm_pdf_z2s) / Z
    # Need to deal with the tails to prevent catestrophic cancellation
    if upper_bound is not None:
        # Using series expansion approximations
        p = B.where(z1s > upper_bound, p_tails(z1s, z2s), p)
        p = B.where(z2s < -upper_bound, p_tails(z1s, z2s), p)
        if upper_bound2 is not None:
            # Using one sided series expansion approximations
            p = B.where(z1s > upper_bound2, p_far_tails(z1s), p)
            p = B.where(z2s < -upper_bound2, p_far_tails(z2s), p)
            # # TODO check it shouldn't be:
            # p = B.where(z2s < -upper_bound2, p_far_tails(-z2s), p)
    return p


def _posterior_mean(g, cov, K):
    """
    Return the approximate posterior mean of m.

    2021 Page Eq.()

    :arg y: (N,) array
    :type y: :class:`numpy.ndarray`
    :arg cov:
    :type cov:
    :arg K:
    :type K:
    """
    weight = cov @ g
    ## TODO: This is 3-4 times slower on CPU, what about with jit compiled CPU or GPU?
    # weight = cho_solve((L_cov, ), y)
    return K @ weight, weight  # (N,), (N,)


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
    if upper_bound is not None:
        # Using series expansion approximations
        sigma_dp = B.where(z1s > upper_bound, dp_tails(z1s, z2s), sigma_dp)
        sigma_dp = B.where(z2s < -upper_bound, dp_tails(z1s, z2s), sigma_dp)
        if upper_bound2 is not None:
            sigma_dp = B.where(z1s == -inf, - z2s * norm_pdf_z2s / Z, sigma_dp)
            sigma_dp = B.where(z2s == inf, z1s * norm_pdf_z1s / Z, sigma_dp)
            # Using one sided series expansion approximations
            sigma_dp = B.where(
                z1s > upper_bound2, dp_far_tails(z1s), sigma_dp)
            sigma_dp = B.where(
                z2s < -upper_bound2, dp_far_tails(-z2s), sigma_dp)
    return sigma_dp


def p_tails(z1, z2):
    """
    Series expansion at infinity. Even for z1, z2 >= 4,
    this is accurate to three decimal places.
    """
    return (
        B.exp(-0.5 * z1**2) - B.exp(-0.5 * z2**2)) / (
            1 / z1 * B.exp(-0.5 * z1**2)* B.exp(h(z1))
            - 1 / z2 * B.exp(-0.5 * z2**2) * B.exp(h(z2)))


def p_far_tails(z):
    """Prevents overflow at large z."""
    return z * B.exp(-h(z))


def dp_tails(z1, z2):
    """Series expansion at infinity."""
    return (
        z1 * B.exp(-0.5 * z1**2) - z2 * B.exp(-0.5 * z2**2)) / (
            1 / z1 * B.exp(-0.5 * z1**2) * B.exp(h(z1))
            - 1 / z2 * B.exp(-0.5 * z2**2) * B.exp(h(z2)))


def dp_far_tails(z):
    """Prevents overflow at large z."""
    return z**2 * B.exp(-h(z))


def update_hyperparameter_posterior_VB(
    posterior_mean, theta, theta_hyperparameters):
    theta = _theta(
        posterior_mean, theta_hyperparameters,
        n_samples=10)
    theta_hyperparameters = _theta_hyperparameters(theta)


def _theta_hyperparameters(
        theta, theta_hyperhyperparameters):
    """
    Return the approximate posterior mean of the kernel
    theta hyperparameters.

    Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
    Probit Regression with Gaussian Process Priors," in Neural Computation,
    vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
    doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(10).

    :arg theta: Posterior mean approximate of theta.
    :type theta: :class:`numpy.ndarray`
    :return: The approximate posterior mean of the hyperhyperparameters psi
        Girolami and Rogers Page 9 Eq.(10).
    """
    return B.divide(
        B.add(1, theta_hyperhyperparameters[0]),
        B.add(theta_hyperhyperparameters[1], theta))


def _theta(
        self, posterior_mean, weight, theta_hyperparameters, N, n_samples=10,
        vectorised=False):
    """
    Return the weights of the importance sampler for theta.

    Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
    Probit Regression with Gaussian Process Priors," in Neural Computation,
    vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
    doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(9).

    :arg posterior_mean: approximate posterior mean.
    :arg theta_hyperparameters: approximate posterior mean of the kernel
        hyperparameters.
    :arg int n_samples: The number of samples for the importance sampling
        estimate, 500 is used in 2005 Page 13.
    """
    thetas = sample_thetas(  # (n_samples, N, N) for ISO case, depends on shape of theta hyper-hyperparameter
        theta_hyperparameters, n_samples)  # (n_samples, )
    log_thetas = B.log(thetas)
    Ks_samples = self.kernel.kernel_matrices(
        self.X_train, self.X_train, thetas)  # (n_samples, N, N)
    Ks_samples = B.add(Ks_samples, self.epsilon * B.eye(N))
    if vectorised:
        raise ValueError("TODO")
    else:
        log_ws = B.empty((n_samples,))
        # Scalar version
        for i in range(n_samples):
            L_K = B.cholesky(Ks_samples[i])
            half_log_det_K = B.sum(B.log(B.diag(L_K)))  # correct sign?
            # TODO 2021 - something not quite right - converges to zero
            # TODO: 28/07/2022 this may have been fixed by a fixing bug in the
            # code that did not update theta? Test
            # TODO: 28/07/2022 also may have made this more stable by using
            # K_inv @ posterior_mean = weight
            log_ws[i] = log_over_sqrt_2_pi - half_log_det_K\
                - 0.5 * posterior_mean.T @ weight
    # Normalise the w vectors
    max_log_ws = B.max(log_ws)
    log_normalising_constant = max_log_ws + B.log(
        B.sum(B.exp(log_ws - max_log_ws), axis=0))
    log_ws = B.subtract(log_ws, log_normalising_constant)
    print(B.sum(B.exp(log_ws)))
    element_prod = B.add(log_thetas, log_ws)
    element_prod = B.exp(element_prod)
    return B.sum(element_prod, axis=0)


def sample_thetas(theta_hyperparameter, n_samples):
    """
    Take n_samples of theta, given the hyperparameter of theta.

    theta_hyperparameter is a rate parameter since, with an uninformative
    prior (sigma=tau=0), then the posterior mean of Q(psi) is
    psi_tilde = 1. / theta_tilde. Therefore, by taking the expected value of
    the prior on theta ~ Exp(psi_tilde),
    we expect to obtain theta_tilde = 1. / psi_tilde. We get this if
    psi_tilde is a rate.

    :arg psi: float (Array) of hyper-hyperparameter(s)
    :type psi: :class:`np.ndarray`
    :arg int n_samples: The number of samples for the importance sample.
    """
    # scale = theta_hyperparameter
    scale = 1. / theta_hyperparameter
    shape = B.shape(theta_hyperparameter)
    if shape == ():
        size = (n_samples,)
    else:
        size = (n_samples, shape[0], shape[1])
    return expon.rvs(scale=scale, size=size)
