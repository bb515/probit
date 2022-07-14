"""Priors for hyperparameters."""
import numpy as np
from scipy.stats import expon, norm
from scipy.stats import gamma
from scipy.special import ndtr, log_ndtr
import warnings


def prior(phi, indices, J, varphi_hyperparameters, noise_std_hyperparameters,
        cutpoints_hyperparameters, scale_hyperparameters, cutpoints_0):
    """
    A priors defined over their usual domains, and so a transformation of
    random variables is used for sampling from proposal distrubutions defined
    over continuous domains. Hyperparameter priors assumed to be independent
    assumption so take the product of prior pdfs.
    """
    # Do not update these hyperparameters by default
    noise_variance = None
    cutpoints = None
    scale = None
    varphi = None
    log_prior_theta = np.zeros(len(phi))
    index = 0
    if indices[0]:
        # Gamma prior is placed on the noise std - evaluate the prior pdf
        noise_std = np.exp(phi[index])
        log_prior_pdf = gamma.logpdf(
            noise_std,
            a=noise_std_hyperparameters[0],
            scale=noise_std_hyperparameters[1])
        log_prior_theta[index] = log_prior_pdf
        # scale = scale_0
        index += 1
    for j in range(1, J):
        if indices[j]:
            cutpoints = cutpoints_0
    if indices[1]:
        cutpoints[1] = phi[index]
        log_prior_pdf = norm.logpdf(
            phi[index],
            loc=cutpoints_hyperparameters[1, 0],
            scale=cutpoints_hyperparameters[1, 1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    for j in range(2, J):
        if indices[j]:
            cutpoints[j] = cutpoints[j-1] + np.exp(phi[index])
            log_prior_pdf = norm.logpdf(
                phi[index],
                loc=cutpoints_hyperparameters[j, 0],
                scale=cutpoints_hyperparameters[j, 1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
    if indices[J]:
        scale_std = np.exp(phi[index])
        scale = scale_std**2
        log_prior_pdf = norm.logpdf(
                phi[index],
                loc=cutpoints_hyperparameters[J, 0],
                scale=cutpoints_hyperparameters[J, 1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    if indices[J + 1]:
        # if 0:  # self.approximator.kernel._general and self.approximator.kernel._ARD:
        #     # In this case, then there is a scale parameter, the first
        #     # cutpoint, the interval parameters,
        #     # and lengthscales parameter for each dimension and class
        #     raise ValueError("TODO")
        # else:
        #     # In this case, then there is a scale parameter, the first
        #     # cutpoint, the interval parameters,
        #     # and a single, shared lengthscale parameter
        varphi = np.exp(phi[index])
        # log_prior_pdf = expon.logpdf(
        #     varphi,
        #     loc=varphi_hyperparameters[0],
        #     scale=varphi_hyperparameters[1])
        log_prior_pdf = gamma.logpdf(
            varphi,
            a=varphi_hyperparameters[0],
            scale=varphi_hyperparameters[1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    return cutpoints, varphi, scale, noise_variance, log_prior_theta


def prior_reparameterised(
        phi, indices, J, varphi_hyperparameters, noise_std_hyperparameters,
        cutpoints_hyperparameters, scale_hyperparameters, cutpoints_0):
    """
    The variables \theta has been reparameterized into \phi, so now are infact
    sampling from the marginal p(\phi|y). Therefore, no jacobian is required.

    A reparametrisation such that all of the hyperparameters can be sampled from the real line,
    therefore there is no transformation (thus no jacobian) when sampling from the proposal distribution.
    Hyperparameter priors assumed to be independent assumption so take the product of prior pdfs.
    """
    # Do not update these hyperparameters by default
    noise_variance = None
    cutpoints = None
    scale = None
    varphi = None
    index = 0
    log_prior_phi = np.zeros(np.size(phi))
    if indices[0]:
        # Gamma prior is placed on the noise std - evaluate the prior pdf
        log_noise_std = phi[index]
        log_prior_pdf = norm.logpdf(
            log_noise_std,
            loc=noise_std_hyperparameters[0],
            scale=noise_std_hyperparameters[1])
        log_prior_phi[index] = log_prior_pdf
        noise_variance = np.exp(log_noise_std)**2
        # scale = scale_0
        if noise_variance < 1.0e-04:
            warnings.warn(
                "WARNING: noise variance is very low - numerical stability"
                " issues may arise (noise_variance={}).".format(
                    noise_variance))
        elif noise_variance > 1.0e3:
            warnings.warn(
                "WARNING: noise variance is very large - numerical "
                "stability issues may arise (noise_variance={}).".format(
                    noise_variance))
        index += 1
    for j in range(1, J):
        if indices[j]:
            cutpoints = cutpoints_0
    if indices[1]:
        cutpoints[1] = phi[index]
        log_prior_pdf = norm.logpdf(
            phi[index],
            loc=cutpoints_hyperparameters[1, 0],
            scale=cutpoints_hyperparameters[1, 1])
        log_prior_phi[index] = log_prior_pdf
        index += 1
    for j in range(2, J):
        if indices[j]:
            cutpoints[j] = cutpoints[j-1] + np.exp(phi[index])
            log_prior_pdf = norm.pdf(
                phi[index],
                loc=cutpoints_hyperparameters[j, 0],
                scale=cutpoints_hyperparameters[j, 1])
            log_prior_phi[index] = log_prior_pdf
            index += 1
    if indices[J]:
        scale_std = np.exp(phi[index])
        scale = scale_std**2
        log_prior_pdf = norm.logpdf(
            phi[index],
            loc=scale_hyperparameters[J, 0],
            scale=scale_hyperparameters[J, 1])
        log_prior_pdf[index] = log_prior_pdf
        index += 1
    if indices[J + 1]:
        # if self.approximator.kernel._general and self.approximator.kernel._ARD:
        #     # In this case, then there is a scale parameter, the first
        #     # cutpoint, the interval parameters,
        #     # and lengthscales parameter for each dimension and class
        #     raise ValueError("TODO")
        # else:
        #     # In this case, then there is a scale parameter, the first
        #     # cutpoint, the interval parameters,
        #     # and a single, shared lengthscale parameter
        varphi = np.exp(phi[index])
        ## Normal distribution in the log domain
        log_prior_pdf = norm.logpdf(
            phi[index],
            loc=varphi_hyperparameters[0],
            scale=varphi_hyperparameters[1])
        log_prior_phi[index] = log_prior_pdf
        index += 1
    return cutpoints, varphi, scale, noise_variance, log_prior_phi

