"""Priors for hyperparameters."""
import numpy as np
from scipy.stats import expon, norm
from scipy.stats import gamma as gamma_
from scipy.special import ndtr, log_ndtr
import warnings

def prior_reparameterised(
        theta, indices, J, psi, noise_std_hyperparameters,
        gamma_hyperparameters, scale_hyperparameters, gamma=None):
    """
    A reparametrisation such that all of the hyperparameters can be sampled from the real line,
    therefore there is no transformation (thus no jacobian) when sampling from the proposal distribution.
    Hyperparameter priors assumed to be independent assumption so take the product of prior pdfs.
    """
    # Do not update these hyperparameters by default
    noise_variance = None
    gamma = None
    scale = None
    varphi = None
    index = 0
    log_prior_theta = np.zeros(len(theta))
    if indices[0]:
        # Gamma prior is placed on the noise std - evaluate the prior pdf
        log_noise_std = theta[index]
        log_prior_pdf = norm.logpdf(
            log_noise_std,
            loc=noise_std_hyperparameters[0],
            scale=noise_std_hyperparameters[1])
        log_prior_theta[index] = log_prior_pdf
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
    if indices[1]:
        if gamma is None:
            raise ValueError("ValueError: gamma must be supplied if ")
            # Get gamma from classifier
            gamma = self.approximator.gamma
        gamma[1] = theta[index]
        log_prior_pdf = norm.logpdf(
            theta[index],
            loc=gamma_hyperparameters[1, 0],
            scale=gamma_hyperparameters[1, 1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    for j in range(2, J):
        if indices[j]:
            if gamma is None:
                # TODO: THERE IS AN ERROR HERE IF indices[1] and indices[2] is true,
                # TODO: due to overwrite!
                # Get gamma from classifier
                gamma = self.approximator.gamma
            gamma[j] = gamma[j-1] + np.exp(theta[index])
            log_prior_pdf = norm.pdf(
                theta[index],
                loc=gamma_hyperparameters[j, 0],
                scale=gamma_hyperparameters[j, 1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
    if indices[J]:
        scale_std = np.exp(theta[index])
        scale = scale_std**2
        log_prior_pdf = norm.logpdf(
            theta[index],
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
        varphi = np.exp(theta[index])
        log_prior_pdf = norm.logpdf(
            theta[index],
            loc=psi[0],
            scale=psi[1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    return gamma, varphi, scale, noise_variance, log_prior_theta

def prior(theta, indices, J, psi, noise_std_hyperparameters,
        gamma_hyperparameters, scale_hyperparameters, gamma=None):
    """
    A priors defined over their usual domains, and so a transformation of random variables is used
    for sampling from proposal distrubutions defined over continuous domains.
    Hyperparameter priors assumed to be independent assumption so take the product of prior pdfs.
    """
    # Do not update these hyperparameters by default
    noise_variance = None
    gamma = None
    scale = None
    varphi = None
    log_prior_theta = np.zeros(len(theta))
    index = 0
    if indices[0]:
        # Gamma prior is placed on the noise std - evaluate the prior pdf
        noise_std = np.exp(theta[index])
        log_prior_pdf = gamma_.logpdf(
            noise_std,
            loc=noise_std_hyperparameters[0],
            scale=noise_std_hyperparameters[1])
        log_prior_theta[index] = log_prior_pdf
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
    if indices[1]:
        if gamma is None:
            # Get gamma from classifier
            gamma = self.approximator.gamma
        gamma[1] = theta[index]
        log_prior_pdf = norm.logpdf(
            theta[index],
            loc=gamma_hyperparameters[1, 0],
            scale=gamma_hyperparameters[1, 1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    for j in range(2, J):
        if indices[j]:
            if gamma is None:
                # Get gamma from classifier
                gamma = self.approximator.gamma
            gamma[j] = gamma[j-1] + np.exp(theta[index])
            log_prior_pdf = norm.logpdf(
                theta[index],
                loc=gamma_hyperparameters[j, 0],
                scale=gamma_hyperparameters[j, 1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
    if indices[J]:
        scale_std = np.exp(theta[index])
        scale = scale_std**2
        log_prior_pdf = norm.logpdf(
                theta[index],
                loc=gamma_hyperparameters[J, 0],
                scale=gamma_hyperparameters[J, 1])
        log_prior_theta[index] = log_prior_pdf
        index += 1
    if indices[J + 1]:
        if self.approximator.kernel._general and self.approximator.kernel._ARD:
            # In this case, then there is a scale parameter, the first
            # cutpoint, the interval parameters,
            # and lengthscales parameter for each dimension and class
            raise ValueError("TODO")
        else:
            # In this case, then there is a scale parameter, the first
            # cutpoint, the interval parameters,
            # and a single, shared lengthscale parameter
            varphi = np.exp(theta[index])
            log_prior_pdf = gamma_.logpdf(
                varphi,
                a=psi[0],
                scale=1./ psi[1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
    return gamma, varphi, scale, noise_variance, log_prior_theta