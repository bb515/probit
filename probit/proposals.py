"""Proposals for hyperparameters."""
import numpy as np


def proposal_reparameterised(theta, indices, L_cov):
    """independence assumption so take the product of prior pdfs."""
    # No need to update parameters as this is done in the prior
    z = np.random.normal(0, 1, len(theta))
    delta = np.dot(L_cov, z)
    theta = theta + delta
    # Since the variables have been reparametrised to be sampled over the real domain, then there is
    # no change of variables and no jacobian
    log_jacobian_theta = 0.0
    return theta, log_jacobian_theta


def proposal(theta, indices, L_cov, J):
    """independence assumption so take the product of prior pdfs."""
    # No need to update parameters as this is done in the prior
    z = np.random.normal(0, 1, len(theta))
    delta = np.dot(L_cov, z)
    theta = theta + delta
    index = 0
    log_jacobian_theta = np.zeros(len(theta))
    # Calculate the jacobian from the theorem of transformation of continuous random variables
    if indices[0]:
        # noise_std is sampled from the domain of log(noise_std) and so the jacobian is
        log_jacobian_theta[index] = -theta[index]  # -ve since jacobian is 1/\sigma
        index += 1
    if indices[1]:
        # gamma_1 is sampled from the domain of gamma_1, so jacobian is unity
        log_jacobian_theta[index] = 0.0
        index += 1
    for j in range(2, J):
        if indices[j]:
            # gamma_j is sampled from the domain of log(gamma_j - gamma_j-1) and so the jacobian is
            log_jacobian_theta[index] = -theta[index] # -ve since jacobian is 1/(gamma_j - gamma_j-1)
            index += 1
    if indices[J]:
        # scale is sampled from the domain of log(scale) and so the jacobian is
        log_jacobian_theta[index] = -theta[index]
        index += 1
    if indices[J + 1]:
        # if self.approximator.kernel._general and self.approximator.kernel._ARD:
        #     # In this case, then there is a scale parameter, the first
        #     # cutpoint, the interval parameters,
        #     # and lengthscales parameter for each dimension and class
        #     raise ValueError("TODO")
        # else:
        # In this case, then there is a scale parameter, the first
        # cutpoint, the interval parameters,
        # and a single, shared lengthscale parameter
        # varphi is sampled from the domain of log(varphi) and so the jacobian is
        log_jacobian_theta[index] = -theta[index]
        index += 1
    return theta, log_jacobian_theta