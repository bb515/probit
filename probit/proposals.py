"""Proposals for hyperparameters."""
import numpy as np
from scipy.linalg import cho_solve, cho_factor, solve_triangular


def proposal_initiate(theta, indices, proposal_cov):
    """TODO: check this code"""
    if np.shape(proposal_cov) == (len(theta),):
        # Independent elliptical Gaussian proposals with standard deviations equal to proposal_L_cov
        cov = np.diagonal(proposal_cov[np.ix_(indices)])
        L_cov = np.sqrt(proposal_cov)
    elif np.shape(proposal_cov) == ():
        # Independent spherical Gaussian proposals with standard deviation equal to proposal_L_cov
        L_cov = np.sqrt(proposal_cov)
    elif np.shape(proposal_cov) == (len(theta), len(theta)):
        # Multivariate Gaussian proposals with cholesky factor equal to proposal_L_cov
        # The proposal distribution is a multivariate Gaussian
        # Take the sub-matrix marginal Gaussian for the indices
        mask = np.outer(indices, indices)
        mask = np.array(mask, dtype=bool)
        cov = proposal_cov[np.ix_(mask)]
        (L_cov, _) = cho_factor(cov)
    else:
        raise ValueError("Unsupported dimensions of proposal_L_cov, got {},"
            " expected square matrix or 1D vector or scalar".format(np.shape(proposal_cov)))
    return L_cov


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
        # cutpoints_1 is sampled from the domain of cutpoints_1, so jacobian is unity
        log_jacobian_theta[index] = 0.0
        index += 1
    for j in range(2, J):
        if indices[j]:
            # cutpoints_j is sampled from the domain of log(cutpoints_j - cutpoints_j-1) and so the jacobian is
            log_jacobian_theta[index] = -theta[index] # -ve since jacobian is 1/(cutpoints_j - cutpoints_j-1)
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