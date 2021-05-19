"""Test file for the miscellaneous tests for the gibbs sampler."""
import numpy as np
from scipy.stats import norm
from ..utilities import (
   matrix_of_differences, sample_U, sample_varphis, function_u1)
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt


def test_adding():
   """Tests the predictive posterior calculations."""
   samples = []
   K = 4
   m = norm.rvs(0, 1, K)
   I = np.eye(K)
   Lambda = np.tile(m, (K, 1))
   Lambda_T = Lambda.T
   print('Lambda', Lambda)
   print('Lambda_T', Lambda_T)
   # Symmetric matrix of differences
   differences = Lambda - Lambda_T
   print('differences', differences)
   # Sample normal random variate
   u = norm.rvs(0, 1, K)
   # u = norm.rvs(0, 1, 1)
   U = np.tile(u, (K, 1))
   print('U', U)
   # U = np.multiply(u, np.ones((K, K)))
   random_variables = np.add(U, differences)
   print('rvs', random_variables)

   # log_random_variables = np.log(random_variables)
   # # Make the diagonals 0 so that they don't contribute to the sum
   # log_random_variables_no_diag = np.fill_diagonal(log_random_variables, 0)
   # print('log_random_variables', log_random_variables_no_diag)
   # # Sum across columns TODO: check that this is correct
   # log_sum = np.sum(log_random_variables_no_diag, axis=0)
   # print('summed', log_sum)
   #
   # # Exponentiate the log_sum and
   # sample = np.exp(log_sum)
   # samples.append(sample)

    # u = norm.rvs(0, 1, (1, K))
    # print(u)
    # # u = norm.rvs(0, 1, 1)
    # U = np.tile(u, (K, 1))
    # print(U)
    #
    # differences = U - U.T
    # print(differences)

### Tests for Variational Bayes relavent utilities functions. Some of these have been SS.###

def plot_vague_prior():
   """
   Scipy stats module takes two parameters,a and scale. From the form of the posterior over the psi (which comes
   from the fact that exponential is a conjugate prior to gamma, and the posterior takes the form of a gamma),
   we know that sigma as written in the paper is a, and tau as written in the paper is a rate parameter = 1./scale
   """
   # as written in the paper
   # sigma is known as k on wikipedia, and is a shape parameter
   sigmas = 10e-3 * np.arange(1, 4)
   # tau is known as beta on wikipedia, and is a rate parameter
   taus = 10e-3 * np.arange(1, 4)
   x = np.linspace(0, 300, 1001)
   for i in range(np.shape(sigmas)[0]):
      y = gamma.pdf(x, a=sigmas[i], scale=1./taus[i])
      plt.plot(x, y, label='sigma = {}, tau = {}'.format(sigmas[i], taus[i]))
   plt.title('The priors with the high sigma and tau are more concentrated around 0 - informative prior')
   plt.ylim((0, 3e-2))
   plt.xlim((0,2))
   plt.legend()
   plt.show()
   for i in range(np.shape(sigmas)[0]):
      y = gamma.pdf(x, a=sigmas[i], scale=1./taus[i])
      plt.plot(x, y, label='sigma = {}, tau = {}'.format(sigmas[i], taus[i]))
   plt.title('The priors with the lower sigma and tau have a much thicker tail- vague prior')
   plt.ylim((0, 3e-5))
   plt.xlim((100, 300))
   plt.legend()
   plt.show()


def test_sample_varphi():
   """Test sampling varphi from the prior."""
   # Draw hyperparameters from prior with hyper-hyperparameter
   M = 2
   K = 3
   # Initiate varphi posterior estimates as all ones
   varphi_tilde = 0.6*np.ones((K, M))
   # Uninformative priors
   sigma_k = np.zeros((K, M))
   tau_k = np.zeros((K, M))
   psi_tilde = (np.ones((K, M)) + sigma_k) / (tau_k + varphi_tilde)
   samples = []
   n_samples = 1000
   for i in range(n_samples):
      samples.append(sample_varphi(psi_tilde))
   samples = np.array(samples)
   varphi_tilde = np.sum(samples, axis=0) / n_samples
   # rtol = 0.1 since sample variance is Var(varphi) / root(n) = (1/ psi_tilde^2) / root(1000) = 0.0114, so std = 0.1
   assert np.allclose(varphi_tilde, 1./psi_tilde, rtol=0.1)


def test_samples_varphi():
   """Test sampling varphi from the prior with tensor version."""
   M = 2
   K = 3
   # Initiate varphi posterior estimates as all ones
   varphi_tilde = 0.6 * np.ones((K, M))
   # Uninformative priors
   sigma_k = np.zeros((K, M))
   tau_k = np.zeros((K, M))
   psi_tilde = (np.ones((K, M)) + sigma_k) / (tau_k + varphi_tilde)
   samples = []
   n_samples = 1000
   samples = samples_varphi(psi_tilde, n_samples)
   varphi_tilde = np.sum(samples, axis=0) / n_samples
   # rtol = 0.1 since sample variance is Var(varphi) / root(n) = (1/ psi_tilde^2) / root(1000) = 0.0114, so std = 0.1
   assert np.allclose(varphi_tilde, 1. / psi_tilde, rtol=0.1)


def test_matrix_of_differences():
   """Test the matrix of difference function produces expected (and not transposed) result."""
   m_n = np.array([-1, 0, 1])

   expected_MOD = np.array([
      [0, -1, -2],
      [1, 0, -1],
      [2, 1, 0]
   ])
   actual_MOD = matrix_of_differences(m_n)
   assert np.allclose(expected_MOD, actual_MOD)
   t_n = np.argmax(m_n)
   actual_vector_difference = actual_MOD[:, t_n]
   expected_vector_difference = np.array([-2, -1, 0])
   assert np.allclose(actual_vector_difference, expected_vector_difference)


def test_function_u1_sum_to_one():
   # TODO: this currently doesn't work
   K = 3
   m_n = np.array([-1, 0, 1])
   difference = matrix_of_differences(m_n)
   print(difference)
   U = sample_U(K)
   print(U)
   f1 = function_u1(difference, U)
   print(f1)
   assert np.allclose(np.sum(f1), 1.0)


def test_functions_sum_to_one():
   # TODO: this currently doesn't work
   K = 3
   m_n = np.array([-1, 0, 1])
   difference = matrix_of_differences(m_n)
   U = sample_U(K)
   t_n = np.argmax(m_n)
   f1 = function_u1_alt(difference, U, np.argmax(t_n))
   vector_difference = difference[:, t_n]
   f2 = function_u2(difference, vector_difference, U, t_n, K)
   f3 = function_u3(difference, vector_difference, U, t_n, K)
   assert np.allclose(np.sum(f1), 1.0)
   assert np.allclose(np.sum(f2), 1.0)
   assert np.allclose(np.sum(f3), 1.0)


# test_samples_varphi()

# u = norm.rvs(0, 1, (K, 1))
# print(u)
# U = np.tile(u, (K,))
# print(U)
#


