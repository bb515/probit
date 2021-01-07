"""Test file for the miscellaneous tests for the gibbs sampler."""
import numpy as np
from scipy.stats import norm

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


# u = norm.rvs(0, 1, (K, 1))
# print(u)
# U = np.tile(u, (K,))
# print(U)
#


