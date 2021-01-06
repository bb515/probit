"""Test file for the gibbs sampler."""
import numpy as np
from gibbs_multinomial import set_gaussian_kernel

def test_set_gaussian_kernel():
    x1 = np.array([1, 0])
    x2 = np.array([0, 1])
    x3 = np.array([0.9, 0.1])
    x4 = np.array([0.1, 0.9])
    # D = 2 and N = 2
    X = np.array([x1, x2, x3, x4])
    # K = 3
    varphi1 = np.array([0.4, 0.4])
    varphi2 = np.array([0.4, 0.5])
    varphi3 = np.array([0.4, 0.6])
    varphi = np.array([varphi1, varphi2, varphi3])
    Cs = set_gaussian_kernel(X, varphi)
    print(Cs[0])
    print(Cs[1])
    print(Cs[2])
    # (3, 2, 2) since N=2 and K=3
    shape = np.shape(Cs)
    assert(shape == (3, 4, 4))


def test_adding():
   """Tests the predictive posterior calculations."""
   samples = []
   K = 4
   m = norm.rvs(0, 1, K)
   I = np.eye(K)
   Lambda = np.tile(M, (K, 1))
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
   log_random_variables = np.log(random_variables)
   # Make the diagonals 0 so that they don't contribute to the sum
   log_random_variables_no_diag = np.fill_diagonal(log_random_variables, 0)
   print('log_random_variables', log_random_variables_no_diag)
   # Sum across columns TODO: check that this is correct
   log_sum = np.sum(log_random_variables_no_diag, axis=0)
   print('summed', log_sum)

   # Exponentiate the log_sum and
   sample = np.exp(log_sum)
   samples.append(sample)

    # u = norm.rvs(0, 1, (1, K))
    # print(u)
    # # u = norm.rvs(0, 1, 1)
    # U = np.tile(u, (K, 1))
    # print(U)
    #
    # differences = U - U.T
    # print(differences)


test_adding()

# u = norm.rvs(0, 1, (K, 1))
# print(u)
# U = np.tile(u, (K,))
# print(U)
#



test_set_gaussian_kernel()