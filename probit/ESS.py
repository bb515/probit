"""
TODO: need to refactor this or delete 22/02/22

Elliptical slice sampler."""
# Make sure to limit CPU usage
import os
from lab.linear_algebra import triangular_solve
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from probit.kernels import LabEQ
from probit.approximators import VBOrdinalGP
from probit.approximators import EPOrdinalGP
from probit.samplers import PseudoMarginalOrdinalGP
from probit.plot import outer_loops, grid_synthetic
from probit.VB import test
import pathlib
from probit.data.utilities import datasets, load_data_synthetic
import matplotlib.pyplot as plt
import time
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from scipy.stats import gamma
import warnings


# def test(Kernel, J, X_true, y_true, varphi_0, noise_variance_0, scale_0):
#     N = len(X_true)
#     # Section 3 - MCMC sampling from p(f, \theta | y)
#     # Initiate kernel
#     kernel = Kernel(varphi=varphi_0, scale=scale_0) 
#     # Import kernel
#     K0 = kernel.kernel_matrix(X_true, X_true)
#     jitter = 1e-6
#     K = K0 + jitter * np.eye(N)
#     # Calculate GP posterior
#     N_test = 1000
#     X_test = np.linspace(-0.5, 1.5, N_test)
#     print(np.shape(y_true))
#     y_true = y_true.flatten()
#     (L_cov, lower) = cho_factor(noise_variance_0 * np.eye(N) + K0)
#     # Unfortunately, it is necessary to take this cho_factor,
#     # only for log_det_K
#     (L_K, lower) = cho_factor(K + jitter * np.eye(N))
#     # log_det_K = 2 * np.sum(np.log(np.diag(L_K)))
#     # log_det_cov = -2 * np.sum(np.log(np.diag(L_cov)))
#     # L_covT_inv = solve_triangular(
#     #     L_cov.T, np.eye(N), lower=True)
#     #cov = np.linalg.inv(noise_variance_0 * np.eye(N) + K0)
#     #cov = solve_triangular(L_cov, L_covT_inv, lower=False)
#     C_news = kernel.kernel_matrix(X_true, X_test)  # (N, N_test)
#     Cov_news = kernel.kernel_matrix(X_test, X_test)  # (N_test, N_test)
#     c_news = np.diag(Cov_news)  # N_test
#     intermediate_vectors = solve_triangular(L_cov.T, C_news, lower=True)
#     intermediate_vectors = solve_triangular(L_cov, intermediate_vectors, lower=False)
#     intermediate_scalars = np.sum(np.multiply(C_news, intermediate_vectors), axis=0)  # (N_test, )
#     K_post = Cov_news - C_news.T @ intermediate_vectors
#     m_post = np.einsum('ij, i -> j', intermediate_vectors, y_true)  # (N, N_test) (N, ) = (N_test,)
#     var_post = c_news - intermediate_scalars
#     std_post = np.sqrt(var_post)
#     plt.plot(X_test, m_post, color='g', label="continuous GP")
#     N_test = N
#     X_test = X_true
#     c_news = np.diag(K)  # N_test
#     intermediate_vectors = solve_triangular(L_cov.T, K, lower=True)
#     intermediate_vectors = solve_triangular(L_cov, intermediate_vectors, lower=False)
#     #intermediate_vectors = cov @ C_news  # (N, N_test)
#     intermediate_scalars = np.sum(np.multiply(K, intermediate_vectors), axis=0)  # (N_test, )
#     K_post = K - K @ intermediate_vectors
#     m_post = np.einsum('ij, i -> j', intermediate_vectors, y_true)  # (N, N_test) (N, ) = (N_test,)
#     # Draw sample from GP posterior
#     z1 = np.random.normal(0, 1, N_test)
#     z2 = np.random.normal(0, 1, N_test)
#     #K_post = Cov_news - np.einsum('ki, kj -> ji', intermediate_vectors, C_news)  # (N, N_test) (N, N_test) = (N_test, N_test)
#     # For some reason cho_factor doesn't seem to work properly here - could investigate
#     L_post = np.linalg.cholesky(K_post + jitter * np.eye(N_test))
#     f_samp1 = m_post + np.dot(L_post, z1)
#     f_samp2 = m_post + np.dot(L_post, z2)
#     log_likelihood2 = get_log_likelihood(f_samp2, classifier)
#     # Now draw samples from the GP posterior by performaing elliptical slice sampling
#     f_samp3, log_likelihood3 = ELLSS_transition_operator(L_K, N, y_true, f_samp2, log_likelihood2)
#     f_samp4, log_likelihood4 = ELLSS_transition_operator(L_K, N, y_true, f_samp3, log_likelihood3)
#     ##plt.scatter(X_true, f_samp1, color='gray', label="sample1", s=3)
#     plt.scatter(X_true, f_samp2, color='b', label="sample2", s=3)
#     plt.scatter(X_true, f_samp3, color='m', label="sample3", s=3)
#     #plt.scatter(X_true, f_samp4, color='y', label="sample4", s=3)
#     # Not how or if need true function
#     # plt.plot(x, y, '--k', label="true function")
#     plt.legend()
#     plt.scatter(X_true, y_true, color='g', s=4)
#     plt.xlim((-0.1, 1.1))
#     plt.ylim((-2.1, 0.6))
#     plt.show()
#     assert 0


def main():
    """Conduct Eliptical slice sampling."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="quantile or decile")
    # parser.add_argument(
    #     "method", help="")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = args.bins
    # method = args.method
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    if dataset in datasets["synthetic"]:
        n_importance_samples = 10
        (X, t,
        X_true, Y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        print(noise_variance_0)
        print(varphi_0)
        noise_variance_0 = 1.0
        psi_0 = np.array([5.0, 1.0])  # shape and rate for gamma prior
        #psi_0 = np.array([3.0, 6.0])  # mean and variance for diffuse prior on log_varphi
        # Initiate kernel
        kernel = Kernel(varphi=varphi_0, psi=psi_0, scale=scale_0)
        indices = np.ones(J + 2)
        # Fix noise_variance
        indices[0] = 0
        # Fix scale
        indices[J] = 0
        # Fix varphi
        #indices[-1] = 0
        # Fix gamma
        indices[1:J] = 0
        # Just varphi
        # domain = ((-4, 4), None)
        # res = (100, None)
        # Initiate sampler
        gamma_hyperparameters = None
        noise_std_hyperparameters = None
        sampler = PseudoMarginalOrdinalGP(gamma_0, noise_variance_0, kernel, X, t, J)
        print(sampler.sample(
            indices, proposal_cov=0.0005, steps=100, first_step=1, num_importance_samples=10, reparameterised=False))

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

