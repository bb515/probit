"""
Ordinal regression concrete examples. Exact inference: Gibbs sampling.

Multiclass oredered probit regression
3 bin example from Cowles 1996 empirical study
showing convergence of the orginal probit with the Gibbs sampler.
Gibbs vs Metropolis within Gibbs convergence for a 3 bin example.
Except we take the mean (intercept) to be fixed, and don't fix the first
cutpoint.
"""
# Make sure to limit CPU usage
import os
from re import A
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = "6"
from numba import set_num_threads
set_num_threads(6)

import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from scipy.stats import multivariate_normal
from probit.approximators import EPOrdinalGP, LaplaceOrdinalGP, VBOrdinalGP
from probit.samplers import (
    GibbsOrdinalGP, EllipticalSliceOrdinalGP,
    SufficientAugmentation, AncilliaryAugmentation, PseudoMarginal)
from probit.plot import outer_loops, grid_synthetic
from probit.Gibbs import plot
from probit.kernels import SEIso
from probit.proposals import proposal_initiate
import pathlib
from probit.data.utilities import datasets, load_data, load_data_synthetic
import time
import matplotlib.pyplot as plt


now = time.ctime()
write_path = pathlib.Path()


def main():
    """
    Plot of the PM as a function of the lengthscale \varphi;
    black solid lines represent the average over 500 repetitions
    and dashed lines represent 2.5th and 97.5th quantiles for
    N_imp = 1 and N_imp = 64. The solid red line is the prior
    density.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="quantile or decile")
    parser.add_argument(
        "method", help="SA or AA")  # TODO: Surrogate
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = args.bins
    method = args.method
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets["benchmark"]:
        (X_trains, t_trains,
        X_tests, t_tests,
        X_true, y_tests,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, Kernel) = load_data(
            dataset, bins)
        burn_steps = 2000
        steps = 1000
        m_0 = np.zeros(len(X_tests))
        y_0 = m_0.copy()
        # outer_loops(
        #     test, GibsOrdinalGP, Kernel, X_trains, t_trains, X_tests,
        #     t_tests, burn_steps, steps,
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        # Initiate kernel
        kernel = Kernel(varphi=varphi_0, scale=scale_0)
        # Initiate classifier
        sampler = GibbsOrdinalGP(
            gamma_0, noise_variance_0, kernel, X_trains[2], t_trains[2], J)
        plot(sampler, m_0, y_0, gamma_0, burn_steps, steps, J, D)
    elif dataset in datasets["synthetic"]:
        (X, t,
        X_true, y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        # Set varphi hyperparameters
        varphi_hyperparameters = np.array([3.4, 2.0])  # [loc, scale] of a normal on np.exp(varphi)
        # Initiate kernel
        kernel = Kernel(varphi=varphi_0, scale=scale_0, varphi_hyperparameters=varphi_hyperparameters)
        # Initiate classifier
        # TODO: temporary, since we know the true latent variables here
        burn_steps = 500
        steps = 1000
        m_0 = y_true.flatten()
        y_0 = y_true.flatten()
        # sampler = GibbsOrdinalGP(gamma_0, noise_variance_0, kernel, X, t, J)
        noise_std_hyperparameters = None
        gamma_hyperparameters = None
        sampler = EllipticalSliceOrdinalGP(
            gamma_0, noise_variance_0,
            noise_std_hyperparameters,
            gamma_hyperparameters, kernel, X, t, J)
        nu_true = sampler.cov @ y_true.flatten()
        m_true = sampler.K @ nu_true
        # plt.scatter(sampler.X_train, m_true)
        # plt.show()
        M = 100
        varphis = np.logspace(-2.0, 2.0, M+1)
        varphis_step = varphis[1:] - varphis[:-1]
        varphis = varphis[:-1]
        p_theta_giv_f = np.empty(M)
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
        domain = ((-2, 2), None)
        res = (10, None)
        theta = sampler.get_theta(indices)
        proposal_cov = 0.05
        proposal_L_cov = proposal_initiate(theta, indices, proposal_cov)

        # Pseudo Marginal approach - VB
        approximator = VBOrdinalGP(
            gamma_0, noise_variance_0,
            kernel, X, t, J)
        hyper_sampler = PseudoMarginal(approximator)
        log_p_pseudo_marginalss = []
        for i, varphi in enumerate(varphis):
            # Need to update sampler hyperparameters
            approximator.hyperparameters_update(varphi=varphi)
            theta=sampler.get_theta(indices)
            log_p_pseudo_marginals = hyper_sampler.tmp_compute_marginal(
                    theta, indices, reparameterised=True)
            log_p_pseudo_marginalss.append(log_p_pseudo_marginals)
        log_p_pseudo_marginalss = np.array(log_p_pseudo_marginalss)
        print("here", log_p_pseudo_marginalss)
        print(np.shape(log_p_pseudo_marginalss))
        log_p_pseudo_marginals_ms = np.mean(log_p_pseudo_marginalss, axis=1)
        log_p_pseudo_marginals_std = np.std(log_p_pseudo_marginalss, axis=1)
        print("here", log_p_pseudo_marginals_ms)
        plt.plot(varphis, log_p_pseudo_marginals_ms, 'k')
        plt.plot(varphis, log_p_pseudo_marginals_ms + log_p_pseudo_marginals_std, '--b')
        plt.plot(varphis, log_p_pseudo_marginals_ms - log_p_pseudo_marginals_std, '--b')
        plt.savefig("tmp0.png")
        plt.show()

        max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginals_ms)
        log_sum_exp = max_log_p_pseudo_marginals + np.log(np.sum(np.exp(log_p_pseudo_marginals_ms - max_log_p_pseudo_marginals)))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginals_ms - log_sum_exp - np.log(varphis_step))

        plt.plot(varphis, p_pseudo_marginals)
        # plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        # plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r')
        plt.savefig("tmp1.png") 
        plt.show()

        max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginalss, axis=0)
        print(np.shape(max_log_p_pseudo_marginals))
        log_sum_exp = np.tile(max_log_p_pseudo_marginals, (M, 1)) + np.tile(np.log(np.sum(np.exp(log_p_pseudo_marginalss - max_log_p_pseudo_marginals), axis=0)), (M, 1))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginalss - log_sum_exp - np.log(varphis_step).reshape(-1, 1))

        p_pseudo_marginals_mean = np.mean(p_pseudo_marginals, axis=1)
        p_pseudo_marginals_std = np.std(p_pseudo_marginals, axis=1)

        plt.plot(varphis, p_pseudo_marginals_mean)
        plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r')
        plt.savefig("tmp2.png")
        plt.show()
        assert 0



        # Pseudo Marginal approach - Laplace
        approximator = LaplaceOrdinalGP(
            gamma_0, noise_variance_0,
            kernel, X, t, J)
        hyper_sampler = PseudoMarginal(approximator)
        log_p_pseudo_marginalss = []
        for i, varphi in enumerate(varphis):
            # Need to update sampler hyperparameters
            approximator.hyperparameters_update(varphi=varphi)
            theta=sampler.get_theta(indices)
            log_p_pseudo_marginals = hyper_sampler.tmp_compute_marginal(
                    theta, indices, reparameterised=True)
            log_p_pseudo_marginalss.append(log_p_pseudo_marginals)
        log_p_pseudo_marginalss = np.array(log_p_pseudo_marginalss)
        print("here", log_p_pseudo_marginalss)
        print(np.shape(log_p_pseudo_marginalss))
        log_p_pseudo_marginals_ms = np.mean(log_p_pseudo_marginalss, axis=1)
        log_p_pseudo_marginals_std = np.std(log_p_pseudo_marginalss, axis=1)
        print("here", log_p_pseudo_marginals_ms)
        plt.plot(varphis, log_p_pseudo_marginals_ms, 'k')
        plt.plot(varphis, log_p_pseudo_marginals_ms + log_p_pseudo_marginals_std, '--b')
        plt.plot(varphis, log_p_pseudo_marginals_ms - log_p_pseudo_marginals_std, '--b')
        plt.savefig("tmp0.png")
        plt.show()

        max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginals_ms)
        log_sum_exp = max_log_p_pseudo_marginals + np.log(np.sum(np.exp(log_p_pseudo_marginals_ms - max_log_p_pseudo_marginals)))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginals_ms - log_sum_exp - np.log(varphis_step))

        plt.plot(varphis, p_pseudo_marginals)
        # plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        # plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r')
        plt.savefig("tmp1.png") 
        plt.show()

        max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginalss, axis=0)
        print(np.shape(max_log_p_pseudo_marginals))
        log_sum_exp = np.tile(max_log_p_pseudo_marginals, (M, 1)) + np.tile(np.log(np.sum(np.exp(log_p_pseudo_marginalss - max_log_p_pseudo_marginals), axis=0)), (M, 1))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginalss - log_sum_exp - np.log(varphis_step).reshape(-1, 1))

        p_pseudo_marginals_mean = np.mean(p_pseudo_marginals, axis=1)
        p_pseudo_marginals_std = np.std(p_pseudo_marginals, axis=1)

        plt.plot(varphis, p_pseudo_marginals_mean)
        plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r')
        plt.savefig("tmp2.png")
        plt.show()

        assert 0
        # Pseudo Marginal approach - EP
        approximator = EPOrdinalGP(
            gamma_0, noise_variance_0,
            kernel, X, t, J)
        hyper_sampler = PseudoMarginal(approximator)
        log_p_pseudo_marginalss = []
        for i, varphi in enumerate(varphis):
            # Need to update sampler hyperparameters
            approximator.hyperparameters_update(varphi=varphi)
            theta=sampler.get_theta(indices)
            log_p_pseudo_marginals = hyper_sampler.tmp_compute_marginal(
                    m_true, theta, indices, proposal_L_cov, reparameterised=True)
            log_p_pseudo_marginalss.append(log_p_pseudo_marginals)
        log_p_pseudo_marginalss = np.array(log_p_pseudo_marginalss)
        print(np.shape(log_p_pseudo_marginalss))
        log_p_pseudo_marginals_ms = np.mean(log_p_pseudo_marginalss, axis=1)
        log_p_pseudo_marginals_std = np.std(log_p_pseudo_marginalss, axis=1)
        plt.plot(varphis, log_p_pseudo_marginals_ms, 'k')
        plt.plot(varphis, log_p_pseudo_marginals_ms + log_p_pseudo_marginals_std, '--b')
        plt.plot(varphis, log_p_pseudo_marginals_ms - log_p_pseudo_marginals_std, '--b') 
        plt.show()

        max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginals_ms)
        log_sum_exp = max_log_p_pseudo_marginals + np.log(np.sum(np.exp(log_p_pseudo_marginals_ms - max_log_p_pseudo_marginals)))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginals_ms - log_sum_exp - np.log(varphis_step))

        plt.plot(varphis, p_pseudo_marginals)
        # plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        # plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r') 
        plt.show()

        max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginalss, axis=0)
        print(np.shape(max_log_p_pseudo_marginals))
        log_sum_exp = np.tile(max_log_p_pseudo_marginals, (M, 1)) + np.tile(np.log(np.sum(np.exp(log_p_pseudo_marginalss - max_log_p_pseudo_marginals), axis=0)), (M, 1))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginalss - log_sum_exp - np.log(varphis_step).reshape(-1, 1))

        p_pseudo_marginals_mean = np.mean(p_pseudo_marginals, axis=1)
        p_pseudo_marginals_std = np.std(p_pseudo_marginals, axis=1)

        plt.plot(varphis, p_pseudo_marginals_mean)
        plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r') 
        plt.show()

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
