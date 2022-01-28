"""
Ordinal regression concrete examples. Comparing different samplers.
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
    if dataset in datasets["synthetic"]:

        # Sampler
        (X, t,
            X_true, y_true,
            gamma_0, varphi_0, noise_variance_0, scale_0,
            J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        # Set varphi hyperparameters
        #varphi_hyperparameters = np.array([0.0, 200.0])  # [loc, scale] of an exponential on varphi
        #varphi_hyperparameters = np.array([1.0, 1./2000.0])  # [shape, rate] of an gamma on varphi
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


        # Load data from file
        # (X, t,
        # gamma_0, varphi_0, noise_variance_0, scale_0,
        # J, D, colors, Kernel) = load_data_paper(dataset)
        (X, t,
        X_true, y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)

        # Set varphi hyperparameters
        varphi_hyperparameters = np.array([1.0, 1./np.sqrt(D)])  # [shape, rate] of an gamma on varphi

        # Initiate kernel
        kernel = Kernel(varphi=varphi_0, scale=scale_0, varphi_hyperparameters=varphi_hyperparameters)

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
        res = (100, None)

        # Sampler
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

        if approach == "AA":  # Ancilliary Augmentation approach
            hyper_sampler = AncilliaryAugmentation(sampler)
            log_p_theta_giv_y_nus = []
            for i, varphi in enumerate(varphis):
                print(i)
                # Need to update sampler hyperparameters
                sampler.hyperparameters_update(varphi=varphi)
                theta=sampler.get_theta(indices)
                log_p_theta_giv_y_nu = hyper_sampler.tmp_compute_marginal(
                    m_true, theta, indices, proposal_L_cov, reparameterised=True)
                log_p_theta_giv_y_nus.append(log_p_theta_giv_y_nu)
            plt.plot(log_p_theta_giv_y_nus)
            plt.show()
            max_log_p_theta_giv_y_nus = np.max(log_p_theta_giv_y_nus)
            log_sum_exp = max_log_p_theta_giv_y_nus + np.log(np.sum(np.exp(log_p_theta_giv_y_nus - max_log_p_theta_giv_y_nus)))
            p_theta_giv_y_nus = np.exp(log_p_theta_giv_y_nus - log_sum_exp - np.log(varphis_step))
            plt.plot(varphis, p_theta_giv_y_nus)
            plt.show()
        elif approach == "SA":  # Sufficient Augmentation approach
            hyper_sampler = SufficientAugmentation(sampler)
            log_p_theta_giv_ms = []
            for i, varphi in enumerate(varphis):
                print(i)
                # Need to update sampler hyperparameters
                sampler.hyperparameters_update(varphi=varphi)
                theta=sampler.get_theta(indices)
                log_p_theta_giv_ms.append(hyper_sampler.tmp_compute_marginal(
                        m_true, theta, indices, proposal_L_cov, reparameterised=True))
            plt.plot(log_p_theta_giv_ms, 'k')
            plt.show()
            max_log_p_theta_giv_ms = np.max(log_p_theta_giv_ms)
            log_sum_exp = max_log_p_theta_giv_ms + np.log(np.sum(np.exp(log_p_theta_giv_ms - max_log_p_theta_giv_ms)))
            p_theta_giv_ms = np.exp(log_p_theta_giv_ms - log_sum_exp - np.log(varphis_step))
            plt.plot(varphis, p_theta_giv_ms)
            plt.show()
        elif approach == "PM":  # Pseudo Marginal approach


            if approximation == "VB":
                approximator = VBOrdinalGP(  # VB approximation
                    gamma_0, noise_variance_0,
                    kernel, X, t, J)
            elif approximation == "Laplace":
                approximator = LaplaceOrdinalGP(  # Laplace MAP approximation
                    gamma_0, noise_variance_0,
                    kernel, X, t, J)
            elif approximation == "EP":
                approximator = EPOrdinalGP(  # EP approximation
                    gamma_0, noise_variance_0,
                    kernel, X, t, J)

        # Initiate hyper-parameter sampler
        hyper_sampler = PseudoMarginal(approximator)

        # plot figures
        figure2(hyper_sampler, approximator, domain, res, indices, reparameterised=False, show=True, write=True)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
