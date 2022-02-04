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
#from numba import set_num_threads
#set_num_threads(6)
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from probit.approximators import EPOrdinalGP, LaplaceOrdinalGP, VBOrdinalGP
from probit.samplers import (
    EllipticalSliceOrdinalGP,
    SufficientAugmentation, AncilliaryAugmentation, PseudoMarginal)
from probit.plot import table1, draw_mixing
import pathlib
from probit.data.utilities import datasets, load_data_paper
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
        "approach", help="SA, AA or PM")
    parser.add_argument(
        "approximation", help="LA, EP or VB")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    approach = args.approach
    approximation = args.approximation
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets["synthetic"]:
        # Load data from file
        (X, Y, t,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_paper(dataset, plot=False)
        X = X[:50, :]  # X, t have already been shuffled
        t = t[:50]
        # Set varphi hyperparameters
        varphi_hyperparameters = np.array([1.0, np.sqrt(D)])  # [shape, scale] of a gamma on varphi
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
        burn_steps = 1  # 5000
        steps = 1000  # 10000
        m_0 = Y.flatten()
        # sampler = GibbsOrdinalGP(gamma_0, noise_variance_0, kernel, X, t, J)
        noise_std_hyperparameters = None
        gamma_hyperparameters = None
        # Define the proposal covariance
        proposal_cov = 4.5
        if approach in ["AA", "SA"]:
            sampler = EllipticalSliceOrdinalGP(
                gamma_0, noise_variance_0,
                noise_std_hyperparameters,
                gamma_hyperparameters, kernel, X, t, J)
            theta = sampler.get_theta(indices)
            # MPI parallel across chains
            if approach == "AA":  # Ancilliary Augmentation approach
                hyper_sampler = AncilliaryAugmentation(sampler)
            else: # approach == "SA":  # Sufficient Augmentation approach
                hyper_sampler = SufficientAugmentation(sampler)
            # Burn-in
            thetas, acceptance_rate = hyper_sampler.sample(
                m_0, indices, proposal_cov, burn_steps, verbose=True)
            theta_0 = thetas[-1]
            # Sample
            thetas, acceptance_rate = hyper_sampler.sample(
                m_0, indices, proposal_cov, steps, theta_0=theta_0, verbose=False)
            np.savez(
                write_path/'theta_N={}_D={}_J={}_{}_Nimp={}_ARD={}_{}_chain={}.npz'.format(
                sampler.N, sampler.D, approach, 0, False, 0),
                X=thetas,
                acceptance_rate=acceptance_rate)
        elif approach == "PM":  # Pseudo Marginal approach
            if approximation == "VB":
                approximator = VBOrdinalGP(  # VB approximation
                    gamma_0, noise_variance_0,
                    kernel, X, t, J)
            elif approximation == "LA":
                approximator = LaplaceOrdinalGP(  # Laplace MAP approximation
                    gamma_0, noise_variance_0,
                    kernel, X, t, J)
            elif approximation == "EP":
                approximator = EPOrdinalGP(  # EP approximation
                    gamma_0, noise_variance_0,
                    kernel, X, t, J)

            # Initiate hyper-parameter sampler
            hyper_sampler = PseudoMarginal(approximator)
            num_importance_samples = 64
            # Burn
            thetas, acceptance_rate = hyper_sampler.sample(
                indices, proposal_cov, burn_steps, num_importance_samples=num_importance_samples)
            theta_0 = thetas[-1]
            thetas, acceptance_rate = hyper_sampler.sample(
                indices, proposal_cov, steps, num_importance_samples=num_importance_samples,
                theta_0=theta_0)
            print("ACC=", acceptance_rate)
            draw_mixing(thetas, varphi_0, reparameterised=True, write_path=write_path)
            np.savez(
                write_path/'theta_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.npz'.format(
                approximator.N, approximator.D, approximator.J, num_importance_samples, False, approach, approximation, 0),
                X=thetas,
                acceptance_rate=acceptance_rate)
            print("finished")

        # plot figures
        table1(write_path, show=True, write=True)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
