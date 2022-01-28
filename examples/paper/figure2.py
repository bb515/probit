"""
Figure 2

Ordinal regression concrete examples. Pseudomarginal inference: comparing the posterior approximation used:

EP vs VB vs MAP Laplace
"""
# Make sure to limit CPU usage if necessary
# import os
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
# os.environ["MKL_NUM_THREADS"] = "6"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "6"
# os.environ["NUMBA_NUM_THREADS"] = "6"
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from probit.approximators import EPOrdinalGP, LaplaceOrdinalGP, VBOrdinalGP
from probit.samplers import PseudoMarginal
from probit.plot import figure2
import pathlib
from probit.data.utilities import datasets, load_data_paper, load_data_synthetic
import time
import sys
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
        "approximation", help="LA, VB or EP")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = "SEIso"
    bins = "tertile" 
    approximation = args.approximation
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
        sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets["synthetic"]:
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

        # (log) domain of grid
        domain = ((-2, 2.55), None)
        # resolution of grid
        res = (100, None)

        num_importance_samples = [64, 1]
        num_data = [135, 50]
        for N in num_data:
            X = X[:N, :]  # X, t have already been shuffled
            t = t[:N]
            for i, Nimp in enumerate(num_importance_samples):
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

                # plot figures
                (Phi_new, p_pseudo_marginals_mean, p_pseudo_marginals_lo, p_pseudo_marginals_hi, p_priors) = figure2(
                    hyper_sampler, approximator, domain, res, indices,
                    num_importance_samples=Nimp, reparameterised=False, show=True, write=True)
                if i==0:
                    plt.plot(Phi_new, p_pseudo_marginals_mean)
                    plt.plot(Phi_new, p_pseudo_marginals_lo, '--b', label="Nimp={}".format(Nimp))
                    plt.plot(Phi_new, p_pseudo_marginals_hi, '--b')
                    plt.plot(Phi_new, p_priors, 'r')
                    axes = plt.gca()
                    y_min_0, y_max_0 = axes.get_ylim()
                    plt.xlabel("length-scale")
                    plt.ylabel("pseudo marginal")
                    plt.title("N = {}, {}".format(N, approximation))
                else:
                    plt.plot(Phi_new, p_pseudo_marginals_lo, '--g', label="Nimp={}".format(Nimp))
                    plt.plot(Phi_new, p_pseudo_marginals_hi, '--g')
                    y_min, y_max = axes.get_ylim()
                    y_min = np.min([y_min, y_min_0])
                    y_max = np.max([y_max, y_max_0])
            plt.vlines(varphi_0, 0.0, 0.010, colors='k')
            plt.legend()
            plt.ylim(y_min, y_max)
            plt.savefig(write_path / "fig2_{}_Nimp={}_N={}.png".format(approximation, Nimp, N))
            plt.show()
            plt.close()

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
        sys.stdout.close()


if __name__ == "__main__":
    main()
