"""
Ordinal regression, pseudo-marginal (PM) inference. EP, VB, MAP Laplace, V.

Fig. 2. Plot of the PM as a function of the length-scale, \ell; black solid
lines represent the average over 500 repetitions and dashed lines represent
2.5th and 97.5th quantiles for $N_{\text{imp}} = 1$ and $N_{\text{imp}} = 64$.
The solid red line is the prior density.
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
from probit.approximators import EPGP, LaplaceGP, VBGP
from probit.sparse import SparseLaplaceGP, SparseVBGP
from probit.gpflow import VGP, SVGP
from probit.kernels_gpflow import SquaredExponential
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
    """>>> python figure2.py figure 2 3 EP --profile"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="3, 13 or 52")
    parser.add_argument(
        "approximation", help="LA, VB, EP or V")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = int(args.bins)
    approximation = args.approximation
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
        sys.stdout = open("{}.txt".format(now), "w")
    # Load data from file
    if dataset in datasets["paper"] or dataset in datasets["synthetic"]:
        if dataset in datasets["paper"]:
            (X, Y, t,
                cutpoints_0, varphi_0, noise_variance_0, scale_0,
                J, D, colors, Kernel) = load_data_paper(dataset, plot=True)
        else:
            (X, t,
            X_true, y_true,
            cutpoints_0, varphi_0, noise_variance_0, scale_0,
            J, D, colors, Kernel) = load_data_synthetic(dataset, bins)

        # Set varphi hyperparameters
        varphi_hyperparameters = np.array([1.0, np.sqrt(D)])  # [shape, rate] of an cutpoints on varphi

        # Initiate kernel
        kernel = Kernel(
            varphi=varphi_0,
            variance=scale_0, varphi_hyperparameters=varphi_hyperparameters)

        indices = np.ones(J + 2)
        # Fix noise_variance
        indices[0] = 0
        # Fix scale
        indices[J] = 0
        # Fix varphi
        #indices[-1] = 0
        # Fix cutpoints
        indices[1:J] = 0

        # (log) domain of grid
        domain = ((-1.5, 1.0), None)
        # resolution of grid
        res = (50, None)

        num_importance_samples = [64]
        num_data = [200]  # TODO: for large values of N, I observe numerical instability. Why? Don't think it is due
        # To self.EPS or self.jitter. Possible is overflow error in a log sum exp?
        for N in num_data:
            X = X[:N, :]  # X, t have already been shuffled
            t = t[:N]
            for i, Nimp in enumerate(num_importance_samples):
                if approximation == "VB":
                    steps = np.max([100, N//10])
                    Approximator = VBGP  # VB approximation
                elif approximation == "LA":
                    steps = np.max([2, N//1000])
                    Approximator = LaplaceGP  # Laplace MAP approximation
                elif approximation == "EP":
                    steps = np.max([10, N//100])  # for N=3000, steps is 300 - could be too large since per iteration is slow.
                    Approximator = EPGP  # EP approximation
                elif approximation == "V":
                    steps = np.max([100, N//10])
                    kernel = SquaredExponential(
                        lengthscales=1./np.sqrt(2 * varphi_0),
                        variance=scale_0,
                        varphi_hyperparameters=varphi_hyperparameters)
                    Approximator = VGP
                elif approximation == "SV":
                    steps = np.max([4000, N//10])
                    kernel = SquaredExponential(
                        lengthscales=1./np.sqrt(2 * varphi_0),
                        variance=scale_0,
                        varphi_hyperparameters=varphi_hyperparameters)
                    Approximator = SVGP
                elif approximation == "SLA":
                    steps = np.max([2, N//1000])
                    Approximator = SparseLaplaceGP
                elif approximation == "SVB":
                    Approximator = SparseVBGP
                    steps = np.max([2, N//1000])

                if "S" in approximation:
                    M = np.max(10, N//1000)
                    approximator = Approximator(
                        M,
                        cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                        kernel=kernel, J=J, data=(X, t))
                else:
                    approximator = Approximator(
                        cutpoints_0, noise_variance_0,
                        kernel, J, (X, t))

                # Initiate hyper-parameter sampler
                hyper_sampler = PseudoMarginal(approximator)

                # plot figures
                (Phi_new, p_pseudo_marginals_mean, p_pseudo_marginals_lo,
                        p_pseudo_marginals_hi, p_priors) = figure2(
                    hyper_sampler, approximator, domain, res, indices,
                    num_importance_samples=Nimp, steps=steps,
                    reparameterised=False, show=True, write=True)
                if i==0:
                    plt.plot(Phi_new, p_pseudo_marginals_mean)
                    plt.plot(Phi_new, p_pseudo_marginals_lo, '--b',
                        label="Nimp={}".format(Nimp))
                    plt.plot(Phi_new, p_pseudo_marginals_hi, '--b')
                    plt.plot(Phi_new, p_priors, 'r')
                    axes = plt.gca()
                    y_min_0, y_max_0 = axes.get_ylim()
                    plt.xlabel("length-scale")
                    plt.ylabel("pseudo marginal")
                    plt.title("N = {}, {}".format(N, approximation))
                    # plt.savefig("test.png")
                    # plt.close()
                else:
                    plt.plot(Phi_new, p_pseudo_marginals_lo, '--g',
                        label="Nimp={}".format(Nimp))
                    plt.plot(Phi_new, p_pseudo_marginals_hi, '--g')
                    y_min, y_max = axes.get_ylim()
                    y_min = np.min([y_min, y_min_0])
                    y_max = np.max([y_max, y_max_0])
                    plt.ylim(y_min, y_max)
            plt.vlines(varphi_0, 0.0, 0.010, colors='k')
            plt.legend()
            plt.show()
            plt.savefig(
                write_path / "fig2_{}_N={}.png".format(approximation, N))
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
