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
from probit.approximators import EPGP, LaplaceGP, VBGP, PEPGP
from probit.sparse import SparseLaplaceGP, SparseVBGP
from probit.gpflow import VGP, SVGP
import gpflow
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


def get_approximator(
        approximation, Kernel, lengthscale_0, signal_variance_0,
        N_train, D):
    # Set varphi hyperparameters
    lengthscale_hyperparameters = np.array([1.0, np.sqrt(D)])  # [shape, rate]
    # Initiate kernel
    kernel = Kernel(
        varphi=lengthscale_0,
        variance=signal_variance_0,
        varphi_hyperparameters=lengthscale_hyperparameters)
    M = None
    if approximation == "EP":
        # steps is the number of swipes over the data until check convergence
        steps = 1
        Approximator = EPGP
    elif approximation == "PEP":
        # steps is the number of swipes over the data until check convergence
        steps = 1
        Approximator = PEPGP
    elif approximation == "VB":
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//10])
        Approximator = VBGP
    elif approximation == "LA":
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    elif approximation == "SLA":
        M = 30  # Number of inducing points
        steps = np.max([2, M//10])
        Approximator = SparseLaplaceGP
    elif approximation == "SVB":
        M = 30  # Number of inducing points
        steps = np.max([10, M])
        Approximator = SparseVBGP
    elif approximation == "V":
        steps = 1000
        Approximator = VGP
        # Initiate kernel
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=lengthscale_0,
            variance=signal_variance_0
        )
    elif approximation == "SV":
        M = 30  # Number of inducing points.
        steps = 10000
        Approximator = SVGP
        # Initiate kernel
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=lengthscale_0,
            variance=signal_variance_0
        )
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected EP, VB, LA, V, SVB, SLA or SV)".format(
                approximation))
    return Approximator, steps, M, kernel


def main():
    """>>> python figure2.py figure 2 3 EP --profile"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="3, 13 or 52")
    # parser.add_argument(
    #     "approximation", help="LA, VB, EP, PEP or V")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = int(args.bins)
    # approximation = args.approximation
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
        sys.stdout = open("{}.txt".format(now), "w")
    # Load data from file
    if dataset in datasets["paper"] or dataset in datasets["synthetic"]:
        if dataset in datasets["paper"]:
            (X, f, g, y,
                cutpoints_0, lengthscale_0, noise_variance_0, variance_0,
                J, D, colors, Kernel) = load_data_paper(dataset, plot=True)
        else:
            (X, y,
            X_true, g_true,
            cutpoints_0, lengthscale_0, noise_variance_0, variance_0,
            J, D, colors, Kernel) = load_data_synthetic(dataset, bins)

        trainables = np.ones(J + 2)
        # Fix noise_variance
        trainables[0] = 0
        # Fix scale
        trainables[J] = 0
        # Fix varphi
        #trainables[-1] = 0
        # Fix cutpoints
        trainables[1:J] = 0

        # (log) domain of grid
        domain = ((-1.5, 0.33), None)
        # resolution of grid
        res = (50, None)

        num_importance_samples = [64, 1]
        num_data = [200]  # TODO: for large values of N, I observe numerical instability. Why? Don't think it is due
        # To self.EPS or self.jitter. Possible is overflow error in a log sum exp?
        for approximation in ["PEP"]:
            for N in num_data:
                # print(y)
                # plt.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], marker='o', s=25,  c='b')
                # plt.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], marker='o', s=25,  c='r')
                # # plt.scatter(X[np.where(y==2), 0], X[np.where(y==2), 1], marker='o', s=25,  c='g')
                # plt.savefig("data1.png")
                # plt.close()
                # Xt = np.c_[y, X]
                # np.random.shuffle(Xt)
                # X = Xt[:N, 1:D+1]
                # y = Xt[:N, 0]
                # y = y.astype(int)
                X = X[:N, :]  # X, t have already been shuffled
                y = y[:N]
                # print(y)
                # plt.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], marker='o', s=25,  c='b')
                # plt.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], marker='o', s=25,  c='r')
                # # plt.scatter(X[np.where(y==2), 0], X[np.where(y==2), 1], marker='o', s=25,  c='g')
                # plt.savefig("data.png")
                # plt.close()

                Approximator, steps, M, kernel = get_approximator(
                    approximation, Kernel, lengthscale_0, variance_0,
                    N, D)
                if "S" in approximation:
                    # Initiate sparse classifier
                    approximator = Approximator(
                        M=M, cutpoints=cutpoints_0,
                        noise_variance=noise_variance_0,
                        kernel=kernel, J=J, data=(X, y),
                        varphi_hyperparameters = np.array([1.0, np.sqrt(D)]))  # [shape, rate])
                elif "PEP" in approximation:
                    alpha = 0.5
                    # Initate PEP classifier
                    approximator = Approximator(
                        cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                        alpha=alpha, gauss_hermite_points=20,
                        kernel=kernel, J=J, data=(X, y),
                        varphi_hyperparameters = np.array([1.0, np.sqrt(D)]))  # [shape, rate])
                else:
                    # Initiate classifier
                    approximator = Approximator(
                        cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                        kernel=kernel, J=J, data=(X, y),
                        varphi_hyperparameters = np.array([1.0, np.sqrt(D)]))  # [shape, rate])
                for i, Nimp in enumerate(num_importance_samples):
                    # Initiate hyper-parameter sampler
                    hyper_sampler = PseudoMarginal(approximator)
                    # plot figures
                    (thetas, p_pseudo_marginals_mean, p_pseudo_marginals_lo,
                            p_pseudo_marginals_hi, p_priors) = figure2(
                        hyper_sampler, approximator, domain, res, trainables,
                        num_importance_samples=Nimp, steps=steps,
                        reparameterised=False, show=True, write=True)
                    if i==0:
                        plt.plot(thetas, p_pseudo_marginals_mean)
                        plt.plot(thetas, p_pseudo_marginals_lo, '--b',
                            label="Nimp={}".format(Nimp))
                        plt.plot(thetas, p_pseudo_marginals_hi, '--b')
                        plt.plot(thetas, p_priors, 'r')
                        axes = plt.gca()
                        y_min, y_max = axes.get_ylim()
                        plt.xlabel("length-scale")
                        plt.ylabel("pseudo marginal")
                        plt.title("N = {}, {}".format(N, approximation))
                        # plt.savefig("test.png")
                        # plt.close()
                    else:
                        plt.plot(thetas, p_pseudo_marginals_lo, '--g',
                            label="Nimp={}".format(Nimp))
                        plt.plot(thetas, p_pseudo_marginals_hi, '--g')
                        y_min_new, y_max_new = axes.get_ylim()
                        y_min = np.min([y_min, y_min_new])
                        y_max = np.max([y_max, y_max_new])
                plt.ylim(0.0, 5.0)
                plt.vlines(lengthscale_0, 0.0, 5.0, colors='k')
                plt.legend()
                plt.show()
                plt.savefig(
                    write_path / "fig2_{}_N={}.png".format(approximation, N))
                plt.savefig(
                    write_path / "fig2_{}_N={}.pdf".format(approximation, N))
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
