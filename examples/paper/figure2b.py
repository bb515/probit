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
import matplotlib

font = {'family' : 'sans-serif',
        'size'   : 22}

# matplotlib.rcParams.update({'font.size': 22})

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
    """>>> python figure2.py figure2og 2 EP --profile"""
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
            (X_, f_, g_, y_,
                cutpoints_0, lengthscale_0, noise_variance_0, variance_0,
                J, D, colors, Kernel) = load_data_paper(dataset, plot=True)
        else:
            (X_, y_,
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
        res = (100, None)

        num_importance_samples = [16]
        num_data = [1, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # TODO: for large values of N, I observe numerical instability. Why? Don't think it is due
        #num_data = [1, 10, 20, 50]
        colors = plt.cm.jet(np.linspace(0,1,len(num_data) + 1))
        # To self.EPS or self.jitter. Possible is overflow error in a log sum exp?
        for approximation in ["EP"]:
            for i, N in enumerate(num_data):
                # print(y)
                # plt.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], marker='o', s=25,  c='b')
                # plt.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], marker='o', s=25,  c='r')
                # # plt.scatter(X[np.where(y==2), 0], X[np.where(y==2), 1], marker='o', s=25,  c='g')
                # plt.savefig("data1.png")
                # plt.close()
                X = X_[:N, :]  # X, t have already been shuffled
                y = y_[:N]
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
                for _, Nimp in enumerate(num_importance_samples):
                    # Initiate hyper-parameter sampler
                    hyper_sampler = PseudoMarginal(approximator)
                    # plot figures
                    (thetas, thetas_step, p_pseudo_marginals_mean, p_pseudo_marginals_lo,
                            p_pseudo_marginals_hi, p_priors) = figure2(
                        hyper_sampler, approximator, domain, res, trainables,
                        num_importance_samples=Nimp, steps=steps,
                        reparameterised=False, show=True, write=True)
                    print(p_pseudo_marginals_mean)
                if i==0:
                    fig = plt.figure()
                    fig.patch.set_facecolor('white')
                    fig.patch.set_alpha(0.0)
                    ax = fig.add_subplot(111)
                    ax.plot(thetas, p_priors, color=colors[0],  label="prior".format(N))
                    ax.plot(thetas, p_pseudo_marginals_mean, color=colors[i + 1], label="N={}".format(N))
                    ax.plot(thetas, p_pseudo_marginals_lo, '--', color=colors[i + 1], alpha=0.4)
                        # ,label="Nimp={}".format(Nimp))
                    ax.plot(thetas, p_pseudo_marginals_hi, '--', color=colors[i + 1], alpha=0.4)
                    y_min, y_max = ax.get_ylim()
                    ax.set_xlabel("length-scale", **font)
                    ax.set_ylabel("pseudo marginal", **font)
                    # plt.title("N = {}, {}".format(N, approximation))
                    # plt.savefig("test.png")
                    # plt.close()
                else:
                    ax.plot(thetas, p_pseudo_marginals_mean, color=colors[i + 1], label="N={}".format(N))
                    ax.plot(thetas, p_pseudo_marginals_lo, '--', color=colors[i + 1], alpha=0.4)
                        # ,label="Nimp={}".format(Nimp))
                    ax.plot(thetas, p_pseudo_marginals_hi, '--', color=colors[i + 1], alpha=0.4)
                    y_min_new, y_max_new = ax.get_ylim()
                    y_min = np.min([y_min, y_min_new])
                    y_max = np.max([y_max, y_max_new])

        ax.set_ylim(0.0, 4.1)
        ax.vlines(lengthscale_0, 0.0, 4.1, colors='k', label="true")
        ax.legend()
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.tick_params(axis='both', which='minor', labelsize=22)
        plt.tight_layout()
        ax.grid()
        fig.savefig(
            write_path / "fig2_{}.png".format(approximation, N),
            facecolor=fig.get_facecolor(), edgecolor='none')
        fig.savefig(
            write_path / "fig2_{}.pdf".format(approximation, N),
            facecolor=fig.get_facecolor(), edgecolor='none')
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
