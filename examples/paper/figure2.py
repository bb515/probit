"""
Ordinal regression, pseudo-marginal (PM) inference. EP, VB, MAP Laplace, V.

Fig. 2. Plot of the PM as a function of the length-scale, \ell; black solid
lines represent the average over 500 repetitions and dashed lines represent
2.5th and 97.5th quantiles for $N_{\text{imp}} = 1$ and $N_{\text{imp}} = 64$.
The solid red line is the prior density.
"""
import os
nthreads = "20"
os.environ["OMP_NUM_THREADS"] = nthreads # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = nthreads # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = nthreads # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = nthreads # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = nthreads
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
import matplotlib
import matplotlib.pyplot as plt


now = time.ctime()
write_path = pathlib.Path()
font = {'family' : 'sans-serif',
        'size'   : 22}
matplotlib.rc('font', **font)


def get_approximator(
        approximation, Kernel, lengthscale_0, signal_variance_0,
        N_train, D):
    # Set theta hyperparameters
    lengthscale_hyperparameters = np.array([1.0, np.sqrt(D)])  # [shape, rate]
    # Initiate kernel
    kernel = Kernel(
        theta=lengthscale_0,
        variance=signal_variance_0,
        theta_hyperparameters=lengthscale_hyperparameters)
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
            (X_, f, g, y_,
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
        # Fix theta
        # trainables[-1] = 0
        # Fix cutpoints
        trainables[1:J] = 0

        noise_std_hyperparameters = np.array([1.2, 1./0.2])
        theta_hyperparameters = np.array([1.0, np.sqrt(D)])

        # (log) domain of grid
        domain = ((-1.5, 0.33), None)
        # resolution of grid
        res = (500, None)

        num_importance_samples = [1, 16]
        num_data = [450]  # TODO: for large values of N, I observe numerical instability. Why? Don't think it is due
        # To self.EPS or self.jitter. Possible is overflow error in a log sum exp?
        approximations = ["EP", "LA"]

        colors = plt.cm.jet(np.linspace(0,1,14))
        fig, ax = plt.subplots(nrows=len(num_data), ncols=len(approximations), figsize=(10, 3))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.0)
        for j, approximation in enumerate(approximations):
            for N in num_data:
                X = X_[:N, :]  # X, t have already been shuffled
                y = y_[:N]
                Approximator, steps, M, kernel = get_approximator(
                    approximation, Kernel, lengthscale_0, variance_0,
                    N, D)
                if "S" in approximation:
                    # Initiate sparse classifier
                    approximator = Approximator(
                        M=M, cutpoints=cutpoints_0,
                        noise_variance=noise_variance_0,
                        kernel=kernel, J=J, data=(X, y),
                        noise_std_hyperparameters = noise_std_hyperparameters,
                        theta_hyperparameters = theta_hyperparameters)
                elif "PEP" in approximation:
                    alpha = 0.5
                    # Initate PEP classifier
                    approximator = Approximator(
                        cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                        alpha=alpha, gauss_hermite_points=20,
                        kernel=kernel, J=J, data=(X, y),
                        noise_std_hyperparameters = noise_std_hyperparameters,
                        )
                else:
                    # Initiate classifier
                    approximator = Approximator(
                        cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                        kernel=kernel, J=J, data=(X, y),
                        noise_std_hyperparameters = noise_std_hyperparameters,
                        )  # [shape, rate])
                phi_true = approximator.get_phi(trainables)
                theta_true = np.exp(phi_true)
                for i, Nimp in enumerate(num_importance_samples):
                    # Initiate hyper-parameter sampler
                    hyper_sampler = PseudoMarginal(approximator)
                    # plot figures
                    (thetas, thetas_step, p_pseudo_marginals_mean,
                            p_pseudo_marginals_lo,
                            p_pseudo_marginals_hi, p_priors) = figure2(
                        hyper_sampler, approximator, domain, res, trainables,
                        num_importance_samples=Nimp, steps=steps,
                        reparameterised=False, show=True, write=True)
                    # Check Riemann sum approximates 1
                    print("Riemann sum={}".format(
                        np.sum(thetas_step * p_pseudo_marginals_mean)))
                    if i==0:
                        ax[j].plot(thetas, p_pseudo_marginals_mean, color=colors[-2])
                        ax[j].plot(thetas, p_pseudo_marginals_lo, '--', color=colors[-2], alpha=0.4,
                            label="Nimp={}".format(Nimp))
                        ax[j].plot(thetas, p_pseudo_marginals_hi, '--', color=colors[-2], alpha=0.4)
                        ax[j].plot(thetas, p_priors, color=colors[0])
                        y_min, y_max = ax[j].get_ylim()
                        # plt.title("N = {}, {}".format(N, approximation))
                        # plt.savefig("test.png")
                        # plt.close()
                    else:
                        ax[j].plot(thetas, p_pseudo_marginals_lo, '--', color=colors[-2],
                            label="Nimp={}".format(Nimp))
                        ax[j].plot(thetas, p_pseudo_marginals_hi, '--', color=colors[-2])
                        y_min_new, y_max_new = ax[j].get_ylim()
                        y_min = np.min([y_min, y_min_new])
                        y_max = np.max([y_max, y_max_new])

                ax[j].set_xticks([])
                ax[j].set_xticks([], minor=True)
                if approximation == "EP":
                    ax[j].set_xlabel("{}".format(approximation))
                else:
                    ax[j].set_xlabel("Laplace")
                ax[j].set_yticks([])
                ax[j].set_yticks([], minor=True)
                ax[j].set_ylim(0.0, 4.1)
                ax[j].vlines(theta_true, 0.0, 4.1, colors='k')
                ax[j].legend()
                ax[j].grid()

        # fig.set_xlabel("theta")
        # fig.set_ylabel("pseudo marginal")
        plt.tight_layout()
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
