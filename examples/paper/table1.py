"""
Ordinal regression concrete examples. Comparing different samplers.
"""
# Make sure to limit CPU usage
import os

from pandas import cut
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
from probit.approximators import PEPGP, EPGP, LaplaceGP, VBGP
from probit.gpflow import VGP
from probit.samplers import (
    EllipticalSliceGP,
    SufficientAugmentation, AncilliaryAugmentation, PseudoMarginal)
from probit.kernels_gpflow import SquaredExponential
from probit.plot import table1, draw_mixing, draw_histogram
import pathlib
from probit.data.utilities import datasets, load_data_paper
import time


now = time.ctime()
write_path = pathlib.Path()


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
    # elif approximation == "SLA":
    #     M = 30  # Number of inducing points
    #     steps = np.max([2, M//10])
    #     Approximator = SparseLaplaceGP
    # elif approximation == "SVB":
    #     M = 30  # Number of inducing points
    #     steps = np.max([10, M])
    #     Approximator = SparseVBGP
    # elif approximation == "V":
    #     steps = 1000
    #     Approximator = VGP
    #     # Initiate kernel
    #     kernel = gpflow.kernels.SquaredExponential(
    #         lengthscales=lengthscale_0,
    #         variance=signal_variance_0
    #     )
    # elif approximation == "SV":
    #     M = 30  # Number of inducing points.
    #     steps = 10000
    #     Approximator = SVGP
    #     # Initiate kernel
    #     kernel = gpflow.kernels.SquaredExponential(
    #         lengthscales=lengthscale_0,
    #         variance=signal_variance_0
    #     )
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected EP, VB, LA, V, SVB, SLA or SV)".format(
                approximation))
    return Approximator, steps, M, kernel


def main():
    """>>> python table.py table1 EP --profile"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "J", help="number of classes")
    parser.add_argument(
        "D", help="number of classes")
    parser.add_argument(
        "approach", help="SA, AA or PM")
    parser.add_argument(
        "approximation", help="LA, EP or VB")
    parser.add_argument(
        "Nimp", help="Number of importance samples 1, 2, 64, ...")
    # parser.add_argument(
    #     "chain", help="int, e.g. 0, 1, 2...")
    parser.add_argument(
        "--N_train", help="int, Number of training data points")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    J = int(args.J)
    D = int(args.D)
    approach = args.approach
    approximation = args.approximation
    num_importance_samples = int(args.Nimp)
    # chain = int(args.chain)
    N_train = int(args.N_train)
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    n_burn = 5000  # 5000
    n_samples = 10000  # 10000
    if 0:
        if dataset in datasets["paper"]:
            # Load data from file
            (X_, f_, g_, y_,
                cutpoints_0, lengthscale_0, noise_variance_0, variance_0,
                J, D, colors, Kernel) = load_data_paper(
                    dataset, J=J, D=D, ARD=False, plot=False)
            X = X_[:N_train, :]
            f = f_[:N_train]
            g = g_[:N_train]
            y = y_[:N_train]

            # # Shuffle data, just in case
            # data = np.c_[X_, f_, g_, y_]
            # np.random.shuffle(data)
            # X = data[:N_train, 0:D]
            # f = data[:N_train, D].flatten()
            # g = data[:N_train, D + 1].flatten()
            # y = data[:N_train, -1].flatten()
            # y = np.array(y, dtype=int)
            import matplotlib.pyplot as plt
            plt.scatter(X[np.where(y==0), 0], X[np.where(y==0), 1], marker='o', s=25,  c='b')
            plt.scatter(X[np.where(y==1), 0], X[np.where(y==1), 1], marker='o', s=25,  c='r')
            # plt.scatter(X[np.where(y==2), 0], X[np.where(y==2), 1], marker='o', s=25,  c='g')
            plt.savefig("data.png")
            plt.close()
            # Set theta hyperparameters
            theta_hyperparameters = np.array([1.0, np.sqrt(D)])  # [shape, rate]) on lengthscale
            noise_std_hyperparameters = np.array([1.2, 1./0.1])
            # noise_std_hyperparameters = None
            cutpoints_hyperparameters = None
            trainables = np.ones(J + 2)
            # Fix noise_variance
            # trainables[0] = 0
            # Fix scale
            trainables[J] = 0
            # Don't fix theta
            # trainables[-1] = 0
            # Fix cutpoints
            trainables[1:J] = 0
            # Sampler
            # sampler = GibbsGP(cutpoints_0, noise_variance_0, kernel, J, (X, t))
            # Define the proposal covariance
            # if approach in ["AA", "SA"]:
            #     # Initiate kernel
            #     kernel = Kernel(
            #         theta=lengthscale_0,
            #         variance=variance_0,
            #         theta_hyperparameters=theta_hyperparameters)
            #     sampler = EllipticalSliceGP(
            #         cutpoints_0, noise_variance_0,
            #         cutpoints_hyperparameters,
            #         noise_std_hyperparameters,
            #         kernel, J, (X, y))
            #     # phi = sampler.get_phi(trainables)
            #     if approach == "AA":  # Ancilliary Augmentation approach
            #         hyper_sampler = AncilliaryAugmentation(sampler)
            #     else: # approach == "SA":  # Sufficient Augmentation approach
            #         hyper_sampler = SufficientAugmentation(sampler)
            #     # TODO: not sure if theta is theta or phi
            #     # Burn-in
            #     phis, acceptance_rate = hyper_sampler.sample(
            #         f, trainables, proposal_cov, n_burn)
            #     phi_0 = phis[-1]
            #     print(phis)
            #     print(np.cov(phis.T))
            #     proposal_cov = 10.0 * np.cov(phis.T)
            #     print(np.corrcoef(phis.T))
            #     print("ACC={}".format(acceptance_rate))
            #     # Sample
            #     phis, acceptance_rate = hyper_sampler.sample(
            #         f, trainables, proposal_cov, n_samples, phi_0=phi_0, verbose=False)
            #     np.savez(
            #         write_path/'theta_N={}_D={}_J={}_ARD={}_{}_chain={}.npz'.format(
            #         sampler.N, sampler.D, sampler.J, False, approach, 0),
            #         X=phis,
            #         acceptance_rate=acceptance_rate)
            if approach == "PM":  # Pseudo Marginal approach
                Approximator, steps, M, kernel = get_approximator(
                    approximation, Kernel, lengthscale_0, variance_0,
                    N_train, D)
                for chain in range(1):
                    if "S" in approximation:
                        # Initiate sparse classifier
                        approximator = Approximator(
                            M=M, cutpoints=cutpoints_0,
                            noise_variance=noise_variance_0,
                            kernel=kernel, J=J, data=(X, y),
                            theta_hyperparameters = theta_hyperparameters,
                            noise_std_hyperparameters = noise_std_hyperparameters,
                            cutpoints_hyperparameters = cutpoints_hyperparameters,
                            )
                    elif "PEP" in approximation:
                        alpha = 0.5
                        # Initate PEP classifier
                        approximator = Approximator(
                            cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                            alpha=alpha, gauss_hermite_points=20,
                            kernel=kernel, J=J, data=(X, y),
                            noise_std_hyperparameters = noise_std_hyperparameters,
                            cutpoints_hyperparameters = cutpoints_hyperparameters,
                            )
                    else:
                        # Initiate classifier
                        approximator = Approximator(
                            cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                            kernel=kernel, J=J, data=(X, y),
                            noise_std_hyperparameters = noise_std_hyperparameters,
                            cutpoints_hyperparameters = cutpoints_hyperparameters,
                            )

                    # Initiate hyper-parameter sampler
                    hyper_sampler = PseudoMarginal(approximator)
                    phi_true = approximator.get_phi(trainables)
                    # # J=3, D=2
                    # proposal_cov = 5.0 * np.array(
                    #     [
                    #         [0.01260413, 0.01655797],
                    #         [0.01655797, 0.86673764]
                    #     ])
                    # J=13, D=2
                    proposal_cov = 4.0 * np.array(
                        [[0.00350904, 0.00712082],
                        [0.00712082, 0.53788416]])
   
                    # Burn
                    phis, acceptance_rate = hyper_sampler.sample(
                        trainables, steps, proposal_cov, n_burn,
                        num_importance_samples=num_importance_samples,
                        reparameterised=False)
                    print(phis)
                    print(np.cov(phis.T))
                    print("ACC={}".format(acceptance_rate))
                    print(proposal_cov)
                    for chain in range(3):
                        draw_mixing(phis, phi_true, logplot=False,
                            write_path=write_path,
                            file_name='burn_trace_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.png'.format(
                            approximator.N, approximator.D, approximator.J,
                            num_importance_samples, False, approach, approximation, chain))
                        draw_histogram(phis, phi_true, logplot=False,
                            write_path=write_path,
                            file_name='burn_histogram_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.png'.format(
                            approximator.N, approximator.D, approximator.J,
                            num_importance_samples, False,
                            approach, approximation, chain), bins=100)
                        phi_0 = phis[-1]
                        phis, acceptance_rate = hyper_sampler.sample(
                            trainables, steps, proposal_cov, n_samples,
                            num_importance_samples=num_importance_samples,
                            phi_0=phi_0, reparameterised=False)
                        print(np.cov(phis.T))
                        print("ACC=", acceptance_rate)
                        # draw_mixing(phis, phi_true, logplot=False,
                        #     write_path=write_path,
                        #     file_name='trace_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.png'.format(
                        #     approximator.N, approximator.D, approximator.J,
                        #     num_importance_samples, False,
                        #     approach, approximation, chain))
                        # draw_histogram(phis, phi_true, logplot=False,
                        #     write_path=write_path,
                        #     file_name='histogram_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.png'.format(
                        #     approximator.N, approximator.D, approximator.J,
                        #     num_importance_samples, False,
                        #     approach, approximation, chain), bins=100)
                        np.savez(
                            write_path/'theta_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.npz'.format(
                            approximator.N, approximator.D, approximator.J,
                            num_importance_samples, False,
                            approach, approximation, chain),
                            X=phis, acceptance_rate=acceptance_rate)

    # plot figures
    if 1: table1(write_path, n_samples, show=True, write=True)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
