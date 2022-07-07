"""Ordinal regression concrete examples. Approximate inference."""
# Make sure to limit CPU usage
import os
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
import pathlib
from probit.plot import outer_loops, grid_synthetic, grid, plot_synthetic, plot, train, test
from probit.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
import sys
import time


now = time.ctime()
write_path = pathlib.Path()


def get_approximator(
        approximation, Kernel, varphi_0, signal_variance_0, N_train):
    # Initiate kernel
    kernel = Kernel(
        varphi=varphi_0, variance=signal_variance_0)
    M = None
    if approximation == "EP":
        from probit.approximators import EPGP
        # steps is the number of swipes over the data until check convergence
        steps = 1
        Approximator = EPGP
    elif approximation == "VB":
        from probit.approximators import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//10])
        Approximator = VBGP
    elif approximation == "LA":
        from probit.approximators import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    elif approximation == "SLA":
        from probit.sparse import SparseLaplaceGP
        M = 30  # Number of inducing points
        steps = np.max([2, M//10])
        Approximator = SparseLaplaceGP
    elif approximation == "SVB":
        from probit.sparse import SparseVBGP
        M = 30  # Number of inducing points
        steps = np.max([10, M])
        Approximator = SparseVBGP
    # elif approximation == "V":
    #     from probit.gpflow import VGP
    #     import gpflow
    #     steps = 100
    #     Approximator = VGP
    #     # Initiate kernel
    #     # kernel = gpflow.kernels.SquaredExponential(
    #     #     lengthscales=varphi_0, variance=signal_variance_0)
    #     kernel = gpflow.kernels.SquaredExponential(
    #         lengthscales=1./np.sqrt(2 * varphi_0),
    #         variance=signal_variance_0
    #     )
    # elif approximation == "SV":
    #     M = 30  # Number of inducing points.
    #     from probit.gpflow import SVGP
    #     import gpflow
    #     steps = 10000
    #     Approximator = SVGP
    #     # Initiate kernel
    #     # kernel = gpflow.kernels.SquaredExponential()
    #     kernel = gpflow.kernels.SquaredExponential(
    #         lengthscales=1./np.sqrt(2 * varphi_0),
    #         variance=signal_variance_0
    #     )
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected EP, VB, LA, V, SVB, SLA or SV)".format(
                approximation))
    return Approximator, steps, M, kernel


def main():
    """Conduct an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="e.g., 13, 53, 101, etc.")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
    parser.add_argument(
        "approximation", help="EP or VB or LA")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = int(args.bins)
    method = args.method
    approximation = args.approximation
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets["benchmark"]:
        (X_trains, y_trains,
        X_tests, y_tests,
        X_true, g_tests,
        cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        J, D, Kernel) = load_data(
            dataset, bins)
        N_train = np.shape(y_trains[0])
        Approximator, steps, M, kernel = get_approximator(
            approximation, Kernel, varphi_0, signal_variance_0, N_train)
        outer_loops(
            Approximator, Kernel, X_trains, y_trains, X_tests, y_tests, steps,
            cutpoints_0, varphi_0, noise_variance_0, signal_variance_0, J, D)
        if "S" in approximation:
            # Initiate sparse classifier
            classifier = Approximator(
                M=M, cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                kernel=kernel, J=J, data=(X_trains[2], y_trains[2]))
        else:
            # Initiate classifier
            classifier = Approximator(
                cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                kernel=kernel, J=J, data=(X_trains[2], y_trains[2]))
        trainables = np.ones(5)  # three
        # trainables = np.ones(15)  # thirteen
        # Fix noise_variance
        trainables[0] = 0
        # Fix cutpoints
        trainables[1:J] = 0
        # Fix signal variance
        trainables[J] = 0
        # Fix varphi
        trainables[-1] = 0
        classifier = train(
            classifier, method, trainables)
        fx, metrics = test(
            classifier, X_tests[2], y_tests[2], g_tests[2], steps)
    elif dataset in datasets["synthetic"]:
        # (X, g, y,
        # cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        # J, D, colors, Kernel) = load_data_paper(dataset, plot=True)
        (X, y,
        X_true, g_true,
        cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        N_train = np.shape(y)[0]
        Approximator, steps, M, kernel = get_approximator(
            approximation, Kernel, varphi_0, signal_variance_0, N_train)
        if "S" in approximation:
            # Initiate sparse classifier
            classifier = Approximator(
                M=M, cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                kernel=kernel, J=J, data=(X, y))
        else:
            # Initiate classifier
            classifier = Approximator(
                cutpoints=cutpoints_0, noise_variance=noise_variance_0,
                kernel=kernel, J=J, data=(X, y))
        trainables = np.ones(J + 2)
        # Fix noise variance
        trainables[0] = 0
        # Fix signal variance
        trainables[J] = 0
        # Fix varphi
        trainables[-1] = 0
        # Fix cutpoints
        trainables[1] = 0
        # Just varphi
        domain = ((-0.0, 0.3), None)
        res = (100, None)
        # #  just signal variance
        # domain = ((0., 1.8), None)
        # res = (20, None)
        # # just std
        # domain = ((-0.1, 1.), None)
        # res = (100, None)
        # # varphi and signal variance
        # domain = ((0, 2), (0, 2))
        # res = (100, None)
        # # cutpoints b_1 and b_2 - b_1
        # domain = ((-0.75, -0.5), (-1.0, 1.0))
        # res = (14, 14)
        # grid_synthetic(classifier, domain, res, steps, trainables, show=False)
        # plot(classifier, domain=None)
        # classifier = train(classifier, method, trainables)
        # classifier = train(
        #     classifier, method, trainables, verbose=True, steps=steps)
        # test(classifier, X, y, g_true, steps)
        # grid_synthetic(classifier, domain, res, steps, trainables, show=True)
        plot_synthetic(
            classifier, dataset, X_true, g_true, steps, colors=colors)
        #plot_synthetic(classifier, dataset, X, Y, colors=colors)
    else:
        raise ValueError("Dataset {} not found.".format(dataset))
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
