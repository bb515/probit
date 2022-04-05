"""
Ordinal regression concrete examples. Approximate inference.
"""
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
from probit.approximators import EPOrdinalGP, VBOrdinalGP, LaplaceOrdinalGP
from probit.plot import outer_loops, grid_synthetic, grid, plot_synthetic, plot, train, test
from probit.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
import sys
import time


now = time.ctime()
write_path = pathlib.Path()


def main():
    """Conduct an EP approximation to the posterior, and optimise hyperparameters."""
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
        (X_trains, t_trains,
        X_tests, t_tests,
        X_true, y_tests,
        cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        J, D, Kernel) = load_data(
            dataset, bins)
        N_train = np.shape(t_trains[0])
        if approximation == "EP":
            steps = np.max([10, N_train//100])  # for N=3000, steps is 300 - could be too large since per iteration is slow.
            Approximator = EPOrdinalGP
        elif approximation == "VB":
            steps = np.max([100, N_train//10])
            Approximator = VBOrdinalGP
        elif approximation == "LA":
            steps = np.max([2, N_train//1000])
            Approximator = LaplaceOrdinalGP
        outer_loops(
            Approximator, Kernel, X_trains, t_trains, X_tests, t_tests, steps,
            cutpoints_0, varphi_0, noise_variance_0, signal_variance_0, J, D)
        # Initiate kernel
        kernel = Kernel(varphi=varphi_0, variance=signal_variance_0)
        # Initiate the classifier with the training data
        classifier = Approximator(
            cutpoints_0, noise_variance_0, kernel,
            J, (X_trains[2], t_trains[2]))
        indices = np.ones(5)  # three
        # indices = np.ones(15)  # thirteen
        # Fix noise_variance
        indices[0] = 0
        # Fix cutpoints
        indices[1:J] = 0
        # Fix signal variance
        indices[J] = 0
        # Fix varphi
        indices[-1] = 0
        classifier = train(
            classifier, method, indices)
        fx, metrics = test(
            classifier, X_tests[2], t_tests[2], y_tests[2], steps)
    elif dataset in datasets["synthetic"]:
        # (X, Y, t,
        # cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        # J, D, colors, Kernel) = load_data_paper(dataset, plot=True)
        (X, t,
        X_true, Y_true,
        cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        # Initiate kernel
        kernel = Kernel(
            varphi=varphi_0, variance=signal_variance_0)
        N_train = np.shape(t)[0]
        if approximation == "EP":
            steps = np.max([100, N_train//100])  # for N=3000, steps is 300 - could be too large since per iteration is slow.
            Approximator = EPOrdinalGP
        elif approximation == "VB":
            steps = np.max([10, N_train//10])
            Approximator = VBOrdinalGP
        elif approximation == "LA":
            steps = np.max([2, N_train//1000])
            Approximator = LaplaceOrdinalGP
        # Initiate classifier
        classifier = Approximator(
            cutpoints_0, noise_variance_0, kernel,
            J, (X, t))
        indices = np.ones(J + 2)
        # Fix noise variance
        indices[0] = 0
        # Fix signal variance
        indices[J] = 0
        # Fix varphi
        #indices[-1] = 0
        # Fix cutpoints
        indices[1:J] = 0
        # Just varphi
        domain = ((-4, 4), None)
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
        # # varphi and std
        # domain = ((0, 2), (0, 2))
        # res = (100, None)
        # grid_synthetic(classifier, domain, res, indices, show=False)
        # plot(classifier, domain=None)
        # classifier = train(classifier, method, indices)
        # test(classifier, X, t, Y_true, steps)
        #grid_synthetic(classifier, domain, res, indices, show=False)
        plot_synthetic(
            classifier, dataset, X_true, Y_true, steps, colors=colors)
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
