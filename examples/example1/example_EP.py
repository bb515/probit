"""
Ordinal regression concrete examples. Approximate inference: EP approximation.
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
from probit.approximators import EPOrdinalGP
from probit.plot import outer_loops, grid_synthetic, train, grid
from probit.EP import test, plot_synthetic, plot
from probit.data.utilities import datasets, load_data, load_data_synthetic
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
        "bins", help="quantile or decile")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
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
    if dataset in datasets["benchmark"]:
        (X_trains, t_trains,
        X_tests, t_tests,
        X_true, y_tests,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, Kernel) = load_data(
            dataset, bins)
        steps = 1000
        outer_loops(
            EPOrdinalGP, Kernel, X_trains, t_trains, X_tests, t_tests, steps,
            gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        # # Initiate kernel
        # kernel = Kernel(varphi=varphi_0, scale=scale_0)
        # # Initiate the classifier with the training data
        # classifier = EPOrdinalGP(
        #     gamma_0, noise_variance_0, kernel, X_trains[2], t_trains[2], J)
        # indices = np.ones(5)  # three
        # # indices = np.ones(15)  # thirteen
        # # Fix noise_variance
        # indices[0] = 0
        # # Fix gamma
        # indices[1:J] = 0
        # # Fix scale
        # indices[J] = 0
        # # Fix varphi
        # indices[-1] = 0
        # classifier = train(
        #     classifier, method, indices)
        # fx, metrics = test(
        #     classifier, X_tests[2], t_tests[2], y_tests[2], steps)
    elif dataset in datasets["synthetic"]:
        (X, t,
        X_true, Y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        steps = 50
        # Initiate kernel
        kernel = Kernel(varphi=varphi_0, scale=scale_0)
        # Initiate classifier
        classifier = EPOrdinalGP(
            gamma_0, noise_variance_0, kernel, X, t, J)
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
        domain = ((-4, 4), None)
        res = (100, None)
        # #  just scale
        # domain = ((0., 1.8), None)
        # res = (20, None)
        # # just std
        # domain = ((-0.1, 1.), None)
        # res = (100, None)
        # # varphi and scale
        # domain = ((0, 2), (0, 2))
        # res = (100, None)
        # # varphi and std
        # domain = ((0, 2), (0, 2))
        # res = (100, None)
        # grid_synthetic(classifier, domain, res, indices, show=False)
        # plot(classifier, domain=None)
        # classifier = train(classifier, method, indices)
        # test(classifier, X, t, Y_true, steps)
        # grid_synthetic(classifier, domain, res, indices, show=False)
        plot_synthetic(classifier, dataset, X_true, Y_true, colors=colors)
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
