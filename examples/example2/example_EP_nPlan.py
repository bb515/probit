"""
nPlan data, ordinal regression. Approximate inference: EP approx.
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
import sys
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import pathlib
from probit.approximators import EPOrdinalGP
from probit.plot import train, outer_loops, outer_loop_problem_size, grid
from probit.EP import test
from probit.data.utilities_nplan import datasets, load_data
import numpy as np



def main():
    """Conduct an EP approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="quantile or decile")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
    parser.add_argument(
        "--N_train", help="Number of training data points")
    parser.add_argument(
        "--N_test", help="Number of testing data points")
    parser.add_argument(
        "--real_valued", help="data is from prior?",
        action='store_const', const=True)
    parser.add_argument(
        "--text_data", help="data is from prior?",
        action='store_const', const=True)
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = args.bins
    text_data = args.text_data
    real_valued = args.real_valued
    if args.N_train is not None:
        N_train = np.int(args.N_train)
    if args.N_test is not None:
        N_test = np.int(args.N_test)
    method = args.method
    N_train = np.int(args.N_train)
    N_test = np.int(args.N_test)
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("test.txt", "w")
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets:
        (X_trains, t_trains,
        X_tests, t_tests,
        X_true, y_tests,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, data, Kernel, *_) = load_data(
            dataset, bins, N_train=N_train, N_test=N_test,
            text_data=text_data, real_valued_only=real_valued)
        # (_, _,
        # X_tests, t_tests,
        # X_true, y_tests,
        # *_) = load_data("acciona_test1",
        #     bins, N_train=N_train, N_test=N_test, text_data=text_data,
        #     real_valued_only=real_valued)
        # print("gamma={}, varphi={}, noise_variance={}, scale={}".format(
        #     gamma_0, varphi_0, noise_variance_0, scale_0))
        steps = 1000
        mean_fx, std_fx, mean_metrics, std_metrics = outer_loops(
            test, EPOrdinalGP, Kernel, method, X_trains, t_trains, X_tests,
            t_tests, y_tests, steps,
            gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        print("fx = {} +/- {}".format(mean_fx, std_fx))
        print("metrics = {} +/- {}".format(mean_metrics, std_metrics))
        # # Initiate kernel
        # kernel = Kernel(varphi=varphi_0, scale=scale_0)
        # # Initiate the classifier with the training data
        # classifier = EPOrdinalGP(
        #     gamma_0, noise_variance_0, kernel,
        #     X_trains[0], t_trains[0], J)
        # test(classifier, X_tests[0], t_tests[0], y_tests[0], steps)
        # indices = np.ones(15, dtype=int)
        # # fix noise_variance
        # indices[0] = 0
        # # fix gammas
        # indices[1:J] = 0
        # # fix scale
        # indices[J] = 0
        # # fix varphi
        # indices[-1] = 0
        # print("indices", indices)
        # mean_fx, std_fx, mean_metrics, std_metrics = outer_loops(
        #     test, EPOrdinalGP, Kernel, method, X_trains, t_trains, X_tests,
        #     t_tests, y_tests, steps,
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        # print("mean_fx=", mean_fx, " std_fx=", std_fx)
        # print("mean_metrics=", mean_metrics, " std_metrics=", std_metrics)
        # outer_loop_problem_size(
        #     test, EPOrdinalGP, Kernel, method, X_trains, t_trains, X_tests,
        #     t_tests, y_tests, steps,
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J, D, size=4.0,
        #     num=8)
        # grid(classifier, X_trains, t_trains, ((-5.0, -2.0), (None)), (30, None),
        #     "bam_varphi=0.001", indices=indices)
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
