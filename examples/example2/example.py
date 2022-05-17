"""
nPlan data, ordinal regression. Approximate inference: VB or EP or LA.
"""
# Make sure to limit CPU usage
import os
num_threads = "6"
os.environ["OMP_NUM_THREADS"] = num_threads # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = num_threads # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = num_threads # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = num_threads # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = num_threads

import argparse
import sys
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import pathlib
from probit.plot import train, outer_loops, outer_loop_problem_size, grid, test
from probit.approximators import VBOrdinalGP, EPOrdinalGP, LaplaceOrdinalGP
from probit.sparse import SparseVBOrdinalGP, SparseLaplaceOrdinalGP
from probit.data.utilities_nplan import datasets, load_data
import numpy as np


def main():
    """Conduct a VB approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "num_classes", help="e.g., 13 or 101")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
    # The type of approximate posterior used
    parser.add_argument(
        "approximation", help="EP or VB or Laplace")
    parser.add_argument(
        "--N_train", help="Number of training data points")
    parser.add_argument(
        "--N_test", help="Number of testing data points")
    parser.add_argument(
        "--real_valued", help="Only use real valued data",
        action='store_const', const=True)
    parser.add_argument(
        "--text_data", help="Only use text data",
        action='store_const', const=True)
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    J = np.int(args.num_classes)
    text_data = args.text_data
    real_valued = args.real_valued
    N_train = args.N_train
    N_test = args.N_test
    if args.N_train is not None:
        N_train = np.int(N_train)
    if args.N_test is not None:
        N_test = np.int(N_test)
    method = args.method
    approximation = args.approximation
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
        cutpoints_0, varphi_0, noise_variance_0, signal_variance_0,
        J, D, data, Kernel, bin_edges) = load_data(dataset,
            J, N_train=N_train, N_test=N_test, text_data=text_data,
            real_valued_only=real_valued)
        # (_, _,
        # X_tests, t_tests,
        # X_true, y_tests,
        # *_) = load_data("acciona_test1",
        #     J, N_train=N_train, N_test=N_test, text_data=text_data,
        #     real_valued_only=real_valued)
        if approximation == "EP":
            steps = np.max([10, N_train//100])  # for N=3000, steps is 300 - could be too large since per iteration is slow.
            Approximator = EPOrdinalGP
        elif approximation == "VB":
            steps = np.max([100, N_train//10])
            Approximator = VBOrdinalGP
        elif approximation == "LA":
            steps = np.max([2, N_train//1000])
            Approximator = LaplaceOrdinalGP
        elif approximation == "SLA":
            steps = np.max([2, N_train//1000])
            Approximator == SparseLaplaceOrdinalGP
        elif approximation == "SVB":
            steps = np.max([100, N_train//10])
            Approximator == SparseVBOrdinalGP
        if 0:
            # test_0 = t_tests[0]
            # on_time = len(test_0[test_0==5]) / len(test_0)
            # print(on_time)
            # # steps = 1000  # Note that this is needed when the noise variance is small, for the fix point to converge quickly
            mean_fx, std_fx, mean_metrics, std_metrics = outer_loops(
                test, Approximator, Kernel, method, X_trains, t_trains, X_tests,
                t_tests, y_tests, steps,
                cutpoints_0, varphi_0, noise_variance_0, signal_variance_0, J, D)
            print("fx = {} +/- {}".format(mean_fx, std_fx))
            print("metrics = {} +/- {}".format(mean_metrics, std_metrics))
        if 1:
            # Initiate kernel
            kernel = Kernel(
                varphi=varphi_0, variance=signal_variance_0)
            if approximation in ["EP, VB, LA"]:
                # Initiate the classifier with the training data
                classifier = Approximator(
                    cutpoints_0, noise_variance_0, kernel,
                    J, (X_trains[0], t_trains[0]))
            else:
                # Initiate the sparse classifier with the training data
                M = 1000  # number of inducing points
                classifier = Approximator(
                    M=M, cutpoints=cutpoints_0,
                    noise_variance=noise_variance_0, kernel=kernel,
                    J=J, data=(X_trains[0], t_trains[0])
                )

            posterior_inv_cov, posterior_mean, *_ = test(
                classifier, X_tests[0], t_tests[0], y_tests[0], steps)

            if 0:
                from probit.data.tf_parse import embedder_initiate_alternate
                import matplotlib.pyplot as plt
                embed = embedder_initiate_alternate()  # text embedder
                text = np.array(
                    [
                        [b'Axial Fans - Prepare technical documents'],
                        [b'MCC1 Bullnose to Roundabout - Complete jointing work and grout all poles'],
                        [b'Sprinkler & Hydrant Valves - Design Alliance support during technical evaluation'],
                        [b'Axial Fans - Design Alliance support during technical evaluation'],
                        [b'Low Point Sump Foam Suppression System - Prepare technical documents - Design Alliance Input'],
                        [b'4 off hydraulics units and steering mechanisms refurbishment/replacement'],
                        [b'Assembly of pipes, valv. and isol. ground'],
                        [b'TMR Road Lighting Poles - Confirm Prices for first batch'],
                        [b'Carpentry'],
                        [b'SHAFT 100 (DN 315) of the pK 1 + pK 200 to 1 + 550'],
                        [b'Cable Support Systems - Contract Conformance'],
                        [b'lay foundations concrete piles'],
                        [b'formwork'],
                        [b'walk to the shops'],
                        [b'eat rice crispies, a bannana and peanut butter']
                    ]
                )
                text = text.flatten()
                text_embedding = embed(text).numpy()
                print(np.shape(text_embedding))
                (Z,
                posterior_predictive_m,
                posterior_std) = classifier.predict(
                    classifier.cutpoints, posterior_inv_cov, posterior_mean,
                    classifier.kernel.varphi,
                    classifier.noise_variance, text_embedding, vectorised=True)
                
                cutpoints = np.concatenate(
                        [np.array([np.exp(-3.0)]),
                        np.array(cutpoints),
                        np.array([np.exp(3.0)])])
                cutpoints_left = cutpoints[:-1]
                width = 0.85*(cutpoints[1:] - cutpoints[:-1])

                for i in range(np.shape(Z)[0]):
                    plt.bar(cutpoints_left, Z[i], align='edge', width=width)
                    plt.xlim(0.0, 5.0)
                    plt.title("{}".format(text[i]))
                    plt.savefig("{}.png".format(i))
                    plt.close()
                assert 0

        if 0:
            # Initiate kernel
            kernel = Kernel(varphi=varphi_0, variance=signal_variance_0)
            # Initiate the classifier with the training data
            classifier = Approximator(
                cutpoints_0, noise_variance_0, kernel, J,
                (X_trains[0], t_trains[0]))
            indices = np.ones(15, dtype=int)
            # # fix noise variance
            # indices[0] = 0
            # fix cutpoints
            indices[1:J] = 0
            # fix signal variance
            indices[J] = 0
            # fix varphi
            # indices[-1] = 0
            outer_loop_problem_size(
                test, Approximator, Kernel, method, X_trains, t_trains, X_tests,
                t_tests, y_tests, steps,
                cutpoints_0, varphi_0, noise_variance_0, signal_variance_0, J, D, size=4.23,
                num=4)
            print("indices", indices)
            grid(classifier, X_trains, t_trains,
                ((-1.0, 1.0), (-1.0, 1.0)), (20, 20),
                "acciona_noise_var_varphi", indices=indices)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
