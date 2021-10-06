"""
Ordinal regression concrete examples. Approximate inference: EP approximation.
"""
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
import pathlib
from probit.EP import (
    outer_loops, grid_synthetic, EP_training, grid, EP_testing,
    test_synthetic)
from probit.data.utilities import (
    datasets, load_data, load_data_synthetic)
import sys
import time


now = time.ctime()
write_path = pathlib.Path()


def main():
    """Conduct an EP estimation/optimisation."""
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
        X_true, Y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, Kernel) = load_data(
            dataset, bins)
        outer_loops(
            Kernel, method,
            X_trains, t_trains,
            X_tests, t_tests,
            gamma_0, varphi_0, noise_variance_0, scale_0,
            J, D)
        # gamma, varphi, noise_variance, scale = EP_training(
        #     method, X_trains[2], t_trains[2], X_tests[2], t_tests[2],
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        # EP_testing(
        #     X_trains[2], t_trains[2], X_tests[2], t_tests[2],
        #     gamma, varphi, noise_variance, scale, J, D)

    elif dataset in datasets["synthetic"]:
        (X, t,
        X_true, Y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        print("gamma_0 = {}".format(gamma_0))
        print("varphi_0 = {}".format(varphi_0))
        print("noise_variance_0 = {}".format(noise_variance_0))
        print("scale_0 = {}".format(scale_0))
        # test_plots(
        #     X_tests[0], X_trains[0], t_tests[0], t_trains[0], Y_trues[0])
        # varphi and std
        # grid_synthetic(J, Kernel, X, t, ((-1, 1), (-2, 2)),
        #     gamma=gamma_0, scale=scale_0)
        indices = np.ones(5)  # three
        # indices = np.ones(15)  # thirteen
        # Fix noise_variance
        indices[0] = 0
        # Fix scale
        indices[J] = 0
        # Fix varphi
        #indices[-1] = 0
        # Fix gamma
        indices[1:J] = 0
        # indices = np.ones(14)
        # gamma, varphi, noise_variance, scale = EP_training(
        #     Kernel, method,
        #     X_trains[2], t_trains[2],
        #     gamma_0, varphi_0,
        #     noise_variance_0, scale_0,
        #     J, indices)
        # EP_testing(
        #     X, t, X, t,
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        grid_synthetic(
            J, Kernel, X, t, ((-2, 0), (None)), (20, None), indices,
            varphi=varphi_0, gamma=gamma_0, noise_variance=noise_variance_0,
            scale=scale_0, show=True)
        # Just scale
        # grid_synthetic(
        #     J, Kernel, X, t, ((-4, 4),(None)), (100, None),
        #     varphi=varphi_0, gamma=gamma_0, noise_variance=noise_variance_0,
        #     scale=None, show=True)
        # Two of the cutpoints
        # grid_synthetic(
        #     J, Kernel, X, t, ((-2, 2), (-3, 1)), (100, None),
        #     varphi=varphi_0, noise_variance=noise_variance_0, scale=scale_0)
        # test_synthetic(
        #     Kernel, method, X, t, X_true, Y_true,
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J, D, colors=colors)
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
