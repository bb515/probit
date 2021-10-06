import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from probit.VB import (
    outer_loops, VB_testing, grid_synthetic)
import pathlib
from probit.data.utilities import (
    datasets, load_data, load_data_synthetic)
import time


now = time.ctime()
write_path = pathlib.Path()

def main():
    """Conduct VB approximate inference and optimisation."""
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
    if dataset in datasets:
        X_trains, t_trains, X_tests, t_tests, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, J, D, Kernel = load_data(
            dataset, bins)
        steps = 1000
        gamma_0 = np.array(gamma_0)
        outer_loops(
            Kernel, X_trains, t_trains, X_tests, t_tests, steps,
            gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
        # gamma, varphi, noise_variance = VB_training(
        #     Kernel, method, X_trains[2], t_trains[2],
        #     gamma_0, varphi_0, noise_variance_0, scale_0, J)
        # VB_testing(
        #     Kernel, X_trains[2], t_trains[2], X_tests[2], t_tests[2],
        #     steps, gamma=gamma, varphi=varphi,
        #     noise_variance=noise_variance, scale=scale, J)
    else:
        # TODO: will need to extract test/train data for outerloops
        (X, t,
        X_true, Y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, bins)
        # test_plots(
        #     Kernel,
        #     X_tests[0], X_trains[0],
        #     t_tests[0], t_trains[0], Y_trues[0])
        indices = np.ones(15)  # three
        # indices = np.ones(15)  # thirteen
        # Fix noise_variance
        indices[0] = 0
        # Fix scale
        indices[J] = 0
        # Fix varphi
        #indices[-1] = 0
        # Fix gamma
        indices[1:J] = 0
        # #  just scale
        # grid_synthetic(
        #     J, Kernel, X, t, ((0., 1.8), None), (20, None),
        #     gamma=gamma_0, varphi=varphi_0,
        #     noise_variance=noise_variance_0, scale_0=scale_0)
        # # just std
        # grid_synthetic(
        #     J, Kernel, X, t, ((-0.1, 1.), None), (100, None)
        #     gamma=gamma_0, varphi=varphi_0, scale=scale_0, show=True)
        # # varphi and scale
        # grid_synthetic(
        #     J, Kernel, X, t, ((0, 2), (0, 2)), (100, None),
        #     gamma=gamma_0, noise_variance=noise_variance_0, scale=scale_0)
        # # varphi and std
        # grid_synthetic(
        #     J, Kernel, X, t, ((0, 2), (0, 2)), (100, None),
        #     gamma=gamma_0, scale=scale_0)
        # Just varphi
        grid_synthetic(
            J, Kernel, X, t, ((-4, 4), None), (100, None), indices,
            gamma=gamma_0, noise_variance=noise_variance_0,
            scale=scale_0, show=True)
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
