"""Ordinal regression concrete examples. Approximate inference."""
# Make sure to limit CPU usage
import os
# # Enable double precision
# from jax.config import config
# config.update("jax_enable_x64", True)

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = "6"

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
        approximation, N_train):
    if approximation == "VB":
        from probit.approximate_implicit import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//1000])
        Approximator = VBGP
    elif approximation == "LA":
        from probit.approximate_implicit import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected VB, LA)".format(
                approximation))
    return Approximator, steps


def main():
    """Conduct an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "J", help="e.g., 13, 53, 101, etc.")
    parser.add_argument(
        "D", help="number of classes")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
    parser.add_argument(
        "approximation", help="EP or VB or LA")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    J = int(args.J)
    D = int(args.D)
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
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, Kernel) = load_data(
            dataset, J)
        X = X_trains[2]
        y = y_trains[2]
    elif dataset in datasets["synthetic"]:
        (X, y,
        X_true, g_true,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, J)
    elif dataset in datasets["paper"]:
        (X, f_, g_true, y,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_paper(
            dataset, J=J, D=D, ARD=False, plot=True)
    else:
        raise ValueError("Dataset {} not found.".format(dataset))
    from mlkernels import EQ
    N_train = np.shape(y)[0]
    Approximator, steps = get_approximator(approximation, N_train)
    # Initiate classifier
    def model(parameters):
        cutpoints = cutpoints_0
        cutpoints = parameters
        noise_variance = noise_variance_0
        theta = theta_0
        signal_variance = signal_variance_0
        # Here you can define the kernel that defines the Gaussian process
        kernel = signal_variance * EQ().stretch(theta)
        # Make sure that model returns the kernel, cutpoints and noise_variance
        return (kernel, cutpoints, noise_variance)
    parameters = cutpoints_0
    classifier = Approximator(model, parameters, J, data=(X, y),
        # single_precision=True
        )
    trainables = [1] * (J + 2)
    # if kernel._ARD:
    #     trainables[-1] = [1, 1]
    #     # Fix theta
    #     trainables[-1] = [0] * int(D)
    trainables[-1] = 1
    # Fix theta
    trainables[-1] = 0
    # Fix noise standard deviation
    trainables[0] = 1
    # Fix signal standard deviation
    trainables[J] = 0
    # Fix cutpoints
    trainables[1:J] = [0] * (J - 1)
    trainables[1] = 0
    print("trainables = {}".format(trainables))
    # just theta
    domain = ((-1, 1), None)
    res = (3, None)
    # theta_0 and theta_1
    # domain = ((-1, 1.3), (-1, 1.3))
    # res = (20, 20)
    # #  just signal standard deviation, domain is log_10(signal_std)
    # domain = ((0., 1.8), None)
    # res = (20, None)
    # just noise std, domain is log_10(noise_std)
    # domain = ((-1., 1.0), None)
    # res = (100, None)
    # # theta and signal std dev
    # domain = ((0, 2), (0, 2))
    # res = (100, None)
    # # cutpoints b_1 and b_2 - b_1
    # domain = ((-0.75, -0.5), (-1.0, 1.0))
    # res = (14, 14)

    # grid_synthetic(classifier, domain, res, steps, trainables, show=False)

    # plot(classifier, domain=None)

    # classifier = train(
    #     classifier, method, trainables, verbose=True, steps=steps)
    # test(classifier, X, y, g_true, steps)

    # Notes: anderson solver worked stably, Newton solver did not. Fixed point iteration worked stably and fastest
    # Newton may be unstable due to the condition number of the matrix. I wonder if I can hard code it instead of using autodiff?

    # z_star = jnp.array([-0.75561608, -0.76437714, -0.21782411, -0.87292511, 0.63080693, -0.97272624,
    #  -0.57807269, -0.06598705, -0.20778863, -0.33945913])

    # #z_star = jnp.zeros(ndim)

    f = classifier.construct()

    # z_star = 0
    # for i in range(100):
    #     z_prev = z_star
    #     z_star = f(1.0, z_star)
    #     print(np.linalg.norm(z_star - z_prev))

    g = classifier.take_grad()
    N = 30
    thetas = np.logspace(-1, 1, 30)
    gs = np.empty(30)
    fs = np.empty(30)
    for i, theta in enumerate(thetas):
        fx, gx = g(theta)
        fs[i] = fx
        gs[i] = gx
        print(g(theta))
    import matplotlib.pyplot as plt
    plt.plot(thetas, fs)
    plt.xscale("log")
    plt.savefig("fx.png")
    plt.close()


    dZ = np.gradient(fs, thetas)
    plt.plot(thetas, dZ)
    plt.plot(thetas, gs)
    plt.xscale("log")
    plt.savefig("gx.png")
    plt.close()
    assert 0

    # def fun(theta):
    #     value_and_grad = g(theta)
    #     return (np.asarray(value_and_grad[0]), np.asarray(value_and_grad[1]))

    # grid_synthetic(
    #     classifier, domain, res, steps, trainables, show=True, verbose=True)

    # plot_synthetic(classifier, dataset, X_true, g_true, steps, colors=colors)

    # outer_loops(
    #     Approximator, Kernel, X_trains, y_trains, X_tests, y_tests, steps,
    #     cutpoints_0, theta_0, noise_variance_0, signal_variance_0, J, D)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
