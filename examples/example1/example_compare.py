"""Ordinal regression concrete examples. Approximate inference."""
# Make sure to limit CPU usage
import os

# Enable double precision
from jax.config import config
config.update("jax_enable_x64", True)

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
from probit_jax.plot import (
    _grid_over_hyperparameters_initiate)
from probit.data.utilities import datasets as datasets_
from probit.data.utilities import load_data as load_data_
from probit.data.utilities import load_data_synthetic as load_data_synthetic_
from probit.data.utilities import load_data_paper as load_data_paper_
from probit.data.utilities import datasets as datasets_
from probit_jax.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
from mlkernels import Kernel as BaseKernel
from probit_jax.utilities import InvalidKernel, check_cutpoints
from probit_jax.implicit.utilities import (
    log_probit_likelihood, grad_log_probit_likelihood, hessian_log_probit_likelihood)
import sys
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4


now = time.ctime()
write_path = pathlib.Path()


def get_approximator(
        approximation, N_train):
    if approximation == "VB":
        from probit_jax.approximators import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//1000])
        Approximator = VBGP
    elif approximation == "LA":
        from probit_jax.approximators import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected VB, LA)".format(
                approximation))
    return Approximator, steps


def get_approximator_(
        approximation, Kernel, theta_0, signal_variance_0, N_train):
    # Initiate kernel
    kernel = Kernel(
        theta=theta_0, variance=signal_variance_0)
    M = None
    if approximation == "VB":
        from probit.approximators import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//1000])
        Approximator = VBGP
    elif approximation == "LA":
        from probit.approximators import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected VB or LA".format(
                approximation))
    return Approximator, steps, M, kernel


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

    # Initiate data and classifier for probit_jax repo

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
    def prior(prior_parameters):
        stretch = prior_parameters
        signal_variance = signal_variance_0
        # Here you can define the kernel that defines the Gaussian process
        kernel = signal_variance * EQ().stretch(stretch)
        # Make sure that model returns the kernel, cutpoints and noise_variance
        return kernel

    # Test prior
    if not (isinstance(prior(1.0), BaseKernel)):
        raise InvalidKernel(prior(1.0))

    # check that the cutpoints are in the correct format
    # for the number of classes, J
    cutpoints_0 = check_cutpoints(cutpoints_0, J)

    classifier = Approximator(prior, log_probit_likelihood,
        # grad_log_likelihood=grad_log_probit_likelihood,
        # hessian_log_likelihood=hessian_log_probit_likelihood,
        data=(X, y))

    # Initiate data and classifier for probit repo
    dataset = "SEIso"
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets_["benchmark"]:
        (X_trains, y_trains,
        X_tests, y_tests,
        X_true, g_tests,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, Kernel) = load_data_(
            dataset, J)
        X = X_trains[2]
        y = y_trains[2]
    elif dataset in datasets_["synthetic"]:
        (X, y,
        X_true, g_true,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_synthetic_(dataset, J)
    elif dataset in datasets_["paper"]:
        (X, f_, g_true, y,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_paper_(
            dataset, J=J, D=D, ARD=False, plot=True)
    else:
        raise ValueError("Dataset {} not found.".format(dataset))
    N_train = np.shape(y)[0]
    Approximator, steps, M, kernel = get_approximator_(
        approximation, Kernel, theta_0, signal_variance_0, N_train)
    if "S" in approximation:
        # Initiate sparse classifier
        _classifier = Approximator(
            M=M, cutpoints=cutpoints_0, noise_variance=noise_variance_0,
            kernel=kernel, J=J, data=(X, y))
    else:
        # Initiate classifier
        _classifier = Approximator(
            cutpoints=cutpoints_0, noise_variance=noise_variance_0,
            kernel=kernel, J=J, data=(X, y), single_precision=False)

    # Notes: anderson solver worked stably, Newton solver did not. Fixed point iteration worked stably and fastest
    # Newton may be unstable due to the condition number of the matrix. I wonder if I can hard code it instead of using autodiff?

    # z_star = jnp.array([-0.75561608, -0.76437714, -0.21782411, -0.87292511, 0.63080693, -0.97272624,
    #  -0.57807269, -0.06598705, -0.20778863, -0.33945913])

    # #z_star = jnp.zeros(ndim)

    # f = classifier.construct()

    # prior_parameters_0 = jnp.sqrt(1./(2 * theta_0))
    # likelihood_parameters_0 = (jnp.sqrt(noise_variance_0), cutpoints_0)
    # print(prior_parameters_0)
    # z_0 = B.zeros(classifier.N)
    # z = jnp.array(B.dense(f((prior_parameters_0, likelihood_parameters_0), z_0))).flatten()
    # print(z)
    # z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    # print(z)
    # z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    # print(z)
    # z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    # print(z)
    # z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    # print(z)
    # plt.scatter(X, z)
    # plt.savefig("testlatent")
    # plt.close()

    # z_star = 0
    # for i in range(100):
    #     z_prev = z_star
    #     z_star = f(1.0, z_star)
    #     print(np.linalg.norm(z_star - z_prev))
    # TODO: not sure why in their example can just initiate to any parameters here.
    params = ((jnp.sqrt(1./(2 * theta_0))), (jnp.sqrt(noise_variance_0), cutpoints_0))
    g = classifier.take_grad()

    trainables = [1] * (J + 2)
    # Fix theta
    # trainables[-1] = 0
    # Fix noise standard deviation
    trainables[0] = 0
    # Fix signal standard deviation
    trainables[J] = 0
    # Fix cutpoints
    trainables[1:J] = [0] * (J - 1)
    trainables[1] = 0
    print("trainables = {}".format(trainables))
    # just theta
    #domain = ((-1, 2), None)
    domain = ((-1, 2), None)
    res = (30, None)
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

    (x1s, x2s,
    xlabel, ylabel,
    xscale, yscale,
    xx, yy,
    phis, fxs,
    gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
        _classifier, res, domain, trainables)

    gs = np.empty(res[0])
    fs = np.empty(res[0])
    for i, phi in enumerate(phis):
        theta = jnp.exp(phi)[0]
        params = ((jnp.sqrt(1./(2 * theta))), (jnp.sqrt(noise_variance_0), cutpoints_0))
        # params = ((jnp.sqrt(1./(2 * theta_0))), (jnp.sqrt(theta), cutpoints_0))
        fx, gx = g(params)
        fs[i] = fx
        gs[i] = gx[0] * (- 0.5 * (2 * theta_0)**(-1./2))  # multiply by a Jacobian
        # gs[i] = gx[1][0] * (0.5 * (theta) ** (1./2))  # multiply by a Jacobian
        print(fx)
        print(gx)
        # print(gx[0])
        # print(gx[1][0])
        # print(gx[1][1])

    plt.plot(phis, fs)
    # plt.xscale("log")
    plt.savefig("testfx.png")
    plt.close()
    plt.plot(phis, gs, label="ad")
    # plt.xscale("log")
    plt.legend()
    plt.savefig("testgx.png")
    plt.close()

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    for i, phi in enumerate(phis):
        fx, gx = _classifier.approximate_posterior(
            phi, trainables, steps, verbose=True)
        fxs[i] = fx
        gxs[i] = gx

    (fxs, gxs,
    x, y,
    xlabel, ylabel,
    xscale, yscale) = (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    #First derivatives: need to calculate them in the log domain if theta is in log domain
    if xscale == "log":
        log_x = np.log(x)
        dfxs_ = np.gradient(fxs, log_x)
        dfsxs = np.gradient(fs, log_x)
    elif xscale == "linear":
        dfxs_ = np.gradient(fxs, x)
        dfsxs = np.gradient(fs, x)
    idx_hat = np.argmin(fxs)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(x, fxs, 'b',  label=r"$\mathcal{F}}$ analytic")
    ax.plot(x, fs, 'g', label=r"$\mathcal{F}$ autodiff")
    ylim = ax.get_ylim()
    ax.vlines(x[idx_hat], 0.99 * ylim[0], 0.99 * ylim[1], 'r',
        alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
    ax.vlines(theta_0, 0.99 * ylim[0], 0.99 * ylim[1], 'k',
        alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
    ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    ax.set_ylabel(r"$\mathcal{F}$")
    ax.legend()
    fig.savefig("bound.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(
        x, dfxs_, 'b--',
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic numeric")
    #ax.set_ylim(ax.get_ylim())
    ax.plot(
        x, gxs, 'b', alpha=0.7,
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic")
    ax.plot(
        x, gs, 'g',
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff")
    ax.plot(
        x, dfsxs, 'g--',
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff numeric")
    # ax.vlines(theta_0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
    # ax.vlines(theta_0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
    #     alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
    # ax.vlines(x[idx_hat], 0.9 * ylim[0], 0.9 * ylim[1], 'r',
    #     alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
    ax.legend()
    fig.savefig("grad.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
 
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
