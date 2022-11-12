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
    #sys.stdout = open("{}.txt".format(now), "w")
 
    # Initiate data and classifier for probit_jax repo
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
        single_precision=True,
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
            kernel=kernel, J=J, data=(X, y), single_precision=True)

    # Notes: fwd_solver, newton_solver work, anderson solver has bug with vmap ValueError

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
    # theta domain and resolution
    domain = ((-1, 2), None)
    res = (30, None)

    (x1s, x2s,
    xlabel, ylabel,
    xscale, yscale,
    xx, yy,
    phis, fxs,
    gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
        _classifier, res, domain, trainables)

    gs = np.empty(res[0])
    fs = np.empty(res[0])

    params = ((jnp.sqrt(1./(2 * theta_0))), (jnp.sqrt(noise_variance_0), cutpoints_0))
    fx, gx = g(params)

    jposterior_mean = classifier.approximate_posterior(params)
    print(jposterior_mean)
    print(fx)
    print(gx)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    fx, gx, weights, posterior_mean, cov = _classifier.approximate_posterior(
        np.array([np.log(theta_0)]), trainables, steps, return_reparameterised=True, verbose=True)
    plt.scatter(X, posterior_mean, label="probit", alpha=0.3)
    plt.scatter(X, jposterior_mean, label="probit_jax", alpha=0.3)
    plt.legend()
    plt.savefig("weights.png")
    plt.close()

    (fxs, gxs,
    x, y,
    xlabel, ylabel,
    xscale, yscale) = (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    #Numerical derivatives: need to calculate them in the log domain if theta is in log domain
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
