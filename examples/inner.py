"""Compare the inner loop of probit vs probit_jax."""
# Enable double precision
from jax.config import config
config.update("jax_enable_x64", True)
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
import pathlib
from probit.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
from mlkernels import Kernel as BaseKernel
from probit_jax.utilities import (
    InvalidKernel, check_cutpoints,
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


def plot_helper(
        resolution, domain, trainables=["lengthscale"]):
    """
    Initiate metadata and hyperparameters for plotting the objective
    function surface over hyperparameters.

    :arg int resolution:
    :arg domain: ((2,)tuple, (2.)tuple) of the start/stop in the domain to
        grid over, for each axis, like: ((start, stop), (start, stop)).
    :type domain:
    :arg trainables: Indicator array of the hyperparameters to sample over.
    :type trainables: :class:`numpy.ndarray`
    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    """
    index = 0
    label = []
    axis_scale = []
    space = []
    phi_space = [] 
    if "noise_std" in trainables:
        # Grid over noise_std
        label.append(r"$\sigma$")
        axis_scale.append("log")
        theta = np.logspace(
            domain[index][0], domain[index][1], resolution[index])
        space.append(theta)
        phi_space.append(np.log(theta))
        index += 1
    if "cutpoints" in trainables:
        # Grid over b_1, the first cutpoint
        label.append(r"$b_{}$".format(1))
        axis_scale.append("linear")
        theta = np.linspace(
            domain[index][0], domain[index][1], resolution[index])
        space.append(theta)
        phi_space.append(theta)
        index += 1
    if "signal_variance" in trainables:
        # Grid over signal variance
        label.append(r"$\sigma_{\theta}^{ 2}$")
        axis_scale.append("log")
        theta = np.logspace(
            domain[index][0], domain[index][1], resolution[index])
        space.append(theta)
        phi_space.append(np.log(theta))
        index += 1
    else:
        if "lengthscale" in trainables:
            # Grid over only kernel hyperparameter, theta
            label.append(r"$\theta$")
            axis_scale.append("log")
            theta = np.logspace(
                domain[index][0], domain[index][1], resolution[index])
            space.append(theta)
            phi_space.append(np.log(theta))
            index +=1
    if index == 2:
        meshgrid_theta = np.meshgrid(space[0], space[1])
        meshgrid_phi = np.meshgrid(phi_space[0], phi_space[1])
        phis = np.dstack(meshgrid_phi)
        phis = phis.reshape((len(space[0]) * len(space[1]), 2))
        theta_0 = np.array(theta_0)
    elif index == 1:
        meshgrid_theta = (space[0], None)
        space.append(None)
        phi_space.append(None)
        axis_scale.append(None)
        label.append(None)
        phis = phi_space[0].reshape(-1, 1)
    else:
        raise ValueError(
            "Too few or too many independent variables to plot objective over!"
            " (got {}, expected {})".format(
            index, "1, or 2"))
    assert len(axis_scale) == 2
    assert len(meshgrid_theta) == 2
    assert len(space) ==  2
    assert len(label) == 2
    return (
        space[0], space[1],
        label[0], label[1],
        axis_scale[0], axis_scale[1],
        meshgrid_theta[0], meshgrid_theta[1],
        phis)


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

    classifier = Approximator((X, y), prior, log_probit_likelihood,
        # grad_log_likelihood = lambda f, y, lp: grad_log_probit_likelihood(f, y, lp, single_precision=False),
        # hessian_log_likelihood = lambda f, y, lp: hessian_log_probit_likelihood(f, y, lp, single_precision=False),
        tolerance=1e-5)

    # Initiate data and classifier for probit repo
    dataset = "SEIso"
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

    params = ((jnp.sqrt(1./(2 * theta_0))), (jnp.sqrt(noise_variance_0), cutpoints_0))
    fx, gx = g(params)

    weight, precision = classifier.approximate_posterior(params)
    posterior_mean, posterior_variance = classifier.predict(
        X,
        params,
        weight, precision)

    print(posterior_mean)
    print(fx)
    gx = gx[0] * (- 0.5 * (2 * theta_0)**(-1./2))  # multiply by the lengthscale Jacobian
    print(gx)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    fx, gx, weights, _posterior_mean, cov = _classifier.approximate_posterior(
        np.array([np.log(theta_0)]), trainables, steps, return_reparameterised=True, verbose=True)
    print(_posterior_mean)
    print(fx)
    print(gx)
    plt.scatter(X, _posterior_mean, marker='X', label="probit", alpha=0.3)
    plt.scatter(X, posterior_mean, label="probit_jax", alpha=0.3)
    plt.legend()
    plt.savefig("compare.png")
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
