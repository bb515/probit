"""GP regression."""
# Uncomment to enable double precision
# from jax.config import config
# config.update("jax_enable_x64", True)
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
import lab as B
from mlkernels import Kernel, EQ
import pathlib
from probit_jax.utilities import (
    InvalidKernel,
    log_gaussian_likelihood)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from probit_jax.approximators import LaplaceGP as GP


# For plotting
BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4

write_path = pathlib.Path()


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


def generate_data(
        N_train,
        D, kernel, noise_std,
        N_show, jitter=1e-6, seed=None):
    """
    Generate data from the GP prior.

    :arg int N_per_class: The number of data points per class.
    :arg splits:
    :arg int J: The number of bins/classes/quantiles.
    :arg int D: The number of data dimensions.
    :arg kernel: The GP prior.
    :arg noise_std: The noise standard deviation.
    """
    if D==1:
        # Generate input data from a linear grid
        X_show = np.linspace(-0.5, 1.5, N_show)
        # reshape X to make it (n, D)
        X_show = X_show[:, None]
    elif D==2:
        # Generate input data from a linear meshgrid
        x = np.linspace(-0.5, 1.5, N_show)
        y = np.linspace(-0.5, 1.5, N_show)
        xx, yy = np.meshgrid(x, y)
        # Pairs
        X_show = np.dstack([xx, yy]).reshape(-1, 2)

    # Sample from the real line, uniformly
    if seed: np.random.seed(seed)  # set seed
    X_train = np.random.uniform(low=0.0, high=1.0, size=(N_train, D))

    # Concatenate X_train and X_show
    X = np.append(X_train, X_show, axis=0)

    # Sample from a multivariate normal
    K = B.dense(kernel(X))
    K = K + jitter * np.identity(np.shape(X)[0])
    L_K = np.linalg.cholesky(K)

    # Generate normal samples for both sets of input data
    if seed: np.random.seed(seed)  # set seed
    z = np.random.normal(loc=0.0, scale=1.0,
        size=X_train.shape[0] + X_show.shape[0])
    f = L_K @ z

    # Store f_show
    f_train = f[:N_train]
    f_show = f[N_train:]

    assert np.shape(f_show) == (np.shape(X_show)[0],)

    # Generate the latent variables
    epsilons = np.random.normal(0, noise_std, N_train)
    y_train = epsilons + f_train
    y_train = y_train.flatten()

    # Reshuffle
    data = np.c_[y_train, X_train, f_train]
    np.random.shuffle(data)
    y_train = data[:, :1].flatten()
    X_train = data[:, 1:D + 1]

    return (
        N_show, X_train, y_train,
        X_show, f_show)


def main():
    """Make an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
 
    # Initiate a misspecified model, using a kernel
    # other than the one used to generate data
    def prior(prior_parameters):
        lengthscale, signal_variance = prior_parameters
        # Here you can define the kernel that defines the Gaussian process
        return signal_variance * EQ().stretch(lengthscale).periodic(0.5)

    # Generate data
    noise_std = 0.2
    (N_show, X, y,
    X_show, f_show) = generate_data(
        N_train=20,
        D=1, kernel=prior((1.0, 1.0)), noise_std=noise_std,
        N_show=1000, jitter=1e-8, seed=None)

    gaussian_process = GP(data=(X, y), prior=prior,
        log_likelihood=log_gaussian_likelihood)

    g = gaussian_process.take_grad()
    g_lengthscale = lambda theta: (g(((theta, 1.0), (noise_std,))))

    # Optimize ELBO
    fun = lambda x: (
        g_lengthscale(x)[0],
        g_lengthscale(x)[1][0][0])
    res = minimize(
        fun, 1.0,
        method='BFGS', jac=True)
    print("\nOptimization output:")
    print(res)

    domain = ((-2, 2), None)
    resolution = (50, None)
    (x, _,
    xlabel, _,
    xscale, _,
    _, _,
    phis) = plot_helper(
        resolution, domain)
    gs = np.empty(resolution[0])
    fs = np.empty(resolution[0])
    for i, phi in enumerate(phis):
        theta = np.exp(phi)[0]
        fx, gx = g_lengthscale(theta)
        fs[i] = fx
        gs[i] = gx[0][0] * theta  # multiply by a Jacobian

    # Calculate numerical derivatives wrt domain of plot
    if xscale == "log":
        log_x = np.log(x)
        dfsxs = np.gradient(fs, log_x)
    elif xscale == "linear":
        dfsxs = np.gradient(fs, x)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(x, fs, 'g', label=r"$\mathcal{F}$ autodiff")
    ylim = ax.get_ylim()
    ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
    ax.vlines(np.float64(res.x), 0.99 * ylim[0], 0.99 * ylim[1], 'r',
        alpha=0.5, label=r"$\hat\theta={:.2f}$".format(np.float64(res.x)))
    ax.vlines(1.0, 0.99 * ylim[0], 0.99 * ylim[1], 'k',
        alpha=0.5, label=r"$\theta={}$".format(1.0))
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
        x, gs, 'g', alpha=0.4,
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ JAX autodiff")
    ax.set_ylim(ax.get_ylim())
    ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
    ax.plot(
        x, dfsxs, 'g--',
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ numerical")
    ax.vlines(1.0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
        alpha=0.5, label=r"$\theta={}$".format(1.0))
    ax.vlines(np.float64(res.x), 0.9 * ylim[0], 0.9 * ylim[1], 'r',
        alpha=0.5, label=r"$\hat\theta={:.2f}$".format(np.float64(res.x)))
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
    ax.legend()
    fig.savefig("grad.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


    params = (res.x, 1.0), (noise_std,)

    # Approximate posterior
    weight, precision = gaussian_process.approximate_posterior(params)
    mean, variance = gaussian_process.predict(
        X_show,
        params,
        weight, precision)

    # Plot result.
    plt.plot(X_show, f_show, label="True", color="orange")
    plt.plot(X_show, mean, label="Prediction", linestyle="--", color="blue")
    plt.scatter(X, y, label="Observations", color="black", s=20)
    plt.fill_between(X_show.flatten(), mean - 2. * np.sqrt(variance + noise_std**2), mean + 2. * np.sqrt(variance + noise_std**2), alpha=0.3, color="blue")
    plt.xlim((0.0, 1.0))
    plt.legend()
    plt.grid()
    plt.savefig("readme_example1_simple_regression.png")

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
