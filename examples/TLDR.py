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
from mlkernels import Kernel, Matern12, EQ
import pathlib
from probit_jax.utilities import (
    InvalidKernel, check_cutpoints,
    log_probit_likelihood, probit_predictive_distributions)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from probit_jax.approximators import LaplaceGP


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


def generate_data(
        N_train_per_class, N_test_per_class,
        J, D, kernel, noise_variance,
        N_show, jitter=1e-6, seed=None):
    """
    Generate data from the GP prior, and choose some cutpoints that
    approximately divides data into equal bins.

    :arg int N_per_class: The number of data points per class.
    :arg splits:
    :arg int J: The number of bins/classes/quantiles.
    :arg int D: The number of data dimensions.
    :arg kernel: The GP prior.
    :arg noise_variance: The noise variance.
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

    N_train = int(J * N_train_per_class)
    N_test = int(J * N_test_per_class)
    N_total = N_train + N_test
    N_per_class = N_train_per_class + N_test_per_class

    # Sample from the real line, uniformly
    if seed: np.random.seed(seed)  # set seed
    X_data = np.random.uniform(low=0.0, high=1.0, size=(N_total, D))

    # Concatenate X_data and X_show
    X = np.append(X_data, X_show, axis=0)

    # Sample from a multivariate normal
    K = B.dense(kernel(X))
    K = K + jitter * np.identity(np.shape(X)[0])
    L_K = np.linalg.cholesky(K)

    # Generate normal samples for both sets of input data
    if seed: np.random.seed(seed)  # set seed
    z = np.random.normal(loc=0.0, scale=1.0,
        size=np.shape(X_data)[0] + np.shape(X_show)[0])
    f = L_K @ z

    # Store f_show
    f_data = f[:N_total]
    f_show = f[N_total:]

    assert np.shape(f_show) == (np.shape(X_show)[0],)

    # Generate the latent variables
    X = X_data
    f = f_data
    epsilons = np.random.normal(0, np.sqrt(noise_variance), N_total)
    y = epsilons + f
    y = y.flatten()

    # Reshuffle
    data = np.c_[y, X, f]
    np.random.shuffle(data)
    y = data[:, :1].flatten()
    X = data[:, 1:D + 1]
    f = data[:, -1].flatten()

    # split train vs test/validate
    f_train = f[:N_train]
    X_train = X[:N_train]
    y_train = y[:N_train]
    f_test = f[N_train:]
    X_test = X[N_train:]
    y_test = y_j[N_train:]

    assert np.shape(X_test) == (N_test, D)
    assert np.shape(X_train) == (N_train, D)
    assert np.shape(y_test) == (N_test,)
    assert np.shape(y_train) == (N_train,)
    assert np.shape(f) == (N_total,)
    assert np.shape(y) == (N_total,)
    return (
        N_show, X_train, y_train,
        X_test, y_test,
        X_show, f_show)


def calculate_metrics(y_test, predictive_distributions):
    y_pred = np.argmax(predictive_distributions, axis=1)
    grid = np.ogrid[0:len(y_test)]
    predictive_likelihood = predictive_distributions[grid, y_test]
    mean_absolute_error = np.sum(np.abs(y_pred - y_test)) / len(y_test)
    print(np.sum(y_pred != y_test), "sum incorrect")
    print(np.sum(y_pred == y_test), "sum correct")
    print("mean_absolute_error ", mean_absolute_error)
    log_predictive_probability = np.sum(np.log(predictive_likelihood))
    print("log_pred_probability ", log_predictive_probability)
    predictive_likelihood = np.sum(predictive_likelihood) / len(y_test)
    print("predictive_likelihood ", predictive_likelihood)
    mean_zero_one = np.sum(y_pred != y_test) / len(y_test)
    print("mean_zero_one_error", mean_zero_one)
    return (
        mean_zero_one,
        mean_absolute_error,
        log_predictive_probability,
        predictive_likelihood)
 

def main():
    """Make an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
 
    J = 3
    D = 1

    cmap = plt.cm.get_cmap('viridis', J)
    colors = []
    for j in range(J):
        colors.append(cmap((j + 0.5)/J))

    # Generate data
    noise_variance = 0.4
    signal_variance = 1.0
    lengthscale = 1.0
    kernel = signal_variance * Matern12().stretch(lengthscale)

    (N_show, X, g_true, y, cutpoints,
    X_test, y_test,
    X_show, f_show) = generate_data(
        N_train_per_class=10, N_test_per_class=100,
        J=3, D=1, kernel=kernel, noise_variance=noise_variance,
        N_show=1000, jitter=1e-6, seed=None)

    plot_ordinal(
        X, y, g_true, X_show, f_show, J, D, colors, cmap, N_show=N_show) 

    # Initiate a misspecified model, using a kernel
    # other than the one used to generate data
    def prior(prior_parameters):
        # Here you can define the kernel that defines the Gaussian process
        return signal_variance * EQ().stretch(prior_parameters)

    # Test prior
    if not (isinstance(prior(1.0), Kernel)):
        raise InvalidKernel(prior(1.0))

    # check that the cutpoints are in the correct format
    # for the number of classes, J
    cutpoints = check_cutpoints(cutpoints, J)
    print("cutpoints={}".format(cutpoints))

    gaussian_process = LaplaceGP(data=(X, y), prior=prior,
        log_likelihood=log_gaussian_likelihood,
        tolerance=1e-5  # tolerance for the jaxopt fixed-point resolution
    )

    g = classifier.take_grad()

    # Optimize ELBO
    params = ((lengthscale)), (np.sqrt(noise_variance), cutpoints)
    print("\nELBO and gradient of the hyper-parameters:")
    print(g(params))
    fun = lambda x: (
        np.float64(g((((x)), (np.sqrt(noise_variance), cutpoints)))[0]),
        np.float64(g((((x)), (np.sqrt(noise_variance), cutpoints)))[1][0]))
    res = minimize(
        fun, lengthscale,
        method='BFGS', jac=True)
    print("\nOptimization output:")
    print(res)

    theta_0 = lengthscale
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
        params = ((theta)), (np.sqrt(noise_variance), cutpoints)
        fx, gx = g(params)
        fs[i] = fx
        gs[i] = gx[0] * theta  # multiply by a Jacobian

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
    ax.vlines(theta_0, 0.99 * ylim[0], 0.99 * ylim[1], 'k',
        alpha=0.5, label=r"$\theta={:.2f}$".format(theta_0))
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
    ax.vlines(theta_0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
        alpha=0.5, label=r"$\theta={:.2f}$".format(theta_0))
    ax.vlines(np.float64(res.x), 0.9 * ylim[0], 0.9 * ylim[1], 'r',
        alpha=0.5, label=r"$\hat\theta={:.2f}$".format(np.float64(res.x)))
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
    ax.legend()
    fig.savefig("grad.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


    params = ((res.x)), (np.sqrt(noise_variance), cutpoints)

    # Approximate posterior
    weight, precision = classifier.approximate_posterior(params)
    posterior_mean, posterior_variance = classifier.predict(
        X_show,
        params,
        weight, precision)
    # Make predictions
    predictive_distributions = probit_predictive_distributions(
        params[1],
        posterior_mean, posterior_variance)
    plot_contour(X_show, predictive_distributions, posterior_mean,
        posterior_variance, X, y, g_true, J, colors)

    # Evaluate model
    posterior_mean, posterior_variance = classifier.predict(
        X_test,
        params,
        weight, precision)
    predictive_distributions = probit_predictive_distributions(
        params[1],
        posterior_mean, posterior_variance)
    print("\nEvaluation of model:")
    calculate_metrics(y_test, predictive_distributions) 

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
