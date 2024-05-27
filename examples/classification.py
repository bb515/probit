"""Ordinal GP regression via approximate inference."""

# Uncomment to enable double precision
from jax.config import config

config.update("jax_enable_x64", True)

from probit.utilities import (
    InvalidKernel,
    check_cutpoints,
    log_probit_likelihood,
    probit_predictive_distributions,
)
import numpy as np
import lab as B
import jax.numpy as jnp
import jax.random as random
from mlkernels import Kernel, Matern12, EQ
from varz import Vars, minimise_l_bfgs_b
import matplotlib.pyplot as plt
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import pathlib
from jax import vmap, value_and_grad

# For plotting
BG_ALPHA = 1.0
MG_ALPHA = 1.0
FG_ALPHA = 0.3

write_path = pathlib.Path()


def plot(
    x,
    predictive_distributions,
    mean,
    variance,
    X_show,
    f_show,
    X_train,
    y_train,
    g_train,
    J,
    colors,
    fname="plot",
):
    posterior_std = jnp.sqrt(variance)
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel("Cumulative probability", fontsize=10)
    ax.stackplot(
        x[:, 0],
        predictive_distributions.T,
        colors=colors,
        labels=(
            r"$p(y=0|\mathcal{D}, \theta)$",
            r"$p(y=1|\mathcal{D}, \theta)$",
            r"$p(y=2|\mathcal{D}, \theta)$",
        ),
    )
    val = 0.5  # where the data lies on the y-axis.
    plt.tight_layout()
    ax.legend()
    fig.savefig("{}_contour.png".format(fname))
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(x[:, 0], mean, color="blue", label="Prediction", linestyle="--")
    ax.fill_between(
        x[:, 0],
        mean - 2 * posterior_std,
        mean + 2 * posterior_std,
        color="blue",
        alpha=FG_ALPHA,
    )
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.scatter(
        X_train, g_train, color=[colors[i] for i in y_train], s=4, label="Observations"
    )
    ax.plot(X_show, f_show, label="True", color="orange", alpha=FG_ALPHA)
    ax.grid(visible=True, which="major", linestyle="-")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        "{}_mean_variance.png".format(fname),
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close()


def plot_obj(x_hat, x_0, x, fs, gs, domain, xlabel, xscale, fname="plot"):
    # Calculate numerical derivatives wrt domain of plot
    if xscale == "log":
        log_x = jnp.log(x)
        dfsx = np.gradient(fs, log_x)
        gs *= x  # Jacobian
    elif xscale == "linear":
        dfsx = np.gradient(fs, x)
    fig = plt.figure()
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(x, fs, "g", label=r"$\mathcal{F}$ autodiff")
    ylim = ax.get_ylim()
    ax.set_xlim((10 ** domain[0][0], 10 ** domain[0][1]))
    ax.set_ylim(ylim)
    ax.vlines(
        x_hat,
        ylim[0],
        ylim[1],
        "r",
        alpha=MG_ALPHA,
        label=r"$\hat\theta={:.2f}$".format(x_hat),
    )
    ax.vlines(
        x_0, ylim[0], ylim[1], "k", alpha=MG_ALPHA, label=r"$\theta={:.2f}$".format(x_0)
    )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xscale(xscale)
    ax.set_ylabel(r"$\mathcal{F}$", fontsize=10)
    ax.set_title("Hyper-parameter optimisation objective")
    ax.legend()
    fig.savefig("readme_objective.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    fig = plt.figure()
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(
        x,
        gs,
        "g",
        alpha=FG_ALPHA,
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ JAX autodiff",
    )
    ax.set_ylim(ax.get_ylim())
    ax.set_xlim((10 ** domain[0][0], 10 ** domain[0][1]))
    ax.plot(
        x,
        dfsx,
        "g--",
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ numerical",
    )
    ylim = ax.get_ylim()
    ax.vlines(
        x_hat,
        ylim[0],
        ylim[1],
        "r",
        alpha=MG_ALPHA,
        label=r"$\hat\theta={:.2f}$".format(x_hat),
    )
    ax.vlines(
        x_0, ylim[0], ylim[1], "k", alpha=MG_ALPHA, label=r"$\theta={:.2f}$".format(x_0)
    )
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$", fontsize=10)
    ax.set_title("Hyper-parameter optimisation gradient")
    ax.legend()
    fig.savefig("readme_grad.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def generate_data(
    key,
    N_train_per_class,
    N_test_per_class,
    J,
    kernel,
    noise_variance,
    N_show,
    jitter=1e-6,
):
    """
    Generate data from the GP prior, and choose some cutpoints that
    approximately divides data into equal bins.

    :arg key: JAX random key, a random seed.
    :arg int N_train_per_class: The number of data points per class.
    :arg int N_test_per_class: The number of data points per class.
    :arg int J: The number of bins/classes/quantiles.
    :arg kernel: The GP prior.
    :arg noise_variance: The noise variance.
    :arg int N_show: The number of data points to plot.
    :arg jitter: For making Gram matrix better conditioned,
        so that Cholesky decomposition can be performed.
        Try 1e-5 for single precision or 1e-10 for double
        precision.
    """
    # Generate input data from a linear grid
    X_show = jnp.linspace(-0.5, 1.5, N_show)
    # reshape X to make it (n, 1)
    X_show = X_show[:, None]

    N_train = int(J * N_train_per_class)
    N_test = int(J * N_test_per_class)
    N_total = N_train + N_test
    N_per_class = N_train_per_class + N_test_per_class

    # Sample from the real line, uniformly
    key, step_key = random.split(key, 2)
    X_data = random.uniform(key, minval=0.0, maxval=1.0, shape=(N_total, 1))

    # Concatenate X_data and X_show
    X = jnp.append(X_data, X_show, axis=0)

    # Sample from a multivariate normal
    K = B.dense(kernel(X))
    K = K + jitter * jnp.identity(jnp.shape(X)[0])
    L_K = jnp.linalg.cholesky(K)

    # Generate normal samples for both sets of input data
    key, step_key = random.split(key, 2)
    z = random.normal(key, shape=(jnp.shape(X_data)[0] + jnp.shape(X_show)[0],))
    f = L_K @ z

    # Store f_show
    f_data = f[:N_total]
    f_show = f[N_total:]

    assert jnp.shape(f_show) == (jnp.shape(X_show)[0],)

    # Generate the latent variables
    X = X_data
    f = f_data
    key, step_key = random.split(key, 2)
    epsilons = jnp.sqrt(noise_variance) * random.normal(key, shape=(N_total,))
    g = epsilons + f
    g = g.flatten()

    # Sort the responses
    idx_sorted = jnp.argsort(g)
    g = g[idx_sorted]
    f = f[idx_sorted]
    X = X[idx_sorted]
    X_js = []
    g_js = []
    f_js = []
    y_js = []
    cutpoints = jnp.empty(J + 1)
    for j in range(J):
        X_js.append(X[N_per_class * j : N_per_class * (j + 1), :1])
        g_js.append(g[N_per_class * j : N_per_class * (j + 1)])
        f_js.append(f[N_per_class * j : N_per_class * (j + 1)])
        y_js.append(j * jnp.ones(N_per_class, dtype=int))

    for j in range(1, J):
        # Find the cutpoints
        cutpoint_j_min = g_js[j - 1][-1]
        cutpoint_j_max = g_js[j][0]
        cutpoints = cutpoints.at[j].set(
            jnp.mean(jnp.array([cutpoint_j_max, cutpoint_j_min]))
        )
    cutpoints = cutpoints.at[0].set(-jnp.inf)
    cutpoints = cutpoints.at[-1].set(jnp.inf)
    X_js = jnp.array(X_js)
    g_js = jnp.array(g_js)
    f_js = jnp.array(f_js)
    y_js = jnp.array(y_js, dtype=int)
    X = X.reshape(-1, X.shape[-1])
    g = g_js.flatten()
    f = f_js.flatten()
    y = y_js.flatten()
    y = jnp.array(y, dtype=int)

    g_train_js = []
    X_train_js = []
    y_train_js = []
    X_test_js = []
    y_test_js = []
    for j in range(J):
        data = jnp.c_[g_js[j], X_js[j], y_js[j]]
        key, step_key = random.split(key, 2)
        random.shuffle(key, data)
        g_j = data[:, :1]
        X_j = data[:, 1 : 1 + 1]
        y_j = data[:, -1]
        # split train vs test/validate
        g_train_j = g_j[:N_train_per_class]
        X_train_j = X_j[:N_train_per_class]
        y_train_j = y_j[:N_train_per_class]
        X_j = X_j[N_train_per_class:]
        y_j = y_j[N_train_per_class:]
        X_train_js.append(X_train_j)
        g_train_js.append(g_train_j)
        y_train_js.append(y_train_j)
        X_test_js.append(X_j)
        y_test_js.append(y_j)

    X_train_js = jnp.array(X_train_js)
    g_train_js = jnp.array(g_train_js)
    y_train_js = jnp.array(y_train_js, dtype=int)
    X_test_js = jnp.array(X_test_js)
    y_test_js = jnp.array(y_test_js, dtype=int)

    X_train = X_train_js.reshape(-1, X_train_js.shape[-1])
    g_train = g_train_js.flatten()
    y_train = y_train_js.flatten()
    X_test = X_test_js.reshape(-1, X_test_js.shape[-1])
    y_test = y_test_js.flatten()

    y_train = jnp.array(y_train, dtype=int)
    y_test = jnp.array(y_test, dtype=int)

    return (N_show, X, g, y, cutpoints, X_test, y_test, X_show, f_show)


def calculate_metrics(y_test, predictive_distributions):
    y_pred = jnp.argmax(predictive_distributions, axis=1)
    predictive_likelihood = predictive_distributions[:, y_test]
    mean_absolute_error = jnp.sum(jnp.abs(y_pred - y_test)) / len(y_test)
    print(jnp.sum(y_pred != y_test), "sum incorrect")
    print(jnp.sum(y_pred == y_test), "sum correct")
    print("mean_absolute_error={:.2f}".format(mean_absolute_error))
    log_predictive_probability = jnp.sum(jnp.log(predictive_likelihood))
    print("log_pred_probability={:.2f}".format(log_predictive_probability))
    mean_zero_one = jnp.sum(y_pred != y_test) / len(y_test)
    print("mean_zero_one_error={:.2f}".format(mean_zero_one))
    return (
        mean_zero_one,
        mean_absolute_error,
        log_predictive_probability,
        predictive_likelihood,
    )


def main():
    """Make an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument("--profile", action="store_const", const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    approximate_inference_method = "Laplace"
    if approximate_inference_method == "Variational Bayes":
        from probit.approximators import VBGP as Approximator
    elif approximate_inference_method == "Laplace":
        from probit.approximators import LaplaceGP as Approximator

    J = 3
    cmap = plt.cm.get_cmap("PiYG", J)
    colors = []
    mapping = []
    for j in range(J):
        colors.append(cmap((j + 0.5) / J))
        mapping.append((j + 0.5) / J)
    print(colors)
    print(mapping)

    # Generate data
    key = random.PRNGKey(1)
    noise_variance = 0.4
    signal_variance = 1.0
    lengthscale = 1.0
    kernel = signal_variance * Matern12().stretch(lengthscale)
    (N_show, X, g_true, y, cutpoints, X_test, y_test, X_show, f_show) = generate_data(
        key,
        N_train_per_class=10,
        N_test_per_class=100,
        J=J,
        kernel=kernel,
        noise_variance=noise_variance,
        N_show=1000,
        jitter=1e-6,
    )

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

    classifier = Approximator(
        data=(X, y),
        prior=prior,
        log_likelihood=log_probit_likelihood,
        # grad_log_likelihood=grad_log_probit_likelihood,
        # hessian_log_likelihood=hessian_log_probit_likelihood,
        tolerance=1e-5,  # tolerance for the jaxopt fixed-point resolution
    )
    negative_evidence_lower_bound = classifier.objective()

    vs = Vars(jnp.float32)

    def model(vs):
        p = vs.struct
        noise_std = jnp.sqrt(noise_variance)
        return (p.lengthscale.positive(1.2)), (noise_std, cutpoints)

    def objective(vs):
        return negative_evidence_lower_bound(model(vs))

    # Approximate posterior
    parameters = model(vs)
    weight, precision = classifier.approximate_posterior(parameters)
    mean, variance = classifier.predict(X_show, parameters, weight, precision)
    obs_variance = variance + noise_variance
    predictive_distributions = probit_predictive_distributions(
        parameters[1], mean, variance
    )
    plot(
        X_show,
        predictive_distributions,
        mean,
        obs_variance,
        X_show,
        f_show,
        X,
        y,
        g_true,
        J,
        colors,
        fname="readme_classification_before",
    )

    # Evaluate model
    mean, variance = classifier.predict(X_test, parameters, weight, precision)
    predictive_distributions = probit_predictive_distributions(
        parameters[1], mean, variance
    )
    print("\nEvaluation of model:")
    calculate_metrics(y_test, predictive_distributions)

    print("Before optimization, \nparameters={}".format(parameters))
    minimise_l_bfgs_b(objective, vs)
    parameters = model(vs)
    print("After optimization, \nparameters={}".format(model(vs)))

    # Approximate posterior
    parameters = model(vs)
    weight, precision = classifier.approximate_posterior(parameters)
    mean, variance = classifier.predict(X_show, parameters, weight, precision)
    predictive_distributions = probit_predictive_distributions(
        parameters[1], mean, variance
    )
    plot(
        X_show,
        predictive_distributions,
        mean,
        obs_variance,
        X_show,
        f_show,
        X,
        y,
        g_true,
        J,
        colors,
        fname="readme_classification_after",
    )

    # Evaluate model
    mean, variance = classifier.predict(X_test, parameters, weight, precision)
    obs_variance = variance + noise_variance
    predictive_distributions = probit_predictive_distributions(
        parameters[1], mean, variance
    )
    print("\nEvaluation of model:")
    calculate_metrics(y_test, predictive_distributions)

    nelbo = lambda x: negative_evidence_lower_bound(
        ((x), (jnp.sqrt(noise_variance), cutpoints))
    )
    fg = vmap(value_and_grad(nelbo))

    domain = ((-2, 2), None)
    resolution = (50, None)
    x = jnp.logspace(domain[0][0], domain[0][1], resolution[0])
    xlabel = r"lengthscale, $\ell$"
    xscale = "log"
    phis = jnp.log(x)

    fgs = fg(x)
    fs = fgs[0]
    gs = fgs[1]
    plot_obj(vs.struct.lengthscale(), lengthscale, x, fs, gs, domain, xlabel, xscale)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(0.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
