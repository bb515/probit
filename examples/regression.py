"""GP regression."""

# Uncomment to enable double precision
from jax.config import config

config.update("jax_enable_x64", True)

from probit.utilities import log_gaussian_likelihood
from probit.approximators import LaplaceGP as GP
import lab as B
import jax.numpy as jnp
import jax.random as random
from mlkernels import EQ
from varz import Vars, minimise_l_bfgs_b
import matplotlib.pyplot as plt
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey


# For plotting
BG_ALPHA = 1.0
MG_ALPHA = 1.0
FG_ALPHA = 0.3


def generate_data(key, N_train, kernel, noise_std, N_show, jitter=1e-10):
    """
    Generate data from the GP prior.

    :arg key: JAX random key, a random seed.
    :arg int N_train: The number of data points.
    :arg kernel: The GP prior.
    :arg noise_std: The noise standard deviation.
    :arg int N_show: The number of data points to plot.
    :arg jitter: For making Gram matrix better conditioned,
        so that Cholesky decomposition can be performed.
        Try 1e-5 for single precision or 1e-10 for double
        precision.
    """
    # Generate input data from a linear grid
    X_show = jnp.linspace(-0.5, 1.5, N_show)
    X_show = X_show[:, None]  # (N, 1)

    # Sample from the real line, uniformly
    key, step_key = random.split(key, 2)
    X_train = random.uniform(key, minval=0.0, maxval=1.0, shape=(N_train, 1))

    # Concatenate X_train and X_show
    X = jnp.append(X_train, X_show, axis=0)

    # Sample from a multivariate normal
    K = B.dense(kernel(X))
    K = K + jitter * jnp.identity(jnp.shape(X)[0])
    L_K = jnp.linalg.cholesky(K)

    # Generate normal samples for both sets of input data
    key, step_key = random.split(key, 2)
    z = random.normal(key, shape=(X_train.shape[0] + X_show.shape[0],))
    f = L_K @ z

    # Store f_show
    f_train = f[:N_train]
    f_show = f[N_train:]

    # Generate the latent variables
    key, step_key = random.split(key, 2)
    epsilons = noise_std * random.normal(key, shape=(N_train,))
    y_train = epsilons + f_train
    y_train = y_train.flatten()

    # Reshuffle
    key, step_key = random.split(key, 2)
    data = jnp.c_[y_train, X_train]
    random.shuffle(key, data)
    y_train = data[:, :1].flatten()
    X_train = data[:, 1:]

    return (X_train, y_train, X_show, f_show, N_show)


def plot(train_data, test_data, mean, variance, fname="plot.png"):
    X, y = train_data
    X_show, f_show = test_data
    # Plot result
    fig, ax = plt.subplots(1, 1)
    ax.plot(X_show, f_show, label="True", color="orange")
    ax.plot(X_show, mean, label="Prediction", linestyle="--", color="blue")
    ax.scatter(X, y, label="Observations", color="black", s=20)
    ax.fill_between(
        X_show.flatten(),
        mean - 2.0 * jnp.sqrt(variance),
        mean + 2.0 * jnp.sqrt(variance),
        alpha=FG_ALPHA,
        color="blue",
    )
    ax.set_xlim((0.0, 1.0))
    ax.grid(visible=True, which="major", linestyle="-")
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(BG_ALPHA)
    ax.patch.set_alpha(MG_ALPHA)
    ax.legend()
    fig.savefig(fname)
    plt.close()


def main():
    """Make an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument("--profile", action="store_const", const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    def prior(prior_parameters):
        lengthscale, signal_variance = prior_parameters
        # Here you can define the kernel that defines the Gaussian process
        return signal_variance * EQ().stretch(lengthscale).periodic(0.5)

    # Generate data
    key = random.PRNGKey(0)
    noise_std = 0.2
    (X, y, X_show, f_show, N_show) = generate_data(
        key, N_train=20, kernel=prior((1.0, 1.0)), noise_std=noise_std, N_show=1000
    )

    gaussian_process = GP(
        data=(X, y), prior=prior, log_likelihood=log_gaussian_likelihood
    )
    negative_evidence = gaussian_process.objective()

    vs = Vars(jnp.float32)

    def model(vs):
        p = vs.struct
        return (p.lengthscale.positive(), p.signal_variance.positive()), (
            p.noise_std.positive(),
        )

    def objective(vs):
        return negative_evidence(model(vs))

    # Approximate posterior
    parameters = model(vs)
    weight, precision = gaussian_process.approximate_posterior(parameters)
    mean, variance = gaussian_process.predict(X_show, parameters, weight, precision)
    noise_variance = vs.struct.noise_std() ** 2
    obs_variance = variance + noise_variance
    plot((X, y), (X_show, f_show), mean, variance, fname="readme_regression_before.png")

    print("Before optimization, \nparams={}".format(parameters))
    minimise_l_bfgs_b(objective, vs)
    parameters = model(vs)
    print("After optimization, \nparams={}".format(model(vs)))

    # Approximate posterior
    weight, precision = gaussian_process.approximate_posterior(parameters)
    mean, variance = gaussian_process.predict(X_show, parameters, weight, precision)
    noise_variance = vs.struct.noise_std() ** 2
    obs_variance = variance + noise_variance
    plot(
        (X, y),
        (X_show, f_show),
        mean,
        obs_variance,
        fname="readme_regression_after.png",
    )

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(0.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
