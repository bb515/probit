"""GP regression."""
# Uncomment to enable double precision
from jax.config import config
config.update("jax_enable_x64", True)
from probit_jax.utilities import log_gaussian_likelihood
from probit_jax.approximators import LaplaceGP as GP
import lab as B
import jax.numpy as jnp
import jax.random as random
from mlkernels import EQ
import matplotlib.pyplot as plt
from varz import Vars, minimise_l_bfgs_b, parametrised, Positive
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey

from jax import grad, jit, vmap
import seaborn as sns
sns.set_style("darkgrid")
cm = sns.color_palette("mako_r", as_cmap=True)

# For plotting
BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4

def plot_kernel_heatmap(function, res, area_min=-2, area_max=2, fname="plot_kernel"):
    # this helper function is here so that we can jit it.
    # We can not jit the whole function since plt.quiver cannot
    # be jitted

    # @partial(jit, static_argnums=[0,])
    def helper(area_min, area_max):
        x = jnp.linspace(area_min, area_max, res)
        x, y = jnp.meshgrid(x, x)
        grid = jnp.stack([x.flatten(), y.flatten()], axis=1)
        K = function(grid)
        return grid, K

    grid, hm = helper(area_min, area_max)
    x_0 = jnp.argmin(hm)
    x_0 = grid[x_0]
    print(jnp.exp(x_0), "min")
    hm = hm.reshape(res, res)
    extent = [area_min, area_max, area_max, area_min]
    plt.scatter(x_0[0], x_0[1])
    plt.imshow(hm, cmap=cm, interpolation='nearest', extent=extent)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig(fname)
    plt.close()

def generate_data(
        key,
        N_train,
        kernel, noise_std,
        N_show, jitter=1e-10):
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
    z = random.normal(key,
        shape=(X_train.shape[0] + X_show.shape[0],))
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

    return (
        X_train, y_train,
        X_show, f_show, N_show)


def plot(train_data, test_data, mean, variance,
         fname="plot.png"):
    X, y = train_data
    X_show, f_show = test_data
    # Plot result
    plt.plot(X_show, f_show, label="True", color="orange")
    plt.plot(X_show, mean, label="Prediction", linestyle="--", color="blue")
    plt.scatter(X, y, label="Observations", color="black", s=20)
    plt.fill_between(
        X_show.flatten(), mean - 2. * jnp.sqrt(variance),
        mean + 2. * jnp.sqrt(variance), alpha=0.3, color="blue")
    plt.xlim((0.0, 1.0))
    plt.legend()
    plt.grid()
    plt.savefig(fname)
    plt.close()


def main():
    """Make an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
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
    (X, y,
     X_show, f_show, N_show) = generate_data(
         key,
         N_train=20,
         kernel=prior((1.0, 1.0)), noise_std=noise_std,
         N_show=1000)

    gaussian_process = GP(data=(X, y), prior=prior,
        log_likelihood=log_gaussian_likelihood)
    evidence = gaussian_process.objective()

    vs = Vars(jnp.float32)

    def model(vs):
        p = vs.struct
        return (p.lengthscale.positive(), p.signal_variance.positive()), (p.noise_std.positive(),)

    def objective(vs):
        return evidence(model(vs))

    # Approximate posterior
    parameters = model(vs)
    weight, precision = gaussian_process.approximate_posterior(parameters)
    mean, variance = gaussian_process.predict(
        X_show,
        parameters,
        weight, precision)
    noise_variance = parameters[1][0]**2
    obs_variance = variance + noise_variance
    plot((X, y), (X_show, f_show), mean, variance, fname="readme_simple_regression_before.png")

    print("Before optimization, \nparams={}".format(parameters))
    minimise_l_bfgs_b(objective, vs)
    parameters = model(vs)
    print("After optimization, \nparams={}".format(model(vs)))
    g = grad(evidence)
    print(g(model(vs)))
    print(parameters)
    x = [0.0, 0.0]
    e = lambda x: evidence(((jnp.exp(x[0]), jnp.exp(x[1])), parameters[1]))
    e = vmap(e)
    plot_kernel_heatmap(e, res=64)
    assert 0

    # Approximate posterior
    weight, precision = gaussian_process.approximate_posterior(parameters)
    mean, variance = gaussian_process.predict(
        X_show,
        parameters,
        weight, precision)
    noise_variance = parameters[1][0]**2
    obs_variance = variance + noise_variance
    plot((X, y), (X_show, f_show), mean, obs_variance, fname="readme_simple_regression_after.png")
    assert 0

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()

