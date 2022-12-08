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
from probit_jax.plot import outer_loops, grid_synthetic, grid, plot_synthetic, plot, train, test
from probit_jax.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
from mlkernels import Kernel as BaseKernel
from probit_jax.utilities import InvalidKernel, check_cutpoints
from probit_jax.implicit.utilities import probit_likelihood, log_probit_likelihood
import sys
import time
import matplotlib.pyplot as plt
## Temp
import jax
import jax.numpy as jnp
from jax import vmap, grad, jit
import lab as B


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

    scalar_likelihood = lambda f, y, params: probit_likelihood(f, y, params)
    scalar_log_likelihood = lambda f, y, params: log_probit_likelihood(f, y, params)
    grad_scalar_log_likelihood = grad(scalar_log_likelihood)
    hessian_scalar_log_likelihood = grad(lambda f, y, params: grad(scalar_log_likelihood)(f, y, params))

    likelihood= vmap(scalar_likelihood, in_axes=(0, 0, None), out_axes=(0))
    log_likelihood= vmap(scalar_log_likelihood, in_axes=(0, 0, None), out_axes=(0))
    grad_log_likelihood = vmap(grad_scalar_log_likelihood, in_axes=(0, 0, None), out_axes=(0))
    hessian_log_likelihood = vmap(hessian_scalar_log_likelihood, in_axes=(0, 0, None), out_axes=(0))
    fs = jnp.linspace(0.0, 5.0, 50)
    ys = jnp.ones(50, dtype=int) * 1
    ps = log_likelihood(fs, ys, [jnp.sqrt(noise_variance_0), cutpoints_0])
    gs = grad_log_likelihood(fs, ys, [jnp.sqrt(noise_variance_0), cutpoints_0])
    hs = hessian_log_likelihood(fs, ys, [jnp.sqrt(noise_variance_0), cutpoints_0])

    # def objective(likelihood_params, log_likelihood, fs, ys):
    #     return jnp.sum(log_likelihood(fs, ys, likelihood_params))
    # # This seems to work, now just need to stitch it all together.
    # fx_gx = jax.value_and_grad(lambda theta: objective(theta, log_likelihood, fs, ys))
    # print(fx_gx([jnp.sqrt(noise_variance_0), cutpoints_0]))  # unforunately, gradient will return `nan` here when y is not all 1s.

    import matplotlib.pyplot as plt
    plt.plot(fs, ps)
    plt.plot(fs, gs)
    plt.plot(fs, hs)
    plt.savefig("testgrads.png")
    plt.close()

    classifier = Approximator(prior, scalar_log_likelihood, data=(X, y))

    # trainables are defined implicitly by the arguments to probit_likelihood_scalar and prior
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

    prior_parameters_0 = (jnp.sqrt(1./(2 * theta_0)))
    likelihood_parameters_0 = (jnp.sqrt(noise_variance_0), cutpoints_0)
    print(prior_parameters_0)
    z_0 = B.zeros(classifier.N)
    z = jnp.array(B.dense(f((prior_parameters_0, likelihood_parameters_0), z_0))).flatten()
    z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    z = B.dense(f((prior_parameters_0, likelihood_parameters_0), z))
    K = B.dense(classifier.prior(prior_parameters_0)(X))
    posterior_mean = K @ z 
    plt.scatter(X, posterior_mean)
    plt.savefig("testlatent")
    plt.close()
    assert 0

    # z_star = 0
    # for i in range(100):
    #     z_prev = z_star
    #     z_star = f(1.0, z_star)
    #     print(np.linalg.norm(z_star - z_prev))
    # TODO: not sure why in their example can just initiate to any parameters here.
    g = classifier.take_grad()
    # g = classifier.take_grad((prior_parameters_0, likelihood_parameters_0))
    print(g((prior_parameters_0, likelihood_parameters_0)))
    N = 40
    thetas = np.logspace(-1, 2, N)
    gs = np.empty(N)
    fs = np.empty(N)
    for i, theta in enumerate(thetas):
        fx, gx = g(((jnp.sqrt(1./(2 * theta))), ((jnp.sqrt(noise_variance_0), cutpoints_0))))
        fs[i] = fx
        gs[i] = gx[0]
        print(gx[0])
        print(gx[1][0])
        print(gx[1][1])
    plt.plot(thetas, fs)
    plt.xscale("log")
    plt.savefig("testfx.png")
    plt.close()
    print(gs)
    plt.plot(thetas, gs)
    plt.xscale("log")
    plt.savefig("testgx.png")
    plt.close()

    assert 0

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
