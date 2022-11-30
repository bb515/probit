"""Ordinal regression concrete examples. Approximate inference."""
# Enable double precision
from jax.config import config
config.update("jax_enable_x64", True)

# TODO delete this
# Make sure to limit CPU usage
import os
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = "6"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION "] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
import lab as B
import pathlib
from probit_jax.plot import (
    _grid_over_hyperparameters_initiate)
from probit.data.utilities import datasets as datasets_
from probit.data.utilities import load_data as load_data_
from probit.data.utilities import load_data_synthetic as load_data_synthetic_
from probit.data.utilities import load_data_paper as load_data_paper_
from probit.data.utilities import datasets as datasets_
from probit_jax.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
from probit_jax.utilities import InvalidKernel, check_cutpoints
from probit_jax.implicit.utilities import (
    log_probit_likelihood, grad_log_probit_likelihood, hessian_log_probit_likelihood,
    posterior_covariance, predict_reparameterised, matrix_inverse, posterior_covariance)
import sys
import time
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm


BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4


now = time.ctime()
write_path = pathlib.Path()


def plot_ordinal(X, y, g, X_show, f_show, J, D, N_show=None):
    cmap = plt.cm.get_cmap('viridis', J)
    colors = []
    for j in range(J):
        colors.append(cmap((j + 0.5)/J))
    if D==1:
        fig, ax = plt.subplots()
        ax.scatter(X, g, color=[colors[i] for i in y])
        ax.plot(X_show, f_show, color='k', alpha=0.4)
        fig.show()
        fig.savefig("scatter.png")
        plt.close()
    elif D==2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter3D(X[:, 0], X[:, 1], g[:], color=[colors[i] for i in y])
        surf = ax.plot_surface(
            X_show[:, 0].reshape(N_show, N_show),
            X_show[:, 1].reshape(N_show, N_show),
            f_show.reshape(N_show, N_show), alpha=0.4)
        fig.colorbar(cm.ScalarMappable(cmap=cmap))  # TODO: how to not normalize this
        plt.savefig("surface.png")
        plt.show()
        plt.close()
    else:
        pass


def _grid_over_hyperparameters_initiate(
        classifier, res, domain, trainables):
    """
    Initiate metadata and hyperparameters for plotting the objective
    function surface over hyperparameters.

    :arg int res:
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
    phi_0 = classifier.get_phi(trainables)
    
    theta_0 = []
    if trainables[0]:
        # Grid over noise_std
        label.append(r"$\sigma$")
        axis_scale.append("log")
        print(domain[index][0])
        theta = np.logspace(domain[index][0], domain[index][1], res[index])
        space.append(theta)
        phi_space.append(np.log(theta))
        theta_0.append(np.exp(phi_0[index]))
        index += 1
    if trainables[1]:
        # Grid over b_1, the first cutpoint
        label.append(r"$b_{}$".format(1))
        axis_scale.append("linear")
        theta = np.linspace(domain[index][0], domain[index][1], res[index])
        space.append(theta)
        phi_space.append(theta)
        theta_0.append(phi_0[index])
        index += 1
    for j in range(2, classifier.J):
        if trainables[j]:
            # Grid over b_j - b_{j-1}, the differences between cutpoints
            label.append(rf"$b_{ {j} } - b_{ {j-1} }$")
            axis_scale.append("log")
            theta = np.logspace(
                domain[index][0], domain[index][1], res[index])
            space.append(theta)
            phi_space.append(np.log(theta))
            theta_0.append(np.exp(phi_0[index]))
            index += 1
    if trainables[classifier.J]:
        # Grid over signal variance
        label.append(r"$\sigma_{\theta}^{ 2}$")
        axis_scale.append("log")
        theta = np.logspace(domain[index][0], domain[index][1], res[index])
        space.append(theta)
        phi_space.append(np.log(theta))
        theta_0.append(np.exp(phi_0[index]))
        index += 1
    if classifier.kernel._ARD is True:
        # In this case, then there is a scale parameter,
        #  the first cutpoint, the interval parameters,
        # and lengthvariances parameter for each dimension and class
        for d in range(classifier.D):
            if trainables[classifier.J + 1][d]:
                # Grid over kernel hyperparameter, theta in this dimension
                label.append(r"$\theta_{}$".format(d))
                axis_scale.append("log")
                theta = np.logspace(domain[index][0], domain[index][1], res[index])
                space.append(theta)
                phi_space.append(np.log(theta))
                theta_0.append(np.exp(phi_0[index]))
                index += 1
    else:
        if trainables[classifier.J + 1]:
            # Grid over only kernel hyperparameter, theta
            label.append(r"$\theta$")
            axis_scale.append("log")
            theta = np.logspace(domain[index][0], domain[index][1], res[index])
            space.append(theta)
            phi_space.append(np.log(theta))
            theta_0.append(np.exp(phi_0[index]))
            index +=1
    if index == 2:
        meshgrid_theta = np.meshgrid(space[0], space[1])
        meshgrid_phi = np.meshgrid(phi_space[0], phi_space[1])
        phis = np.dstack(meshgrid_phi)
        phis = phis.reshape((len(space[0]) * len(space[1]), 2))
        fxs = np.empty(len(phis))
        gxs = np.empty((len(phis), 2))
        theta_0 = np.array(theta_0)
    elif index == 1:
        meshgrid_theta = (space[0], None)
        space.append(None)
        phi_space.append(None)
        axis_scale.append(None)
        label.append(None)
        phis = phi_space[0].reshape(-1, 1)
        fxs = np.empty(len(phis))
        gxs = np.empty(len(phis))
        theta_0 = theta_0[0]
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
        phis, fxs, gxs, theta_0, phi_0)


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
    nu = np.random.normal(loc=0.0, scale=1.0,
        size=np.shape(X_data)[0] + np.shape(X_show)[0])
    f = L_K @ nu

    # Store f_show
    f_data = f[:N_total]
    f_show = f[N_total:]

    assert np.shape(f_show) == (np.shape(X_show)[0],)

    # Generate the latent variables
    X = X_data
    f = f_data
    epsilons = np.random.normal(0, np.sqrt(noise_variance), N_total)
    g = epsilons + f
    g = g.flatten()


    # Sort the responses
    idx_sorted = np.argsort(g)
    g = g[idx_sorted]
    f = f[idx_sorted]
    X = X[idx_sorted]
    X_js = []
    g_js = []
    f_js = []
    y_js = []
    cutpoints = np.empty(J + 1)
    for j in range(J):
        X_js.append(X[N_per_class * j:N_per_class * (j + 1), :D])
        g_js.append(g[N_per_class * j:N_per_class * (j + 1)])
        f_js.append(f[N_per_class * j:N_per_class * (j + 1)])
        y_js.append(j * np.ones(N_per_class, dtype=int))

    for j in range(1, J):
        # Find the cutpoints
        cutpoint_j_min = g_js[j - 1][-1]
        cutpoint_j_max = g_js[j][0]
        cutpoints[j] = np.average([cutpoint_j_max, cutpoint_j_min])
    cutpoints[0] = -np.inf
    cutpoints[-1] = np.inf
    print("cutpoints={}".format(cutpoints))
    X_js = np.array(X_js)
    g_js = np.array(g_js)
    f_js = np.array(f_js)
    y_js = np.array(y_js, dtype=int)
    X = X.reshape(-1, X.shape[-1])
    g = g_js.flatten()
    f = f_js.flatten()
    y = y_js.flatten()
    # Reshuffle
    data = np.c_[g, X, y, f]
    np.random.shuffle(data)
    g = data[:, :1].flatten()
    X = data[:, 1:D + 1]
    y = data[:, D + 1].flatten()
    y = np.array(y, dtype=int)
    f = data[:, -1].flatten()

    g_train_js = []
    X_train_js = []
    y_train_js = []
    X_test_js = []
    y_test_js = []
    for j in range(J):
        data = np.c_[g_js[j], X_js[j], y_js[j]]
        np.random.shuffle(data)
        g_j = data[:, :1]
        X_j = data[:, 1:D + 1]
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

    X_train_js = np.array(X_train_js)
    g_train_js = np.array(g_train_js)
    y_train_js = np.array(y_train_js, dtype=int)
    X_test_js = np.array(X_test_js)
    y_test_js = np.array(y_test_js, dtype=int)

    X_train = X_train_js.reshape(-1, X_train_js.shape[-1])
    g_train = g_train_js.flatten()
    y_train = y_train_js.flatten()
    X_test = X_test_js.reshape(-1, X_test_js.shape[-1])
    y_test = y_test_js.flatten()

    data = np.c_[g_train, X_train, y_train]
    np.random.shuffle(data)
    g_train = data[:, :1].flatten()
    X_train = data[:, 1:D + 1]
    y_train = data[:, -1]

    data = np.c_[X_test, y_test]
    np.random.shuffle(data)
    X_test = data[:, 0:D]
    y_test = data[:, -1]

    g_train = np.array(g_train)
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=int)

    assert np.shape(X_test) == (N_test, D)
    assert np.shape(X_train) == (N_train, D)
    assert np.shape(g_train) == (N_train,)
    assert np.shape(y_test) == (N_test,)
    assert np.shape(y_train) == (N_train,)
    assert np.shape(X_js) == (J, N_per_class, D)
    assert np.shape(g_js) == (J, N_per_class)
    assert np.shape(X) == (N_total, D)
    assert np.shape(g) == (N_total,)
    assert np.shape(f) == (N_total,)
    assert np.shape(y) == (N_total,)
    return (
        N_show, N_total, X_js, g_js, X, f, g, y, cutpoints,
        X_train, g_train, y_train,
        X_test, y_test,
        X_show, f_show)


#(get the probit_jax approximator)
def get_approximator(
        approximate_inference_method, N_train):
    """Get approximator class

    Args:
        approximate_inference_method:
    
    """
    if approximate_inference_method == "VB":
        from probit_jax.approximators import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//1000])
        print("steps: ", steps)
        Approximator = VBGP
    elif approximate_inference_method == "LA":
        from probit_jax.approximators import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected VB, LA)".format(
                approximate_inference_method))
    return Approximator, steps


def main():
    """Conduct an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
 
    J = 3
    D = 1
    method = "L-BFGS-B"

    # Generate data
    from mlkernels import Matern52
    kernel = 1.0 * Matern52().stretch(0.2)
    (N_show, N_total, X_js, g_js, X, f, g, y, cutpoints,
    X_train, g_train, y_train,
    X_test, y_test,
    X_show, f_show) = generate_data(
        N_train_per_class=10, N_test_per_class=100,
        J=3, D=1, kernel=kernel, noise_variance=1.0,
        N_show=1000, jitter=1e-6, seed=None)

    plot_ordinal(
        X, y, g, X_show, f_show, J, D, N_show=N_show) 

    # Let the model be mispecified, using a kernel
    # other than the one used to generate data

    # approximate_inference_method is "LA" or "VB"
    Approximator, steps = get_approximator(approximate_inference_method="VB", N_train=np.shape(y_train)[0])

    from mlkernels import EQ, BaseKernel
    # Initiate classifier
    def prior(prior_parameters):
        stretch = prior_parameters
        signal_variance = 1.0
        # Here you can define the kernel that defines the Gaussian process model
        kernel = signal_variance * EQ().stretch(stretch)
        # Make sure that model returns the kernel, cutpoints and noise_variance
        return kernel

    # Check prior is valid
    if not (isinstance(prior(prior_parameters=1.0), BaseKernel)):
        raise InvalidKernel(prior(prior_parameters=1.0))

    print("cutpoints_0={}".format(cutpoints_0))
    # Check cutpoints are valid
    cutpoints_0 = check_cutpoints(cutpoints_0, J)
    print("cutpoints_0={}".format(cutpoints_0))

    # Initiate classifier
    classifier = Approximator(prior, log_probit_likelihood,
        data=(X_train, y_train), single_precision=False)

    g = classifier.take_grad()

    # hyperparameter domain and resolution
    domain = ((-1, 2), None) # x-axis domain range
    res = (30, None) # increments in domain

    (x1s, x2s,
    xlabel, ylabel,
    xscale, yscale,
    xx, yy,
    phis, fxs,
    gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
    _classifier, res, domain, trainables)

    # print("theta_0 outisde", theta_0)
    # # get test for probit
    # phi_test = phis[phis.shape[0]//2]
    # theta_test = jnp.exp(phi_test)[0]

    # # get parameters for probit_jax
    params = ((jnp.sqrt(1./(2 * theta_0))), (jnp.sqrt(noise_variance_0), cutpoints_0))
    # params_test = ((jnp.sqrt(1./(2 * (theta_test)))), (jnp.sqrt(noise_variance_0), cutpoints_0))    

    # # probit_jax - latent variables
    # fxp, gxp = g(params)
    # fxp_test, gxp = g(params_test)
    latent_jax = classifier.get_latents(params)
    #latent_jax_test = classifier.get_latents(params_test)

    # # probit - latent variables
    fx, gx, latent_probit, _ = _classifier.approximate_posterior(
    phi_0, trainables, steps, verbose=False, return_reparameterised = True)
    # # fx_test, gx_test, latent_probit_test, _ = _classifier.approximate_posterior(
    # phi_test, trainables, steps, verbose=False, return_reparameterised = True)
    
    #print("model evidence difference: ", fx - fxp)
    # probit - predictive distribution
    #cov = noise_variance_0**2*jnp.identity(N_train) + _classifier.prior_coveriance # sigma^2I + K)^{-1}
    #_, pred_mean, _ = _classifier.predict(
    #X, cov, f, reparameterised=True, whitened=False)    

    def _plot(title, probit, probit_jax):
        x = [_[0] for _ in X]

        #_x = jnp.linspace(0, len(latents)-1, len(latents))
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(x, probit, c='b', marker="X", markersize=8, alpha=1, linestyle = "None", label=r"analytic")
        ax.plot(x, probit_jax, c='g', marker ="o", alpha=0.7, linestyle = "None", label=r"autodiff")
        ax.hlines(sum(probit)/len(probit), min(x), max(x), 'r', alpha=0.5, label=r"analytic mean")
        ax.hlines(sum(probit_jax)/len(probit_jax), min(x), max(x), 'k',
            alpha=0.5, label=r"autodiff mean")
        ax.set_ylabel(r"Latent Variables")
        ax.legend()
        fig.savefig(title,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

    #_plot("steps=100",latent_probit, latent_jax)
    _plot("LA_newton_method_latents_new", latent_probit, latent_jax)

    def plot_diff(phis, plot=True):
        dfs = np.empty(res[0]) # list to store the differences of the likelihoods
        fis = []
        for i, phi in enumerate(phis):
            theta = jnp.exp(phi)[0]
            params = ((jnp.sqrt(1./(2 * theta))), (jnp.sqrt(noise_variance_0), cutpoints_0)) # params[0]-->prior. params[1]-->likelihood
            
            fxp, gxp, weight, _ = _classifier.approximate_posterior(
            phi, trainables, steps, verbose=False, return_reparameterised = True)
            fx, gx = g(params) # g is a function. passing the arguments for the solver to perform fixed_point_interation
            gs = gx[0] * (- 0.5 * (2 * theta)**(-1./2))  # multiply by the lengthscale Jacobian
            
            df = fxp - fx
            dfs[i] = df
            
            if df <= 5:
                fis.append(phi)
        
        if plot == True:
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.grid()
            ax.plot(x1s, dfs, 'b',  label=r"Evidence diff")
            ax.set_ylabel(r"Model evidence difference (probit-probit_jax)")
            ax.set_xlabel(xlabel)
            ax.set_xscale(xscale)
            ax.legend()
            fig.savefig("Difference_VB.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

        return fis
    
    #plot_diff(phis)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    def plot_bound_grad(title1="bound_LA_1", title2="grad_LA_1"):
        domain = ((-1, 2), None) # x-axis domain range
        res = (30, None) # increments in domain
        trainables = [0,0,0,0,1] # vary signal std 

        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        phis, fxs,
        gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
        _classifier, res, domain, trainables)
        
        print(phis)
        print("theta_inside", theta_0)
        print("phi_inside", phi_0)
        
        gs = np.empty(res[0])
        fs = np.empty(res[0])

        #outer loop for probit_jax
        for i, phi in enumerate(phis):
            theta = jnp.exp(phi)[0]
            
            params = ((jnp.sqrt(1./(2 * theta))), (jnp.sqrt(noise_variance_0), cutpoints_0)) # params[0]-->prior. params[1]-->likelihood
            fx, gx = g(params) # g is a function. passing the arguments for the solver to perform fixed_point_interation
            gs[i] = gx[0] * (- 0.5 * (2 * theta)**(-1./2))  # multiply by the lengthscale Jacobian
            fs[i] = fx
        
        # outer loop for probit
        for i, phi in enumerate(phis):
            fx, gx, weight, _ = _classifier.approximate_posterior(
                phi, trainables, steps, verbose=False, return_reparameterised = True)
            
            fxs[i] = fx
            gxs[i] = gx

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
        ax.plot(x, fxs, 'b', marker="o", label=r"$\mathcal{F}}$ analytic")
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
        fig.savefig(title1,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(
            x, dfxs_, 'b--', marker="o", 
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic numeric")
        ax.set_ylim(ax.get_ylim())
        ax.plot(
            x, gxs, 'b', alpha=0.7, marker="o",
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic")
        ax.plot(
            x, gs, 'g',
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff")
        ax.plot(
            x, dfsxs, 'g--',
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff numeric")
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
        ax.legend()
        fig.savefig(title2,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
    
    #plot_bound_grad("bound_LA_fwd", "grad_LA_fwd")

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    sys.stdout.close()


if __name__ == "__main__":
    main()
