"""Ordinal regression concrete examples. Approximate inference."""
# Enable double precision
from jax.config import config
config.update("jax_enable_x64", True)

# Useful arguments to limit CPU usage
# import os
# os.environ["OMP_NUM_THREADS"] = "6"
# os.environ["OPENBLAS_NUM_THREADS"] = "6" 
# os.environ["MKL_NUM_THREADS"] = "6"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
# os.environ["NUMEXPR_NUM_THREADS"] = "6"
# os.environ["NUMBA_NUM_THREADS"] = "6"

# Useful arguments to limit GPU usage
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION "] = "0.5"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
import lab as B
import pathlib
from probit_jax.utilities import InvalidKernel, check_cutpoints
from probit_jax.implicit.utilities import (
    log_probit_likelihood, grad_log_probit_likelihood, hessian_log_probit_likelihood)
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import lab as B
from mlkernels import Kernel, Matern52, EQ

import warnings
import time
from scipy.optimize import minimize


# For plotting
BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


now = time.ctime()
write_path = pathlib.Path()


def train(classifier, method, trainables, verbose=True, steps=None, max_sec=5000):
    """
    Hyperparameter training via gradient descent of the objective function, a
    negative log marginal likelihood (or bound thereof).

    :arg classifier:
    :type classifier: :class:`probit.estimators.Approximator`
        or :class:`probit.samplers.Sampler`
    :arg str method: "CG" or "L-BFGS-B" seem to be widely used and work well.
    :arg trainables: Binary array or list of trainables indicating which
        hyperparameters to fix, and which to optimize.
    :type trainables: :class:`numpy.ndarray` or list
    :arg bool verbose: Verbosity bool, default True
    :arg int steps: The number of steps to run the algorithm on the inner loop.
    :arg float max_sec: The max time to do optimization for, in seconds.
        TODO: test.
    :return: The hyperparameters after optimization.
    :rtype: tuple (
        :class:`numpy.ndarray`, float or :class:`numpy.ndarray`, float, float)
    """
    minimize_stopper = MinimizeStopper(max_sec=max_sec)
    phi = classifier.get_phi(trainables)
    args = (trainables, steps, verbose)
    res = minimize(
        classifier.approximate_posterior, phi,
        args, method=method, jac=True,
        callback=minimize_stopper.__call__)
    return classifier


def plot_synthetic(
    classifier, dataset, X_true, Y_true, steps, colors=colors):
    """
    Plots for synthetic data.

    TODO: needs generalizing to other datasets other than Chu.
    """
    (fx, gx,
    weights, (cov, is_reparametrised)
    ) = classifier.approximate_posterior(
            None, None, steps,
            return_reparameterised=True, verbose=True)

    if classifier.J == 3:
        if classifier.D == 1:
            x_lims = (-0.5, 1.5)
            N = 1000
            x = np.linspace(x_lims[0], x_lims[1], N)
            X_new = x.reshape((N, classifier.D))
            # Test
            (Z,
            posterior_predictive_m,
            posterior_std) = classifier.predict(
                X_new, cov, weights)
            print(np.sum(Z, axis=1), 'sum')
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.set_xlim(x_lims)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(r"$x$", fontsize=16)
            ax.set_ylabel(r"$p(y|x, X, y)$", fontsize=16)
            ax.stackplot(x, Z.T,
                        labels=(
                            r"$p(y=0|x, X, y)$",
                            r"$p(y=1|x, X, y)$",
                            r"$p(y=2|x, X, y)$"),
                        colors=(
                            colors[0], colors[1], colors[2])
                        )
            val = 0.5  # where the data lies on the y-axis.
            for j in range(classifier.J):
                ax.scatter(
                    classifier.X_train[np.where(classifier.y_train == j)],
                    np.zeros_like(classifier.X_train[np.where(
                        classifier.y_train == j)]) + val,
                    s=15, facecolors=colors[j], edgecolors='white')
            plt.tight_layout()
            fig.savefig(
                "Cumulative distribution plot of ordinal class "
                "distributions for x_new=[{}, {}].png".format(
                    x_lims[0], x_lims[1]),
                    facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()
            plt.close()

            np.savez(
                "tertile.npz",
                x=X_new, y=posterior_predictive_m, s=posterior_std)
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.plot(X_new, posterior_predictive_m, 'r')
            ax.fill_between(
                X_new[:, 0], posterior_predictive_m - 2*posterior_std,
                posterior_predictive_m + 2*posterior_std,
                color='red', alpha=MG_ALPHA)
            ax.scatter(X_true, Y_true, color='b', s=4)
            ax.set_ylim(-2.2, 2.2)
            ax.set_xlim(-0.5, 1.5)
            for j in range(classifier.J):
                ax.scatter(
                    classifier.X_train[np.where(classifier.y_train == j)],
                    np.zeros_like(classifier.X_train[
                        np.where(classifier.y_train == j)]),
                    s=15, facecolors=colors[j], edgecolors='white')
            plt.tight_layout()
            fig.savefig(
                "Scatter plot of data compared to posterior mean.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()
            plt.close()

        elif classifier.D == 2:
            x_lims = (-4.0, 6.0)
            y_lims = (-4.0, 6.0)
            # x_lims = (-0.5, 1.5)
            # y_lims = (-0.5, 1.5)
            N = 200
            x = np.linspace(x_lims[0], x_lims[1], N)
            y = np.linspace(y_lims[0], y_lims[1], N)
            xx, yy = np.meshgrid(x, y)
            X_new = np.dstack([xx, yy]).reshape(-1, 2)
            # Test
            (Z,
            posterior_predictive_m,
            posterior_std) = classifier.predict(
                X_new, cov, weights)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(
                X_new[:, 0].reshape(N, N),
                X_new[:, 1].reshape(N, N),
                Z.T[0, :].reshape(N, N), alpha=FG_ALPHA, color=colors[0],
                    label=r"$p(y=0|x, X, t)$")
            surf = ax.plot_surface(
                X_new[:, 0].reshape(N, N),
                X_new[:, 1].reshape(N, N),
                Z.T[1, :].reshape(N, N), alpha=FG_ALPHA, color=colors[1],
                    label=r"$p(y=1|x, X, t)$")
            surf = ax.plot_surface(
                X_new[:, 0].reshape(N, N),
                X_new[:, 1].reshape(N, N),
                Z.T[2, :].reshape(N, N), alpha=FG_ALPHA, color=colors[2],
                    label=r"$p(y=2|x, X, t)$")
            cmap = plt.cm.get_cmap('viridis', classifier.J)
            import matplotlib as mpl
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap))  # TODO: how to not normalize this
            for j in range(classifier.J):
                ax.scatter3D(
                    classifier.X_train[np.where(classifier.y_train == j), 0],
                    classifier.X_train[np.where(classifier.y_train == j), 1],
                    Y_true[np.where(
                        classifier.y_train == j)],
                    s=15, facecolors=colors[j], edgecolors='white')
            plt.tight_layout()
            fig.savefig("surface.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()

            # ax.set_xlim(x_lims)
            # ax.set_ylim(y_lims)
            # ax.set_ylim(0.0, 1.0)
            # ax.set_xlabel(r"$x$", fontsize=16)
            # ax.set_ylabel(r"$p(y|x, X, y)$", fontsize=16)
            # ax.stackplot(x, Z.T,
            #             labels=(
            #                 r"$p(y=0|x, X, t)$",
            #                 r"$p(y=1|x, X, t)$",
            #                 r"$p(y=2|x, X, t)$"),
            #             colors=(
            #                 colors[0], colors[1], colors[2])
            #             )
            # plt.tight_layout()
            # fig.savefig(
            #     "Ordered Gibbs Cumulative distribution plot of class "
            #     "distributions for x_new=[{}, {}].png".format(
            #         x_lims[0], x_lims[1]))
            # plt.show()
            # plt.close()
            def colorTriangle(r,g,b):
                image = np.stack([r,g,b],axis=2)
                return image/image.max(axis=2)[:,:,None]
            ax.imshow(
                colorTriangle(
                    Z.T[0, :].reshape(N, N), Z.T[1, :].reshape(N, N),
                    Z.T[2, :].reshape(N, N)),
                    origin='lower',extent=(-4.0,6.0,-4.0,6.0))
            fig.savefig("color_triangle.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()
            plt.close()
    else:
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, classifier.D))
        # Test
        (Z,
        posterior_predictive_m,
        posterior_std) = classifier.predict(
            X_new, cov, weights)
        print(np.sum(Z, axis=1), 'sum')
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_xlim(x_lims)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$x$", fontsize=16)
        ax.set_ylabel(r"$p(y_{*}={}|x, X, y)$", fontsize=16)
        if classifier.J == 13:
            plt.stackplot(x, Z.T,
                            labels=(
                            r"$p(y_{*}=0|x, X, y)$",
                            r"$p(y_{*}=1|x, X, y)$",
                            r"$p(y_{*}=2|x, X, y)$",
                            r"$p(y_{*}=3|x, X, y)$",
                            r"$p(y_{*}=4|x, X, y)$",
                            r"$p(y_{*}=5|x, X, y)$",
                            r"$p(y_{*}=6|x, X, y)$",
                            r"$p(y_{*}=7|x, X, y)$",
                            r"$p(y_{*}=8|x, X, y)$",
                            r"$p(y_{*}=9|x, X, y)$",
                            r"$p(y_{*}=10|x, X, y)$",
                            r"$p(y_{*}=11|x, X, y)$",
                            r"$p(y_{*}=12|x, X, y)$"),
                        colors=colors
                        )
        else:
            ax.stackplot(x, Z.T, colors=colors)
        val = 0.5  # where the data lies on the y-axis.
        for j in range(classifier.J):
            ax.scatter(
                classifier.X_train[np.where(classifier.y_train == j)],
                np.zeros_like(
                    classifier.X_train[np.where(
                        classifier.y_train == j)]) + val,
                        s=15, facecolors=colors[j],
                    edgecolors='white')
        fig.savefig(
            "Ordered Gibbs Cumulative distribution plot of class "
            "distributions for x_new=[{}, {}].png"
                .format(x_lims[0], x_lims[1]),
                facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()
        plt.close()

        np.savez(
            "{}.npz".format(classifier.J),
            x=X_new, y=posterior_predictive_m, s=posterior_std)
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.plot(X_new, posterior_predictive_m, 'r')
        ax.fill_between(
            X_new[:, 0], posterior_predictive_m - 2*posterior_std,
            posterior_predictive_m + 2*posterior_std,
            color='red', alpha=MG_ALPHA)
        ax.scatter(X_true, Y_true, color='b', s=4)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlim(-0.5, 1.5)
        for j in range(classifier.J):
            ax.scatter(
                classifier.X_train[np.where(classifier.y_train == j)],
                np.zeros_like(classifier.X_train[np.where(classifier.y_train == j)]),
                s=15,
                facecolors=colors[j],
                edgecolors='white')
        ax.savefig("scatter_versus_posterior_mean.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()
        plt.close()
    return fx


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


def plot_helper(
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
        theta_0 = np.array(theta_0)
    elif index == 1:
        meshgrid_theta = (space[0], None)
        space.append(None)
        phi_space.append(None)
        axis_scale.append(None)
        label.append(None)
        phis = phi_space[0].reshape(-1, 1)
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
        phis, theta_0, phi_0)


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
    approximate_inference_method = "VB"
    method = "L-BFGS-B"

    # Generate data
    noise_variance = 1.0
    kernel = 1.0 * Matern52().stretch(0.2)
    (N_show, N_total, X_js, g_js, X, f, g, y, cutpoints,
    X_train, g_train, y_train,
    X_test, y_test,
    X_show, f_show) = generate_data(
        N_train_per_class=10, N_test_per_class=100,
        J=3, D=1, kernel=kernel, noise_variance=noise_variance,
        N_show=1000, jitter=1e-6, seed=None)

    plot_ordinal(
        X, y, g, X_show, f_show, J, D, N_show=N_show) 

    if approximate_inference_method=="VB":
        from probit_jax.approximators import VBGP as Approximator
        # TODO: max steps?
        # steps is the number of fix point iterations until check convergence
        # steps = np.max([10, N_train//1000])
    elif approximate_inference_method=="LA":
        from probit_jax.approximators import LaplaceGP as Approximator
        # steps = np.max([2, N_train//1000])

    # Initiate a misspecified model, using a kernel
    # other than the one used to generate data
    def prior(prior_parameters):
        stretch = prior_parameters
        signal_variance = 1.0
        # Here you can define the kernel that defines the Gaussian process
        kernel = signal_variance * EQ().stretch(stretch)
        # Make sure that model returns the kernel, cutpoints and noise_variance
        return kernel

    # Test prior
    if not (isinstance(prior(1.0), Kernel)):
        raise InvalidKernel(prior(1.0))

    print("cutpoints_0={}".format(cutpoints_0))
    # check that the cutpoints are in the correct format
    # for the number of classes, J
    cutpoints_0 = check_cutpoints(cutpoints_0, J)
    print("cutpoints_0={}".format(cutpoints_0))

    classifier = Approximator(prior, log_probit_likelihood,
        single_precision=True,
        # grad_log_likelihood=grad_log_probit_likelihood,
        # hessian_log_likelihood=hessian_log_probit_likelihood,
        data=(X, y))

    # Notes: fwd_solver, newton_solver work, anderson solver has bug with vmap ValueError
    g = classifier.take_grad()

    (x, _,
    xlabel, _,
    xscale, _,
    _, _,
    phis, theta_0, phi_0) = plot_helper(
        res, domain)

    domain = ((-1, 2), None)
    res = (30, None)
    gs = np.empty(res[0])
    fs = np.empty(res[0])
    for i, phi in enumerate(phis):
        theta = jnp.exp(phi)[0]
        params = ((theta)), (jnp.sqrt(noise_variance), cutpoints)
        fx, gx = g(params)
        fs[i] = fx
        gs[i] = gx[0] * 1./theta  # multiply by a Jacobian

    # Numerical derivatives: need to calculate them in the log domain
    # if theta is in log domain
    if xscale == "log":
        log_x = np.log(x)
        dfsxs = np.gradient(fs, log_x)
    elif xscale == "linear":
        dfsxs = np.gradient(fs, x)
    idx_hat = np.argmin(fs)

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax = fig.add_subplot(111)
    ax.grid()
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
        x, gs, 'g', alpha=0.4,
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff")
    ax.set_ylim(ax.get_ylim())
    ax.plot(
        x, dfsxs, 'g--',
        label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff numeric")
    ax.vlines(theta_0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
        alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
    ax.vlines(x[idx_hat], 0.9 * ylim[0], 0.9 * ylim[1], 'r',
        alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
    ax.legend()
    fig.savefig("grad.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    def calculate_metrics(y_test, t_test, Z, cutpoints):
        t_pred = np.argmax(Z, axis=1)
        grid = np.ogrid[0:len(t_test)]
        # Other error
        predictive_likelihood = Z[grid, t_test]
        mean_absolute_error = np.sum(np.abs(t_pred - t_test)) / len(t_test)
        root_mean_squared_error = np.sqrt(
            np.sum(pow(t_pred - t_test, 2)) / len(t_test))
        print("root_mean_squared_error", root_mean_squared_error)
        print("mean_absolute_error ", mean_absolute_error)
        log_predictive_probability = np.sum(np.log(predictive_likelihood))
        print("log_pred_probability ", log_predictive_probability)
        predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
        print("predictive_likelihood ", predictive_likelihood)
        print("av_prob_of_correct ", predictive_likelihood)
        print(np.sum(t_pred != t_test), "sum incorrect")
        mean_zero_one = np.sum(t_pred != t_test) / len(t_test)
        print("mean_zero_one_error", mean_zero_one)
        print(np.sum(t_pred == t_test), "sum correct")
        mean_zero_one = np.sum(t_pred == t_test) / len(t_test)
        print("mean_zero_one_correct", mean_zero_one)
        return (
            mean_zero_one,
            root_mean_squared_error,
            mean_absolute_error,
            log_predictive_probability,
            predictive_likelihood)


    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()


class TookTooLong(Warning):
    pass


class MinimizeStopper(object):
    def __init__(self, max_sec=100):
        self.max_sec = max_sec
        self.start   = time.time()

    def __call__(self, xk):
        # callback to terminate if max_sec exceeded
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)
