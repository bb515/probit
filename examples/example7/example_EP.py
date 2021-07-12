"""
Multiclass ordered probit regression 3 bin example from Cowles 1996 empirical study.
Variational inference implementation.
"""
import argparse
import cProfile
from io import StringIO
#from pstats import Stats, SortKey
import numpy as np
from scipy.stats import multivariate_normal
from probit.estimators import EPMultinomialOrderedGP
from probit.kernels import SEIso
import matplotlib.pyplot as plt
import pathlib
from scipy.optimize import minimize
from probit.utilities import generate_prior_data, generate_synthetic_data
import importlib.resources as pkg_resources

write_path = pathlib.Path()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def split(list, K):
    """Split a list into quantiles."""
    divisor, remainder = divmod(len(list), K)
    return np.array(list[i * divisor + min(i, remainder):(i+1) * divisor + min(i + 1, remainder)] for i in range(K))

datasets = [
    "abalone",
    "auto",
    "diabetes",
    "housing",
    "machine",
    "pyrim",
    "stocks",
    "triazines",
    "wpbc"
]


plot_lims = {
    "abalone": ((15.0, 65.0), (15.0, 70.0)),
    "auto": ((15.0, 65.0), (15.0, 70.0)),
    "diabetes": ((15.0, 65.0), (15.0, 70.0)),
    "housing": ((15.0, 65.0), (15.0, 70.0)),
    "machine": ((15.0, 65.0), (15.0, 70.0)),
    "pyrim": ((15.0, 65.0), (15.0, 70.0)),
    "stocks": ((15.0, 65.0), (15.0, 70.0)),
    "triazines": ((15.0, 65.0), (15.0, 70.0)),
    "wpbc": ((15.0, 65.0), (15.0, 70.0)),
}

def load_data(dataset, bins):
    if dataset == "abalone":
        from probit.data import abalone
        with pkg_resources.path(abalone, 'abalone.npz') as path:
            data_continuous = np.load(path)
        D = 10
        varphi_0 = 2.0/D
        noise_variance_0 = 1.0
        if bins == "quantile":
            from probit.data.abalone import quantile
            with pkg_resources.path(quantile, 'abalone.npz') as path:
                data = np.load(path)
            K = 5
        elif bins == "decile":
            from probit.data.abalone import decile
            with pkg_resources.path(decile, 'abalone.npz') as path:
                data = np.load(path)
            K = 10
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "auto":
        from probit.data import auto
        with pkg_resources.path(auto, 'auto.npz') as path:
            data_continuous = np.load(path)
        D = 7
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if bins == "quantile":
            K = 5
            from probit.data.auto import quantile
            with pkg_resources.path(quantile, 'auto.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            from probit.data.auto import decile
            with pkg_resources.path(decile, 'auto.npz') as path:
                data = np.load(path)
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "diabetes":
        D = 2
        varphi_0 = 6.7e-06
        noise_variance_0 = 1.0
        from probit.data import diabetes
        with pkg_resources.path(diabetes, 'diabetes.DATA.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            K = 5
            from probit.data.diabetes import quantile
            with pkg_resources.path(quantile, 'diabetes.data.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            from probit.data.diabetes import decile
            with pkg_resources.path(decile, 'diabetes.data.npz') as path:
                data = np.load(path)
            K = 10
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "housing":
        D = 13
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        from probit.data import bostonhousing
        with pkg_resources.path(bostonhousing, 'housing.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            K = 5
            from probit.data.bostonhousing import quantile
            with pkg_resources.path(quantile, 'housing.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            from probit.data.bostonhousing import decile
            with pkg_resources.path(decile, 'housing.npz') as path:
                data = np.load(path)
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "machine":
        D = 6
        varphi_0 = 2.0 / D
        noise_variance_0 = 2.0
        from probit.data import machinecpu
        with pkg_resources.path(machinecpu, 'machine.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            from probit.data.machinecpu import quantile
            with pkg_resources.path(quantile, 'machine.npz') as path:
                data = np.load(path)
            K = 5
        elif bins == "decile":
            from probit.data.machinecpu import decile
            with pkg_resources.path(decile, 'machine.npz') as path:
                data = np.load(path)
            K = 10
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "pyrim":
        from probit.data import pyrimidines
        with pkg_resources.path(pyrimidines, 'pyrim.npz') as path:
            data_continuous = np.load(path)
        D = 27
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if bins == "quantile":
            from probit.data.pyrimidines import quantile
            with pkg_resources.path(quantile, 'pyrim.npz') as path:
                data = np.load(path)
            K = 5
        elif bins == "decile":
            from probit.data.pyrimidines import decile
            with pkg_resources.path(decile, 'pyrim.npz') as path:
                data = np.load(path)
            K = 10
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "stocks":
        from probit.data import stocksdomain
        with pkg_resources.path(stocksdomain, 'stock.npz') as path:
            data_continuous = np.load(path)
        K = 5
        D = 9
        noise_variance_0 = 0.01  # 2.0  0.03
        varphi_0 = 0.00045  # 0.0001  # varphi_0 = 0.00045
        if bins == "quantile":
            from probit.data.stocksdomain import quantile
            with pkg_resources.path(quantile, 'stock.npz') as path:
                data = np.load(path)
            K = 5
        elif bins == "decile":
            from probit.data.stocksdomain import decile
            with pkg_resources.path(decile, 'stock.npz') as path:
                data = np.load(path)
            K = 10
            #data = np.load("./data/10bin/stock.npz")
        gamma_0 = [-np.inf, -1.17119928, -0.65961478, 0.1277627, 0.64710874, np.inf]
        # gamma_0 = np.array([-np.inf, -0.5, -0.02, 0.43, 0.96, np.inf])
        # gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "triazines":
        from probit.data import triazines
        with pkg_resources.path(triazines, 'triazines.npz') as path:
            data_continuous = np.load(path)
        D = 60
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if bins == "quantile":
            from probit.data.triazines import quantile
            with pkg_resources.path(quantile, 'triazines.npz') as path:
                data = np.load(path)
            K = 5
        elif bins == "decile":
            K = 10
            data = np.load(write_path / "./data/10bin/triazines.npz")
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "wpbc":
        D = 32
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        from probit.data import wisconsin
        with pkg_resources.path(wisconsin, 'wpbc.npz') as path:
            data_continuous = np.load(path)
        data_continuous = np.load(write_path / "./data/continuous/wpbc.npz")
        if bins == "quantile":
            from probit.data.wisconsin import quantile
            with pkg_resources.path(quantile, 'wpbc.npz') as path:
                data = np.load(path)
            K = 5
        elif bins == "decile":
            from probit.data.wisconsin import decile
            with pkg_resources.path(decile, 'wpbc.npz') as path:
                data = np.load(path)
            K = 10
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])

    X_trains = data["X_train"]
    t_trains = data["t_train"]
    X_tests = data["X_test"]
    t_tests = data["t_test"]

    # Python indexing
    t_tests = t_tests - 1
    t_trains = t_trains - 1
    t_tests = t_tests.astype(int)
    t_trains = t_trains.astype(int)

    # Number of splits
    N_splits = len(X_trains)
    assert len(X_trains) == len(X_tests)

    X_true = data_continuous["X"]
    Y_true = data_continuous["y"]  # this is not going to be the correct one
    # Y_trues = []
    #
    # for k in range(20):
    #     y = []
    #     for i in range(len(X_trains[0, :, :])):
    #         for j, two in enumerate(X_true):
    #             one = X_trains[k, i]
    #             if np.allclose(one, two):
    #                 y.append(Y_true[j])
    #     Y_trues.append(y)
    # Y_trues = np.array(Y_trues)
    return X_trains, t_trains, X_tests, t_tests, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D


def load_data_synthetic(dataset, data_from_prior):
    if dataset == "tertile":
        from probit.data import tertile
        K = 3
        D = 1
        N_per_class = 30  # 64
        varphi_0 = 28.247881910538307  # 7.0 #  19.59821963518377  # 30.0
        scale = 1.0
        noise_variance_0 = 0.11103503642649291  # 1.0 #  0.07548142258576254  #0.1
        kernel = SEIso(varphi_0, scale, sigma=10e-6, tau=10e-6)
        # gamma_0 = np.array([-np.inf, 0.0, 2.29, np.inf])
        if data_from_prior == True:
            with pkg_resources.path(tertile, 'tertile_prior.npz') as path:
                data = np.load(path)
            # Generate the synethetic data
            # X_k, Y_true_k, X, Y_true, t, gamma_0 = generate_prior_data(
            #     N_per_class, K, D, kernel, noise_variance=noise_variance_0)
            # np.savez(write_path / "data_tertile_prior.npz", X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t, gamma_0=gamma_0)
            X_k = data["X_k"]  # Contains (90,) array of binned x values
            # Y_true_k = data["Y_k"]  # Contains (90,) array of binned y values
            X = data["X"]  # Contains (90,) array of x values
            t = data["t"]  # Contains (90,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
            gamma_0 = data["gamma_0"]
            gamma = [-np.inf, - 0.43160987, 0.2652492, np.inf]
            # gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
        else:
            with pkg_resources.path(tertile, 'tertile.npz') as path:
                data = np.load(path)
            X_k = data["X_k"]  # Contains (256, 7) array of binned x values
            # Y_true_k = data["Y_k"]  # Contains (256, 7) array of binned y values
            X = data["X"]  # Contains (1792,) array of x values
            t = data["t"]  # Contains (1792,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
            N_total = int(N_per_class * K)
    elif dataset == "septile":
        from probit.data import septile
        with pkg_resources.path(septile, 'septile.npz') as path:
            data = np.load(path)
        K = 7
        D = 1
        N_per_class = 32
        varphi_0 = 30.0
        scale = 20.0
        noise_variance_0 = 1.0
        # Generate the synethetic data
        #X_k, Y_true_k, X, Y_true, t = generate_synthetic_data(N_per_class, K, D, kernel)
        #np.savez(write_path / "data_septile.npz", X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t)
        gamma_0 = np.array([-np.inf, 0.0, 1.0, 2.0, 4.0, 5.5, 6.5, np.inf])
        X_k = data["X_k"]  # Contains (256, 7) array of binned x values
        # Y_true_k = data["Y_k"]  # Contains (256, 7) array of binned y values
        X = data["X"]  # Contains (1792,) array of x values
        t = data["t"]  # Contains (1792,) array of ordinal response variables, corresponding to Xs values
        Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
        X_true = None  # TODO
        N_total = int(N_per_class * K)

    # # Plot
    # colors_ = [colors[i] for i in t]
    # plt.scatter(X, Y_true, color=colors_)
    # plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
    # plt.xlabel(r"$x$", fontsize=16)
    # plt.ylabel(r"$y$", fontsize=16)
    # plt.show()

    # # Plot from the binned arrays
    # for k in range(K):
    #     plt.scatter(X_k[k], Y_true_k[k], color=colors[k], label=r"$t={}$".format(k))
    # plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
    # plt.legend()
    # plt.xlabel(r"$x$", fontsize=16)
    # plt.ylabel(r"$y$", fontsize=16)
    # plt.show()
    return X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D


def EP_plotting(dataset, X_train, t_train, X_true, Y_true, gamma, varphi, noise_variance, K, D, scale):
    kernel = SEIso(varphi, scale, sigma=10e-6, tau=10e-6)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    steps = variational_classifier.N
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    while error / steps > variational_classifier.EPS ** 2:
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
         precision_EP, amplitude_EP, containers) = variational_classifier.estimate(
            steps, gamma, varphi, noise_variance, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP, write=True)
        plt.scatter(X_train, posterior_mean)
        plt.scatter(X_true, Y_true)  # TODO: I changed X_train to X_true, correct?
        plt.ylim(-3, 3)
        plt.show()
        print("iteration {}, error={}".format(iteration, error / steps))
    weights, precision_EP, Lambda_cholesky, Lambda = variational_classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
    t1, t2, t3, t4, t5 = variational_classifier.compute_integrals(
        gamma, Sigma, precision_EP, posterior_mean, noise_variance)
    fx = variational_classifier.evaluate_function(precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
    (xlims, ylims) = plot_lims[dataset]

    N = 75
    x1 = np.linspace(xlims[0], xlims[1], N)
    x2 = np.linspace(ylims[0], ylims[1], N)
    xx, yy = np.meshgrid(x1, x2)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, 2))
    X_new_ = np.zeros((N * N, D))
    X_new_[:, :2] = X_new

    Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi,
                                       noise_variance, X_new_, Lambda, vectorised=True)
    Z_new = Z.reshape((N, N, K))
    print(np.sum(Z, axis=1), 'sum')
    for i in range(K):
        fig, axs = plt.subplots(1, figsize=(6, 6))
        plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
        plt.scatter(X_train[np.where(t_train == i)][:, 0], X_train[np.where(t_train == i)][:, 1], color='red')
        # plt.scatter(X_train[np.where(t == i + 1)][:, 0], X_train[np.where(t == i + 1)][:, 1], color='blue')
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.title("Contour plot - Expectation propagation")
        plt.savefig("contour_EP_{}.png".format(i))
        plt.show()


def EP_plotting_synthetic(dataset, X, t, Y_true, gamma, varphi, noise_variance, K, D,
                steps=5000, scale=1.0, sigma=10e-6, tau=10e-6):
    print("scale={}".format(scale))
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X, t, kernel)
    steps = variational_classifier.N
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    while error / steps > variational_classifier.EPS**2:
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
         precision_EP, amplitude_EP, containers) = variational_classifier.estimate(
            steps, gamma, varphi, noise_variance, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP, write=True)
        plt.scatter(X, posterior_mean)
        plt.scatter(X, Y_true)
        plt.ylim(-3, 3)
        plt.show()
        print("iteration {}, error={}".format(iteration, error / steps))
    weights, precision_EP, Lambda_cholesky, Lambda = variational_classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
    t1, t2, t3, t4, t5 = variational_classifier.compute_integrals(
        gamma, Sigma, precision_EP, posterior_mean, noise_variance)
    fx = variational_classifier.evaluate_function(precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)

    if dataset == "tertile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_new, Lambda,
                                           vectorised=True)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(x_lims)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
                  .format(x_lims[0], x_lims[1]))
        plt.stackplot(x, Z.T,
                      labels=(
                          r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
                      colors=(
                          colors[0], colors[1], colors[2])
                      )
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
        plt.scatter(X[np.where(t == 0)], np.zeros_like(X[np.where(t == 0)]) + val, facecolors=colors[0],
                    edgecolors='white')
        plt.scatter(X[np.where(t == 1)], np.zeros_like(X[np.where(t == 1)]) + val, facecolors=colors[1],
                    edgecolors='white')
        plt.scatter(X[np.where(t == 2)], np.zeros_like(X[np.where(t == 2)]) + val, facecolors=colors[2],
                    edgecolors='white')
        plt.savefig("cumulative_stackplot.png")
        plt.show()

    elif dataset == "septile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_new, Lambda,
                                           vectorised=True)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(x_lims)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
                  .format(x_lims[1], x_lims[0]))
        plt.stackplot(x, Z.T,
                      labels=(
                          r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$", r"$p(t=3|x, X, t)$",
                          r"$p(t=4|x, X, t)$", r"$p(t=5|x, X, t)$", r"$p(t=6|x, X, t)$"),
                      colors=(
                          colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6])
                      )
        plt.legend()
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
        for i in range(7):
            plt.scatter(
                X[np.where(t == i)], np.zeros_like(X[np.where(t == i)]) + val, facecolors=colors[i], edgecolors='white')
        plt.show()
    return fx


def EP_testing(
        X_train, t_train, X_test, t_test, gamma, varphi, noise_variance,
        K, D, scale=1.0, sigma=10e-6, tau=10e-6):
    grid = np.ogrid[0:len(X_test[:, :])]
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    print("varphi", kernel.varphi, varphi)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    steps = variational_classifier.N
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    while error / steps > variational_classifier.EPS**2:
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
         precision_EP, amplitude_EP, containers) = variational_classifier.estimate(
            steps, gamma, varphi, noise_variance, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP, write=True)
        print("iteration {}, error={}".format(iteration, error / steps))
    weights, precision_EP, Lambda_cholesky, Lambda = variational_classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
    t1, t2, t3, t4, t5 = variational_classifier.compute_integrals(
        gamma, Sigma, precision_EP, posterior_mean, noise_variance)
    fx = variational_classifier.evaluate_function(precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
    # Test
    Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi,
                                       noise_variance, X_test, Lambda, vectorised=True)  # (n_test, K)
    predictive_likelihood = Z[grid, t_test]
    predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
    print("predictive_likelihood ", predictive_likelihood)
    (x_lims, y_lims) = plot_lims["dataset"]

    N = 75
    x1 = np.linspace(x_lims[0], x_lims[1], N)
    x2 = np.linspace(y_lims[0], y_lims[1], N)
    xx, yy = np.meshgrid(x1, x2)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, 2))
    X_new_ = np.zeros((N * N, D))
    X_new_[:, :2] = X_new
    # Mean zero-one error
    t_star = np.argmax(Z, axis=1)
    print(t_star)
    print(t_test)
    zero_one = (t_star == t_test)
    mean_zero_one = zero_one * 1.
    mean_zero_one = np.sum(mean_zero_one) / len(t_test)
    mean_zero_one = 1. - mean_zero_one
    print("mean_zero_one_error", mean_zero_one)
    # Other error
    mean_absolute_error = np.sum(np.abs(t_star - t_test)) / len(t_test)
    print("mean_absolute_error ", mean_absolute_error)

    Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi,
                                       noise_variance, X_new_, Lambda, vectorised=True)
    Z_new = Z.reshape((N, N, K))
    print(np.sum(Z, axis=1), 'sum')
    for i in range(K):
        fig, axs = plt.subplots(1, figsize=(6, 6))
        plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
        plt.scatter(X_train[np.where(t_train == i)][:, 0], X_train[np.where(t_train == i)][:, 1], color='red')
        plt.scatter(X_test[np.where(t_test == i)][:, 0], X_test[np.where(t_test == i)][:, 1], color='blue')
        # plt.scatter(X_train[np.where(t == i + 1)][:, 0], X_train[np.where(t == i + 1)][:, 1], color='blue')

        # plt.xlim(0, 2)
        # plt.ylim(0, 2)
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.title("Contour plot - Expectation propagation")
        plt.savefig("contour_EP_{}.png".format(i))
        plt.show()
    return fx, zero_one, predictive_likelihood, mean_absolute_error


def EP_training(X_train, t_train, gamma_0, varphi_0, noise_variance_0, K, scale=1.0):
    """
    An example ordinal training function.

    :return:
    """
    theta = []
    theta.append(np.log(np.sqrt(noise_variance_0)))
    theta.append(gamma_0[1])
    for i in range(2, K):
        theta.append(np.log(gamma_0[i] - gamma_0[i - 1]))
    theta.append(np.log(varphi_0))
    theta = np.array(theta)
    kernel = SEIso(varphi_0, scale, sigma=10e-6, tau=10e-6)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    # Use L-BFGS-B
    res = minimize(variational_classifier.hyperparameter_training_step, theta, method='L-BFGS-B', jac=True, options={
        'maxiter':10})
    theta = res.x
    noise_variance = np.exp(theta[0])
    gamma = np.empty((K + 1,))  # including all of the cutpoints
    gamma[0] = np.NINF
    gamma[-1] = np.inf
    gamma[1] = theta[1]
    for i in range(2, K):
        gamma[i] = gamma[i - 1] + np.exp(theta[i])
    varphi = np.exp(theta[K])
    return gamma, varphi, noise_variance

def EP_training_varphi(X_train, t_train, varphi_0=1e-3, scale=1.0, sigma=10e-6, tau=10e-6):
    """
    An example ordinal training function.

    :return:
    """
    varphi = varphi_0
    # gamma = [-np.inf, -0.28436501, 0.36586332, 3.708507, 4.01687246, np.inf]
    # noise_variance = 4.927489010195959,
    gamma = np.array([-np.inf, -0.28436501, 0.36586332, 3.708507, 4.01687246, np.inf])
    noise_variance = 0.01
    theta = []
    theta.append(np.log(varphi))
    theta = np.array(theta)
    print("theta_0", theta)
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    # Use L-BFGS-B
    res = minimize(variational_classifier.hyperparameter_training_step_varphi, theta, method='L-BFGS-B', jac=True,
                   options={'maxiter':10})
    theta = res.x
    varphi = np.exp(theta[0])
    return gamma, varphi, noise_variance


def test(X_trains, t_trains, X_tests, t_tests, split, gamma_0, varphi_0, noise_variance_0, K, D, scale=1.0):
    X_train = X_trains[split, :, :]
    t_train = t_trains[split, :]
    X_test = X_tests[split, :, :]
    t_test = t_tests[split, :]

    gamma, varphi, noise_variance = EP_training(
        X_train, t_train, gamma_0, varphi_0, noise_variance_0, K, scale=scale)

    fx, zero_one, predictive_likelihood, mean_abs = EP_testing(
        X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, D, scale=scale)

    return gamma, varphi, noise_variance, zero_one, predictive_likelihood, mean_abs, fx


def test_varphi(X_trains, t_trains, X_tests, t_tests, K, scale=1.0):
    split = 2
    X_train = X_trains[split, :, :]
    t_train = t_trains[split, :]
    X_test = X_tests[split, :, :]
    t_test = t_tests[split, :]

    gamma, varphi, noise_variance = EP_training_varphi(
        X_train, t_train, scale=scale)

    bound, zero_one, predictive_likelihood, mean_abs = EP_testing(
        X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, scale=scale)

    print("gamma", gamma)
    print("noise_variance", noise_variance)
    print("varphi", varphi)
    print("zero_one", zero_one)
    print("predictive likelihood", predictive_likelihood)
    print("mean_abs", mean_abs)
    print("bound", bound)
    assert 0


def outer_loops(X_trains, t_trains, X_tests, t_tests, gamma_0, varphi_0, noise_variance_0, K, D):
    bounds = []
    zero_ones = []
    predictive_likelihoods = []
    mean_abss = []
    varphis = []
    noise_variances = []
    gammas = []
    for split in range(20):
        gamma, varphi, noise_variance, zero_one, predictive_likelihood, mean_abs, fx = test(
            X_trains, t_trains, X_tests, t_tests, split,
            gamma_0=gamma_0, varphi_0=varphi_0, noise_variance_0=noise_variance_0, K=K, D=D, scale = 1.0)
        bounds.append(fx)
        zero_ones.append(zero_one)
        predictive_likelihoods.append(predictive_likelihood)
        mean_abss.append(mean_abs)
        varphis.append(varphi)
        noise_variances.append(noise_variance)
        gammas.append(gamma[1:-1])
    bounds = np.array(bounds)
    zero_ones = np.array(zero_ones)
    predictive_likelihoods = np.array(predictive_likelihoods)
    predictive_likelihoods = np.array(predictive_likelihoods)
    mean_abss = np.array(mean_abss)
    varphis = np.array(varphis)
    noise_variances = np.array(noise_variances)
    gammas = np.array(gammas)
    avg_bound = np.average(bounds)
    std_bound = np.std(bounds)
    avg_predictive_likelihood = np.average(predictive_likelihoods)
    std_predictive_likelihood = np.std(predictive_likelihoods)
    avg_zero_one = np.average(zero_ones)
    std_zero_one = np.std(zero_ones)
    avg_mean_abs = np.average(mean_abss)
    std_mean_abs = np.std(mean_abss)
    print(avg_bound, std_bound, "bound, std")
    print(avg_predictive_likelihood, std_predictive_likelihood, "predictive likelihood, std")
    print(avg_zero_one, std_zero_one, "zero one, std")
    print(avg_mean_abs, std_mean_abs, "mean abs, std")
    avg_varphi = np.average(varphis)
    std_varphi = np.std(varphis)
    avg_noise_variances = np.average(noise_variances)
    std_noise_variances = np.std(noise_variances)
    avg_gammas = np.average(gammas, axis=1)
    std_gammas = np.std(gammas, axis=1)
    print(avg_varphi, std_varphi, "varphi, std")
    print(avg_noise_variances, std_noise_variances, "noise_variances, std")
    print(avg_gammas, std_gammas, "gammas, std")
    return 0


def SSouter_loops(X_trains, t_trains, X_tests, t_tests, Y_true, gamma_0):
    grid = np.ogrid[0:len(X_tests[0, :, :])]
    avg_bounds_Z = []
    avg_zero_one_Z = []
    avg_predictive_likelihood_Z = []
    avg_mean_abs_Z = []
    max_bounds = []
    max_zero_ones = []
    max_predictive_likelihoods = []
    max_mean_abss = []

    for split in range(20):
        X_train = X_trains[split, :, :]
        t_train = t_trains[split, :]
        X_test = X_tests[split, :, :]
        t_test = t_tests[split, :]
        # Y_true = Y_trues[split, :]

        lower_x1 = -10
        lower_x2 = -1
        upper_x1 = 0
        upper_x2 = 6
        N = 20
        x1 = np.logspace(lower_x1, upper_x1, N)
        x2 = np.logspace(lower_x2, upper_x2, N)
        xx, yy = np.meshgrid(x1, x2)
        X_new = np.dstack((xx, yy))
        X_new = X_new.reshape((N * N, 2))
        # Outer loop
        sigma = 10e-6
        tau = 10e-6

        bounds_Z = []
        zero_one_Z = []
        mean_abs_Z = []
        predictive_likelihood_Z = []
        tick = False

        for x_new in X_new:
            print(x_new)
            kernel = SEIso(x_new[0], x_new[1], sigma=sigma, tau=tau)
            # Initiate classifier
            variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
            steps = 50
            y_0 = Y_true.flatten()
            m_0 = y_0
            gamma, m_tilde, Sigma_tilde, C_tilde, y_tilde, varphi_tilde, bound, containers = variational_classifier.estimate(
                m_0, gamma_0, steps, varphi_0=x_new[0], fix_hyperparameters=True, write=False)
            bounds_Z.append(bound)
            # ms, ys, varphis, psis, bounds = containers

            # Test
            Z = variational_classifier.predict(gamma, Sigma_tilde, y_tilde, varphi_tilde, X_test,
                                               vectorised=True)  # (n_test, K)
            predictive_likelihood = Z[grid, t_test]
            predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
            predictive_likelihood_Z.append(predictive_likelihood)

            # Mean zero-one error
            t_star = np.argmax(Z, axis=1)
            zero_one = (t_star == t_test)
            mean_zero_one = zero_one * 1
            mean_zero_one = np.sum(mean_zero_one) / len(t_test)
            zero_one_Z.append(mean_zero_one)

            # Other error
            mean_absolute_error = np.sum(np.abs(t_star - t_test)) / len(t_test)
            mean_abs_Z.append(mean_absolute_error)
        avg_bounds_Z.append(bounds_Z)
        avg_zero_one_Z.append(zero_one_Z)
        avg_predictive_likelihood_Z.append(predictive_likelihood_Z)
        avg_mean_abs_Z.append(mean_abs_Z)

        bounds_Z = np.array(bounds_Z)
        predictive_likelihood_Z = np.array(predictive_likelihood_Z)
        zero_one_Z = np.array(zero_one_Z)
        mean_abs_Z = np.array(mean_abs_Z)


        max_bound = np.max(bounds_Z)
        max_predictive_likelihood = np.max(predictive_likelihood_Z)
        max_zero_one = np.max(zero_one_Z)
        max_mean_abs = np.min(mean_abs_Z)
        max_bounds.append(max_bound)
        max_zero_ones.append(max_zero_one)
        max_mean_abss.append(max_mean_abs)
        max_predictive_likelihoods.append(max_predictive_likelihood)
        argmax_bound = np.argmax(bounds_Z)
        argmax_predictive_likelihood = np.argmax(predictive_likelihood_Z)
        argmax_zero_one = np.argmax(zero_one_Z)

        bounds_Z = bounds_Z.reshape((N, N))
        predictive_likelihood_Z = predictive_likelihood_Z.reshape((N, N))
        zero_one_Z = zero_one_Z.reshape((N, N))

        # fig, axs = plt.subplots(1, figsize=(6, 6))
        # plt.contourf(x1, x2, predictive_likelihood_Z)
        # plt.scatter(X_new[argmax_predictive_likelihood, 0], X_new[argmax_predictive_likelihood, 1], c='r')
        # axs.set_xscale('log')
        # axs.set_yscale('log')
        # plt.xlabel(r"$\log{\varphi}$", fontsize=16)
        # plt.ylabel(r"$\log{s}$", fontsize=16)
        # plt.title("Contour plot - Predictive likelihood of test set")
        # plt.savefig("contour_predictive_likelihood.png")
        # plt.show()

        # fig, axs = plt.subplots(1, figsize=(6, 6))
        # plt.contourf(x1, x2, bounds_Z)
        # plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
        # axs.set_xscale('log')
        # axs.set_yscale('log')
        # plt.xlabel(r"$\log{\varphi}$", fontsize=16)
        # plt.ylabel(r"$\log{s}$", fontsize=16)
        # plt.title("Contour plot - Variational lower bound")
        # plt.savefig("variational_bound.png")
        # plt.show()
        #
        # fig, axs = plt.subplots(1, figsize=(6, 6))
        # plt.contourf(x1, x2, zero_one_Z)
        # plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
        # axs.set_xscale('log')
        # axs.set_yscale('log')
        # plt.xlabel(r"$\log{\varphi}$", fontsize=16)
        # plt.ylabel(r"$\log{s}$", fontsize=16)
        # plt.title("Contour plot - mean zero-one accuracy")
        # plt.savefig("mean_zero_one.png")
        # plt.show()

    avg_max_bound = np.average(np.array(max_bounds))
    std_max_bound = np.std(np.array(max_bounds))
    avg_max_predictive_likelihood = np.average(np.array(max_predictive_likelihoods))
    std_max_predictive_likelihood = np.std(np.array(max_predictive_likelihoods))
    avg_max_zero_one = np.average(np.array(max_zero_ones))
    std_max_zero_one = np.std(np.array(max_zero_ones))
    avg_max_mean_abs = np.average(np.array(max_mean_abss))
    std_max_mean_abs = np.std(np.array(max_mean_abss))
    print(avg_max_bound, std_max_bound, "average max bound, std")
    print(avg_max_predictive_likelihood, std_max_predictive_likelihood, "average max predictive likelihood, std")
    print(avg_max_zero_one, std_max_zero_one, "average max zero one, std")
    print(avg_max_mean_abs, std_max_mean_abs, "average max mean abs, std")

    avg_bounds_Z = np.array(avg_bounds_Z)
    avg_predictive_likelihood_Z = np.array(avg_predictive_likelihood_Z)
    avg_zero_one_Z = np.array(avg_zero_one_Z)
    avg_mean_abs_Z = np.array(avg_mean_abs_Z)
    avg_bounds_Z = np.average(avg_bounds_Z, axis=0)
    avg_predictive_likelihood_Z = np.average(avg_predictive_likelihood_Z, axis=0)
    avg_zero_one_Z = np.average(avg_zero_one_Z, axis=0)
    avg_mean_abs_Z = np.average(avg_mean_abs_Z, axis=0)

    std_max_bound = np.std(np.array(avg_bounds_Z))
    std_max_predictive_likelihood = np.std(np.array(avg_predictive_likelihood_Z))
    std_max_zero_one = np.std(np.array(avg_zero_one_Z))
    std_max_mean_abs = np.std(np.array(avg_mean_abs_Z))

    argmax_bound = np.argmax(avg_bounds_Z)
    argmax_predictive_likelihood = np.argmax(avg_predictive_likelihood_Z)
    argmax_zero_one = np.argmax(avg_zero_one_Z)
    argmax_mean_abs = np.argmax(avg_mean_abs_Z)

    max_bound = np.max(avg_bounds_Z)
    max_predictive_likelihood = np.max(avg_predictive_likelihood_Z)
    max_zero_one = np.max(avg_zero_one_Z)
    max_mean_abs = np.min(avg_mean_abs_Z)
    print(max_bound, std_max_bound, "Max avg bound, std", X_new[argmax_bound], "parameters")
    print(max_predictive_likelihood, std_max_predictive_likelihood, "Max avg predictive likelihood, std", X_new[argmax_predictive_likelihood], "parameters")
    print(max_zero_one, std_max_zero_one, "Max avg zero one, std", X_new[argmax_zero_one], "parameters")
    print(max_mean_abs, std_max_mean_abs, "Max avg mean abs, std", X_new[argmax_mean_abs], "parameters")

    avg_bounds_Z = avg_bounds_Z.reshape((N, N))
    avg_predictive_likelihood_Z = avg_predictive_likelihood_Z.reshape((N, N))
    avg_zero_one_Z = avg_zero_one_Z.reshape((N, N))
    avg_mean_abs_Z = avg_mean_abs_Z.reshape((N, N))
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_predictive_likelihood_Z)
    plt.scatter(X_new[argmax_predictive_likelihood, 0], X_new[argmax_predictive_likelihood, 1], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - Predictive likelihood of test set")
    plt.savefig("predictive_likelihood.png")
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_bounds_Z)
    plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - Variational lower bound")
    plt.savefig("variational_bound.png")
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_zero_one_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - mean zero-one accuracy")
    plt.savefig("mean_zero_one.png")
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_mean_abs_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - mean absolute error accuracy")
    plt.savefig("mean_absolute.png")
    plt.show()

def grid_toy(X_train, t_train, gamma, range_log_varphi, range_log_noise_std, scale=1.0):
    """Grid of optimised lower bound across the hyperparameters with cutpoints set."""
    sigma = 10e-6
    tau = 10e-6
    res = 30
    varphi_0 = 1.0
    kernel = SEIso(varphi_0, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    Z, grad, x, y = variational_classifier.grid_over_hyperparameters(
        gamma, range_log_varphi, range_log_noise_std, res)
    fig, axs = plt.subplots(1, figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, Z, rstride=1, cstride=1, alpha=0.4,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    plt.show()
    norm = np.linalg.norm(np.array((grad[:, 0], grad[:, 1])), axis=0)
    u = grad[:, 0] / norm
    v = grad[:, 1] / norm

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    ax.contourf(x, y, np.log(Z), 100, cmap='viridis', zorder=1)
    ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
    ax.plot(0.1, 30, 'm')

    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)
    plt.xlabel(r"$\sigma$", fontsize=16)
    plt.ylabel(r"$\varphi$", fontsize=16)
    plt.title("Contour plot - EP lower bound on the log likelihood")
    plt.savefig("contour_bound.png")
    plt.show()


def test_toy(dataset, X_train, t_train, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D, scale=1.0):
    """Test toy."""
    # gamma, varphi, noise_variance = EP_training(X_train, t_train, gamma_0, varphi_0, noise_variance_0, K, scale=1.0)
    # print("gamma = {}, gamma_0 = {}".format(gamma, gamma_0))
    # print("varphi = {}, varphi_0 = {}".format(varphi, varphi_0))
    # print("noise_variance = {}, noise_variance_0 = {}".format(noise_variance, noise_variance_0))
    # print("gamma_0 = {}, varphi_0 = {}, noise_variance_0 = {}".format(gamma_0, varphi_0, noise_variance_0))
    fx = EP_plotting(dataset, X_train, t_train, X_true, Y_true, gamma_0,
                     varphi=varphi_0, noise_variance=noise_variance_0, K=K, D=D, scale=scale)
    print("fx={}".format(fx))


def test_plots(dataset, X_test, X_train, t_test, t_train, Y_true, gamma, varphi, noise_variance, K):
    grid = np.ogrid[0:len(X_test)]
    sigma = 10e-6
    tau = 10e-6
    kernel = SEIso(varphi, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    steps = 50
    y_0 = Y_true.flatten()
    m_0 = y_0
    (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma,
     mean_EP, precision_EP, amplitude_EP, containers) = variational_classifier.estimate(
        steps, gamma, varphi, noise_variance, fix_hyperparameters=False, write=True)
    weights, precision_EP, L, Lambda = variational_classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean)

    if dataset in datasets:
        lower_x1 = 0.0
        upper_x1 = 16.0
        lower_x2 = -30
        upper_x2 = 0
        N = 60

        x1 = np.linspace(lower_x1, upper_x1, N)
        x2 = np.linspace(lower_x2, upper_x2, N)
        xx, yy = np.meshgrid(x1, x2)
        X_new = np.dstack((xx, yy))
        X_new = X_new.reshape((N * N, 2))

        # Test
        Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda,
                                           vectorised=True)
        predictive_likelihood = Z[grid, t_test]
        predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
        print("predictive_likelihood ", predictive_likelihood)

        # Mean zero-one error
        t_star = np.argmax(Z, axis=1)
        print(t_star)
        print(t_test)
        zero_one = (t_star == t_test)
        mean_zero_one = zero_one * 1
        mean_zero_one = np.sum(mean_zero_one) / len(t_test)
        print("mean_zero_one ", mean_zero_one)

        Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda,
                                           vectorised=True)
        Z_new = Z.reshape((N, N, K))
        print(np.sum(Z, axis=1), 'sum')
        for i in range(K):
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
            plt.scatter(X_train[np.where(t_train == i)][:, 0], X_train[np.where(t_train == i)][:, 1], color='red')
            plt.scatter(X_test[np.where(t_test == i)][:, 0], X_test[np.where(t_test == i)][:, 1], color='blue')
            #plt.scatter(X_train[np.where(t == i + 1)][:, 0], X_train[np.where(t == i + 1)][:, 1], color='blue')

            # plt.xlim(0, 2)
            # plt.ylim(0, 2)
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.title("Contour plot - Variational")
            plt.show()


def main():
    """Conduct an EP estimation/optimisation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="quantile or decile")
    parser.add_argument(
        "--data_from_prior", help="data is from prior?", action='store_const', const=True)
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = args.bins
    data_from_prior = args.data_from_prior
    write_path = pathlib.Path(__file__).parent.absolute()
    if dataset in datasets:
        X_trains, t_trains, X_tests, t_tests, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D = load_data(
            dataset, bins)
        outer_loops(X_trains, t_trains, X_tests, t_tests, gamma_0, varphi_0, noise_variance_0, K, D)
        # gamma, varphi, noise_variance = EP_training(
        #     X_trains[2], t_trains[2], X_tests[2], t_tests[2], gamma_0, varphi_0, noise_variance_0, K)
        # EP_testing(
        #     X_trains[2], t_trains[2], X_tests[2], t_tests[2], gamma, varphi, noise_variance, K, scale=1.0)
    else:
        X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D = load_data_synthetic(dataset, data_from_prior)
        # test_plots(dataset, X_tests[0], X_trains[0], t_tests[0], t_trains[0], Y_trues[0])
        grid_toy(X, t, gamma_0, [-2, 2], [-1, 1], scale=1.0)
        # test_toy(dataset, X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D, scale=1.0)

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()


if __name__ == "__main__":
    main()

