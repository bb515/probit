"""
Ordered probit regression concrete examples. Approximate inference: VB approximation.
"""
import argparse
import cProfile
from io import StringIO
from operator import pos
from pstats import Stats, SortKey
import numpy as np
from probit.estimators import VBOrderedGP
from probit.kernels import SEIso
import matplotlib.pyplot as plt
import pathlib
from scipy.optimize import minimize
from probit.data.utilities import (generate_prior_data, generate_synthetic_data, get_Y_trues, colors, datasets,
    metadata, load_data, load_data_synthetic, training, training_varphi)
import sys
import time


now = time.ctime()
write_path = pathlib.Path()


def VB_plot(dataset, X_train, t_train, X_true, Y_true, m_0, gamma, steps, varphi, noise_variance, K, D, scale):
    """Plots for Chu data."""
    kernel = SEIso(varphi, scale, sigma=10e-6, tau=10e-6)
    # Initiate classifier
    variational_classifier = VBOrderedGP(X_train, t_train, kernel)
    y_0 = Y_true.flatten()
    m_0 = y_0
    # TODO: correct these output arguments
    gamma, m_tilde, dm_tilde, Sigma_tilde, cov, C_tilde, calligraphic_Z, y_tilde, p, varphi_tilde, *_ = variational_classifier.estimate(
        steps, gamma, varphi, noise_variance=noise_variance, m_tilde_0=m_0, fix_hyperparameters=True, write=False)
    fx = variational_classifier.evaluate_function(m_tilde, Sigma_tilde, C_tilde, calligraphic_Z)
    (xlims, ylims) = metadata[dataset]["plot_lims"]
    N = 75
    x1 = np.linspace(xlims[0], xlims[1], N)
    x2 = np.linspace(ylims[0], ylims[1], N)
    xx, yy = np.meshgrid(x1, x2)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, 2))
    X_new_ = np.zeros((N * N, D))
    X_new_[:, :2] = X_new
    Z = variational_classifier.predict(gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new_)
    Z_new = Z.reshape((N, N, K))
    print(np.sum(Z, axis=1), 'sum')
    for i in range(K):
        fig, axs = plt.subplots(1, figsize=(6, 6))
        plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
        plt.scatter(X_train[np.where(t_train == i)][:, 0], X_train[np.where(t_train == i)][:, 1], color='red')
        # plt.scatter(X_train[np.where(t == i + 1)][:, 0], X_train[np.where(t == i + 1)][:, 1], color='blue')
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.savefig("contour_EP_{}.png".format(i))
        plt.close()
    return fx


def VB_plot_synthetic(dataset, X, t, X_true, Y_true, m_tilde_0, gamma, steps, varphi, noise_variance, K, D,
                        scale=1.0, sigma=10e-6, tau=10e-6, colors=colors):
    """Plots for synthetic data."""
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    print("noise_variance", noise_variance)
    variational_classifier = VBOrderedGP(noise_variance, X, t, kernel)
    (m_tilde, dm_tilde, Sigma_tilde, cov, C, y_tilde, p, varphi_tilde, containers) = variational_classifier.estimate(
        steps, gamma, varphi, noise_variance=noise_variance, m_tilde_0=m_tilde_0, fix_hyperparameters=True, write=True)
    print(np.shape(X))
    print(np.shape(m_tilde))
    plt.scatter(X, m_tilde)
    plt.plot(X_true, Y_true)
    plt.show()
    (ms, ys, varphis, psis, fxs) = containers
    plt.plot(fxs)
    plt.title("Variational lower bound on the marginal likelihood")
    plt.show()
    # print(Sigma_tilde)
    # print(C)
    # print(varphi_tilde, "varphi")
    noise_std = np.sqrt(noise_variance)
    calligraphic_Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, *_ = variational_classifier._calligraphic_Z(
                    gamma, noise_std, m_tilde,
                    upper_bound=variational_classifier.upper_bound, upper_bound2=variational_classifier.upper_bound2)
    N = variational_classifier.N
    fx, _ = variational_classifier.evaluate_function(N, m_tilde, Sigma_tilde, C, calligraphic_Z, noise_variance)
    if dataset == "tertile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        print("y", y_tilde)
        print("varphi", varphi_tilde)
        print("noisevar", noise_variance)
        Z, posterior_predictive_m, posterior_std = variational_classifier.predict(gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(x_lims)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.stackplot(x, Z.T,
                      labels=(
                          r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
                      colors=(
                          colors[0], colors[1], colors[2])
                      )
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
        for k in range(K):
            plt.scatter(X[np.where(t == k)], np.zeros_like(X[np.where(t == k)]) + val, facecolors=colors[k],
                        edgecolors='white')
        plt.savefig("Ordered Gibbs Cumulative distribution plot of class distributions for x_new=[{}, {}].png"
                  .format(x_lims[0], x_lims[1]))
        plt.show()
        plt.close()
        plt.plot(X_new, posterior_predictive_m, 'r')
        plt.fill_between(X_new[:, 0], posterior_predictive_m - 2*posterior_std, posterior_predictive_m + 2*posterior_std,
                 color='red', alpha=0.2)
        plt.plot(X_true, Y_true, 'b')
        plt.ylim(-0.5, 1.5)
        plt.savefig("scatter_versus_posterior_mean.png")
        plt.show()
        plt.close()
    elif dataset == "septile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        Z, posterior_predictive_m, posterior_std = variational_classifier.predict(gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(x_lims)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.stackplot(x, Z.T,
            labels=(
                r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$", r"$p(t=3|x, X, t)$",
                r"$p(t=4|x, X, t)$", r"$p(t=5|x, X, t)$", r"$p(t=6|x, X, t)$"),
            colors=(
                colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6]))
        plt.legend()
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
        for i in range(K):
            plt.scatter(
                X[np.where(t == i)], np.zeros_like(X[np.where(t == i)]) + val, facecolors=colors[i], edgecolors='white')
        plt.savefig("Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}].png"
                  .format(x_lims[1], x_lims[0]))
        plt.close()
    elif dataset=="thirteen":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        print("y", y_tilde)
        print("varphi", varphi_tilde)
        print("noisevar", noise_variance)
        Z, posterior_predictive_m, posterior_std = variational_classifier.predict(gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(x_lims)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.stackplot(x, Z.T,
                      labels=(
                          r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
                      colors=colors
                      )
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
        for k in range(K):
            plt.scatter(X[np.where(t == k)], np.zeros_like(X[np.where(t == k)]) + val, facecolors=colors[k],
                        edgecolors='white')
        plt.savefig("Ordered Gibbs Cumulative distribution plot of class distributions for x_new=[{}, {}].png"
                  .format(x_lims[0], x_lims[1]))
        plt.show()
        plt.close()
        plt.plot(X_new, posterior_predictive_m, 'r')
        plt.fill_between(X_new[:, 0], posterior_predictive_m - 2*posterior_std, posterior_predictive_m + 2*posterior_std,
                 color='red', alpha=0.2)
        plt.plot(X_true, Y_true, 'b')
        plt.ylim(-0.5, 1.5)
        plt.savefig("scatter_versus_posterior_mean.png")
        plt.show()
        plt.close()
    return fx


def VB_testing(
        dataset, X_train, t_train, X_test, t_test, m_tilde_0, gamma, steps, varphi, noise_variance,
        K, D, scale=1.0, sigma=10e-6, tau=10e-6):
    grid = np.ogrid[0:len(X_test[:, :])]
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = VBOrderedGP(X_train, t_train, kernel)
    m_tilde, dm_tilde, Sigma_tilde, cov, C_tilde, calligraphic_Z, y_tilde, p, varphi_tilde, *_ = variational_classifier.estimate(
        steps, gamma, varphi, noise_variance=noise_variance, m_tilde_0=m_tilde_0, fix_hyperparameters=True, write=False)
    fx = variational_classifier.evaluate_function(m_tilde, Sigma_tilde, C_tilde, calligraphic_Z, verbose=True)
    # Test
    Z, posterior_predictive_m, posterior_std = variational_classifier.predict(gamma, cov, y_tilde, varphi, noise_variance, X_test)  # (N_test, K)
    predictive_likelihood = Z[grid, t_test]
    predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
    print("predictive_likelihood ", predictive_likelihood)
    (x_lims, y_lims) = metadata[dataset]["plot_lims"]
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
    zero_one = (t_star != t_test)
    mean_zero_one = zero_one * 1.
    mean_zero_one = np.sum(mean_zero_one) / len(t_test)
    print("mean_zero_one_error", mean_zero_one)
    # Other error
    mean_absolute_error = np.sum(np.abs(t_star - t_test)) / len(t_test)
    print("mean_absolute_error ", mean_absolute_error)
    Z, posterior_predictive_m, posterior_std = variational_classifier.predict(gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new_)
    Z_new = Z.reshape((N, N, K))
    print(np.sum(Z, axis=1), 'sum')
    for i in range(K):
        fig, axs = plt.subplots(1, figsize=(6, 6))
        plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
        plt.scatter(X_train[np.where(t_train == i)][:, 0], X_train[np.where(t_train == i)][:, 1], color='red')
        plt.scatter(X_test[np.where(t_test == i)][:, 0], X_test[np.where(t_test == i)][:, 1], color='blue')
        # plt.scatter(X_train[np.where(t == i + 1)][:, 0], X_train[np.where(t == i + 1)][:, 1], color='blue')
        # plt.xlim(xlims)
        # plt.ylim(ylims)
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.savefig("contour_VB_{}.png".format(i))
        plt.close()
    return fx, zero_one, predictive_likelihood, mean_absolute_error


def test(dataset, X_trains, t_trains, X_tests, t_tests, split, m_0, gamma_0, varphi_0, noise_variance_0, steps, K, D, scale=1.0):
    X_train = X_trains[split, :, :]
    t_train = t_trains[split, :]
    X_test = X_tests[split, :, :]
    t_test = t_tests[split, :]
    #gamma = gamma_0
    #varphi = varphi_0
    #noise_variance = noise_variance_0

    kernel = SEIso(varphi_0, scale, sigma=10e-6, tau=10e-6)
    # Initiate classifier
    variational_classifier = VBOrderedGP(X_train, t_train, kernel)

    gamma, varphi, noise_variance = training(
        variational_classifier, X_train, t_train, m_0, gamma_0, varphi_0, noise_variance_0, K, scale=scale)
    fx, zero_one, predictive_likelihood, mean_abs = VB_testing(
        dataset, X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, D, scale=scale)
    return gamma, varphi, noise_variance, zero_one, predictive_likelihood, mean_abs, fx


def test_varphi(dataset, variational_classifier, method, gamma, varphi_0, noise_variance, X_trains, t_trains, X_tests,
        t_tests, K, scale=1.0):
    split = 2
    X_train = X_trains[split, :, :]
    t_train = t_trains[split, :]
    X_test = X_tests[split, :, :]
    t_test = t_tests[split, :]
    gamma, varphi, noise_variance = training_varphi(
        dataset, method, variational_classifier, gamma, varphi_0, noise_variance)
    bound, zero_one, predictive_likelihood, mean_abs = VB_testing(
        dataset, X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, scale=scale)
    print("gamma", gamma)
    print("noise_variance", noise_variance)
    print("varphi", varphi)
    print("zero_one", zero_one)
    print("predictive likelihood", predictive_likelihood)
    print("mean_abs", mean_abs)
    print("bound", bound)
    assert 0


def outer_loops(dataset, X_trains, t_trains, X_tests, t_tests, gamma_0, varphi_0, noise_variance_0, K, D):
    bounds = []
    zero_ones = []
    predictive_likelihoods = []
    mean_abss = []
    varphis = []
    noise_variances = []
    gammas = []
    for split in range(20):
        gamma, varphi, noise_variance, zero_one, predictive_likelihood, mean_abs, fx = test(
            dataset, X_trains, t_trains, X_tests, t_tests, split,
            gamma_0=gamma_0, varphi_0=varphi_0, noise_variance_0=noise_variance_0, K=K, D=D, scale=1.0)
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
    avg_gammas = np.average(gammas, axis=0)
    std_gammas = np.std(gammas, axis=0)
    print(avg_varphi, std_varphi, "varphi, std")
    print(avg_noise_variances, std_noise_variances, "noise_variances, std")
    print(avg_gammas, std_gammas, "gammas, std")
    return 0


def outer_loops_Rogers(X_trains, t_trains, X_tests, t_tests, Y_true, gamma, plot=False):
    steps = 50
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
        for x_new in X_new:
            noise_std = x_new[0]
            noise_variance = noise_std**2
            varphi = x_new[1]
            kernel = SEIso(varphi, scale=1.0, sigma=sigma, tau=tau)
            # Initiate classifier
            y_0 = Y_true.flatten()
            m_0 = y_0
            variational_classifier = VBOrderedGP(X_train, t_train, kernel)
            m_tilde, dm_tilde, Sigma_tilde, cov, C_tilde, calligraphic_Z, y_tilde, p, varphi_tilde, *_ = variational_classifier.estimate(
                steps, gamma, varphi, noise_variance=noise_variance, m_tilde_0=m_0, fix_hyperparameters=True,
                write=False)
            fx = variational_classifier.evaluate_function(m_tilde, Sigma_tilde, C_tilde, calligraphic_Z, verbose=True)
            bounds_Z.append(fx)
            # Test
            Z, posterior_predictive_m, posterior_std = variational_classifier.predict(gamma, cov, y_tilde, varphi_tilde, noise_variance, X_test)
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
        if plot==True:
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, predictive_likelihood_Z)
            plt.scatter(X_new[argmax_predictive_likelihood, 0], X_new[argmax_predictive_likelihood, 1], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - Predictive likelihood of test set.png")
            plt.close()
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, bounds_Z)
            plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - Variational lower bound.png")
            plt.close()
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, zero_one_Z)
            plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - mean zero-one accuracy.png")
            plt.close()
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
    plt.savefig("Contour plot - Predictive likelihood of test set.png")
    plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_bounds_Z)
    plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - Variational lower bound.png")
    plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_zero_one_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - mean zero-one accuracy.png")
    plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_mean_abs_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - mean absolute error accuracy.png")
    plt.close()

def grid_synthetic(X_train, t_train, range_x1, range_x2,
                   gamma=None, varphi=None, noise_variance=None, scale=1.0, fix_s=True, show=False):
    """Grid of optimised lower bound across the hyperparameters with cutpoints set."""
    sigma = 10e-6
    tau = 10e-6
    res = 100
    # Just for initiation
    kernel = SEIso(varphi=1.0, scale=scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = VBOrderedGP(noise_variance, X_train, t_train, kernel)
    Z, grad, x, y, xlabel, ylabel, xscale, yscale = variational_classifier.grid_over_hyperparameters(
        range_x1, range_x2, res, gamma_0=gamma, varphi_0=varphi, noise_variance_0=noise_variance,
        scale_0=scale, fix_s=fix_s)
    print("xscale={}, yscale={}".format(xscale, yscale))
    if ylabel is None:
        plt.plot(x, Z)
        plt.savefig("grid_over_hyperparameters.png")
        if show: plt.show()
        plt.close()
        # norm = np.abs(np.max(grad))
        # u = grad / norm
        plt.plot(x, Z, 'b')
        plt.xscale(xscale)
        plt.ylabel(r"\mathcal{F}(\varphi)")
        plt.savefig("bound.png")
        if show: plt.show()
        plt.close()
        plt.plot(x, grad, 'r')
        plt.xscale(xscale)
        plt.xlabel(xlabel)
        plt.ylabel(r"\frac{\partial \mathcal{F}}{\partial varphi}")
        plt.savefig("grad.png")
        plt.close()
        #Normalization:
        dx = np.diff(x) # use np.diff(x) if x is not uniform
        #First derivatives: need to calculate them in the log domain
        log_x = np.log(x)
        dlog_x = np.diff(log_x)
        print(dlog_x)
        dZ_ = np.gradient(Z, log_x)
        dZ = np.diff(Z) / dlog_x
        plt.figure()
        plt.plot(log_x, dZ_, 'r.', label='np.grad')
        plt.plot(log_x[:-1], dZ, 'r--', label='np.diff, 1')
        plt.xlabel(xlabel)
        plt.ylabel(r"\frac{\partial \mathcal{F}}{\partial varphi}")
        plt.savefig("grad_finite_diff.png")
        if show: plt.show()
        plt.close()
        plt.plot(log_x, grad)
        plt.xlabel(xlabel)
        plt.ylabel(r"\frac{\partial \mathcal{F}}{\partial varphi}")
        plt.plot(log_x, dZ_, 'r.', label='np.grad')
        plt.plot(log_x[:-1], dZ, 'r', label='np.diff, 1')
        plt.savefig("both.png")
        if show: plt.show()
        plt.close()
        plt.plot(log_x, Z, 'b')
        plt.plot(log_x, grad, 'r', label='grad')
        plt.plot(log_x[:-1], dZ, 'r--', label='np.diff, 1')
        plt.plot(log_x, dZ_, 'r.', label='np.grad')
        plt.xlabel(xlabel)
        plt.ylabel(r"\mathcal{F}(\varphi), \frac{\partial \mathcal{F}}{\partial varphi}")
        plt.savefig("bound_grad.png")
        if show: plt.show()
        plt.close()
    else:
        fig, axs = plt.subplots(1, figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, Z, rstride=1, cstride=1, alpha=0.4,
                        cmap='viridis', edgecolor='none')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.savefig("grid_over_hyperparameters.png")
        if show: plt.show()
        plt.close()
        norm = np.linalg.norm(np.array((grad[:, 0], grad[:, 1])), axis=0)
        u = grad[:, 0] / norm
        v = grad[:, 1] / norm
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.contourf(x, y, Z, 100, cmap='viridis', zorder=1)
        ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        ax.plot(0.1, 30, 'm')
        plt.xscale(xscale)
        plt.xlim(1, 100.0)
        plt.yscale(yscale)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.savefig("Contour plot - VB lower bound on the log likelihood.png")
        if show: plt.show()
        plt.close()
        # fig, ax = plt.subplots(1, 1)
        # ax.set_aspect(1)
        # ax.contourf(x, y, np.log(Z), 100, cmap='viridis', zorder=1)
        # ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        # ax.plot(0.1, 30, 'm')
        # plt.xscale(xscale)
        # plt.yscale(yscale)
        # plt.xlabel(xlabel, fontsize=16)
        # plt.ylabel(ylabel, fontsize=16)
        # plt.savefig("Contour plot - log VB lower bound on the log likelihood.png")
        # if show: plt.show()
        # plt.close()


def VB_training(dataset, method, X_train, t_train, gamma_0, varphi_0, noise_variance_0, K, scale=1.0):
    """
    An example ordinal training function.

    Returns the hyperparameters trained via gradient descent of the ELBO.

    :return: gamma, varphi, noise_variance
    """
    # Initiate kernel
    kernel = SEIso(varphi_0, scale=scale, sigma=10e-6, tau=10e-6)
    # Initiate classifier
    variational_classifier = VBOrderedGP(X_train, t_train, kernel)
    gamma, varphi, noise_variance = training(
        dataset, method, variational_classifier, gamma_0, varphi_0, noise_variance_0, K)
    return gamma, varphi, noise_variance


def VB_training_varphi(dataset, method, X_train, t_train, gamma, varphi_0, noise_variance, scale=1.0):
    """
    An example ordinal training function.

    :return: gamma, varphi, noise_variance
    """
    varphi = varphi_0
    gamma = np.array([-np.inf, -0.28436501, 0.36586332, 3.708507, 4.01687246, np.inf])
    noise_variance = 0.01
    theta = []
    theta.append(np.log(varphi))
    theta = np.array(theta)
    print("theta_0", theta)
    kernel = SEIso(varphi, scale=scale, sigma=10e-6, tau=10e-6)
    # Initiate classifier
    variational_classifier = VBOrderedGP(X_train, t_train, kernel)
    gamma, varphi, noise_variance = training_varphi(
        dataset, method, variational_classifier, gamma, varphi_0, noise_variance)
    return gamma, varphi, noise_variance


def test_synthetic(
    dataset, method, X_train, t_train, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D, scale=1.0, colors=colors):
    """Test some particular hyperparameters."""
    gamma = gamma_0
    varphi = varphi_0
    noise_variance = noise_variance_0
    steps = 1000
    # kernel = SEIso(varphi_0, sigma=10e-6, tau=10e-6)
    # # Initiate classifier
    # variational_classifier = VBOrderedGP(noise_variance_0, X_train, t_train, kernel)
    # gamma, varphi, noise_variance = VB_training(
    #     dataset, method, variational_classifier, gamma_0, varphi_0, noise_variance_0, K)
    # print("gamma = {}, gamma_0 = {}".format(gamma, gamma_0))
    # print("varphi = {}, varphi_0 = {}".format(varphi, varphi_0))
    # print("noise_variance = {}, noise_variance_0 = {}".format(noise_variance, noise_variance_0))
    # # print("gamma_0 = {}, varphi_0 = {}, noise_variance_0 = {}".format(gamma_0, varphi_0, noise_variance_0))
    fx = VB_plot_synthetic(dataset, X_train, t_train, X_true, Y_true, None, gamma, steps,
        varphi, noise_variance, K, D, scale=1.0, sigma=10e-6, tau=10e-6, colors=colors)
    print("fx={}".format(fx))
    return fx


# def test_plots(dataset, X_test, X_train, t_test, t_train, Y_true, gamma, varphi, noise_variance, K):
#     """TODO: looks like it needs fixing for VB"""
#     grid = np.ogrid[0:len(X_test)]
#     kernel = SEIso(varphi, sigma=10e-6, tau=10e-6)
#     # Initiate classifier
#     variational_classifier = VBOrderedGP(X_train, t_train, kernel)
#     steps = 50
#     y_0 = Y_true.flatten()
#     m_0 = y_0
#     (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma,
#      mean_EP, precision_EP, amplitude_EP, containers) = variational_classifier.estimate(
#         steps, gamma, varphi, noise_variance, fix_hyperparameters=False, write=True)
#     weights, precision_EP, L, Lambda = variational_classifier.compute_EP_weights(
#         precision_EP, mean_EP, grad_Z_wrt_cavity_mean)

#     if dataset in datasets:
#         lower_x1 = 0.0
#         upper_x1 = 16.0
#         lower_x2 = -30
#         upper_x2 = 0
#         N = 60

#         x1 = np.linspace(lower_x1, upper_x1, N)
#         x2 = np.linspace(lower_x2, upper_x2, N)
#         xx, yy = np.meshgrid(x1, x2)
#         X_new = np.dstack((xx, yy))
#         X_new = X_new.reshape((N * N, 2))

#         # Test
#         Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda,
#                                            vectorised=True)
#         predictive_likelihood = Z[grid, t_test]
#         predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
#         print("predictive_likelihood ", predictive_likelihood)

#         # Mean zero-one error
#         t_star = np.argmax(Z, axis=1)
#         print(t_star)
#         print(t_test)
#         zero_one = (t_star == t_test)
#         mean_zero_one = zero_one * 1
#         mean_zero_one = np.sum(mean_zero_one) / len(t_test)
#         print("mean_zero_one ", mean_zero_one)

#         Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda,
#                                            vectorised=True)
#         Z_new = Z.reshape((N, N, K))
#         print(np.sum(Z, axis=1), 'sum')
#         for i in range(K):
#             fig, axs = plt.subplots(1, figsize=(6, 6))
#             plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
#             plt.scatter(X_train[np.where(t_train == i)][:, 0], X_train[np.where(t_train == i)][:, 1], color='red')
#             plt.scatter(X_test[np.where(t_test == i)][:, 0], X_test[np.where(t_test == i)][:, 1], color='blue')
#             #plt.scatter(X_train[np.where(t == i + 1)][:, 0], X_train[np.where(t == i + 1)][:, 1], color='blue')
#             # plt.xlim(0, 2)
#             # plt.ylim(0, 2)
#             plt.xlabel(r"$x_1$", fontsize=16)
#             plt.ylabel(r"$x_2$", fontsize=16)
#             plt.savefig("Contour plot - Variational.png")
#             plt.close()


def main():
    """Conduct an EP estimation/optimisation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="quantile or decile")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
    parser.add_argument(
        "--data_from_prior", help="data is from prior?", action='store_const', const=True)
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = args.bins
    data_from_prior = args.data_from_prior
    method = args.method
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets:
        X_trains, t_trains, X_tests, t_tests, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D = load_data(  # TODO: update with colors
            dataset, bins)
        outer_loops(dataset, X_trains, t_trains, X_tests, t_tests, gamma_0, varphi_0, noise_variance_0, K, D)
        # gamma, varphi, noise_variance = VB_training(dataset, method, X_trains[2], t_trains[2], gamma_0, varphi_0,
        #     noise_variance_0, K)
        # VB_testing(
        #     dataset, X_trains[2], t_trains[2], X_tests[2], t_tests[2], gamma=gamma, varphi=varphi,
        #     noise_variance=noise_variance, K)
    else:
        X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, scale_0, K, D, colors = load_data_synthetic(dataset, data_from_prior)
        print(gamma_0)
        print(varphi_0)
        print(noise_variance_0)
        print(scale_0)
        # print(noise_variance_0)
        # plt.scatter(X, Y_true)
        # plt.show()
        # test_plots(dataset, X_tests[0], X_trains[0], t_tests[0], t_trains[0], Y_trues[0])
        # just scale
        # grid_synthetic(X, t, [0., 1.8], None, gamma=gamma_0, varphi=varphi_0, noise_variance=noise_variance_0,
        #     fix_s=False)
        # just std
        # grid_synthetic(X, t, [-1., 1.], None, gamma=gamma_0, varphi=varphi_0, scale=1.0)
        # varphi and scale
        # grid_synthetic(X, t, [0, 2], [0, 2], gamma=gamma_0, noise_variance=noise_variance_0, fix_s=False)
        # varphi and std
        # grid_synthetic(X, t, [0, 2], [0, 2], gamma=gamma_0, scale=30.0)
        # # Just varphi
        # grid_synthetic(X, t, [-8, 2], None, gamma=gamma_0, noise_variance=noise_variance_0, scale=scale_0, fix_s=True, show=True)
        # # Two of the cutpoints
        # grid_synthetic(X, t, [-2, 2], [-3, 1], varphi=varphi_0, noise_variance=noise_variance_0, scale=1.0)
        test_synthetic(dataset, method, X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D, scale=1.0, colors=colors)
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()
