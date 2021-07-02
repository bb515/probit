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

write_path = pathlib.Path()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def split(list, K):
    """Split a list into quantiles."""
    divisor, remainder = divmod(len(list), K)
    return np.array(list[i * divisor + min(i, remainder):(i+1) * divisor + min(i + 1, remainder)] for i in range(K))


argument = "diabetes_quantile"
# argument = "stocks_quantile"

if argument == "diabetes_quantile":
    K = 5
    D = 2
    #gamma_0 = np.array([-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf])
    gamma_0 = np.array([-np.inf, 1.0, 4.5, 5.0, 5.6, np.inf])
    #gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1.*2./K, -1.0 + 2.*2./K, -1.0 + 3.*2./K, np.inf])
    data = np.load(write_path / "./data/5bin/diabetes.data.npz")
    data_continuous = np.load("./data/continuous/diabetes.DATA.npz")
elif argument == "stocks_quantile":
    K = 5
    D = 9
    gamma_0 = np.array([np.NINF, 1.0, 2.0, 3.0, 4.0, np.inf])
    data = np.load(write_path / "./data/5bin/stock.npz")
    data_continuous = np.load(write_path / "./data/continuous/stock.npz")
elif argument == "tertile":
    K = 3
    D = 1
    N_per_class = 64
    gamma_0 = np.array([-np.inf, 0.0, 2.29, np.inf])
    # # Generate the synethetic data
    # X_k, Y_true_k, X, Y_true, t = generate_synthetic_data(N_per_class, K, kernel)
    # np.savez(write_path / "data_tertile.npz", X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t)
    data = np.load("data_tertile.npz")
elif argument == "septile":
    K = 7
    D = 1
    N_per_class = 32
    # Generate the synethetic data
    #X_k, Y_true_k, X, Y_true, t = generate_synthetic_data(N_per_class, K, kernel)
    #np.savez(write_path / "data_septile.npz", X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t)
    data = np.load("data_septile.npz")
    gamma_0 = np.array([-np.inf, 0.0, 1.0, 2.0, 4.0, 5.5, 6.5, np.inf])

if argument == "diabetes_quantile" or argument == "stocks_quantile":
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

    # X = data["X"]
    # t = data["t"]
    # N_total = len(X)
    # # Since python indexes from 0
    # t = t - 1
    # print(len(X), len(t))
    # X_test = data_test["X"]
    # t_test = data_test["t"]
    # t_test = t_test - 1

    X_true = data_continuous["X"]
    Y_true = data_continuous["y"]  # this is not going to be the correct one
    Y_trues = []

    for k in range(20):
        y = []
        for i in range(len(X_trains[0, :, :])):
            for j, two in enumerate(X_true):
                one = X_trains[k, i]
                if np.allclose(one, two):
                    y.append(Y_true[j])
        Y_trues.append(y)
    Y_trues = np.array(Y_trues)

if argument not in ["diabetes_quantile", "stocks_quantile"]:
    X_k = data["X_k"]  # Contains (256, 7) array of binned x values
    #Y_true_k = data["Y_k"]  # Contains (256, 7) array of binned y values
    X = data["X"]  # Contains (1792,) array of x values
    t = data["t"]  # Contains (1792,) array of ordinal response variables, corresponding to Xs values
    Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
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

def ordinal_EP_testing(
        X_train, t_train, X_test, t_test, gamma, varphi, noise_variance,
        K, steps=500, scale=1.0, sigma=10e-6, tau=10e-6):
    grid = np.ogrid[0:len(X_test[:, :])]
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    print("varphi", kernel.varphi)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    approximate_marginal_likelihoods = []
    while error / steps > 5e-3:  # variational_classifier.EPS**2:  #TODO: is this really correct?
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
         precision_EP, amplitude_EP, containers) = variational_classifier.estimate(
            steps, gamma, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP, noise_variance=noise_variance,
            write=True)
        # print("iteration {}, error={}".format(iteration, error / steps))
        variational_classifier.compute_EP_weights(precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        approximate_marginal_likelihoods.append(variational_classifier.approximate_evidence(
            mean_EP, precision_EP, amplitude_EP, Sigma))
    bound = approximate_marginal_likelihoods[-1]
    plt.title(r"Variational lower bound $\scr{F}$", fontsize=16)
    plt.plot(bound)
    plt.show()
    # Test
    Z = variational_classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi,
                                       noise_variance, X_test, vectorised=True)  # (n_test, K)
    predictive_likelihood = Z[grid, t_test]
    predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
    print("predictive_likelihood ", predictive_likelihood)
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

    Z = variational_classifier.predict(gamma_0, Sigma, mean_EP, precision_EP, varphi,
                                       noise_variance, X_new, vectorised=True)
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
        plt.legend()
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.title("Contour plot - Expectation propagation")
        plt.show()
    return bound, zero_one, predictive_likelihood, mean_absolute_error


def ordinal_EP_training(X_train, t_train, X_test, t_test, gamma_0, K, varphi_0=1.0/D, noise_variance_0=1.0,
                        scale=1.0, sigma=10e-6, tau=10e-6):
    """
    An example ordinal training function.

    :return:
    """
    varphi = varphi_0
    noise_variance = noise_variance_0
    theta = []
    theta.append(np.log(noise_variance))
    print("gamma_0", gamma_0)
    theta.append(gamma_0[1])
    for i in range(2, K):
        theta.append(np.log(gamma_0[i] - gamma_0[i - 1]))
    theta.append(np.log(varphi))
    theta = np.array(theta)
    print("theta_0", theta)
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    # Use L-BFGS-B
    res = minimize(variational_classifier.hyperparameter_training_step, theta, method='CG', jac=True, options={
        'maxiter':25})
    theta = res.x
    noise_variance = np.exp(theta[0])
    gamma = np.empty((K + 1,))  # including all of the cutpoints
    gamma[0] = np.NINF
    gamma[-1] = np.inf
    gamma[1] = theta[1]
    for i in range(2, K):
        gamma[i] = gamma[i - 1] + np.exp(theta[i])
    varphi = np.exp(theta[K])
    return gamma, noise_variance, varphi


def test_bed(gamma_0=np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]), scale=1.0):
    """Testing for the error with gradients blowing up."""
    split = 2
    X_train = X_trains[split, :, :]
    t_train = t_trains[split, :]
    X_test = X_tests[split, :, :]
    t_test = t_tests[split, :]
    Y_true = Y_trues[split, :]

    gamma, noise_variance, varphi = ordinal_EP_training(
        X_train, t_train, X_test, t_test, gamma_0, K, scale=scale)

    bound, zero_one, predictive_likelihood, mean_abs = ordinal_EP_testing(
        X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, scale=scale)

    print(gamma)
    print(zero_one)
    print(predictive_likelihood)
    print(mean_abs)
    print(bound)
    assert 0


def outer_loops(scale=1.0):
    grid = np.ogrid[0:len(X_tests[0, :, :])]
    bounds = []
    zero_ones = []
    predictive_likelihoods = []
    mean_abss = []
    varphis = []
    noise_variances = []
    gammas = []
    for split in range(20):
        X_train = X_trains[split, :, :]
        t_train = t_trains[split, :]
        X_test = X_tests[split, :, :]
        t_test = t_tests[split, :]
        Y_true = Y_trues[split, :]

        gamma, noise_variance, varphi = ordinal_EP_training(
            X_train, t_train, X_test, t_test, gamma_0, K, scale=scale)

        bound, zero_one, predictive_likelihood, mean_abs = ordinal_EP_testing(
            X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, scale=scale)

        bounds.append(bound)
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
    print(avg_bound, std_bound, "average max bound, std")
    print(avg_predictive_likelihood, std_predictive_likelihood, "average max predictive likelihood, std")
    print(avg_zero_one, std_zero_one, "average max zero one, std")
    print(avg_mean_abs, std_mean_abs, "average max mean abs, std")
    avg_varphi = np.average(varphis)
    std_varphi = np.std(varphis)
    avg_noise_variances = np.average(noise_variances)
    std_noise_variances = np.std(noise_variances)
    avg_gammas = np.average(gammas)
    std_gammas = np.std(gammas)
    print(avg_varphi, std_varphi, "average varphi, std")
    print(avg_noise_variances, std_noise_variances, "average noise_variances, std")
    print(avg_gammas, std_gammas, "average gammas, std")
    return 0


def SSouter_loops():
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
        Y_true = Y_trues[split, :]

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
        # plt.show()

        # fig, axs = plt.subplots(1, figsize=(6, 6))
        # plt.contourf(x1, x2, bounds_Z)
        # plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
        # axs.set_xscale('log')
        # axs.set_yscale('log')
        # plt.xlabel(r"$\log{\varphi}$", fontsize=16)
        # plt.ylabel(r"$\log{s}$", fontsize=16)
        # plt.title("Contour plot - Variational lower bound")
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
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_bounds_Z)
    plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - Variational lower bound")
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_zero_one_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - mean zero-one accuracy")
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_mean_abs_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.title("Contour plot - mean absolute error accuracy")
    plt.show()


def test_plots(X_test, X_train, t_test, t_train, Y_true):
    grid = np.ogrid[0:len(X_test)]
    varphi = 1.83298071e-05
    scale = 3.79269019e+01
    sigma = 10e-6
    tau = 10e-6
    kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)
    # Initiate classifier
    variational_classifier = EPMultinomialOrderedGP(X_train, t_train, kernel)
    steps = 50
    y_0 = Y_true.flatten()
    m_0 = y_0
    gamma, m_tilde, Sigma_tilde, C_tilde, y_tilde, varphi_tilde, bound, containers = variational_classifier.estimate(
        m_0, gamma_0, steps, varphi_0=varphi, fix_hyperparameters=False, write=True)
    ms, ys, varphis, psis, bounds = containers

    plt.title("variational lower bound")
    plt.title(r"Variational lower bound $\scr{F}$", fontsize=16)
    plt.plot(bound)
    plt.show()

    plt.plot(varphis)
    plt.title(r"$\varphi$", fontsize=16)
    plt.show()

    plt.plot(psis)
    plt.title(r"$\phi$", fontsize=16)
    plt.show()

    if argument in ["diabetes_quantile", "stocks_quantile"]:
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
        Z = variational_classifier.predict(gamma, Sigma_tilde, y_tilde, varphi_tilde, X_test, vectorised=True)  # (n_test, K)

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

        Z = variational_classifier.predict(gamma, Sigma_tilde, y_tilde, varphi_tilde, X_new, vectorised=True)
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
            plt.legend()
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.title("Contour plot - Variational")
            plt.show()

    elif argument == "tertile":
        plt.scatter(X[np.where(t == 0)], m_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
        plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
        plt.scatter(X[np.where(t == 2)], m_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\tilde{m}$", fontsize=16)
        plt.title("GP regression posterior sample mean mbar, plotted against x")
        plt.show()

        plt.scatter(X[np.where(t == 0)], y_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
        plt.scatter(X[np.where(t == 1)], y_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
        plt.scatter(X[np.where(t == 2)], y_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\tilde{y}$", fontsize=16)
        plt.title("Latent variable posterior sample mean ybar, plotted against x")
        plt.show()

        lower_x = -0.5
        upper_x = 1.5
        N = 1000
        x = np.linspace(lower_x, upper_x, N)
        X_new = x.reshape((N, D))
        Z = variational_classifier.predict(Sigma_tilde, y_tilde, varphi_tilde, X_new, vectorised=True)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(lower_x, upper_x)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
                  .format(lower_x, upper_x))
        plt.stackplot(x, Z.T,
                      labels=(
                          r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
                      colors=(
                          colors[0], colors[1], colors[2])
                      )
        plt.legend()
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
        plt.scatter(X[np.where(t == 0)], np.zeros_like(X[np.where(t == 0)]) + val, facecolors=colors[0], edgecolors='white')
        plt.scatter(X[np.where(t == 1)], np.zeros_like(X[np.where(t == 1)]) + val, facecolors=colors[1], edgecolors='white')
        plt.scatter(X[np.where(t == 2)], np.zeros_like(X[np.where(t == 2)]) + val, facecolors=colors[2], edgecolors='white')
        plt.show()

    elif argument == "septile":
        #plt.scatter(X, m_tilde)
        #plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)])
        for i in range(7):
            plt.scatter(X[np.where(t == i)], m_tilde[np.where(t == i)], color=colors[i], label=r"$t={}$".format(i + 1))
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\tilde{m}$", fontsize=16)
        plt.title("GP regression posterior sample mean mbar, plotted against x")
        plt.legend()
        plt.show()

        for i in range(7):
            plt.scatter(X[np.where(t == i)], y_tilde[np.where(t == i)], color=colors[i], label=r"$t={}$".format(i))
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$\tilde{y}$", fontsize=16)
        plt.title("Latent variable posterior sample mean ybar, plotted against x")
        plt.legend()
        plt.show()

        lower_x = -0.5
        upper_x = 1.5
        N = 1000
        x = np.linspace(lower_x, upper_x, N)
        X_new = x.reshape((N, D))
        Z = variational_classifier.predict(Sigma_tilde, y_tilde, varphi_tilde, X_new, vectorised=True)
        print(np.sum(Z, axis=1), 'sum')
        plt.xlim(lower_x, upper_x)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
        plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
                  .format(lower_x, upper_x))
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
            plt.scatter(X[np.where(t == i)], np.zeros_like(X[np.where(t == i)]) + val, facecolors=colors[i], edgecolors='white')
        plt.show()

# Debug
test_bed()

split = 2
X_train = X_trains[split, :, :]
t_train = t_trains[split, :]
X_test = X_tests[split, :, :]
t_test = t_tests[split, :]
Y_true = Y_trues[split, :]
gamma = np.array([np.NANF, 0.91957117, 4.33669622, 6.93837388, 8.10256752, np.inf])
noise_variance = 2.0730371538626398
varphi = 0.054306417960043166
scale = 1.0
bound, zero_one, predictive_likelihood, mean_abs = ordinal_EP_testing(
        X_train, t_train, X_test, t_test, gamma, varphi, noise_variance, K, scale=scale)


assert 0



#test_plots(X_tests[0], X_trains[0], t_tests[0], t_trains[0], Y_trues[0])
#outer_loops(scale=1.0)



#scale = 1.0
#scale = 3.79269019e+01
# ordinal_EP_training(X_trains[2], t_trains[2], X_tests[2], t_tests[2], gamma_0, K, steps=100,
#                     scale=scale)


# gamma = np.array([np.NINF, -0.02051816, 4.22900768, 8.72172009, 10.15449307, np.inf])
# noise_variance = 3.8466548282762365
# varphi = 0.0008118442260631808

# gamma = np.array([np.NINF, -0.01662607, 4.07932558, 7.04164376, 8.25634517, np.inf])
# noise_variance = 2.9791932424310215
# varphi = 0.000844587880740459

# scale=1.0
# gamma = np.array([np.NINF, -0.34955714, 0.93657207, 3.79460213, 5.00797433, np.inf])
# noise_variance = 3.4702749861740054
# varphi = 0.0008392067708278249
#
# ordinal_EP_testing(
#     X_trains[2], t_trains[2], X_tests[2], t_tests[2], gamma, varphi, noise_variance, K, scale=scale)
