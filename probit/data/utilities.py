"""Utility functions for data."""
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from probit.kernels import SEIso, SEARD, Linear, Polynomial
from scipy.optimize import minimize
import warnings
import time

# For plotting
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# The names of the different synthetic datasets
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


metadata = {
    "abalone": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (4177, 10),
        "init": None,
        "max_sec": 500,
    },
    "auto": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (392, 7),
        "init": None,
        "max_sec": 100,
    },
    "diabetes": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (43, 2),
        "init": None,
        "max_sec": 100,
    },
    "housing": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (506, 13),
        "init": None,
        "max_sec": 100,
    },
    "machine": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (209, 6),
        "init": None,
        "max_sec":100,
    },
    "pyrim": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (74, 27),
        "init": None,
        "max_sec":100,
    },
    "stocks": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (950, 9),
        "init": None,
        "max_sec":5000,
    },
    "triazines": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (186, 60),
        "init": None,
        "max_sec": 100,
    },
    "wpbc": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (194, 32),
        "init": None,
        "max_sec":100,
    },
    "tertile": {
        "plot_lims": (-0.5, 1.5),
        "size": (135, 3),  # (90, 3) if not new
        "init": None,
        "max_sec":500,
    },
    "thirteen": {
        "plot_lims": (-0.5, 1.5),
        "size": (585, 13),
        "init": None,
        "max_sec":500,
    },
    "septile": {
        "plot_lims": (-0.5, 1.5),
        "size": (256, 7),
        "init": None,
        "max_sec":500,
    },
}


def training(dataset, method, variational_classifier, gamma_0, varphi_0, noise_variance_0, K):
    """
    An example ordinal training function.
    :arg variational_classifier:
    :type variational_classifier: :class:`probit.estimators.Estimator` or :class:`probit.samplers.Sampler`
    :arg gamma_0:
    :type gamma_0:
    :arg varphi_0:
    :type varphi_0:
    :arg noise_variance_0:
    :type noise_variance_0:
    :arg K:
    :type K:
    :arg scale:
    :type scale:
    :arg method:
    :type method:
    :arg int maxiter:
    :return:
    """
    print("K", K)
    print(gamma_0, "gamma")
    # init stopper
    minimize_stopper = MinimizeStopper(max_sec=metadata[dataset]["max_sec"])
    theta = []
    theta.append(np.log(np.sqrt(noise_variance_0)))
    theta.append(gamma_0[1])
    for i in range(2, K):
        theta.append(np.log(gamma_0[i] - gamma_0[i - 1]))
    theta.append(np.log(varphi_0))
    theta = np.array(theta)
    res = minimize(variational_classifier.hyperparameter_training_step, theta, method=method, jac=True,
        callback = minimize_stopper.__call__)
    theta = res.x
    noise_std = np.exp(theta[0])
    noise_variance = noise_std**2
    gamma = np.empty((K + 1,))
    gamma[0] = np.NINF
    gamma[-1] = np.inf
    gamma[1] = theta[1]
    for i in range(2, K):
        gamma[i] = gamma[i - 1] + np.exp(theta[i])
    varphi = np.exp(theta[K])
    return gamma, varphi, noise_variance


def training_varphi(dataset, method, variational_classifier, gamma, varphi_0, noise_variance):
    """
    An example ordinal training function.
    :arg variational_classifier:
    :type variational_classifier: :class:`probit.estimators.Estimator` or :class:`probit.samplers.Sampler`
    :arg method:
    :type method:
    :arg gamma_0:
    :type gamma_0:
    :arg varphi_0:
    :type varphi_0:
    :arg noise_variance_0:
    :type noise_variance_0:
    :arg K:
    :type K:
    :arg scale:
    :type scale:
    
    :return
    """
    minimize_stopper = MinimizeStopper(max_sec=metadata[dataset]["max_sec"])
    theta = np.array([np.log(varphi_0)])
    args = (gamma, noise_variance)
    res = minimize(
        variational_classifier.hyperparameter_training_step_varphi, theta, args, method=method, jac=True,
        callback = minimize_stopper.__call__)
    theta = res.x
    varphi = np.exp(theta[0])
    return gamma, varphi, noise_variance


def get_Y_trues(X_trains, X_true, Y_true):
    """Get Y_trues (N/K, K) from full array of true y values."""
    Y_trues = []
    for k in range(19):
        y = []
        for i in range(len(X_trains[-1, :, :])):
            for j, two in enumerate(X_true):
                one = X_trains[k, i]
                if np.allclose(one, two):
                    y.append(Y_true[j])
        Y_trues.append(y)
    Y_trues = np.array(Y_trues)
    return Y_trues


def generate_prior_data_new(N_per_class, N_test, splits, K, D, kernel, noise_variance, N_show=2000):
    """
    Generate data from the GP prior, and choose some cutpoints that approximately divides data into equal bins.

    :arg int N_per_class: The number of data points per class.
    :arg int K: The number of bins/classes/quantiles.
    :arg int D: The number of data dimensions.
    :arg kernel: The GP prior.
    """
    epsilon = 1e-6
    N_total = int(K * N_per_class)
    # Sample from the real line, uniformly
    # X = np.random.uniform(0, 12, N_total)
    X_show = np.linspace(-0.5, 1.5, N_show)  # N_show points to show the predictive power
    #X = np.random.random(N_total)  # N_total points unformly random over [0, 1]
    # X_show = X_show[:, None]
    #X = X[:, None]  # reshape X to make it n*D
    C0_show = kernel.kernel_matrix(X_show, X_show)
    #C0 = kernel.kernel_matrix(X, X)
    C_show = C0_show + epsilon * np.identity(N_show)
    #C = C0 + epsilon * np.identity(N_total)
    # Cholesky
    #Chol = np.linalg.cholesky(C)
    Chol_show = np.linalg.cholesky(C_show)
    # Generate normal samples
    #z = np.random.normal(loc=0, scale=1, size=N_total)
    z_show = np.random.normal(loc=0, scale=1, size=N_show)
    Z_show = np.dot(Chol_show, z_show)
    print(np.shape(Z_show), "Z_show")
    print(np.shape(X_show), "X_show")
    N_third = np.int(N_show/3)
    Xt = np.c_[Z_show[N_third:-N_third], X_show[N_third:-N_third]]
    print(np.shape(Xt), "Xt")
    np.random.shuffle(Xt)
    Z = Xt[:N_total, :1]
    X = Xt[:N_total, 1:D + 1]
    print(np.shape(Z))
    print(np.shape(X))
    #Z = np.dot(Chol, z)  # Mean zero
    plt.title("Sample from prior GP")
    plt.scatter(X[:], Z[:], c='b', s=4)
    plt.plot(X_show[:], Z_show[:])
    plt.show()
    # Model latent variable responses
    epsilons = np.random.normal(0, np.sqrt(noise_variance), N_total)
    epsilons = epsilons[:, None]
    # Model latent variable responses
    plt.title("Sample from prior GP")
    Y_true = epsilons + Z
    Y_true = Y_true.flatten()
    sort_indeces = np.argsort(Y_true)
    plt.scatter(X, Y_true, c='b', s=4)
    plt.plot(X_show, Z_show)
    plt.show()
    # Sort the responses
    Y_true = Y_true[sort_indeces]
    X = X[sort_indeces]
    print(np.shape(X), "X")
    X_k = []
    Y_true_k = []
    t_k = []
    gamma = np.empty(K + 1)
    for k in range(K):
        X_k.append(X[N_per_class * k:N_per_class * (k + 1), :D])
        Y_true_k.append(Y_true[N_per_class * k:N_per_class * (k + 1)])
        t_k.append(k * np.ones(N_per_class, dtype=int))
    for k in range(1, K):
        # Find the first cutpoint and set it equal to 0.0
        cutpoint_k_min = Y_true_k[k - 1][-1]
        cutpoint_k_max = Y_true_k[k][0]
        gamma[k] = np.average([cutpoint_k_max, cutpoint_k_min])
    gamma[0] = -np.inf
    gamma[-1] = np.inf
    print("gamma={}".format(gamma))
    cmap = plt.cm.get_cmap('PiYG', K)    # K discrete colors
    colors = []
    for k in range(K):
        colors.append(cmap((k+0.5)/K))
        plt.scatter(X_k[k], Y_true_k[k], color=cmap((k+0.5)/K))
    plt.show()
    Xs_k = np.array(X_k)
    Ys_k = np.array(Y_true_k)
    t_k = np.array(t_k, dtype=int)
    X = Xs_k.flatten()
    Y = Ys_k.flatten()
    t = t_k.flatten()
    # Prepare data
    Y_tests = []
    X_tests = []
    t_tests = []
    Y_trains = []
    X_trains = []
    t_trains = []
    for i in range(splits):
        Xt = np.c_[Y, X, t]
        np.random.shuffle(Xt)
        Y = Xt[:, :1]
        X = Xt[:, 1:D + 1]
        t = Xt[:, -1]
        Y_test = Y[:N_test]
        X_test = X[:N_test, :]
        t_test = t[:N_test]
        Y_train = Y[N_test:]
        X_train = X[N_test:, :]
        t_train = t[N_test:]
        Y_tests.append(Y_test)
        X_tests.append(X_test)
        t_tests.append(t_test)
        Y_trains.append(Y_train)
        X_trains.append(X_train)
        t_trains.append(t_train)
    t = np.array(t, dtype=int)
    t_tests = np.array(t_tests, dtype=int)
    t_trains = np.array(t_trains, dtype=int)
    X_tests = np.array(X_tests)
    X_trains = np.array(X_trains)
    Y_tests = np.array(Y_tests)
    Y_trains = np.array(Y_trains)
    print(np.shape(X_tests))
    print(np.shape(X_trains))
    print(np.shape(Y_tests))
    print(np.shape(Y_trains))
    print(np.shape(t_tests))
    print(np.shape(t_trains))
    print(np.shape(X_k))
    print(np.shape(Y_true_k))
    print(np.shape(t_k))
    print(colors)
    colors_ = [colors[i] for i in t_trains[0, :]]
    plt.scatter(X_trains[0, :, 0], Y_trains[0, :], color=colors_)
    plt.show()
    plot_ordinal(X, t, X_k, Y_true_k, K, D, colors=colors)
    return (X_k, Y_true_k, X, Y, t, gamma, X_tests, Y_tests, t_tests,
        X_trains, Y_trains, t_trains, C0_show, X_show, Z_show, colors)


def generate_prior_data(N_per_class, K, D, kernel, noise_variance):
    """
    Generate data from the GP prior, and choose some cutpoints that approximately divides data into equal bins.

    :arg int N_per_class: The number of data points per class.
    :arg int K: The number of bins/classes/quantiles.
    :arg int D: The number of data dimensions.
    :arg kernel: The GP prior.
    """
    N_total = int(K * N_per_class)
    # Sample from the real line, uniformly
    # X = np.random.uniform(0, 12, N_total)
    X = np.linspace(0., 1., N_total)  # 500 points evenly spaced over [0,1]
    X = X[:, None]  # reshape X to make it n*D
    mu = np.zeros((N_total))  # vector of the means
    C = kernel.kernel_matrix(X, X)
    Z = np.random.multivariate_normal(mu, C)
    plt.figure()  # open new plotting window
    plt.title("Sample from prior GP")
    plt.plot(X[:], Z[:])
    plt.show()
    epsilons = np.random.normal(0, np.sqrt(noise_variance), N_total)
    # Model latent variable responses
    plt.title("Sample from prior GP")
    Y_true = epsilons + Z
    sort_indeces = np.argsort(Y_true)
    plt.scatter(X, Y_true)
    plt.show()
    # Sort the responses
    Y_true = Y_true[sort_indeces]
    X = X[sort_indeces]
    X_k = []
    Y_true_k = []
    t_k = []
    gamma = np.empty(K + 1)
    for k in range(K):
        X_k.append(X[N_per_class * k:N_per_class * (k + 1)])
        Y_true_k.append(Y_true[N_per_class * k:N_per_class * (k + 1)])
        t_k.append(k * np.ones(N_per_class, dtype=int))
    for k in range(1, K):
        # Find the first cutpoint and set it equal to 0.0
        cutpoint_k_min = Y_true_k[k - 1][-1]
        cutpoint_k_max = Y_true_k[k][0]
        gamma[k] = np.average([cutpoint_k_max, cutpoint_k_min])
    gamma[0] = -np.inf
    gamma[-1] = np.inf
    print("gamma={}".format(gamma))
    for k in range(K):
        plt.scatter(X_k[k], Y_true_k[k], color=colors[k])
    plt.show()
    Xs_k = np.array(X_k)
    Ys_k = np.array(Y_true_k)
    t_k = np.array(t_k, dtype=int)
    X = Xs_k.flatten()
    Y = Ys_k.flatten()
    t = t_k.flatten()
    # Prepare data
    Xt = np.c_[Y, X, t]
    print(np.shape(Xt))
    np.random.shuffle(Xt)
    Y_true = Xt[:, :1]
    X = Xt[:, 1:D + 1]
    t = Xt[:, -1]
    print(np.shape(X))
    print(np.shape(t))
    print(np.shape(Y_true))
    t = np.array(t, dtype=int)
    print(t)
    colors_ = [colors[i] for i in t]
    print(colors_)
    plt.scatter(X[:, 0], Y_true, color=colors_)
    plt.show()
    plot_ordinal(X, t, X_k, Y_true_k, K, D)
    return X_k, Y_true_k, X, Y_true, t, gamma


def generate_synthetic_data(N_per_class, K, D, kernel, noise_variance):
    """
    Generate synthetic data for this model.

    This function will generate data such that the ground truth of the first cutpoint is at zero.

    :arg int N_per_class: The number of data points per class.
    :arg int K: The number of bins/classes/quantiles.
    :arg int D: The number of data dimensions.
    :arg kernel: The GP prior.
    """
    N_total = int(K * N_per_class)

    # Sample from the real line, uniformly
    #X = np.random.uniform(0, 12, N_total)
    X = np.linspace(0., 1., N_total)  # 500 points evenly spaced over [0,1]
    X = X[:, None]  # reshape X to make it n*D
    mu = np.zeros((N_total))  # vector of the means

    C = kernel.kernel_matrix(X, X)

    print(np.shape(mu))
    print(np.shape(C))
    print("1")
    cutpoint_0 = np.inf
    while np.abs(cutpoint_0) > 5.0:
        print(cutpoint_0)
        Z = np.random.multivariate_normal(mu, C)
        plt.figure()  # open new plotting window
        plt.plot(X[:], Z[:])
        plt.show()
        epsilons = np.random.normal(0, np.sqrt(noise_variance), N_total)
        # Model latent variable responses
        Y_true = epsilons + Z
        sort_indeces = np.argsort(Y_true)
        plt.scatter(X, Y_true)
        plt.show()
        # Sort the responses
        Y_true = Y_true[sort_indeces]
        X = X[sort_indeces]
        X_k = []
        Y_true_k = []
        t_k = []
        for k in range(K):
            X_k.append(X[N_per_class * k:N_per_class * (k + 1)])
            Y_true_k.append(Y_true[N_per_class * k:N_per_class * (k + 1)])
            t_k.append(k * np.ones(N_per_class, dtype=int))
        # Find the first cutpoint and set it equal to 0.0
        cutpoint_0_min = Y_true_k[0][-1]
        cutpoint_0_max = Y_true_k[1][0]
        print(cutpoint_0_max, cutpoint_0_min)
        cutpoint_0 = np.mean([cutpoint_0_max, cutpoint_0_min])
    Y_true = np.subtract(Y_true, cutpoint_0)
    Y_true_k = np.subtract(Y_true_k, cutpoint_0)
    for k in range(K):
        plt.scatter(X_k[k], Y_true_k[k], color=colors[k])
    plt.show()
    Xs_k = np.array(X_k)
    Ys_k = np.array(Y_true_k)
    t_k = np.array(t_k, dtype=int)
    X = Xs_k.flatten()
    Y = Ys_k.flatten()
    t = t_k.flatten()
    # Prepare data
    Xt = np.c_[Y, X, t]
    print(np.shape(Xt))
    np.random.shuffle(Xt)
    Y_true = Xt[:, :1]
    X = Xt[:, 1:D + 1]
    t = Xt[:, -1]
    print(np.shape(X))
    print(np.shape(t))
    print(np.shape(Y_true))
    t = np.array(t, dtype=int)
    print(t)
    colors_ = [colors[i] for i in t]
    print(colors_)
    plt.scatter(X, Y_true, color=colors_)
    plt.show()
    return X_k, Y_true_k, X, Y_true, t


def load_data(dataset, bins):
    if dataset == "abalone":
        from probit.data import abalone
        with pkg_resources.path(abalone, 'abalone.npz') as path:
            data_continuous = np.load(path)
        D = 10
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (  # Unstable
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    100.0,
                    10.0
                ),
                "1085.1": (
                    [-np.inf, -0.93798221, 1.0093883, 1.5621378, 1.99464624, np.inf],
                    2.0,
                    0.2,
                ),
                "1085.0": (
                    np.array([-np.inf, -0.72645941,  0.2052068 ,  0.99282042,  1.79754696, np.inf]) ,
                    0.07043020535401254 ,
                    0.23622657165969294 ,
                ),
                "1073.0": (
                    np.array([-np.inf, -0.60729699, -0.3207209 , -0.2210021 , -0.10743028, np.inf]) ,
                    0.135,
                    0.005,
                ),
                "1046.0": (
                    np.array([-np.inf, -0.60729699, -0.3207209 , -0.2210021 , -0.10743028, np.inf]) ,
                    0.11,
                    0.005,
                ),
            }
            linear_hyperperameters = {
                "init": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf],
                    0.5 / D,
                    np.ones((10,))
                ),
                "init_alt": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf],
                    100.0,
                    np.ones((10,))
                ),
            }
            polynomial_hyperparameters = {
                "init": (  # Unstable
                    [-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf],
                    0.5 / D,
                    np.ones((10,))
                ),
                "init_alt": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf],
                    100.0,
                    np.ones((10,))
                ),
            }
            ARD_hyperparameters = {
                "init": (  # Unstable
                    [-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf],
                    0.5 / D,
                    np.ones((10,))
                ),
                "init_alt": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf],
                    100.0,
                    np.ones((10,))
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["1046.0"]
            from probit.data.abalone import quantile
            with pkg_resources.path(quantile, 'abalone.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            from probit.data.abalone import decile
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            with pkg_resources.path(decile, 'abalone.npz') as path:
                data = np.load(path)
    elif dataset == "auto":
        from probit.data import auto
        with pkg_resources.path(auto, 'auto.DATA.npz') as path:
            data_continuous = np.load(path)
        D = 7
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (  # Trying lower varphi to find local minima there - this worked
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.0001,
                    0.15
                ),
                "390.7": (
                    [-np.inf, -0.75623025, 0.17624028, 0.97368228, 1.76188594, np.inf],
                    0.0709125443999348,
                    0.20713212477690768
                ),
                "389.0": (  # gives a zo of 0.64 and ma of 0.822, varphi tended downwards. pl 0.25
                    [-np.inf, -0.75623025, 0.17624028, 0.97368228, 1.76188594, np.inf],
                    0.065,
                    0.15
                ),
                "300.9": (
                    np.array([-np.inf, -0.01715729,  0.99812162,  1.42527213,  1.80608152, np.inf]),
                    1.0e-05,
                    0.01,
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["300.9"]
            from probit.data.auto import quantile
            with pkg_resources.path(quantile, 'auto.data.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.auto import decile
            with pkg_resources.path(decile, 'auto.data.npz') as path:
                data = np.load(path)
        gamma_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf])
    elif dataset == "diabetes":
        D = 2
        from probit.data import diabetes
        with pkg_resources.path(diabetes, 'diabetes.DATA.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    100.0,
                    10.0
                ),
                "57.66": (
                    [-np.inf, -0.96965513, -0.59439608, 0.10485131, 0.55336265, np.inf],
                    7.05301883339537e-06,
                    0.33582851890990895
                ),
                "53.07": (
                    [-np.inf, -0.39296099, -0.34374783, -0.26326698, -0.20514771, np.inf],
                    1.8099468519640467e-05,
                    0.00813540519298387
                ),
                "52.66": (
                    [-np.inf, -0.99914905, -0.49661647, 0.84539859, 1.66267616, np.inf],
                    0.6800987545547965,
                    0.1415239624095029
                ),
                 "52.32": (
                    [-np.inf, -0.94, -0.47, 0.79, 1.55, np.inf],
                    0.378154023606708,
                    0.103
                ),
            }
            (gamma_0, varphi_0, noise_variance_0) = hyperparameters["52.32"]
            from probit.data.diabetes import quantile
            with pkg_resources.path(quantile, 'diabetes.data.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.diabetes import decile
            with pkg_resources.path(decile, 'diabetes.data.npz') as path:
                data = np.load(path)
    elif dataset == "housing":
        D = 13
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        from probit.data import bostonhousing
        with pkg_resources.path(bostonhousing, 'housing.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "538.6": (
                    [-np.inf, -0.9, 0.4, 0.7, 1.2, np.inf],
                    0.01,
                    0.2
                ),
                "462.9": (  # Tried to find local minima here but its not good m0 0.40, ma 0.519
                    [-np.inf, -0.65824624, 0.71570933, 1.2696072, 1.65280723, np.inf],
                    0.0015,
                    0.1,
                ),
                "init_alt": (
                    [-np.inf, -0.65824624, 0.71570933, 1.2696072, 1.65280723, np.inf],
                    10.0,
                    0.1,
                )
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit.data.bostonhousing import quantile
            with pkg_resources.path(quantile, 'housing.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.bostonhousing import decile
            with pkg_resources.path(decile, 'housing.npz') as path:
                data = np.load(path)
    elif dataset == "machine":
        D = 6
        varphi_0 = 2.0 / D
        noise_variance_0 = 2.0
        from probit.data import machinecpu
        with pkg_resources.path(machinecpu, 'machine.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, 0.82762696, 1.3234133 , 1.69382192, 2.02491639, np.inf]) ,
                    0.000008,
                    0.6,
                ),
                "211.6": (
                    [-np.inf, 0.86220899, 1.38172233, 1.76874495, 2.11477391, np.inf], 
                    0.08212108678729564,
                    0.6297364232519436,
                ),
                "199.7": (
                    np.array([-np.inf, 0.82481368, 1.31287655, 1.67778397, 2.00289883, np.inf]),
                    0.0007963123721287592,
                    0.20036340095411048,
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit.data.machinecpu import quantile
            with pkg_resources.path(quantile, 'machine.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.machinecpu import decile
            with pkg_resources.path(decile, 'machine.npz') as path:
                data = np.load(path)
    elif dataset == "pyrim":
        from probit.data import pyrimidines
        with pkg_resources.path(pyrimidines, 'pyrim.npz') as path:
            data_continuous = np.load(path)
        D = 27
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    100.0,
                    10.0
                ),
                "99.0": (
                    np.array([-np.inf, -0.96921449, -0.50848339, -0.18839114,  0.13407544, np.inf]),
                    0.01852678001715496,
                    0.13877975100366893
                ),
                "101.0": (
                    [-np.inf, -1.11405451, -0.21082173, 0.36749149, 0.88030159, np.inf],
                    10.0,
                    0.12213407601036991,
                ),
                "92.4": (
                    np.array([-np.inf, -0.41399957, -0.25391163, -0.15800952, -0.0700965, np.inf]) ,
                    0.018,
                    0.006 ,
                ),
                "90.9": (
                    np.array([-np.inf, -0.41399957, -0.25391163, -0.15800952, -0.0700965, np.inf]) ,
                    0.005,
                    0.001 ,
                ),
                "89.9": (
                    np.array([-np.inf, -0.41399957, -0.25391163, 0.0800952, 0.1700965, np.inf]) ,
                    0.5,
                    0.001 ,
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["89.9"]
            from probit.data.pyrimidines import quantile
            with pkg_resources.path(quantile, 'pyrim.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.pyrimidines import decile
            with pkg_resources.path(decile, 'pyrim.npz') as path:
                data = np.load(path)
    elif dataset == "stocks":
        from probit.data import stocksdomain
        with pkg_resources.path(stocksdomain, 'stock.npz') as path:
            data_continuous = np.load(path)
        D = 9
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "720.0": (
                    np.array([-np.inf, -0.5, -0.02, 0.43, 0.96, np.inf]),
                    0.03,
                    0.0001
                ),
                "565.0" : (
                    [-np.inf, -1.17119928, -0.65961478, 0.1277627, 0.64710874, np.inf],
                    0.00045,
                    0.01
                ),
                "561.3" : (
                    [-np.inf, -1.15383995, -0.77984497, -0.45804968, -0.0881168, np.inf],
                    0.0004859826674075834,
                    0.009998200271564093,
                ),
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "1245.4": (
                     [-np.inf, -0.91005114, -0.28471923, 0.45182957, 1.14570708, np.inf],
                    0.192573878772786,
                    9.989576742047504
                ),
                "init_alt" : (
                    [-np.inf, -1.15383995, -0.77984497, -0.45804968, -0.0881168, np.inf],
                    0.06,
                    0.01,
                ),
                "540.8" : (
                    np.array([-np.inf, -0.85938512, -0.11336137,  0.88349296,  1.62309025, np.inf]) ,
                    0.035268828486755506,
                    0.010540656172175512,
                ),
                "537.0" : (
                    np.array([-np.inf, -1.3952776 , -0.71642281,  0.1649921 ,  0.78461035, np.inf]) ,
                    0.003896416837153817,
                    0.01569970567257151,
                ),
                 "init_alt" : (
                    np.array([-np.inf, -0.9 , -0.8,  0.0 ,  0.3, np.inf]) ,
                    0.001,
                    0.0125,
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit.data.stocksdomain import quantile
            with pkg_resources.path(quantile, 'stock.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.stocksdomain import decile
            with pkg_resources.path(decile, 'stock.npz') as path:
                data = np.load(path)
    elif dataset == "triazines":
        from probit.data import triazines
        with pkg_resources.path(triazines, 'triazines.npz') as path:
            data_continuous = np.load(path)
        D = 60
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    [-np.inf, -1.0157062, -0.408891, 0.08201205, 0.45461363, np.inf],
                    0.8,
                    0.0004
                ),
                "192.0": (
                    [-np.inf, -1.0157062, -0.908891, -0.78201205, -0.45461363, np.inf],
                    0.0008,
                    0.0004
                ),
                "179.1": (
                    [-np.inf, -1.01688376, -0.90419162, -0.76962449, -0.34357796, np.inf],
                    0.008334663676431068,
                    0.04633175816789728,
                ),
                "175.2": (  # m0 -0.02, ma -0.05
                    [-np.inf, -1.0157062, -0.908891, -0.78201205, -0.45461363, np.inf],
                    0.00833,
                    0.0475,
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit.data.triazines import quantile
            with pkg_resources.path(quantile, 'triazines.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.triazines import decile
            with pkg_resources.path(decile, 'triazines.npz') as path:
                data = np.load(path)
    elif dataset == "wpbc":
        D = 32
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        from probit.data import wisconsin
        with pkg_resources.path(wisconsin, 'wpbc.npz') as path:
            data_continuous = np.load(path)
        if bins == "quantile":
            K = 5
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.001,
                    0.02
                ),
                "266.4": (  #m0 0.64 ma1.35 nearly there need ma1.0
                    [-np.inf, -0.32841066, 0.17593563, 0.76336227, 1.21093938, np.inf],
                    0.0155,
                    0.2,
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit.data.wisconsin import quantile
            with pkg_resources.path(quantile, 'wpbc.npz') as path:
                data = np.load(path)
        elif bins == "decile":
            K = 10
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                              -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, -1.0 + 6. * 2. / K, -1.0 + 7. * 2. / K,
                              -1.0 + 8. * 2. / K, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.wisconsin import decile
            with pkg_resources.path(decile, 'wpbc.npz') as path:
                data = np.load(path)
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
    Y_true = data_continuous["y"]  # this is not going to be the correct one(?) - see next line
    # Y_trues = get_Y_trues(X_trains, X_true, Y_true)
    return X_trains, t_trains, X_tests, t_tests, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D


def generate_synthetic_data_SEARD(N_per_class, K, D, varphi=[30.0, 20.0], noise_variance=1.0, scale=1.0):
    """Generate synthetic SEARD dataset."""
    # Generate the synethetic data
    kernel = SEARD(varphi, scale=scale, sigma=10e-6, tau=10e-6)
    X_k, Y_true_k, X, Y_true, t, gamma_0 = generate_prior_data(
        N_per_class, K, D, kernel, noise_variance=noise_variance)
    from probit.data import tertile
    with pkg_resources.path(tertile) as path:
        np.savez(
            path / 'data_polynomial_{}dim_{}bin_prior.npz'.format(D, K), X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t, gamma_0=gamma_0)
    return X_k, Y_true_k, X, Y_true, t, gamma_0


def generate_synthetic_data_polynomial(N_per_class, K, D, noise_variance=1.0, scale=1.0,
        intercept=0.0, order=2.0):
    """Generate synthetic Polynomial dataset."""
    # Generate the synethetic data
    kernel = Polynomial(intercept=intercept, order=order, scale=scale, sigma=10e-6, tau=10e-6)
    X_k, Y_true_k, X, Y_true, t, gamma_0 = generate_prior_data(
        N_per_class, K, D, kernel, noise_variance=noise_variance)
    from probit.data import tertile
    with pkg_resources.path(tertile) as path:
        np.savez(
            path / 'data_polynomial_{}dim_{}bin_prior.npz'.format(D, K), X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t, gamma_0=gamma_0)
    return X_k, Y_true_k, X, Y_true, t, gamma_0


def generate_synthetic_data_linear(N_per_class, K, D, noise_variance=1.0, scale=1.0, intercept=0.0):
    """Generate synthetic Linear dataset."""
    # Generate the synethetic data
    kernel = Linear(intercept=intercept, scale=scale, sigma=10e-6, tau=10e-6)
    X_k, Y_true_k, X, Y_true, t, gamma_0 = generate_prior_data(
        N_per_class, K, D, kernel, noise_variance=noise_variance)
    from probit.data import tertile
    with pkg_resources.path(tertile) as path:
        np.savez(
            path / 'data_linear_{}dim_{}bin_prior.npz'.format(D, K), X_k=X_k,
            Y_k=Y_true_k, X=X, Y=Y_true, t=t, gamma_0=gamma_0)
    return X_k, Y_true_k, X, Y_true, t, gamma_0


def generate_synthetic_data(N_per_class, K, D, varphi=30.0, noise_variance=1.0, scale=1.0):
    """Generate synthetic dataset."""
    # Generate the synethetic data
    kernel = SEIso(varphi, scale=scale, sigma=10e-6, tau=10e-6)
    X_k, Y_true_k, X, Y_true, t, gamma_0 = generate_prior_data(
        N_per_class, K, D, kernel, noise_variance=noise_variance)
    from probit.data import tertile
    # with pkg_resources.path(tertile) as path:
    #     np.savez(
    #         path / 'data_tertile_prior_2.npz', X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t, gamma_0=gamma_0)
    np.savez('data_tertile_prior_2.npz', X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t,
        gamma=gamma_0, varphi=varphi, scale=scale, noise_variance=noise_variance)
    return X_k, Y_true_k, X, Y_true, t, gamma_0


def generate_synthetic_data_new(N_per_class, N_test, splits, K, D, varphi=30.0, noise_variance=1.0, scale=1.0):
    """Generate synthetic dataset."""
    kernel = SEIso(varphi, scale=scale, sigma=10e-6, tau=10e-6)
    (X_k, Y_true_k, X, Y, t, gamma,
    X_tests, Y_tests, t_tests,
    X_trains, Y_trains, t_trains,
    C0_show, X_show, Z_show, colors) = generate_prior_data_new(
        N_per_class, N_test, splits, K, D, kernel, noise_variance=noise_variance)
    np.savez('data_thirteen_prior_new.npz', X_k=X_k, Y_k=Y_true_k, X=X, Y=Y, t=t,
        X_tests=X_tests, Y_tests=Y_tests, t_tests=t_tests,
        X_trains=X_trains, Y_trains=Y_trains, t_trains=t_trains,
        C0_show=C0_show,
        X_show=X_show,
        Z_show=Z_show,
        noise_variance=noise_variance,
        scale=scale,
        varphi=varphi,
        gamma=gamma,
        colors=colors)
    return (X_k, Y_true_k, X, Y, t, gamma, X_tests, Y_tests, t_tests, X_trains, Y_trains, t_trains, C0_show,
        X_show, Z_show, colors)


def load_data_synthetic(dataset, data_from_prior, plot=False):
    """Load synethetic data."""
    if dataset == "tertile":
        from probit.data import tertile
        K = 3
        D = 1
        if data_from_prior == True:
            #with pkg_resources.path(tertile, 'tertile_prior_s=1_sigma2=0.1_varphi=30_new.npz') as path:
            with pkg_resources.path(tertile, 'tertile_prior_s=1_sigma2=0.1_varphi=30.npz') as path:  # works for varphi
            #with pkg_resources.path(tertile, 'tertile_prior_s=1_sigma2=1_varphi=30.npz') as path:  # is varphi actually 1?
            #with pkg_resources.path(tertile, 'tertile_prior_s=30_sigma2=10_varphi=30.npz') as path:
                #SS: tertile_prior_30.npz, tertile_prior.npz
                data = np.load(path)
            
            # X_show = data["X_show"]
            # Z_show = data["Z_show"]
            
            N_per_class = 30
            X_k = data["X_k"]  # Contains (90, 3) array of binned x values
            Y_true_k = data["Y_k"]  # Contains (90, 3) array of binned y values
            X = data["X"]  # Contains (90,) array of x values
            t = data["t"]  # Contains (90,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
            X_true = X
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0,
                    1.0,
                ),
                "true": (
                    data["gamma"],
                    data["varphi"],
                    data["noise_variance"],  # np.sqrt(0.1) = 0.316 
                    data["scale"],
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, np.inf]),
                    100.0,
                    10.0,
                    1.0
                ),
            }
            plt.scatter(X, Y_true)
            plt.show()
            gamma_0, varphi_0, noise_variance_0, scale_0 = hyperparameters["true"]
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        else:
            with pkg_resources.path(tertile, 'tertile.npz') as path:
                data = np.load(path)
            N_per_class = 64
            X_k = data["X_k"]  # Contains (172, 3) array of binned x values
            Y_true_k = data["Y_k"]  # Contains (172, 3) array of binned y values
            X = data["X"]  # Contains (172,) array of x values
            t = data["t"]  # Contains (172,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (172,) array of y values, corresponding to Xs values
            X_true = X
            N_total = int(N_per_class * K)
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "true": (
                    np.array([-np.inf, 0.0, 2.29, np.inf]),
                    30.0,
                    0.1
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, np.inf]),
                    100.0,
                    10.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    elif dataset == "thirteen":
        from probit.data import thirteen
        K = 13
        D = 1
        if data_from_prior == True:
            with pkg_resources.path(thirteen, 'thirteen_prior_s=1_sigma2=0.1_varphi=30_new.npz') as path:
                data = np.load(path)
            X_show = data["X_show"]
            Z_show = data["Z_show"]
            N_per_class = 45
            X_k = data["X_k"]  # Contains (13, 45) array of binned x values
            Y_true_k = data["Y_k"]  # Contains (13, 45) array of binned y values
            X = data["X"]  # Contains (585,) array of x values
            t = data["t"]  # Contains (585,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (585,) array of y values, corresponding to Xs values (not in order)
            colors = data["colors"]
            X_true = X
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0,
                    1.0,
                ),
                "true": (
                    data["gamma"],
                    data["varphi"],
                    data["noise_variance"],  # np.sqrt(0.1) = 0.316 
                    data["scale"],
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, np.inf]),
                    100.0,
                    10.0,
                    1.0
                ),
            }
            plt.scatter(X, Y_true)
            plt.show()
            gamma_0, varphi_0, noise_variance_0, scale_0 = hyperparameters["true"]
    elif dataset == "septile":
        K = 7
        D = 1
        from probit.data import septile
        with pkg_resources.path(septile, 'septile.npz') as path:
            data = np.load(path)
        N_per_class = 32
        # The scale was not 1.0, it was 20.0(?)
        scale = 20.0
        X_k = data["X_k"]  # Contains (256, 7) array of binned x values
        Y_true_k = data["Y_k"]  # Contains (256, 7) array of binned y values
        X = data["X"]  # Contains (1792,) array of x values
        t = data["t"]  # Contains (1792,) array of ordinal response variables, corresponding to Xs values
        Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values
        X_true = X
        N_total = int(N_per_class * K)
        hyperparameters = {
            "init": (
                np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                          -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, np.inf]),
                0.5 / D,
                1.0
            ),
            "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K,
                          -1.0 + 4. * 2. / K, -1.0 + 5. * 2. / K, np.inf]),
                    100.0,
                    10.0
                ),
            "true": (
                np.array([-np.inf, 0.0, 1.0, 2.0, 4.0, 5.5, 6.5, np.inf]),
                30.0,
                0.1
            ),
        }
        gamma_0, varphi_0, noise_variance_0 = hyperparameters["true"]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if plot:
        plot_ordinal(X, t, X_k, Y_true_k, K, D)
    return X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, scale_0, K, D, colors


def plot_s(kernel, N_total=500, n_samples=10):
    for i in range(n_samples):
        X = np.linspace(0., 1., N_total)  # 500 points evenly spaced over [0,1]
        X = X[:, None]  # reshape X to make it n*D
        mu = np.zeros((N_total))  # vector of the means
        C = kernel.kernel_matrix(X, X)
        Z = np.random.multivariate_normal(mu, C)
        plt.plot(X[:], Z[:])
    plt.show()


def plot_ordinal(X, t, X_k, Y_k, K, D, colors=colors):
    N_total = len(t)
    colors_ = [colors[i] for i in t]
    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], t, color=colors_)
    plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
    plt.xlabel(r"$x$", fontsize=16)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    plt.ylabel(r"$t$", fontsize=16)
    plt.show()
    # plt.savefig("N_total={}, K={}, D={} Ordinal response data.png".format(N_total, K, D))
    plt.close()
    # Plot from the binned arrays
    for k in range(K):
        plt.scatter(X_k[k][:, 0], Y_k[k], color=colors[k], label=r"$t={}$".format(k))
    plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
    plt.legend()
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$y$", fontsize=16)
    plt.show()
    # plt.savefig("N_total={}, K={}, D={} Ordinal response data_.png".format(N_total, K, D)
    plt.close()


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



if __name__ == "__main__":
    print("Hello")
    generate_synthetic_data_new(
        N_per_class=45, N_test=15*13, splits=20, K=13, D=1, varphi=30.0, noise_variance=0.1, scale=1.0)
    # generate_synthetic_data(30, 3, 1, varphi=30.0, noise_variance=1.0, scale=1.0)
    # generate_synthetic_data_linear(30, 3, 2, noise_variance=0.1, scale=1.0, intercept=0.0)
    # kernel = Linear(intercept=0.0, scale=1.0, sigma=10e-6, tau=10e-6)
    # plot_s(kernel)
    # generate_synthetic_data_polynomial(30, 3, 2, noise_variance=0.1, scale=1.0, intercept=0.0)
    print("HELLO")

# generate_synthetic_data_polynomial(30, 3, 2, noise_variance=0.1, scale=1.0, intercept=0.0)

# generate_synthetic_data_SEARD(30, 3, 2, noise_variance=1.0)
