"""Utility functions for data."""
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from probit.kernels import SEIso
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
        "size": (90, 3),
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
    plt.scatter(X, Y_true, color=colors_)
    plt.show()
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


def generate_synthetic_data(N_per_class, K, D, varphi=30.0, noise_variance=1.0, scale=1.0):
    """Generate synthetic dataset."""
    # Generate the synethetic data
    kernel = SEIso(varphi, scale=scale, sigma=10e-6, tau=10e-6)
    X_k, Y_true_k, X, Y_true, t, gamma_0 = generate_prior_data(
        N_per_class, K, D, kernel, noise_variance=noise_variance)
    from probit.data import tertile
    with pkg_resources.path(tertile, 'data_tertile_prior_.npz') as path:
        np.savez(
            path, X_k=X_k, Y_k=Y_true_k, X=X, Y=Y_true, t=t, gamma_0=gamma_0)
    return X_k, Y_true_k, X, Y_true, t, gamma_0


def load_data_synthetic(dataset, data_from_prior, plot=False):
    """Load synethetic data."""
    if dataset == "tertile":
        from probit.data import tertile
        K = 3
        D = 1
        if data_from_prior == True:
            with pkg_resources.path(tertile, 'tertile_prior.npz') as path:
                data = np.load(path)
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
                    1.0
                ),
                "true": (
                    data["gamma_0"],  # [-np.inf, - 0.43160987, 0.2652492, np.inf]
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
        gamma_0, varphi_0, noise_variance_0 = hyperparameters["True"]
    if plot:
        # Plot
        colors_ = [colors[i] for i in t]
        plt.scatter(X, Y_true, color=colors_)
        plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.show()
        plt.close()
        # Plot from the binned arrays
        for k in range(K):
            plt.scatter(X_k[k], Y_true_k[k], color=colors[k], label=r"$t={}$".format(k))
        plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
        plt.legend()
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.show()
        plt.close()
    return X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D


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
