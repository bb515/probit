"""Utility functions for data."""
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from probit.kernels import SEIso
from scipy.optimize import minimize


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
    },
    "auto": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (392, 7),
        "init": None,
    },
    "diabetes": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (43, 2),
        "init": None,
    },
    "housing": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (506, 13),
        "init": None,
    },
    "machine": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (209, 6),
        "init": None,
    },
    "pyrim": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (74, 27),
        "init": None,
    },
    "stocks": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (950, 9),
        "init": None,
    },
    "triazines": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (186, 60),
        "init": None,
    },
    "wpbc": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (194, 32,),
        "init": None,
    },
}


def training(variational_classifier, X_train, t_train, gamma_0, varphi_0, noise_variance_0, K, scale=1.0):
    """
    An example ordinal training function.
    :arg variational_classifier:
    :type variational_classifier: :class:`probit.estimators.Estimator` or :class:`probit.samplers.Sampler`
    :arg X_train:
    :type X_train:
    :arg t_train:
    :type t_train:
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
    :return:
    """
    theta = []
    theta.append(np.log(np.sqrt(noise_variance_0)))
    theta.append(gamma_0[1])
    for i in range(2, K):
        theta.append(np.log(gamma_0[i] - gamma_0[i - 1]))
    theta.append(np.log(varphi_0))
    theta = np.array(theta)
    # Use L-BFGS-B
    res = minimize(variational_classifier.hyperparameter_training_step, theta, method='L-BFGS-B', jac=True, options = {
        'ftol': 1e-5,
        'maxfun': 20})
    theta = res.x
    noise_std = np.exp(theta[0])
    noise_variance = noise_std**2
    gamma = np.empty((K + 1,))  # including all of the cutpoints
    gamma[0] = np.NINF
    gamma[-1] = np.inf
    gamma[1] = theta[1]
    for i in range(2, K):
        gamma[i] = gamma[i - 1] + np.exp(theta[i])
    varphi = np.exp(theta[K])
    return gamma, varphi, noise_variance


def training_varphi(variational_classifier, X_train, t_train, gamma_0, varphi_0, noise_variance_0, K, scale=1.0):
    """
    An example ordinal training function.
    :arg variational_classifier:
    :type variational_classifier: :class:`probit.estimators.Estimator` or :class:`probit.samplers.Sampler`
    :arg X_train:
    :type X_train:
    :arg t_train:
    :type t_train:
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
    theta = np.array([np.log(varphi_0)])
    # Use L-BFGS-B
    res = minimize(variational_classifier.hyperparameter_training_step_varphi(gamma=gamma_0, noise_variance=noise_variance_0), theta, method='L-BFGS-B', jac=True,
                   options={'maxiter':10})
    theta = res.x
    varphi = np.exp(theta[0])
    return gamma_0, varphi, noise_variance_0


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
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            with pkg_resources.path(decile, 'abalone.npz') as path:
                data = np.load(path)
    elif dataset == "auto":
        from probit.data import auto
        with pkg_resources.path(auto, 'auto.npz') as path:
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.auto import quantile
            with pkg_resources.path(quantile, 'auto.npz') as path:
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.auto import decile
            with pkg_resources.path(decile, 'auto.npz') as path:
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
                "57.86": (
                    np.array([-np.inf, -0.92761785, -0.71569034, -0.23952063, 0.05546283, np.inf]),
                    7.0e-06,
                    0.137
                ),
                "57.66": (
                    [-np.inf, -0.96965513, -0.59439608, 0.10485131, 0.55336265, np.inf],
                    7.05301883339537e-06,
                    0.33582851890990895
                ),
                "56.47": (
                    [-np.inf, -0.47585805, -0.41276548, -0.25253468, -0.15562599, np.inf],
                    1.1701782815822784e-05,
                    0.009451605099929043
                ),
                "53.07": (
                    [-np.inf, -0.39296099, -0.34374783, -0.26326698, -0.20514771, np.inf],
                    1.8099468519640467e-05,
                    0.00813540519298387
                ),
            }
            (gamma_0, varphi_0, noise_variance_0) = hyperparameters["57.66"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
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
                "NA1": (
                    np.array([-np.inf, -0.5, -0.02, 0.43, 0.96, np.inf]),
                    0.03,
                    0.0001
                ),
                "NA0" : (
                    [-np.inf, -1.17119928, -0.65961478, 0.1277627, 0.64710874, np.inf],
                    0.00045,
                    0.01
                ),
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / K, -1.0 + 2. * 2. / K, -1.0 + 3. * 2. / K, np.inf]),
                    0.5 / D,
                    1.0
                ),
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["NA1"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
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
            }
            gamma_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit.data.wisconsin import decile
            with pkg_resources.path(decile, 'wpbc.npz') as path:
                data = np.load(path)
            K = 10
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
