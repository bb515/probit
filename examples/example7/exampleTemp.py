"""
Multiclass oredered probit regression 3 bin example from Cowles 1996 empirical study
showing convergence of the orginal probit with the Gibbs sampler.
"""
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from scipy.stats import multivariate_normal
from probit.samplers import GibbsMultinomialOrderedGPTemp
from probit.kernels import SEIso
import matplotlib.pyplot as plt
import pathlib

write_path = pathlib.Path()

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def generate_synthetic_data(N_per_class, K, kernel):
    """
    Generate synthetic data for this model.

    :arg int N_per_class: The number of data points per class.
    :arg int K: The number of bins/classes/quantiles.
    :arg int D: The number of dimensions of the covariates.
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
        epsilons = np.random.normal(0, 1, N_total)
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


def split(list, K):
    """Split a list into quantiles."""
    divisor, remainder = divmod(len(list), K)
    return np.array(list[i * divisor + min(i, remainder):(i+1) * divisor + min(i + 1, remainder)] for i in range(K))


# This is the general kernel for a GP prior for the multi-class problem
# varphi = 30.0
# scale = 20.0

varphi = 0.01
scale = 3.0
sigma = 10e-6
tau = 10e-6

kernel = SEIso(varphi, scale, sigma=sigma, tau=tau)

#argument = "diabetes_quantile"
argument = "stocks_quantile"

if argument == "diabetes_quantile":
    K = 5
    D = 2
    gamma_0 = np.array([np.NINF, 3.8, 4.5, 5.0, 5.6, np.inf])
    data = np.load("data_diabetes_train.npz")
    # data_test = np.load("data_diabetes_test.npz")  # SS
    # data_continuous = np.load("data_diabetes_continuous.npz")
    data = np.load(write_path / "/data/5bin/diabetes.data.npz")
    data_continuous = np.load("/data/continuous/diabetes.DATA.npz")
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

X = data["X_train"][0]
t = data["t_train"][0]

X_test = data["X_test"][0]
t_test = data["t_test"][0]
# X = data["X_train"]
# t = data["t_test"]
# X_test = data["X_test"]
# t_test = data["t_test"]

X_true = data_continuous["X"]
Y_true = data_continuous["y"]  # this is not going to be the correct one
N_total = len(X)

y = []

for i in range(len(X)):
    for j in range(len(X_true)):
        one = X[i]
        two = X_true[j]
        if np.allclose(one, two):
            y.append(Y_true[j])


# for i in range(len(X_test)):
#     for j in range(len(X_true)):
#         one = X_test[i]
#         two = X_true[j]
#         if np.allclose(one, two):
#             t_test.append[j]


y_true = np.array(y)

print(y_true)

# X_k = data["X_k"]  # Contains (256, 7) array of binned x values
# Y_true_k = data["Y_k"]  # Contains (256, 7) array of binned y values
# X = data["X"]  # Contains (1792,) array of x values
# t = data["t"]  # Contains (1792,) array of ordinal response variables, corresponding to Xs values
# Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
# N_total = int(N_per_class * K)
# print(Y_true_k[1][-1], Y_true_k[2][0], "cutpoint 2")

# Initiate classifier
gibbs_classifier = GibbsMultinomialOrderedGPTemp(K, X, t, kernel)
steps_burn = 100
steps = 5000
y_0 = t.flatten()
#y_0 = Y_true.flatten()

# # Plot
# colors_ = [colors[i] for i in t]
# plt.scatter(X, Y_true, color=colors_)
# plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
# plt.xlabel(r"$x$", fontsize=16)
# plt.ylabel(r"$y$", fontsize=16)
# plt.show()

# Plot from the binned arrays

plt.scatter(X[np.where(t == 1)][:, 0], X[np.where(t == 1)][:, 1])
plt.scatter(X[np.where(t == 2)][:, 0], X[np.where(t == 2)][:, 1])
plt.scatter(X[np.where(t == 3)][:, 0], X[np.where(t == 3)][:, 1])
plt.scatter(X[np.where(t == 4)][:, 0], X[np.where(t == 4)][:, 1])
plt.scatter(X[np.where(t == 5)][:, 0], X[np.where(t == 5)][:, 1])

# for k in range(K):
#     plt.scatter(X_k[k], Y_true_k[k], color=colors[k], label=r"$t={}$".format(k))
# plt.title("N_total={}, K={}, D={} Ordinal response data".format(N_total, K, D))
plt.legend()
plt.xlabel(r"$x_1$", fontsize=16)
plt.ylabel(r"$x_2$", fontsize=16)
plt.show()

# m_0 = np.random.rand(N_total)
# Problem with this is that the intial guess must be close to the true values
# As a result we have to approximate the latent function.
if argument == "diabetes_quantile":
    m_0 = y_true
    y_0 = y_true
elif argument == "stocks_quantile":
    m_0 = y_true
    y_0 = y_true
elif argument == "tertile":
    m_0 = y_0
elif argument == "septile":
    m_0 = y_0

# Burn in
m_samples, y_samples, gamma_samples = gibbs_classifier.sample(m_0, y_0, gamma_0, steps_burn)
#m_samples, y_samples, gamma_samples = gibbs_classifier.sample_metropolis_within_gibbs(m_0, y_0, gamma_0, 0.5, steps_burn)
m_0_burned = m_samples[-1]
y_0_burned = y_samples[-1]
gamma_0_burned = gamma_samples[-1]

# Sample
m_samples, y_samples, gamma_samples = gibbs_classifier.sample(m_0_burned, y_0_burned, gamma_0_burned, steps)
#m_samples, y_samples, gamma_samples = gibbs_classifier.sample_metropolis_within_gibbs(m_0, y_0, gamma_0, 0.5, steps)
m_tilde = np.mean(m_samples, axis=0)
y_tilde = np.mean(y_samples, axis=0)
gamma_tilde = np.mean(gamma_samples, axis=0)

if argument == "diabetes_quantile" or argument == "stocks_quantile":
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(gamma_samples[:, 1])
    ax[0].set_ylabel(r"$\gamma_1$", fontsize=16)
    ax[1].plot(gamma_samples[:, 2])
    ax[1].set_ylabel(r"$\gamma_2$", fontsize=16)
    plt.title('Mixing for cutpoint posterior samples $\gamma$')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(gamma_samples[:, 3])
    ax[0].set_ylabel(r"$\gamma_3$", fontsize=16)
    ax[1].plot(gamma_samples[:, 4])
    ax[1].set_ylabel(r"$\gamma_4$", fontsize=16)
    plt.title('Mixing for cutpoint posterior samples $\gamma$')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(gamma_samples[:, 5])
    ax[0].set_ylabel(r"$\gamma_5$", fontsize=16)
    ax[1].plot(gamma_samples[:, 6])
    ax[1].set_ylabel(r"$\gamma_6$", fontsize=16)
    plt.title('Mixing for cutpoint posterior samples $\gamma$')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    g_star = -1. * np.ones(3)
    n0, g0, patches = ax[0].hist(gamma_samples[:, 1], 20, density="probability", histtype='stepfilled')
    n1, g1, patches = ax[1].hist(gamma_samples[:, 2], 20, density="probability", histtype='stepfilled')
    g_star[0] = g0[np.argmax(n0)]
    g_star[1] = g1[np.argmax(n1)]
    ax[0].axvline(g_star[0], color='k', label=r"Maximum $\gamma_1$")
    ax[1].axvline(g_star[1], color='k', label=r"Maximum $\gamma_2$")
    ax[0].set_xlabel(r"$\gamma_1$", fontsize=16)
    ax[1].set_xlabel(r"$\gamma_2$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    plt.title(r"$\gamma$ posterior samples")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    m_star = -1. * np.ones(3)
    n0, m00, patches = ax[0].hist(m_samples[:, 0], 20, density="probability", histtype='stepfilled')
    n1, m01, patches = ax[1].hist(m_samples[:, 1], 20, density="probability", histtype='stepfilled')
    n2, m20, patches = ax[2].hist(m_samples[:, 2], 20, density="probability", histtype='stepfilled')
    m_star[0] = m00[np.argmax(n0)]
    m_star[1] = m01[np.argmax(n1)]
    m_star[2] = m20[np.argmax(n2)]
    ax[0].axvline(m_star[0], color='k', label=r"Maximum $m_0$")
    ax[1].axvline(m_star[1], color='k', label=r"Maximum $m_1$")
    ax[2].axvline(m_star[2], color='k', label=r"Maximum $m_2$")
    ax[0].set_xlabel(r"$m_0$", fontsize=16)
    ax[1].set_xlabel(r"$m_1$", fontsize=16)
    ax[2].set_xlabel(r"$m_2$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.title(r"$m$ posterior samples")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    y_star = -1. * np.ones(3)
    n0, y00, patches = ax[0].hist(y_samples[:, 0], 20, density="probability", histtype='stepfilled')
    n1, y01, patches = ax[1].hist(y_samples[:, 1], 20, density="probability", histtype='stepfilled')
    n2, y20, patches = ax[2].hist(y_samples[:, 2], 20, density="probability", histtype='stepfilled')
    y_star[0] = y00[np.argmax(n0)]
    y_star[1] = y01[np.argmax(n1)]
    y_star[2] = y20[np.argmax(n2)]
    ax[0].axvline(y_star[0], color='k', label=r"Maximum $y_0$")
    ax[1].axvline(y_star[1], color='k', label=r"Maximum $y_1$")
    ax[2].axvline(y_star[2], color='k', label=r"Maximum $y_2$")
    ax[0].set_xlabel(r"$y_0$", fontsize=16)
    ax[1].set_xlabel(r"$y_1$", fontsize=16)
    ax[2].set_xlabel(r"$y_2$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.title(r"$y$ posterior samples")
    plt.show()

    # plt.scatter(X[np.where(t == 0)], m_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
    # plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
    # plt.scatter(X[np.where(t == 2)], m_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
    # plt.xlabel(r"$x$", fontsize=16)
    # plt.ylabel(r"$\tilde{m}$", fontsize=16)
    # plt.title("GP regression posterior sample mean mbar, plotted against x")
    # plt.show()
    #
    # plt.scatter(X[np.where(t == 0)], y_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
    # plt.scatter(X[np.where(t == 1)], y_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
    # plt.scatter(X[np.where(t == 2)], y_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
    # plt.xlabel(r"$x$", fontsize=16)
    # plt.ylabel(r"$\tilde{y}$", fontsize=16)
    # plt.title("Latent variable posterior sample mean ybar, plotted against x")
    # plt.show()

    lower_x1 = 0.0
    upper_x1 = 16.0
    lower_x2 = -30
    upper_x2 = 0
    N = 60
    x1 = np.linspace(lower_x1, upper_x1, N)
    x2 = np.linspace(lower_x2, upper_x2, N)
    xx, yy = np.meshgrid(x1, x2)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, D))

    # Test
    Z = gibbs_classifier.predict(y_samples, gamma_samples, X_test, vectorised=True)  # (n_test, K)

    # Mean zero-one error
    t_star = np.argmax(Z, axis=1)
    print(t_star)
    print(t_test)
    zero_one = np.logical_and(t_star, t_test)
    mean_zero_one = zero_one * 1
    mean_zero_one = np.sum(mean_zero_one) / len(t_test)
    print(mean_zero_one)

    # X_new = x.reshape((N, D))
    print(np.shape(gamma_samples), 'shape gamma')
    Z = gibbs_classifier.predict(y_samples, gamma_samples, X_new, vectorised=True)
    Z_new = Z.reshape((N, N, K))
    print(np.sum(Z, axis=1), 'sum')

    for i in range(6):
        fig, axs = plt.subplots(1, figsize=(6, 6))
        plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
        plt.scatter(X[np.where(t == i)][:, 0], X[np.where(t == i)][:, 1], color='red')
        plt.scatter(X[np.where(t == i + 1)][:, 0], X[np.where(t == i + 1)][:, 1], color='blue')
        # plt.scatter(X[np.where(t == 2)][:, 0], X[np.where(t == 2)][:, 1])
        # plt.scatter(X[np.where(t == 3)][:, 0], X[np.where(t == 3)][:, 1])
        # plt.scatter(X[np.where(t == 4)][:, 0], X[np.where(t == 4)][:, 1])
        # plt.scatter(X[np.where(t == 5)][:, 0], X[np.where(t == 5)][:, 1])

        # plt.xlim(0, 2)
        # plt.ylim(0, 2)
        plt.legend()
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.title("Contour plot - Gibbs")
        plt.show()

    # plt.xlim(lower_x, upper_x)
    # plt.ylim(0.0, 1.0)
    # plt.xlabel(r"$x$", fontsize=16)
    # plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
    # plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
    #           .format(lower_x, upper_x))
    # plt.stackplot(x, Z.T,
    #               labels=(
    #                   r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
    #               colors=(
    #                   colors[0], colors[1], colors[2])
    #               )
    # plt.legend()
    # val = 0.5  # this is the value where you want the data to appear on the y-axis.
    # plt.scatter(X[np.where(t == 0)], np.zeros_like(X[np.where(t == 0)]) + val, facecolors=colors[0], edgecolors='white')
    # plt.scatter(X[np.where(t == 1)], np.zeros_like(X[np.where(t == 1)]) + val, facecolors=colors[1], edgecolors='white')
    # plt.scatter(X[np.where(t == 2)], np.zeros_like(X[np.where(t == 2)]) + val, facecolors=colors[2], edgecolors='white')
    # plt.show()

elif argument == "tertile":
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(gamma_samples[:, 1])
    ax[0].set_ylabel(r"$\gamma_1$", fontsize=16)
    ax[1].plot(gamma_samples[:, 2])
    ax[1].set_ylabel(r"$\gamma_1$", fontsize=16)
    plt.title('Mixing for cutpoint posterior samples $\gamma$')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    g_star = -1. * np.ones(3)
    n0, g0, patches = ax[0].hist(gamma_samples[:, 1], 20, density="probability", histtype='stepfilled')
    n1, g1, patches = ax[1].hist(gamma_samples[:, 2], 20, density="probability", histtype='stepfilled')
    g_star[0] = g0[np.argmax(n0)]
    g_star[1] = g1[np.argmax(n1)]
    ax[0].axvline(g_star[0], color='k', label=r"Maximum $\gamma_1$")
    ax[1].axvline(g_star[1], color='k', label=r"Maximum $\gamma_2$")
    ax[0].set_xlabel(r"$\gamma_1$", fontsize=16)
    ax[1].set_xlabel(r"$\gamma_2$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    plt.title(r"$\gamma$ posterior samples")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    m_star = -1. * np.ones(3)
    n0, m00, patches = ax[0].hist(m_samples[:, 0], 20, density="probability", histtype='stepfilled')
    n1, m01, patches = ax[1].hist(m_samples[:, 1], 20, density="probability", histtype='stepfilled')
    n2, m20, patches = ax[2].hist(m_samples[:, 2], 20, density="probability", histtype='stepfilled')
    m_star[0] = m00[np.argmax(n0)]
    m_star[1] = m01[np.argmax(n1)]
    m_star[2] = m20[np.argmax(n2)]
    ax[0].axvline(m_star[0], color='k', label=r"Maximum $m_0$")
    ax[1].axvline(m_star[1], color='k', label=r"Maximum $m_1$")
    ax[2].axvline(m_star[2], color='k', label=r"Maximum $m_2$")
    ax[0].set_xlabel(r"$m_0$", fontsize=16)
    ax[1].set_xlabel(r"$m_1$", fontsize=16)
    ax[2].set_xlabel(r"$m_2$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.title(r"$m$ posterior samples")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    y_star = -1. * np.ones(3)
    n0, y00, patches = ax[0].hist(y_samples[:, 0], 20, density="probability", histtype='stepfilled')
    n1, y01, patches = ax[1].hist(y_samples[:, 1], 20, density="probability", histtype='stepfilled')
    n2, y20, patches = ax[2].hist(y_samples[:, 2], 20, density="probability", histtype='stepfilled')
    y_star[0] = y00[np.argmax(n0)]
    y_star[1] = y01[np.argmax(n1)]
    y_star[2] = y20[np.argmax(n2)]
    ax[0].axvline(y_star[0], color='k', label=r"Maximum $y_0$")
    ax[1].axvline(y_star[1], color='k', label=r"Maximum $y_1$")
    ax[2].axvline(y_star[2], color='k', label=r"Maximum $y_2$")
    ax[0].set_xlabel(r"$y_0$", fontsize=16)
    ax[1].set_xlabel(r"$y_1$", fontsize=16)
    ax[2].set_xlabel(r"$y_2$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.title(r"$y$ posterior samples")
    plt.show()

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
    print(np.shape(gamma_samples), 'shape gamma')
    Z = gibbs_classifier.predict(y_samples, gamma_samples, X_new, vectorised=True)
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
    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    for i in range(6):
        ax[i].plot(gamma_samples[:, i + 1])
        ax[i].set_ylabel(r"$\gamma_{}$".format(i + 1), fontsize=16)
    plt.title('Mixing for cutpoint posterior samples $\gamma$')
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax[i].plot(m_samples[:, i])
        ax[i].set_ylabel(r"$m{}$".format(i), fontsize=16)
    plt.title('Mixing for GP posterior samples $m$')
    plt.show()

    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    g_star = -1. * np.ones(6)
    for i in range(6):
        ni, gi, patches = ax[i].hist(gamma_samples[:, i + 1], 20, density="probability", histtype='stepfilled')
        g_star[i] = gi[np.argmax(ni)]
        ax[i].axvline(g_star[i], color='k', label=r"Maximum $\gamma_{}$".format(i+1))
        ax[i].set_xlabel(r"$\gamma_{}$".format(i+1), fontsize=16)
        ax[i].legend()
    plt.title(r"$\gamma$ posterior samples")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    m_star = -1. * np.ones(3)
    for i in range(3):
        ni, mi, patches = ax[i].hist(m_samples[:, i + 1], 20, density="probability", histtype='stepfilled')
        m_star[i] = mi[np.argmax(ni)]
        ax[i].axvline(m_star[i], color='k', label=r"Maximum $m_{}$".format(i + 1))
        ax[i].set_xlabel(r"$m_{}$ posterior samples".format(i + 1), fontsize=16)
        ax[i].legend()
    plt.title(r"$m$ posterior samples")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    y_star = -1. * np.ones(3)
    for i in range(3):
        ni, yi, patches = ax[i].hist(y_samples[:, i + 1], 20, density="probability", histtype='stepfilled')
        y_star[i] = yi[np.argmax(ni)]
        ax[i].axvline(y_star[i], color='k', label=r"Maximum $y_{}$".format(i + 1))
        ax[i].set_xlabel(r"$y_{}$ posterior samples".format(i + 1), fontsize=16)
        ax[i].legend()
    plt.title(r"$y$ posterior samples")
    plt.show()

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
    print(np.shape(gamma_samples), 'shape gamma')
    Z = gibbs_classifier.predict(y_samples, gamma_samples, X_new, vectorised=True)
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



# fig, ax = plt.subplots(5, 10, sharex='col', sharey='row')
# # axes are in a two-dimensional array, indexed by [row, col]
# for i in range(5):
#     for j in range(10):
#         ax[i, j].bar(np.array([0, 1, 2], dtype=np.intc), Z[i*5 + j])
#         ax[i, j].set_xticks(np.array([0, 1, 2], dtype=np.intc), minor=False)
# plt.show()

#
# plt.hist(X, bins=20)
# plt.xlabel(r"$y$", fontsize=16)
# plt.ylabel("counts")
# plt.show()

# for k in range(K):
#     _ = plt.subplots(1, figsize=(6, 6))
#     plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
#     plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
#     plt.scatter(X2[:, 0], X2[:, 1], color='g', label=r"$t=2$", zorder=10)
#     plt.contourf(x, y, Z[k], zorder=1)
#     plt.xlim(0, 2)
#     plt.ylim(0, 2)
#     plt.legend()
#     plt.xlabel(r"$x_1$", fontsize=16)
#     plt.ylabel(r"$x_2$", fontsize=16)
#     plt.title("Contour plot - Gibbs")
#     plt.show()

