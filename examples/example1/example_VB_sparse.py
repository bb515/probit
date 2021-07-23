"""Binomial Probit regression variational Bayes using Sparse GPs example."""
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from probit.estimators import VBMultinomialSparseGP
from probit.kernels import SEIso
from probit.kernels import SEARDMultinomialSS as SEARDMultinomial
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pathlib


write_path = pathlib.Path()


def main():
    """Get approximate posteriors and predictive."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    # Classes
    K = 2
    # Datapoints total
    N = 100
    # Dimension of the data
    D = 2

    X = 2.0 * np.random.rand(N, D)

    varphi = np.ones((K, D))
    #varphi=1.0
    #kernel = SEIso(varphi=1.0, scale=1.0, sigma=0.0, tau=1.0)
    kernel = SEARDMultinomialTemp(varphi=varphi, scale=1.0, sigma=np.array([1e-5, 1e-5]), tau=np.array([1e-5, 1e-5]))

    M_true = multivariate_normal.rvs(mean=None, cov=kernel.kernel_matrix(X, X)[0])
    Y_true = M_true  # + multivariate_normal.rvs(mean=None, cov=np.eye(len(X)))
    t = (Y_true > 0)
    X0 = X[t == 0]
    X1 = X[t == 1]
    plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$")
    plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$")
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.ylabel(r"$x_2$", fontsize=16)
    plt.show()

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(X[:, 0], X[:, 1], M_true, color="green")
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.title("M_true")
    plt.show()
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(X[:, 0], X[:, 1], Y_true, color="green")
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.title("Y_true")
    plt.show()
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(X[:, 0], X[:, 1], t, color="green")
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.title("t_true")
    plt.show()

    variational_classifier = VBMultinomialSparseGP(X, t, kernel)
    # # Sample M_0 from prior
    M_0 = multivariate_normal.rvs(mean=None, cov=np.eye(N), size=2)
    M_0 = M_0.T
    # Number of ADF update steps
    steps = 3
    # M_0, Sigma_tilde, C_tilde, Y_tilde, varphi_0, psi_0 = variational_classifier.estimate(M_0, steps)
    # np.savez(write_path/"initial_estimate.npz", M_0=M_0, varphi_0=varphi_0, psi_0=psi_0)

    # initial_estimate = np.load(write_path/"initial_estimate.npz")
    # M_0 = initial_estimate["M_0"]
    # varphi_0 = initial_estimate["varphi_0"]
    # psi_0 = initial_estimate["psi_0"]
    #
    # print(M_0)
    # print(varphi_0)
    # print(psi_0)
    varphi_0 = 1.0 * varphi
    (M_tilde, M_ADF, s_ADF, Sigma_tilde, C_tilde,
     Y_tilde, varphi_tilde, psi_tilde, containers) = variational_classifier.estimate(
        M_0, steps, varphi_0=varphi_0, informative_selection=False, fix_hyperparameters=True, write=True)
    (Ms, Ys, M_ADFs, s_ADFs, varphis, psis, bounds) = containers
    plt.plot(bounds)
    plt.title('bounds')
    plt.show()
    plt.plot(Ms)
    plt.plot(Ys)
    plt.show()
    # plt.plot(psis)
    # plt.show()
    N = 20
    # x = np.linspace(-0.1, 1.99, N)
    # y = np.linspace(-0.1, 1.99, N)
    x = np.linspace(-1.2, 4.0, N)
    y = np.linspace(-1.2, 4.0, N)
    xx, yy = np.meshgrid(x, y)
    # print(Y_tilde)
    # print(M_tilde)
    # print(varphi_tilde)
    # print(psi_tilde)

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(X[:, 0], X[:, 1], Y_tilde[:, 0], color="green")
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.title("Y_tilde")
    plt.show()

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(X[:, 0], X[:, 1], M_tilde[:, 0], color="green")
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.title("M_tilde")
    plt.show()

    X_test = np.dstack((xx, yy))
    X_test = X_test.reshape((N*N, D))

    Z = variational_classifier.predict(Sigma_tilde, Y_tilde, varphi_tilde, X_test)

    Z = np.reshape(Z, (N, N, K))
    print(Z[:, :, 0])
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
    plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
    plt.contourf(x, y, Z[:, :, 0], zorder=1)
    #plt.xlim(0, 2)
    #plt.ylim(0, 2)
    plt.legend()
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.ylabel(r"$x_2$", fontsize=16)
    plt.title("Contour plot - Variational Bayes")
    plt.show()

    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
    plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
    plt.contourf(x, y, Z[:, :, 1], zorder=1)
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)
    plt.legend()
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.ylabel(r"$x_2$", fontsize=16)
    plt.title("Contour plot - Variational Bayes")
    plt.show()

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
