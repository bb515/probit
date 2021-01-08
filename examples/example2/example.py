"""Binomial Probit regression Gibbs example with parallel sampling (using MPI)."""
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey

import numpy as np
from probit.samplers import GibbsMultinomialGP
from probit.kernels import SEIso
import matplotlib.pyplot as plt

def main():
    """Get Gibbs samples and predictive."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    # Classes
    K = 3
    # Datapoints per class
    N = 50
    # Dimension of the data
    D = 2

    # Uniform quadrant dataset - linearly seperable
    X0 = np.random.rand(N, D)
    X1 = np.ones((N, D)) + np.random.rand(N, D)
    offset = np.array([0, 1])
    offsets = np.tile(offset, (N, 1))
    X2 = offsets + np.random.rand(N, D)
    t0 = np.zeros(len(X0))
    t1 = np.ones(len(X1))
    t2 = 2 * np.ones(len(X2))

    plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$")
    plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$")
    plt.scatter(X2[:, 0], X2[:, 1], color='g', label=r"$t=2$")
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.ylabel(r"$x_2$", fontsize=16)
    plt.show()

    # Prepare data
    X = np.r_[X0, X1, X2]
    t = np.r_[t0, t1, t2]
    # Shuffle data - may only be necessary for matrix conditioning
    Xt = np.c_[X, t]
    np.random.shuffle(Xt)
    X = Xt[:, :D]
    t = Xt[:, -1]

    # This is the kernel for a GP prior for the multi-class problem
    kernel = SEIso(varphi=1.0, s=1.0)
    gibbs_classifier = GibbsMultinomialGP(X, t, kernel)
    steps_burn = 100
    steps = 100
    M_0 = np.zeros((N, K))
    # Burn in
    M_samples, Y_samples = gibbs_classifier.sample(M_0, steps_burn)
    M_0_burned = M_samples[-1]
    M_samples, Y_samples = gibbs_classifier.sample(M_0_burned, steps)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    m_star = -1. * np.ones(3)
    n0, m00, patches = ax[0].hist(M_samples[:, 0, 0], 20, density="probability", histtype='stepfilled')
    n1, m01, patches = ax[1].hist(M_samples[:, 0, 1], 20, density="probability", histtype='stepfilled')
    n2, m20, patches = ax[2].hist(M_samples[:, 2, 0], 20, density="probability", histtype='stepfilled')
    m_star[0] = m00[np.argmax(n0)]
    m_star[1] = m01[np.argmax(n1)]
    m_star[2] = m20[np.argmax(n2)]
    ax[0].axvline(m_star[0], color='k', label=r"Maximum $m_00$")
    ax[1].axvline(m_star[1], color='k', label=r"Maximum $m_01$")
    ax[2].axvline(m_star[2], color='k', label=r"Maximum $m_02$")
    ax[0].set_xlabel(r"$m_00$", fontsize=16)
    ax[1].set_xlabel(r"$m_01$", fontsize=16)
    ax[2].set_xlabel(r"$m_20$", fontsize=16)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

    N = 20
    x = np.linspace(-0.1, 1.99, N)
    y = np.linspace(-0.1, 1.99, N)
    xx, yy = np.meshgrid(x, y)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, D))
    Z = gibbs_classifier.predict_vector(Y_samples, X_new, n_samples=200)
    Z = np.reshape(Z, (N, N, K))
    Z = np.moveaxis(Z, 2, 0)
    for k in range(K):
        *_ = plt.subplots(1, figsize=(6, 6))
        plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
        plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
        plt.scatter(X2[:, 0], X2[:, 1], color='g', label=r"$t=2$", zorder=10)
        plt.contourf(x, y, Z[k], zorder=1)
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.legend()
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.title("Contour plot - Gibbs")
        plt.show()

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()