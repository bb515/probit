"""Binomial Probit regression Gibbs example with parallel sampling (using MPI)."""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from probit import GibbsClassifier
from probit.kernels import Binary
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpi4py import MPI

def main():
    """Get Gibbs samples and predictive."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Classes
    K = 2
    # Datapoints per class
    N = 50
    # Datapoints total
    N_total = N*K
    # Dimension of the data
    D = 2
    # Uniform quadrant dataset - linearly seperable
    X0 = np.random.rand(N, D)
    X1 = np.ones((N, D)) + np.random.rand(N, D)
    offset = np.array([0, 1])
    offsets = np.tile(offset, (N, 1))
    t0 = np.zeros(len(X0))
    t1 = np.ones(len(X1))

    if rank == 0:
        plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$")
        plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$")
        plt.legend()
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.xlabel(r"$x_1$", fontsize=16)
        plt.ylabel(r"$x_2$", fontsize=16)
        plt.show()

    # Extend the input vectors
    X0_tilde = np.c_[np.ones(len(X0)), X0]
    X1_tilde = np.c_[np.ones(len(X1)), X1]
    # Prepare data
    X = np.r_[X0_tilde, X1_tilde]
    t = np.r_[t0, t1]
    # Shuffle data - may only be necessary for matrix conditioning
    Xt = np.c_[X, t]
    np.random.shuffle(Xt)
    X = Xt[:, :3]
    t = Xt[:, -1]



    # This is the kernel for a GP prior for the binary class problem
    kernel = Binary(varphi=1.0)
    gibbs_classifier = GibbsClassifier(kernel, X, t, binomial=1)

    steps_burn = 100
    steps = 1000
    M_0 = np.zeros((N, K))
    # Burn in
    M_samples, Y_samples = gibbs_classifier.sample_gp(M_0, steps_burn)
    M_0_burned = M_samples[-1]
    M_samples, Y_samples = gibbs_classifier.sample_gp(M_0_burned, steps)
    # Gather samples
    M_recvbuf = np.empty((steps * size, 3), dtype=np.float64)
    Y_recvbuf = np.empty((steps * size, N_total), dtype=np.float64)
    # Gather samples from different processes
    comm.Gather(M_samples, M_recvbuf, root=0)
    comm.Gather(Y_samples, Y_recvbuf, root=0)

    # Broadcast all samples to all processes if we wanted to parallelize the predictions.
    # comm.Bcast(beta_samples, root=0)
    # comm.Bcast(Y_samples, root=0)
    if rank == 0:
        # TODO: find a way to plot a histogram of the samples.
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        m_star = -1. * np.ones(3)
        n0, m00, patches = ax[0].hist(M_samples[:, 0, 0], 20, density="probability", histtype='stepfilled')
        n1, m01, patches = ax[1].hist(M_samples[:, 0, 1], 20, density="probability", histtype='stepfilled')
        n2, m20, patches = ax[2].hist(M_samples[:, 2, 0], 20, density="probability", histtype='stepfilled')
        m_star[0] = m00[np.where(n0 == n0.max())]
        m_star[1] = m01[np.where(n1 == n1.max())]
        m_star[2] = m20[np.where(n2 == n2.max())]
        ax[0].axvline(m_star[0], color='k', label=r"Maximum $\beta$")
        ax[1].axvline(m_star[1], color='k', label=r"Maximum $\beta$")
        ax[2].axvline(m_star[2], color='k', label=r"Maximum $\beta$")
        ax[0].set_xlabel(r"$\m_00$", fontsize=16)
        ax[1].set_xlabel(r"$\m_01$", fontsize=16)
        ax[2].set_xlabel(r"$\m_20$", fontsize=16)
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()

        N = 20
        x = np.linspace(-0.1, 1.99, N)
        y = np.linspace(-0.1, 1.99, N)
        xx, yy = np.meshgrid(x, y)
        ones = np.ones((N, N))
        X_new = np.dstack((ones, xx, yy))
        X_new = X_new.reshape((N*N, D + 1))
        Z = np.zeros((N * N))
        for i, x_new in enumerate(X_new):
            predictive_multinomial_distributions = gibbs_classifier.predict_gibbs(
                varphi, s, sigma, X_test, X_train, Y_samples, scalar=None)
            Z[i] = gibbs_classifier.predict(M_samples, x_new)
        Z = np.reshape(Z, (N, N))

        fig, axs = plt.subplots(1, figsize=(6, 6))
        plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
        plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
        plt.contourf(x, y, Z, zorder=1)
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