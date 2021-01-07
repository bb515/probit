"""Binomial Probit regression Gibbs example with parallel sampling (using MPI)."""
import argparse
import cProfile
from io import StringIO
import numpy as np
from probit.samplers import GibbsBinomial
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


    if rank == 0:
        # Uniform quadrant dataset - linearly seperable
        X0 = np.random.rand(N, D)  # AGD slightly cleaner code here
        X1 = np.ones((N, D)) + np.random.rand(N, D)
        offset = np.array([0, 1])
        offsets = np.tile(offset, (N, 1))
        t0 = np.zeros(len(X0))
        t1 = np.ones(len(X1))
        # Need to broadcast data to each dimension.

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

    gibbs_classifier = GibbsBinomial(X, t)

    # Sample beta from prior
    beta = multivariate_normal.rvs(mean=[0, 0, 0], cov=np.eye(3))

    # Take n samples, returning the beta and y samples
    n = 500
    beta_samples, Y_samples = gibbs_classifier.sample(beta, n)
    # Gather samples
    beta_recvbuf = np.empty((n * size, 3), dtype=np.float64)
    Y_recvbuf = np.empty((n * size, N_total), dtype=np.float64)
    # Gather samples from different processes
    comm.Gather(beta_samples, beta_recvbuf, root=0)
    comm.Gather(Y_samples, Y_recvbuf, root=0)
    # Broadcast all samples to all processes if we wanted to parallelize the predictions.
    # comm.Bcast(beta_samples, root=0)
    # comm.Bcast(Y_samples, root=0)
    if rank == 0:
        norm_samples = np.array([beta / np.linalg.norm(beta) for beta in beta_recvbuf])
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        beta_star = np.zeros(3)
        n0, b0, patches = ax[0].hist(norm_samples[:, 0], 20, density="probability", histtype='stepfilled')
        n1, b1, patches = ax[1].hist(norm_samples[:, 1], 20, density="probability", histtype='stepfilled')
        n2, b2, patches = ax[2].hist(norm_samples[:, 2], 20, density="probability", histtype='stepfilled')
        beta_star[0] = b0[np.where(n0 == n0.max())]
        beta_star[1] = b1[np.where(n1 == n1.max())]
        beta_star[2] = b2[np.where(n2 == n2.max())]
        ax[0].axvline(beta_star[0], color='k', label=r"Maximum $\beta$")
        ax[1].axvline(beta_star[1], color='k', label=r"Maximum $\beta$")
        ax[2].axvline(beta_star[2], color='k', label=r"Maximum $\beta$")
        ax[0].set_xlabel(r"$\beta_0$", fontsize=16)
        ax[1].set_xlabel(r"$\beta_1$", fontsize=16)
        ax[2].set_xlabel(r"$\beta_2$", fontsize=16)
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
            Z[i] = gibbs_classifier.predict(beta_samples, x_new)

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
