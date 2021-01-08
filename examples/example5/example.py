"""Variational multinomial example."""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from probit import VariationalBayesClassifier
from probit.kernels import Multivariate
import matplotlib.pyplot as plt


def main():
    """Get Gibbs samples and predictive."""
    parser = argparse.ArgumentParser()
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    # The n argument toggles between MPI and serial implementations
    parser.add_argument('n', help="The number of processes used in mpiexec",
                        type=int)
    args = parser.parse_args()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    if args.n:
        num_processes = args.n
    else:
        num_processes = 0

    # train = np.load(read_path / 'train.npz')
    # test = np.load(read_path / 'test.npz')
    #
    # X_train = train['X']
    # t_train = train['t']
    # X0_train = train['X0']
    # X1_train = train['X1']
    # X2_train = train['X2']
    #
    # X_test = test['X']
    # t_test = test['t']
    # X0_test = test['X0']
    # X1_test = test['X1']
    # X2_test = test['X2']
    ## Plotting
    # plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$")
    # plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$")
    # plt.scatter(X2[:, 0], X2[:, 1], color='g', label=r"$t=2$")
    # plt.legend()
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)
    # plt.xlabel(r"$x_1$", fontsize=16)
    # plt.ylabel(r"$x_2$", fontsize=16)
    # plt.show()

    # for i, t_predi in enumerate(t_pred):
    #     if t_predi == 0:
    #         plt.scatter(X_train[i, 0], X_train[i, 1], color='b')
    #     elif t_predi == 1:
    #         plt.scatter(X_train[i, 0], X_train[i, 1], color='r')
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    # plt.xlabel(r"$x_1$",fontsize=16)
    # plt.ylabel(r"$x_2$",fontsize=16)
    # plt.title("Thresholded class predictions using VB")
    # plt.show()
    #
    # print('X_test', X_test[:, :2])
    # print('t_test', t_test)
    # print('dist', distributions)
    # print('likelihood', log_predictive_likelihood)
    # assert 0
    #
    # Z = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         s = np.exp(log_s[i])
    #         varphi = np.exp(log_varphi[j])
    #         Z = get_log_likelihood(1.0, 1.0, X_train, t_train, X_test, t_test, 2, 2, K)
    #         print(Z)
    # fig, axs = plt.subplots(1, figsize=(6, 6))
    # plt.contourf(log_ss, log_varphis, Z, zorder=1)

    K = 3
    D = 10 # is it?
    varphi = np.ones((K, D)) # TODO: choose varphi
    kernel = Multivariate(varphi)
    gibbs_classifier = VariationalBayesClassifier(kernel, X, t)

    # Get posterior estimates (iterate until convergence)


    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
