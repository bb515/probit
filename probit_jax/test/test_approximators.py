"""
TODO: Update (add regression unit tests).

Tests for the approximators class."""
from probit.approximators import VBBinomialGP
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import pytest
import pathlib
write_path = pathlib.Path()


@pytest.fixture(
    scope="session")
def basic_binary_classifier(data_path, kernel):
    """Create a basic 2D binary classification problem."""
    # # Classes
    # K=2
    # # Datapoints per class
    # N = 50
    # # Datapoints total
    # N_total = N * K
    # # Dimension of the data
    # D = 2
    # # Uniform quadrant dataset - linearly seperable
    # X0 = np.random.rand(N, D)  # AGD slightly cleaner code here
    # X1 = np.ones((N, D)) + np.random.rand(N, D)
    # offset = np.array([0, 1])
    # offsets = np.tile(offset, (N, 1))
    # t0 = np.zeros(len(X0))
    # t1 = np.ones(len(X1))
    # # Prepare data
    # X = np.r_[X0, X1]
    # t = np.r_[t0, t1]
    # # Shuffle data - may only be necessary for matrix conditioning
    # Xt = np.c_[X, t]
    # np.random.shuffle(Xt)
    # X = Xt[:, :3]
    # t = Xt[:, -1]
    # plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$")
    # plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$")
    # plt.legend()
    # plt.xlim(0, 2)
    # plt.ylim(0, 2)
    # plt.xlabel(r"$x_1$", fontsize=16)
    # plt.ylabel(r"$x_2$", fontsize=16)
    # plt.show()
    # np.savez(write_path / "data_basic_binary_classification.npz", X=X, t=t, X0=X0, X1=X1)
    data = np.load(write_path / "data_basic_binary_classification.npz")

    X = data['X']
    t = data['t']
    X0 = data['X0']
    X1 = data['X1']

    basic_binary_classifier = VBBinomialGP(X, t, kernel)
    return basic_binary_classifier


class TestVBBinomialGP:
    """Test the functions from VBBinomialGP."""

    def test_VB(self):
        """Test a basic Variational Bayes classifier."""
        classifier = basic_binary_classifier
        M_tilde = np.ones()
        M_tilde, Sigma_tilde, C_tilde, Y_tilde = classifier.estimate(M_tilde)

        Z = classifier.predict(Sigma_tilde, Y_tilde, n_samples=1000, vectorised=True)
        pass


basic_binary_classifier(write_path)


