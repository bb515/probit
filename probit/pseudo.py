"""Pseudo marginal inference."""
from abc import ABC, abstractmethod
# from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
import random
from scipy.linalg.decomp_svd import svdvals
from scipy.stats import norm, multivariate_normal
from tqdm import trange
import warnings
import math
from probit.data.utilities import load_data_synthetic
from scipy.spatial import distance_matrix, distance
import matplotlib.pyplot as plt

dataset = "thirteen"
data_from_prior = True

# Get the existing data
X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, scale_0, K, D, colors = load_data_synthetic(dataset, data_from_prior)

#np.savez("true_thirteen.npz", x=X_true, y=Y_true)

# Get the data
#true = np.load("true.npz")
true = np.load("true_thirteen.npz")
EP1 = np.load("EP_tertile.npz")
EP2 = np.load("EP_thirteen.npz")
VB1 = np.load("VB_tertile.npz")
VB2 = np.load("VB_thirteen.npz")

x = true["x"]
y = true["y"]
x_EP1 = EP1["x"]
y_EP1 = EP1["y"]
x_EP2 = EP2["x"]
y_EP2 = EP2["y"]
x_VB1 = VB1["x"]
y_VB1 = VB1["y"]
x_VB2 = VB2["x"]
y_VB2 = VB2["y"]


def kernel_matrix(X1, X2, scale, varphi):
    """
    Get Gram matrix efficiently using scipy's distance matrix function.

    :arg X1: (N1, D) data matrix.
    :type X1: class:`numpy.ndarray`
    :arg X2: (N2, D) data matrix. Can be the same as X1.
    :type X2: class:`numpy.ndarray`
    :return: (N1, N2) Gram matrix.
    :rtype: class:`numpy.ndarray`
    """
    def distance_mat(X1, X2):
        """
        Return a distance matrix using scipy spatial.

        :arg X1: The (N1, D) input array for the distance matrix.
        :type X1: :class:`numpy.ndarray`
        :arg X2: The (N2, D) input array for the distance matrix.
        :type X2: :class:`numpy.ndarray` or float
        :return: Euclidean distance matrix (N1, N2).
        :rtype: :class:`numpy.ndarray`.
        """
        # Case that D = 1
        print(np.shape(X1))
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        print(np.shape(X1))
        return distance_matrix(X1, X2)
    distance_mat = distance_mat(X1, X2)
    return np.multiply(scale, np.exp(-1. * varphi * distance_mat**2))


def kernel_vector(x_new, X, scale, varphi):
    """
    Get the kernel vector given an input vector (x_new) input matrix (X).

    :arg x_new: The new (1, D) point drawn from the data space.
    :type x_new: :class:`numpy.ndarray`
    :arg X: (N, D) data matrix.
    :type X: class:`numpy.ndarray`
    :return: the (N,) covariance vector.
    :rtype: class:`numpy.ndarray`
    """
    N = np.shape(X)[0]
    X_new = np.tile(x_new, (N, 1))
    # This is probably horribly inefficient
    D = distance.cdist(X_new, X)[0]
    return np.multiply(scale, np.exp(-1. * varphi * D**2))

# Calculate GP posterior
x_GP = X_true
C0 = kernel_matrix(X_true, X_true, scale=1.0, varphi=30.0)
C = C0 + 1e-6 * np.identity(len(X_true))
X_test = np.linspace(-0.5, 1.5, 1000)
print(np.shape(Y_true))
Y_true = Y_true.flatten()

cov_ = np.linalg.inv(C + noise_variance_0 * np.identity(len(X_true)))  # just check you are using the correct cov on the gradients.

C_news = kernel_matrix(X_true, X_test, scale_0, varphi_0)  # (N, N_test)
c_news = np.diag(kernel_matrix(X_test, X_test, scale_0, varphi_0)) # (N_test, )
intermediate_vectors = cov_ @ C_news  # (N, N_test)
intermediate_scalars = np.sum(np.multiply(C_news, intermediate_vectors), axis=0)  # (N_test, )
GP_m = np.einsum('ij, i -> j', intermediate_vectors, Y_true)  # (N, N_test) (N, ) = (N_test,)
GP_var = c_news - intermediate_scalars
GP_std = np.sqrt(GP_var)

# Plot the result
#plt.plot(x_EP1, y_EP1, label="ordinal EP")
#plt.plot(x_VB1, y_VB1, label="ordinal VB")
plt.plot(x_EP2, y_EP2, label="ordinal EP")
plt.plot(x_VB2, y_VB2, label="ordinal VB")
plt.plot(X_test, GP_m, color='g', label="continuous GP")
plt.plot(x, y, '--k', label="true function")
plt.legend()
plt.scatter(X_true, Y_true, color='g', s=4)
plt.xlim((-0.1, 1.1))
plt.ylim((-1.2, 1.0))
plt.show()
