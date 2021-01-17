"""TODO: Multinomial Probit regression using Gibbs sampling with GP priors."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, multivariate_normal
import pathlib
from probit.utilities import log_heaviside_probit_likelihood

read_path = pathlib.Path()

## Superseded, but may contain useful information.
# def kernel(varphi, k, X_i, X_j):
#     """Get the ij'th element of C, given the X_i and X_j, indices and hyper-parameters."""
#     sum = 0
#     D = np.shape(X_i)[0]
#     for d in range(D):
#         sum += varphi[k, d] * pow((X_i[d] - X_j[d]), 2)
#     C_ij = np.exp(-1. * sum)
#     return C_ij
#
#
# def kernel_matrix(X, varphi):
#     """Generate Gaussian kernel matrices as a numpy array.
#
#     This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
#     quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.
#
#     e.g.
#     for k in range(K):
#         Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))
#
#     :param X: (N, D) dimensional numpy.ndarray which holds the feature vectors.
#     :param varphi: (K, D) dimensional numpy.ndarray which holds the
#         covariance hyperparameters.
#     :returns Cs: A (K, N, N) array of K (N, N) covariance matrices.
#     """
#     Cs = []
#     K = np.shape(varphi)[0] # length of the classes
#     N = np.shape(X)[0]
#     D = np.shape(X)[1]
#     # The general covariance function has a different length scale for each dimension.
#     for k in range(K):
#         # for each x_i
#         C = -1.* np.ones((N, N))
#         for i in range(N):
#             for j in range(N):
#                 C[i, j] = kernel(varphi, k, X[i], X[j])
#         Cs.append(C)
#
#     return Cs
#
#
# def kernel_vector_matrix(x_new, X, varphi):
#     """Generate Gaussian kernel matrices as a numpy array.
#
#     This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
#     quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.
#
#     e.g.
#     for k in range(K):
#         Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))
#
#     :param x_new: (1, D) dimensional numpy.ndarray of the new feature vector.
#     :param X: (N, D) dimensional numpy.ndarray which holds the data feature vectors.
#     :param varphi: (K, D) dimensional numpy.ndarray which holds the
#         covariance hyperparameters.
#     :returns Cs: A (K, N, N) array of K (N, N) covariance matrices.
#     """
#     Cs_new = []
#     K = np.shape(varphi)[0] # length of the classes
#     N = np.shape(X)[0]
#     D = np.shape(X)[1]
#     # The general covariance function has a different length scale for each dimension.
#     for k in range(K):
#         C_new = -1.* np.ones(N)
#         for i in range(N):
#             C_ij = kernel(varphi, k, X[i], x_new[0])
#             C_new[i] = kernel(varphi, k, X[i], x_new[0])
#         Cs_new.append(C_new)
#     return Cs_new


def generate_data_n_draws(n):
    """Generate the draw from the ToyBox dataset."""
    # sample uniformly
    X = []
    X0 = []
    X1 = []
    X2 = []
    t = np.random.randint(3, size=n)
    for t_i in t:
        if t_i == 0:
            dist = 0
            while not ((dist < 0.5) and (dist > 0.1)):
                x = np.random.uniform(low=-1.0, high=1.0, size=2)
                dist = np.linalg.norm(x)
            # Finally, draw normally from the final 8
            x = np.append(x, norm.rvs(0, 1, 8))
            X0.append(x)
        elif t_i == 1:
            dist = 0
            while not ((dist < 1.0) and (dist > 0.6)):
                x = np.random.uniform(low=-1.0, high=1.0, size=2)
                dist = np.linalg.norm(x)
                print(dist)
            # Finally, draw normally from the final 8
            x = np.append(x, norm.rvs(0, 1, 8))
            X1.append(x)
        elif t_i == 2:
            x = norm.rvs(0, 0.1, 2)
            # Finally, draw normally from the final 8
            x = np.append(x, norm.rvs(0, 1, 8))
            X2.append(x)
        X.append(x)
    # By now Xs is a (N, 10) matrix
    X = np.array(X)
    X0 = np.array(X0)
    X1 = np.array(X1)
    X2 = np.array(X2)
    return X, t, X0, X1, X2

# Generate 4620 + 240 draws from the above distribution
# Sample from the three target values
X, t, X0, X1, X2 = generate_data_n_draws(4620)
np.savez(read_path / 'test.npz', X=X, t=t, X0=X0, X1=X1, X2=X2)

X, t, X0, X1, X2 = generate_data_n_draws(240)
np.savez(read_path / 'train.npz', X=X, t=t, X0=X0, X1=X1, X2=X2)
print(np.shape(X0), np.shape(X1), np.shape(X2))
## Plotting
plt.scatter(X0[:,0], X0[:,1], color='b', label=r"$t=0$")
plt.scatter(X1[:,0], X1[:,1], color='r', label=r"$t=1$")
plt.scatter(X2[:,0], X2[:,1], color='g', label=r"$t=2$")
plt.legend()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.show()

D = np.shape(X)[1]
N = np.shape(X)[0]
print(D, N)


# Range of hyper-parameters over which to explore the space
log_s = np.linspace(-1, 5, 10)
log_varphi = np.linspace(-1, 5, 10)

def get_log_likelihood(s, varphi, X_train, t_train, n_burn, n_samples):
    """Given the hyper-parameter values, return a the predictive likelihood."""
    # Calculate the covariance matrix for the posterior predictive GP
    I = np.eye(N)
    C = simple_kernel_matrix(X_train, varphi, s)
    sigma = np.linalg.inv(I + C)
    cov = C @ sigma

    M_0 = np.zeros((N, D))
    # Burn in of Gibbs sampler of 2000 samples
    samples = simple_gibbs(M_0, n_burn, t_train, cov)

    M_samples = np.array([sample[0] for sample in samples])
    M_0_burned_in = M_samples[-1]
    print(M_0_burned_in)

    # 1000 samples for inference
    samples = simple_gibbs(M_0_burned_in, n_samples, t_train, cov)

    M_samples = np.array([sample[0] for sample in samples])
    Y_samples = np.array([sample[1] for sample in samples])

    # Predict the classes for the new data points, X_new
    x_new = np.zeros(8)
    x_new = np.append(x_new, np.array([0.0, 0.0]))
    x_new = np.array([x_new])

    predictive_multinomial_distribution = simple_predict_gibbs(varphi, sigma, X, x_new, Y_samples)
    print(predictive_multinomial_distribution)

get_log_likelihood(1.0, 1.0, X_train, t_train, 20, 10)
# N = 20
# x, y = np.mgrid[-0.1:2.1:2.2/N, -0.1:2.1:2.2/N]
# pos = np.dstack((x, y))
#
# Z = np.zeros((N,N))
# for i in range(N):
#     for j in range(N):
#         Z[i,j] = predict_gibbs(beta_samples, np.array( (x[i,j], y[i,j]) ) )
#
# fig, axs = plt.subplots(1,figsize=(6,6))
# plt.scatter(X0[:,0], X0[:,1], color='b', label=r"$t=0$", zorder=10)
# plt.scatter(X1[:,0], X1[:,1], color='r', label=r"$t=1$", zorder=10)
# plt.contourf(x,y,Z,zorder=1)
# plt.xlim(0,2)
# plt.ylim(0,2)
# plt.legend()
# plt.xlabel(r"$x_1$",fontsize=16)
# plt.ylabel(r"$x_2$",fontsize=16)
# plt.title("Contour plot - Gibbs")
# plt.show()
