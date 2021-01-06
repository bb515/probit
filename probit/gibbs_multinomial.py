"""Multinomial Probit regression using Gibbs sampling with GP priors."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, multivariate_normal
from utilities import kernel, kernel_matrix, kernel_vector_matrix


def log_heaviside_probit_likelihood(u, t, G):
    """The log(p(t|u)) when t=1 indicates inclass and t=0 indicates outclass."""
    v_sample = np.dot(G, u)
    ones = np.ones(len(v_sample))
    phi = norm.cdf(v_sample)
    one_minus_phi = np.subtract(ones, phi)
    log_one_minus_phi = np.log(one_minus_phi)
    log_phi = np.log(phi)
    log_likelihood = (np.dot(t, log_phi)
                      + np.dot(np.subtract(ones, t), log_one_minus_phi))
    return log_likelihood


def gibbs(M_0, n_iters, t, covs):
    """
    Sampling occurs in blocks over the parameters: Y (auxilliaries) and M.
    :param initial: The initial location of the sampler.
    :param n_iters: the number of iterations.
    :param X: The data objects in R^D
    :param varphi: (K, D) np.array fo
    :param t: The target variables
    :return: Gibbs samples and acceptance rate.
    """
    samples = []
    M = M_0 # (N, K)
    K = np.shape(M)[1]
    I = np.eye(K)
    for iter in range(n_iters):
        # Empty Y vector to collect Yi samples over
        Y = []
        print('iteration', iter)
        for i, m in enumerate(M): # can replace with i in range N
            # Class index, k, is the target class
            k_true = t[i]
            # Initiate yi at 0
            yi = np.zeros(K)
            # TODO: this is a bit hacky
            yi[k_true] = -1.0
            # Sample from the cone of the truncated multivariate Gaussian
            # TODO: find out a way to compare yi[k_true] to all the other values apart from itself, as that is the
            # theory
            while yi[k_true] < np.max(yi):
                # sample Y jointly
                yi = multivariate_normal.rvs(mean=m, cov=I)
            # Add sample to the Y vector
            Y.append(yi)
        # By the end of this, Y is (N, K) matrix
        Y = np.array(Y)

        # Calculate statistics, then sample other conditional
        M_T = []
        for k in range(K):
            GP_posterior_mean = cov[k] @ Y.T[k]
            m_k = multivariate_normal.rvs(mean=mean, cov=cov[k])
            # Add sample to the M vector
            M_T.append(m_k)
        # By the end of this, M_T is a (K, N) matrix
        M_T = np.array(M_T)
        M = M_T.T
        samples.append((M, Y))
    return samples


def predict_gibbs(varphi, sigmas, X, x_new, Y_samples):
    X_new = np.append(X, x_new, axis=0)
    Cs_new = kernel_vector_matrix(x_new, X, varphi)
    K = np.shape(Y_samples[0])[1]
    n_posterior_samples = np.shape(Y_samples)[0]
    # Sample pmf over classes
    distribution_over_classes_samples = []
    # For each sample
    for Y in Y_samples:
        # Initiate m with null values
        m = -1. * np.ones(K)
        for k, y_k in enumerate(Y.T):
            mean_k = y_k.T @ sigmas[k] @ Cs_new[k]
            var_k = kernel(varphi, D, k, x_new[0], x_new[0]) - Cs_new[k].T @ sigmas[k] @ Cs_new[k]
            m[k] = norm.rvs(loc=mean_k , scale=var_k)
        # Take an expectation wrt the rv u
        # TODO: How do we know that 1000 samples is enough to converge? Do some empirical testing.
        distribution_over_classes_samples.append(expectation_wrt_u(m, n_samples=1000))
    monte_carlo_estimate = (1. / n_posterior_samples) * np.sum(distribution_over_classes_samples, axis=0)
    # TODO: Could also get a variance from the MC estimate
    return monte_carlo_estimate


def expectation_wrt_u(m, n_samples):
    """m is an (K, 1) np.ndarray filled with m_k^{new, s} where s is the sample, and k is the class indicator."""
    # Find matrix of coefficients
    K  = len(m)
    I = np.eye(K)
    Lambda = np.tile(m, (K, 1))
    Lambda_T = Lambda.T
    # Symmetric matrix of differences
    differences = Lambda - Lambda_T
    # Take samples
    samples = []
    for i in range(n_samples):
        # Sample normal random variate
        u = norm.rvs(0, 1, K)
        # u = norm.rvs(0, 1, 1)
        U = np.tile(u, (K, 1))
        # U = np.multiply(u, np.ones((K, K)))
        ones = np.ones((K, K))
        # TODO: consider doing a log-transform to do stable multiplication
        random_variables = np.add(U, differences)
        np.fill_diagonal(random_variables, 1)
        sample = np.prod(random_variables, axis=0)
        samples.append(sample)

    # TODO: check that this sum is correct
    distribution_over_classes = 1 / n_samples * np.sum(samples, axis=0)
    return(distribution_over_classes)

# Rough number of datapoints to generate for each class
n = 50

# Uniform quadrant dataset - linearly seperable
X0 = np.random.rand(n, 2)
X1 = np.ones((50, 2)) + np.random.rand(n, 2)
offset = np.array([0, 1])
offsets = np.tile(offset, (n, 1))
X2 = offsets + np.random.rand(n, 2)
#X0 = np.array([(np.random.rand(),np.random.rand()) for _ in range(n)])
#X1 = np.array([(1+np.random.rand(),1+np.random.rand()) for _ in range(n)])
t0 = np.zeros(len(X0))
t1 = np.ones(len(X1))
t2 = 2 * np.ones(len(X2))

## Plotting
plt.scatter(X0[:,0], X0[:,1], color='b', label=r"$t=0$")
plt.scatter(X1[:,0], X1[:,1], color='r', label=r"$t=1$")
plt.scatter(X2[:,0], X2[:,1], color='g', label=r"$t=2$")
plt.legend()
plt.xlim(0,2)
plt.ylim(0,2)
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.show()

# I don't think we need to extend the input vectors, since this was just for a constant in a linear model
# Extend the input vectors
# X0_tilde = np.c_[np.ones(len(X0)),X0]
# X1_tilde = np.c_[np.ones(len(X1)),X1]
# X2_tilde = np.c_[np.ones(len(X2)),X2]

# prepare data
# X = np.r_[X0_tilde, X1_tilde, X2_tilde]
X = np.r_[X0, X1, X2]
t = np.r_[t0, t1, t2]

D = np.shape(X)[1]
N = np.shape(X)[0]
K = 3

# Shuffle data - may only be necessary for matrix conditioning
Xt = np.c_[X,t]
np.random.shuffle(Xt)
X = Xt[:,:D]
t = np.intc(Xt[:,-1]) # make sure they are integer values

# Calculate the covariance matrix for the posterior predictive GP
# TODO: find a better definition of varphi, for now initiate as ones
varphi = np.ones((K, D))
I = np.eye(N)
Cs = kernel_matrix(X, varphi)
sigmas = []
covs = []
for C in Cs:
    sigma = np.linalg.inv(I + C)
    cov = C @ sigma
    sigmas.append(sigma)
    covs.append(cov)

# Sample M0 from prior, but first sample Y0, maybe its not necessary and maybe it causes problems
M_0T = []
# Empty Y vector to collect Yi samples over
Y_0 = []
for i in range(N):
    yi = multivariate_normal.rvs(mean=np.zeros(K), cov=np.eye(K))
    # Add sample to the Y vector
    Y_0.append(yi)
# By the end of this, Y is (N, K) matrix
Y_0 = np.array(Y_0)

for k in range(K):
    mean = covs[k] @ Y_0.T[k]
    mk = multivariate_normal.rvs(mean, covs[k])
    # Add sample to the M vector
    M_0T.append(mk)
# By the end of this, M_T is a (K, N) matrix
M_0T = np.array(M_0T)
M_0 = M_0T.T

M_0 = np.zeros((150, 3))

# Take n samples, returning the parameter m and y samples
n = 500
samples = gibbs(M_0, n, t, cov)

M_samples = np.array([sample[0] for sample in samples])
Y_samples = np.array([sample[1] for sample in samples])

# Predict the classes for the new data points, X_new
x_new = np.array([[0.50, 1.50]])

predictive_multinomial_distribution = predict_gibbs(varphi, sigmas, X, x_new, Y_samples)

print(predictive_multinomial_distribution)

# Predict the classes for the new data points, X_new
x_new = np.array([[1.50, 1.50]])

predictive_multinomial_distribution = predict_gibbs(varphi, sigmas, X, x_new, Y_samples)

print(predictive_multinomial_distribution)

# Predict the classes for the new data points, X_new
x_new = np.array([[0.50, 0.50]])

predictive_multinomial_distribution = predict_gibbs(varphi, sigmas, X, x_new, Y_samples)

print(predictive_multinomial_distribution)

# Predict the classes for the new data points, X_new
x_new = np.array([[1.50, 0.50]])

predictive_multinomial_distribution = predict_gibbs(varphi, sigmas, X, x_new, Y_samples)

print(predictive_multinomial_distribution)

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
