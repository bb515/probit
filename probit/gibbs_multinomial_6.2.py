"""Multinomial Probit regression using Gibbs sampling with GP priors."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, multivariate_normal
from scipy.spatial import distance_matrix, distance
import pathlib

read_path = pathlib.Path()


def simple_kernel(varphi, X_i, X_j):
    """Get the ij'th element of C, given the X_i and X_j, indices and hyper-parameters."""
    sum = 0
    D = np.shape(X_i)[0]
    for d in range(D):
        sum += pow((X_i[d] - X_j[d]), 2)
    sum *= varphi
    C_ij = np.exp(-1. * sum)
    return C_ij


def kernel(varphi, k, X_i, X_j):
    """Get the ij'th element of C, given the X_i and X_j, indices and hyper-parameters."""
    sum = 0
    D = np.shape(X_i)[0]
    for d in range(D):
        sum += varphi[k, d] * pow((X_i[d] - X_j[d]), 2)
    C_ij = np.exp(-1. * sum)
    return C_ij


def simple_kernel_matrix(X, varphi, s):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function.

    :param X: are the features drawn from feature space.
    :param varphi: is the length scale common to all dimensions and classes.
    :param s: is the vertical scale common to all classes.
    """
    D = distance_matrix(X, X)
    return np.multiply(s, np.exp(-1. * varphi * pow(D, 2)))


def simple_kernel_vector_matrix(x_new, X, varphi, s):
    """
    :param X: are the objects drawn from feature space.
    :param x_new: is the new object drawn from the feature space.
    :param varphi: is the length scale common to all dimensions and classes.
    :param s: is the vertical scale common to all classes.
    :return: the C_new vector.
    """
    N = np.shape(X)[0]
    X_new = np.tile(x_new, (N, 1))
    # This is probably horribly inefficient
    D = distance.cdist(X_new, X)[0]
    return np.multiply(s, np.exp(-1. * varphi * pow(D, 2)))


def kernel_matrix(X, varphi):
    """Generate Gaussian kernel matrices as a numpy array.

    This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
    quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

    e.g.
    for k in range(K):
        Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

    :param X: (N, D) dimensional numpy.ndarray which holds the feature vectors.
    :param varphi: (K, D) dimensional numpy.ndarray which holds the
        covariance hyperparameters.
    :returns Cs: A (K, N, N) array of K (N, N) covariance matrices.
    """
    Cs = []
    K = np.shape(varphi)[0] # length of the classes
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    # The general covariance function has a different length scale for each dimension.
    for k in range(K):
        # for each x_i
        C = -1.* np.ones((N, N))
        for i in range(N):
            for j in range(N):
                C[i, j] = kernel(varphi, k, X[i], X[j])
        Cs.append(C)

    return Cs


def kernel_vector_matrix(x_new, X, varphi):
    """Generate Gaussian kernel matrices as a numpy array.

    This is a one of calculation that can't be factorised in the most general case, so we don't mind that is has a
    quadruple nested for loop. In less general cases, then scipy.spatial.distance_matrix(x, x) could be used.

    e.g.
    for k in range(K):
        Cs.append(np.exp(-pow(D, 2) * pow(phi[k])))

    :param x_new: (1, D) dimensional numpy.ndarray of the new feature vector.
    :param X: (N, D) dimensional numpy.ndarray which holds the data feature vectors.
    :param varphi: (K, D) dimensional numpy.ndarray which holds the
        covariance hyperparameters.
    :returns Cs: A (K, N, N) array of K (N, N) covariance matrices.
    """
    Cs_new = []
    K = np.shape(varphi)[0] # length of the classes
    N = np.shape(X)[0]
    D = np.shape(X)[1]
    # The general covariance function has a different length scale for each dimension.
    for k in range(K):
        C_new = -1.* np.ones(N)
        for i in range(N):
            C_ij = kernel(varphi, k, X[i], x_new[0])
            C_new[i] = kernel(varphi, k, X[i], x_new[0])
        Cs_new.append(C_new)
    return Cs_new


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


def simple_gibbs(M_0, n_iters, t, cov):
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
            mean = cov @ Y.T[k]
            m_k = multivariate_normal.rvs(mean=mean, cov=cov)
            # Add sample to the M vector
            M_T.append(m_k)
        # By the end of this, M_T is a (K, N) matrix
        M_T = np.array(M_T)
        M = M_T.T
        samples.append((M, Y))
    return samples


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
            mean = cov[k] @ Y.T[k]
            m_k = multivariate_normal.rvs(mean=mean, cov=covs[k])
            # Add sample to the M vector
            M_T.append(m_k)
        # By the end of this, M_T is a (K, N) matrix
        M_T = np.array(M_T)
        M = M_T.T
        samples.append((M, Y))
    return samples


def predict_gibbs(varphi, sigmas, X, x_new, Y_samples):
    X_new = np.append(X, x_new, axis=0)
    Cs_new = simple_kernel_vector_matrix(x_new, X, varphi, s)
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
            var_k = kernel(varphi, k, x_new[0], x_new[0]) - Cs_new[k].T @ sigmas[k] @ Cs_new[k]
            m[k] = norm.rvs(loc=mean_k , scale=var_k)
        # Take an expectation wrt the rv u
        # TODO: How do we know that 1000 samples is enough to converge? Do some empirical testing.
        distribution_over_classes_samples.append(expectation_wrt_u(m, n_samples=1000))
    monte_carlo_estimate = (1. / n_posterior_samples) * np.sum(distribution_over_classes_samples, axis=0)
    # TODO: Could also get a variance from the MC estimate
    return monte_carlo_estimate


def simple_predict_gibbs(varphi, sigma, X, x_new, Y_samples):
    X_new = np.append(X, x_new, axis=0)
    Cs_new = simple_kernel_vector_matrix(x_new, X, varphi, s)
    K = np.shape(Y_samples[0])[1]
    n_posterior_samples = np.shape(Y_samples)[0]
    # Sample pmf over classes
    distribution_over_classes_samples = []
    # For each sample
    for Y in Y_samples:
        # Initiate m with null values
        m = -1. * np.ones(K)
        for k, y_k in enumerate(Y.T):
            mean_k = y_k.T @ sigma @ Cs_new
            var_k = simple_kernel(varphi, x_new[0], x_new[0]) - Cs_new.T @ sigma @ Cs_new
            m[k] = norm.rvs(loc=mean_k , scale=var_k)
        # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
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
