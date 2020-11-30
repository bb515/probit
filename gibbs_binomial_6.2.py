"""Binomial Probit regression using Gibbs sampling with GP priors."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, multivariate_normal
from scipy.spatial import distance_matrix, distance
import pathlib
import time


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


def simple_kernel_matrix(X_1, X_2, varphi, s):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function.

    :param X: are the features drawn from feature space.
    :param varphi: is the length scale common to all dimensions and classes.
    :param s: is the vertical scale common to all classes.
    """
    D = distance_matrix(X_1, X_2)
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
        if iter % 10 == 0:
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


def scalar_predict_gibbs(varphi, s, sigma, x_new, X, Y_samples):
    x_new = np.array([x_new])
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


def vector_predict_gibbs(varphi, s, sigma, X_test, X_train, Y_samples):
    X_new = np.append(X_test, X_train, axis=0)
    N_test = np.shape(X_test)[0]
    N_train = np.shape(X_train)[0]
    ##print(N_test, 'N_test')
    Cs_new = simple_kernel_matrix(X_train, X_test, varphi, s)
    cs_new = np.diagonal(simple_kernel_matrix(X_test, X_test, varphi, s))
    # (N_train, N_test)
    intermediate_matrix = sigma @ Cs_new
    # Assertion that Cs_new is (X_train, X_test)
    assert(np.shape(Cs_new) == (np.shape(X_train)[0], np.shape(X_test)[0]))
    K = np.shape(Y_samples[0])[1]
    n_posterior_samples = np.shape(Y_samples)[0]
    # Sample pmf over classes
    distribution_samples = []
    # For each sample
    for Y in Y_samples:
        # Initiate m with null values
        ms = -1. * np.ones((N_test, K))
        for k, y_k in enumerate(Y.T):
            # (1, N_test)
            time1 = time.time()
            mean_k = y_k.T @ intermediate_matrix
            time2 = time.time()
            for i in range(N_test):
                var_ki = cs_new[i] - np.dot(Cs_new.T[i], intermediate_matrix.T[i])
                ms[i, k] = norm.rvs(loc=mean_k[i] , scale=var_ki)
        # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
        distribution_samples.append(multidimensional_expectation_wrt_u(ms, n_samples=1000))
    monte_carlo_estimates = (1. / n_posterior_samples) * np.sum(distribution_samples, axis=0)
    # TODO: Could also get a variance from the MC estimate
    return monte_carlo_estimates


def predict_gibbs(varphi, s, sigma, X_test, X_train, Y_samples, scalar=None):
    if not scalar:
        predictive_multinomial_distributions =  vector_predict_gibbs(varphi, s, sigma, X_test, X_train, Y_samples)
    else:
        N_test = np.shape(X_test)[0]
        predictive_multinomial_distributions = []
        for i in range(N_test):
            predictive_multinomial_distributions.append(
                scalar_predict_gibbs(varphi, s, sigma, X_test[i], X_train, Y_samples))
    return predictive_multinomial_distributions


def sample_U(K, same_across_classes=None):
    if not same_across_classes:
        u = norm.rvs(0, 1, 1)
        U = np.multiply(u, np.ones((K, K)))
    else:
        # This might be a better option as there is no restriction on u across classes since it is just a random sample
        # What effect will it have?
        u = norm.rvs(0, 1, K)
        U = np.tile(u, (K, 1))
    return U


def multidimensional_expectation_wrt_u(ms, n_samples):
    """ms is an (N_test, K) np.ndarray filled with m_k^{new_i, s} where s is the sample, k is the class indicator
    and i is the index of the test object."""
    # Find matrix of coefficients
    K = np.shape(ms)[1]
    I = np.eye(K)
    N_test = np.shape(ms)[0]
    ms = ms.reshape((N_test, 1, K))
    print(np.shape(ms), 'shape_ms')
    # Lambdas is an (n_test, K, K) stack of Lambda matrices
    Lambdas = np.tile(ms, (1, K, 1))
    print(np.shape(Lambdas), 'shape Lambdas')
    Lambdas_T = Lambdas.transpose((0, 2, 1))
    print(np.shape(Lambdas_T), 'shape LambdasT')
    # Symmetric matrices of differences
    differences = np.subtract(Lambdas_T, Lambdas)
    # Take samples
    sampless = []
    for i in range(n_samples):
        # U is (K, K)
        U = sample_U(K)
        # assume its okay to use the same random variable over all of these data points.
        # Us is (N_test, K, K)
        Us = np.tile(U, (N_test, 1, 1))
        random_variables = np.add(Us, differences)
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        # Employ a for loop here as I couldn't work out the vectorised way
        # the operation isn't so expensive.
        for j in range(N_test):
            np.fill_diagonal(log_cum_dists[j], 0)
        # Sum across the elements of interest, which is the inner most rows (axis=2)
        # log samples will be (N_test, K) array of a (log) distribution sample for each test point
        log_samples = np.sum(log_cum_dists, axis=2)
        samples = np.exp(log_samples)
        sampless.append(samples)
    # Calculate the MC estimate, should be a (N_test, K) array, so sum is across the samples (axis=0)
    distributions_over_classes = 1 / n_samples * np.sum(sampless, axis=0)
    assert(np.shape(distributions_over_classes) == (N_test, K))
    assert 0
    return distributions_over_classes


def expectation_wrt_u(m, n_samples):
    """m is an (K, ) np.ndarray filled with m_k^{new, s} where s is the sample, and k is the class indicator."""
    # Find matrix of coefficients
    K  = len(m)
    I = np.eye(K)
    Lambda = np.tile(m, (K, 1))
    Lambda_T = Lambda.T
    # antisymmetric matrix of differences, the rows contain the elements of the product of interest
    difference = Lambda - Lambda_T
    # Take samples
    samples = []
    for i in range(n_samples):
        U = sample_U(K)
        random_variables = np.add(U, difference)
        cum_dist = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dist = np.log(cum_dist)
        np.fill_diagonal(log_cum_dist, 0)
        # Sum across the elements of the log product of interest (rows, so axis=1)
        log_sample = np.sum(log_cum_dist, axis=0)
        sample = np.exp(log_sample)
        samples.append(sample)

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
    for i, t_i in enumerate(t):
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
            # Finally, draw normally from the final 8
            x = np.append(x, norm.rvs(0, 1, 8))
            X1.append(x)
        elif t_i == 2:
            x = norm.rvs(0, 0.1, 2)
            # Finally, draw normally from the final 8
            x = np.append(x, norm.rvs(0, 1, 8))
            X2.append(x)
            # Finally, set t_i to 1
            t[i] = 1
        X.append(x)
    # By now Xs is a (N, 10) matrix
    X = np.array(X)
    X0 = np.array(X0)
    X1 = np.array(X1)
    X2 = np.array(X2)
    return X, t, X0, X1, X2


def get_log_likelihood(s, varphi, X_train, t_train, X_test, t_test, n_burn, n_samples, K):
    """Given the hyper-parameter values, return a the predictive likelihood."""
    # Calculate the covariance matrix for the posterior predictive GP
    N = np.shape(X_train)[0]
    N_test = np.shape(X_test)[0]
    I = np.eye(N)
    C = simple_kernel_matrix(X_train, X_train, varphi, s)
    sigma = np.linalg.inv(I + C)
    cov = C @ sigma

    M_0 = np.zeros((N, K))
    # Burn in of Gibbs sampler of 2000 samples
    samples = simple_gibbs(M_0, n_burn, t_train, cov)

    M_samples = np.array([sample[0] for sample in samples])
    M_0_burned_in = M_samples[-1]

    # 1000 samples for inference
    samples = simple_gibbs(M_0_burned_in, n_samples, t_train, cov)

    M_samples = np.array([sample[0] for sample in samples])
    Y_samples = np.array([sample[1] for sample in samples])

    predictive_multinomial_distributions = predict_gibbs(varphi, s, sigma, X_test, X_train, Y_samples, scalar=None)

    # Get the hard predicted classes
    t_pred = np.argmax(predictive_multinomial_distributions, axis=1)
    log_predictive_likelihood = 0
    for i, t_testi in enumerate(t_test):
        log_predictive_likelihood += np.log(predictive_multinomial_distributions[i][t_testi])

    return log_predictive_likelihood, predictive_multinomial_distributions, t_pred

# Generate 4620 + 240 draws from the above distribution
# Sample from the three target values
# X, t, X0, X1, X2 = generate_data_n_draws(4620)
# np.savez(read_path / 'test.npz', X=X, t=t, X0=X0, X1=X1, X2=X2)

# X, t, X0, X1, X2 = generate_data_n_draws(240)
# np.savez(read_path / 'train.npz', X=X, t=t, X0=X0, X1=X1, X2=X2)

## Plotting


train = np.load(read_path / 'train.npz')
test = np.load(read_path / 'test.npz')

X_train = train['X']
t_train = train['t']
X0_train = train['X0']
X1_train = train['X1']
X2_train = train['X2']

X_test = test['X']
t_test = test['t']
X0_test = test['X0']
X1_test = test['X1']
X2_test = test['X2']

plt.scatter(X0_train[:,0], X0_train[:,1], color='b', label=r"$t=0$")
plt.scatter(X1_train[:,0], X1_train[:,1], color='r', label=r"$t=1$")
plt.scatter(X2_train[:,0], X2_train[:,1], color='r')
plt.legend()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.show()

N = 100
# Range of hyper-parameters over which to explore the space
log_s = np.linspace(-1, 5, N)
log_varphi = np.linspace(-1, 5, N)

log_ss, log_varphis = np.meshgrid(log_s, log_varphi)
X_test = X_test[:50]
t_test = t_test[:50]

# number of classes
K = 2

log_predictive_likelihood, distributions, t_pred = get_log_likelihood(1.0, 1.0, X_train, t_train, X_test, t_test, 1, 1000, K)

for i, t_predi in enumerate(t_pred):
    if t_predi == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='b')
    elif t_predi == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='r')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.title("Thresholded class predictions using Gibbs")
plt.show()

print('X_test', X_test[:, :2])
print('t_test', t_test)
print('dist', distributions)
print('likelihood', log_predictive_likelihood)
assert 0

Z = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        s = np.exp(log_s[i])
        varphi = np.exp(log_varphi[j])
        Z = get_log_likelihood(1.0, 1.0, X_train, t_train, X_test, t_test, 2, 2, K)
        print(Z)
fig, axs = plt.subplots(1, figsize=(6, 6))
plt.contourf(log_ss, log_varphis, Z, zorder=1)

# N = 20
# x, y = np.mgrid[-0.1:2.1:2.2/N, -0.1:2.1:2.2/N]
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
