"""Variational approximation."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal, expon
from utilities import (
    sample_U, function_u3, function_u2, sample_varphi, samples_varphi)
from kernels import binary
import matplotlib.pyplot as plt


def ws_general_unnormalised(psi_tilde, X, kernel, n_samples, M_tilde):
    """Return the w values of the sample.

    :arg psi_tilde: Posterior mean estimate of psi
    :arg X: The datum.
    :arg kernel: The kernel class object.
    """
    N = np.shape(X)[0]
    # Draw from varphi
    varphi_samples = samples_varphi(psi_tilde)
    for varphi_sample in varphi_samples:
        kernel.varphi = varphi_sample
        Cs = kernel.kernel_matrix(X)
        mean = np.zeros(N)
        M_tilde = m_tilde[K]
        for i, C in enumerate(Cs):
            normal_pdf = multivariate_normal.pdf(M_tilde[i], mean=mean, cov=C)
    return normal_pdf


def varphi_tilde(varphi_samples):
    """Return the posterior mean estimate of varphi via importance sampling."""
    sum = 0
    n_samples = np.shape(varphi_samples)[0]
    for i in range(n_samples):
        varphi_sample = varphi_samples[i]
        sum += varphi_sample * w(varphi_sample)
    return sum


def M_tilde(kernel, Y_tilde, varphi_tilde, X):
    """Return the posterior mean estimate of M."""
    # Update the varphi with new values
    kernel.varphi = varphi_tilde
    # calculate updated C and sigma
    C = kernel.kernel_matrix(X)
    I = np.eye(np.shape(X)[0])
    sigma = np.linalg.inv(I + C)
    return C @ sigma @ Y_tilde


def Y_tilde(M_tilde):
    """Calculate y_tilde elements as defined on page 9 of the paper."""
    N, K = np.shape(Y_tilde)
    Y_tilde = -1. * np.ones((N, K))
    # TODO: I can vectorise expectation_p_m with tensors later.
    for i in range(N):
        m_tilde_n = M_tilde[i, :]
        t_n = np.argmax(m_tilde_n)
        expectation_3 = expectation_p_m(function_u3, m_tilde_n, t_n, n_samples=1000)
        expectation_2 = expectation_p_m(function_u2, m_tilde_n, t_n, n_samples=1000)
        # Equation 5
        y_tilde_n = m_tilde_n - np.divide(expectation_2, expectation_3)
        # Equation 6 follows
        # This part of the differences sum must be 0, since sum is over j \neq i
        y_tilde_n[t_n] = m_tilde_n[t_n]
        diff_sum = np.sum(y_tilde_n - m_tilde_n)
        y_tilde_n[t_n] = m_tilde_n[t_n] - diff_sum
        Y_tilde[i, :] = y_tilde_n
    return Y_tilde


def expectation_p_m(function, m_n, t_n, n_samples):
    """
    m is an (K, ) np.ndarray filled with m_k^{new, s} where s is the sample, and k is the class indicator.

    function is a numpy function. e,g, function_u1(difference, U) which is the numerator of eq () and
    function_u2(difference, U) which is the denominator of eq ()
    """
    # Factored out calculations
    difference = matrix_of_differences(m_n)
    t_n = np.argmax(m_n)
    vector_difference = difference[:, t_n]
    K = len(m_n)
    # Take samples
    samples = []
    for i in range(n_samples):
        U = sample_U(K, different_across_classes=1)
        function_eval = function(difference, vector_difference, U, t_n, K)
        samples.append(function_eval)

    distribution_over_classes = 1 / n_samples * np.sum(samples, axis=0)
    return(distribution_over_classes)


def expn_u_M_tilde(M_tilde, n_samples):
    """
    Return a sample estimate of a function of the r.v. u ~ N(0, 1)

    :param function: a function of u
    """

    def function(u, M_tilde):
        norm.pdf(u, loc=M)

    sum = 0
    for i in range(n_samples):
        sum += function(u, M_tilde)
    return sum / n_samples


def M(sigma, Y_tilde):
    """Q(M) where Y_tilde is the expectation of Y with respect to the posterior component over Y."""
    M_tilde = sigma @ Y_tilde
    # The approximate posterior component Q(M) is a product of normal distributions over the classes
    # But how do I translate this function into code? Must need to take an expectation wrt to it
    return M_tilde


def expn_M(sigma, m_tilde, n_samples):
    # Draw samples from the random variable, M ~ Q(M)
    # Use M to calculate f(M)
    # Take monte carlo estimate
    K = np.shape(m_tilde)[0]
    return None


train = np.load(read_path / 'train.npz')
test = np.load(read_path / 'test.npz')

for i, t_predi in enumerate(t_pred):
    if t_predi == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='b')
    elif t_predi == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], color='r')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.title("Thresholded class predictions using VB")
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
