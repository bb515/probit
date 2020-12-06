"""Variational approximation."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal, expon
from utilities import (
    sample_U, function_u3, function_u2, sample_varphi)
from kernels import binary
import matplotlib.pyplot as plt


def w_binary_unnormalised(psi_tilde, X, kernel, m_tilde):
    # """Return the w value of the sample for a m_tilde. m_tilde is (N, ) in the binary case. Won't need this unless
    # treating varphi as a r.v. with psi as a hyperparameter
    #
    # :arg psi_tilde: Posterior mean estimate of psi
    # :arg X: The datum.
    # :arg kernel: The kernel class object.
    # """
    # N = np.shape(X)[0]
    # # Draw from varphi
    # varphi_sample = sample_varphi(psi_tilde)
    # kernel.varphi = varphi_sample
    # C = kernel.kernel_matrix(X)
    # mean = np.zeros(N)
    # normal_pdf = multivariate_normal.pdf(m_tilde, mean=mean, cov=C)
    # return normal_pdf
    return None


def varphi_tilde(varphi_samples):
    """Return the posterior mean estimate of varphi via importance sampling. Won't need unless treating varphi as
    a r.v."""
    # sum = 0
    # n_samples = np.shape(varphi_samples)[0]
    # for i in range(n_samples):
    #     varphi_sample = varphi_samples[i]
    #     sum += varphi_sample * w(varphi_sample)
    # return sum
    return None


def m_tilde(C, y_tilde):
    """Return the posterior mean estimate of M."""
    N = np.shape(C)[0]
    I = np.eye(N)
    sigma = np.linalg.inv(I + C)
    return C @ sigma @ y_tilde


def y_tilde(m_tilde):
    """Calculate y_tilde elements as defined on page 9 of the paper."""
    t_n = np.sign(m_tilde)
    y_tilde = m_tilde + np.divide(
        np.multiply(t_n, norm.pdf(m_tilde, loc=0, scale=1)), norm.cdf(np.multiply(t_n, m_tilde)))
    return y_tilde


def variational_predictive_distribution(x_new, C, kernel):
    """Return the prodictive distribution (a single probability that t=1)."""
    N = np.shape(C)[0]
    sigma = np.linalg.inv(np.eye(N) + C)
    C_new = kernel.kernel_vector()
    c_new = kernel.kernel(x_new, x_new)
    m_tilde_new = y_tilde.T @ sigma @ C_new
    var_tilde_new = c_new - C_new.T @ sigma @ C_new
    # predictive distribution
    expectation_p


def get_log_likelihood(kernel, X_train, t_train, X_test, t_test, n_iters):
    """Get the log likelihood of VB approximation given the hyperparameters."""
    C = kernel.kernel_matrix(X_train)
    N = np.shape(X_train)[0]
    # Initiate y_tilde at 0
    y_tilde = np.zeros(N)
    for i in n_iters:
        # update m_tilde
        m_tilde_ = m_tilde(C, y_tilde)
        # update y_tilde
        y_tilde_ = y_tilde(m_tilde)



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

varphi = np.exp(log_varphi[0])
s = np.exp(log_s[0])



Z = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        s = np.exp(log_s[i])
        varphi = np.exp(log_varphi[j])
        kernel = binary(varphi=varphi, s=s)
        Z = get_log_likelihood(kernel, X_train, t_train, X_test, t_test, 2)
        print(Z)
fig, axs = plt.subplots(1, figsize=(6, 6))
plt.contourf(log_ss, log_varphis, Z, zorder=1)



