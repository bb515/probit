from utilities import (
    matrix_of_differences, function_u1, function_u2, function_u3, function_u1_alt, sample_U, sample_varphi,
    samples_varphi)
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

def plot_vague_prior():
    """
    Scipy stats module takes two parameters,a and scale. From the form of the posterior over the psi (which comes
    from the fact that exponential is a conjugate prior to gamma, and the posterior takes the form of a gamma),
    we know that sigma as written in the paper is a, and tau as written in the paper is a rate parameter = 1./scale
    """
    # as written in the paper
    # sigma is known as k on wikipedia, and is a shape parameter
    sigmas = 10e-3 * np.arange(1, 4)
    # tau is known as beta on wikipedia, and is a rate parameter
    taus = 10e-3 * np.arange(1, 4)
    x = np.linspace(0, 300, 1001)
    for i in range(np.shape(sigmas)[0]):
        y = gamma.pdf(x, a=sigmas[i], scale=1./taus[i])
        plt.plot(x, y, label='sigma = {}, tau = {}'.format(sigmas[i], taus[i]))
    plt.title('The priors with the high sigma and tau are more concentrated around 0 - informative prior')
    plt.ylim((0, 3e-2))
    plt.xlim((0,2))
    plt.legend()
    plt.show()
    for i in range(np.shape(sigmas)[0]):
        y = gamma.pdf(x, a=sigmas[i], scale=1./taus[i])
        plt.plot(x, y, label='sigma = {}, tau = {}'.format(sigmas[i], taus[i]))
    plt.title('The priors with the lower sigma and tau have a much thicker tail- vague prior')
    plt.ylim((0, 3e-5))
    plt.xlim((100, 300))
    plt.legend()
    plt.show()


def test_sample_varphi():
    """Test sampling varphi from the prior."""
    # Draw hyperparameters from prior with hyper-hyperparameter
    M = 2
    K = 3
    # Initiate varphi posterior estimates as all ones
    varphi_tilde = 0.6*np.ones((K, M))
    # Uninformative priors
    sigma_k = np.zeros((K, M))
    tau_k = np.zeros((K, M))
    psi_tilde = (np.ones((K, M)) + sigma_k) / (tau_k + varphi_tilde)
    samples = []
    n_samples = 1000
    for i in range(n_samples):
        samples.append(sample_varphi(psi_tilde))
    samples = np.array(samples)
    varphi_tilde = np.sum(samples, axis=0) / n_samples
    # rtol = 0.1 since sample variance is Var(varphi) / root(n) = (1/ psi_tilde^2) / root(1000) = 0.0114, so std = 0.1
    assert np.allclose(varphi_tilde, 1./psi_tilde, rtol=0.1)


def test_samples_varphi():
    """Test sampling varphi from the prior with tensor version."""
    M = 2
    K = 3
    # Initiate varphi posterior estimates as all ones
    varphi_tilde = 0.6 * np.ones((K, M))
    # Uninformative priors
    sigma_k = np.zeros((K, M))
    tau_k = np.zeros((K, M))
    psi_tilde = (np.ones((K, M)) + sigma_k) / (tau_k + varphi_tilde)
    samples = []
    n_samples = 1000
    samples = samples_varphi(psi_tilde, n_samples)
    varphi_tilde = np.sum(samples, axis=0) / n_samples
    # rtol = 0.1 since sample variance is Var(varphi) / root(n) = (1/ psi_tilde^2) / root(1000) = 0.0114, so std = 0.1
    assert np.allclose(varphi_tilde, 1. / psi_tilde, rtol=0.1)

def test_matrix_of_differences():
    """Test the matrix of difference function produces expected (and not transposed) result."""
    m_n = np.array([-1, 0, 1])

    expected_MOD = np.array([
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]
    ])
    actual_MOD = matrix_of_differences(m_n)
    assert np.allclose(expected_MOD, actual_MOD)
    t_n = np.argmax(m_n)
    actual_vector_difference = actual_MOD[:, t_n]
    expected_vector_difference = np.array([-2, -1, 0])
    assert np.all(actual_vector_difference, expected_vector_difference)


def test_function_u1_sum_to_one():
    K = 3
    m_n = np.array([-1, 0, 1])
    difference = matrix_of_differences(m_n)
    U = sample_U(K)
    f1 = function_u1(difference, U)
    assert np.close(np.sum(f1), 1.0)


def test_functions_sum_to_one():
    K = 3
    m_n = np.array([-1, 0, 1])
    difference = matrix_of_differences(m_n)
    U = sample_U(K)
    t_n = np.argmax(m_n)
    f1 = function_u1_alt(difference, U, np.argmax(t_n))
    vector_difference = difference[:, t_n]
    f2 = function_u2(difference, vector_difference, U, t_n, K)
    f3 = function_u3(difference, vector_difference, U, t_n, K)
    assert np.close(np.sum(f1), 1.0)
    assert np.close(np.sum(f2), 1.0)
    assert np.close(np.sum(f3), 1.0)

test_samples_varphi()
