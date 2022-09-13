"""Tests for the kernels module."""
from ..kernels import (
    Kernel, SEARDMultinomial, SEIsoBinomial)
import pytest
import numpy as np
from scipy.spatial import distance_matrix, distance


@pytest.fixture(scope="module")
def SEARDmultinomial_kernel():
    """Create an instance of the SEARDmultinomial kernel."""
    # Let K = 3, D = 2
    varphi0 = np.array([0.4, 0.4])
    varphi1 = np.array([0.4, 0.5])
    varphi2 = np.array([0.4, 0.6])
    varphi = np.array([varphi0, varphi1, varphi2])
    s = 2.0
    SEARDmultinomial_kernel = SEARDMultinomial(varphi, s)
    return SEARDmultinomial_kernel


@pytest.fixture(scope="module")
def SEISObinomial_kernel():
    """Create an instance of the SEISObinomial kernel."""
    s = 0.2
    varphi = 0.1
    SEISObinomial_kernel = SEIsoBinomial(varphi, s)
    return SEISObinomial_kernel


class TestKernel:
    """ABC class tests."""

    def test_not_implemented_error(self):
        """Ensure the ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            Kernel()


class TestSEISObinomial:
    """SEISObinomial tests."""

    def test_kernel(self, SEISObinomial_kernel):
        """Regression test for the basic kernel."""
        x1 = np.array([0, 0])
        x2 = np.array([1.0, 1.0])
        expected = 0.2 * np.exp(-0.1 * np.power(distance.euclidean(x1, x2), 2))
        actual = SEISObinomial_kernel.kernel(x1, x2)
        assert expected == actual

    def test_kernel_matrix(self, SEISObinomial_kernel):
        """Regression test for kernel matrix function."""
        X1 = np.array([[0, 0], [0, 1], [0, 2]])
        X2 = X1.copy()
        expected = 0.2 * np.exp(-0.1 * np.power(distance_matrix(X1, X1), 2))
        actual = SEISObinomial_kernel.kernel_matrix(X1, X2)
        assert np.allclose(expected, actual)


class TestSEARDmultinomial:
    """SEARDmultinomial tests."""

    def test_kernel(self, SEARDmultinomial_kernel):
        """Regression test for basic kernel method."""
        x1 = np.array([1, 0])
        x2 = np.array([0, 1])
        D = len(x1)
        s = 2.0
        actual_k0 = SEARDmultinomial_kernel.kernel(0, x1, x2)
        actual_k1 = SEARDmultinomial_kernel.kernel(1, x1, x2)
        actual_k2 = SEARDmultinomial_kernel.kernel(2, x1, x2)
        varphi0 = np.array([0.4, 0.4])
        varphi1 = np.array([0.4, 0.5])
        varphi2 = np.array([0.4, 0.6])
        varphi = np.array([varphi0, varphi1, varphi2])
        k0 = s * np.exp(-1. * np.sum([(varphi[0, d] * np.power((x1[d] - x2[d]), 2)) for d in range(D)]))
        k1 = s * np.exp(-1. * np.sum([(varphi[1, d] * np.power((x1[d] - x2[d]), 2)) for d in range(D)]))
        k2 = s * np.exp(-1. * np.sum([(varphi[2, d] * np.power((x1[d] - x2[d]), 2)) for d in range(D)]))
        expected = np.array([k0, k1, k2])
        actual = np.array([actual_k0, actual_k1, actual_k2])
        assert np.allclose(expected, actual)


    def test_kernel_matrix(self, SEARDmultinomial_kernel):
        x1 = np.array([1, 0])
        x2 = np.array([0, 1])
        x3 = np.array([0.9, 0.1])
        x4 = np.array([0.1, 0.9])
        X = np.array([x1, x2, x3, x4])
        Cs = SEARDmultinomial_kernel.kernel_matrix(X, X)
        D = 2
        N = 4
        s = 2.0
        varphi0 = np.array([0.4, 0.4])
        varphi1 = np.array([0.4, 0.5])
        varphi2 = np.array([0.4, 0.6])
        Cs_0_expected = -1. * np.ones((N, N))
        Cs_1_expected = -1. * np.ones((N, N))
        Cs_2_expected = -1. * np.ones((N, N))
        for i in range(N):
            for j in range(N):
                Cs_0_expected[i, j] = s * np.exp(-1. * sum(
                    [varphi0[d] * np.power((X[i, d] - X[j, d]), 2) for d in range(D)]))
                Cs_1_expected[i, j] = s * np.exp(-1. * sum(
                    [varphi1[d] * np.power((X[i, d] - X[j, d]), 2) for d in range(D)]))
                Cs_2_expected[i, j] = s * np.exp(-1. * sum(
                    [varphi2[d] * np.power((X[i, d] - X[j, d]), 2) for d in range(D)]))
        shape = np.shape(Cs)
        assert shape == (3, 4, 4)  # since N=4 and K=3
        assert np.allclose(Cs_0_expected, Cs[0])
        assert np.allclose(Cs_1_expected, Cs[1])
        assert np.allclose(Cs_2_expected, Cs[2])



