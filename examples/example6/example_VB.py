"""Multiclass Probit regression Variational Bayes example."""
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from scipy.stats import multivariate_normal
from probit.estimators import VBMultinomialOrderedGP
from probit.kernels import SEIso
import matplotlib.pyplot as plt

K = 3
D = 1
means = np.array([
    [1, 1],
    [2, 2],
    [3, 3]
])
sizes = np.array([80, 80, 80])
N_total = np.sum(sizes)
variance = 0.07
covar = variance * np.eye(2)
colors = ['b', 'r', 'g', 'k', 'c', 'm']

# Multiclass case
X = np.array([[]])
Xs = np.empty((N_total, 2))
lower = np.array([0, 80, 160, 240])
upper = np.array([80, 160, 240, 260])
for i in range(K):
    Xi = multivariate_normal.rvs(mean=means[i], cov=covar, size=sizes[i])
    Xs[lower[i]:upper[i], :] = Xi
    X = np.append(X, Xi[:, 0])

# # delta spike
# Xi = np.ones((sizes[3], 2))
# Xs[lower[3]:upper[3], :] = Xi
# X = np.append(X, Xi[:, 0])

print(np.shape(Xs))
t = np.empty(N_total, dtype=np.intc)
X0 = []
X1 = []
X2 = []
for n in range(N_total):
    X_ny = Xs[n, 1]
    X_n = Xs[n]
    if X_ny < 1.5:
        t[n] = 0
        X0.append(X_n)
    elif X_ny < 2.5:
        t[n] = 1
        X1.append(X_n)
    else:
        t[n] = 2
        X2.append(X_n)

X0 = np.array(X0)
X1 = np.array(X1)
X2 = np.array(X2)
plt.scatter(X0[:, 0], X0[:, 1], color=colors[0], label=r"$t={}$".format(0))
plt.scatter(X1[:, 0], X1[:, 1], color=colors[1], label=r"$t={}$".format(1))
plt.scatter(X2[:, 0], X2[:, 1], color=colors[2], label=r"$t={}$".format(2))
plt.legend()
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$y$", fontsize=16)
plt.show()

# Prepare data
Xt = np.c_[X, t]
np.random.shuffle(Xt)
X = Xt[:, :D]
t = Xt[:, -1]

# This is the general kernel for a GP prior for the multi-class problem
varphi = 10000.0
s = 1.0
sigma = 10e-6
tau = 10e-6
cutpoint = np.array([0.0, 1.5, 2.5, 5.0])
kernel = SEIso(varphi, s, sigma=sigma, tau=tau)
variation_bayes_classifier = VBMultinomialOrderedGP(cutpoint, X, t, kernel)
steps = 5
m_0 = np.zeros((N_total,))
m_tilde, Sigma_tilde, C_tilde, y_tilde, varphi_tilde = variation_bayes_classifier.estimate(m_0, steps)

print(varphi_tilde, 'varphi_tilde')

plt.scatter(X, m_tilde)
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$\tilde{m}$", fontsize=16)
plt.title("GP regression posterior mean m tilde, plotted against x")
plt.show()
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$\tilde{y}$", fontsize=16)
plt.title("Latent variable posterior mean y tilde, plotted against x")
plt.scatter(X, y_tilde)
plt.show()

N = 50
x = np.linspace(-0.1, 5.1, N)
X_new = x.reshape((N, D))
Z = variation_bayes_classifier.predict(Sigma_tilde, y_tilde, varphi_tilde, X_new, vectorised=True)
print(np.sum(Z, axis=1), 'sum')
#Z = np.moveaxis(Z, 2, 0)
for k in range(K):
    _ = plt.subplots(1, figsize=(6, 6))
    plt.plot(x, Z[:, k])
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$p(t={}|x, X, t)$".format(k), fontsize=16)
    plt.title("Contour plot - Ordered VB")
    plt.show()

fig, ax = plt.subplots(5, 10, sharex='col', sharey='row')
# axes are in a two-dimensional array, indexed by [row, col]
for i in range(5):
    for j in range(10):
        ax[i, j].bar(np.array([0, 1, 2], dtype=np.intc), Z[i*5 + j])
        ax[i, j].set_xticks(np.array([0, 1, 2], dtype=np.intc), minor=False)
plt.show()

plt.hist(X, bins=20)
plt.xlabel(r"$y$", fontsize=16)
plt.ylabel("counts")
plt.show()