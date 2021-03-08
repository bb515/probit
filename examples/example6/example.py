"""Multiclass Probit regression Gibbs example."""
import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from scipy.stats import multivariate_normal
from probit.samplers import GibbsMultinomialOrderedGP
from probit.kernels import SEIso
import matplotlib.pyplot as plt
import pathlib

path = pathlib.Path()

K = 3
D = 1
sizes = np.array([80, 80, 80])
N_total = np.sum(sizes)
colors = ['b', 'r', 'g', 'k', 'c', 'm']

# means = np.array([
#     [1, 1],
#     [2, 2],
#     [3, 3]
# ])
#
# variance = 0.07
# covar = variance * np.eye(2)
#
# # Multiclass case
# X = np.array([[]])
# Xs = np.empty((N_total, 2))
# lower = np.array([0, 80, 160, 240])
# upper = np.array([80, 160, 240, 260])
# for i in range(K):
#     Xi = multivariate_normal.rvs(mean=means[i], cov=covar, size=sizes[i])
#     Xs[lower[i]:upper[i], :] = Xi
#     X = np.append(X, Xi[:, 0])
#
# # # delta spike
# # Xi = np.ones((sizes[3], 2))
# # Xs[lower[3]:upper[3], :] = Xi
# # X = np.append(X, Xi[:, 0])
#
# t = np.empty(N_total, dtype=np.intc)
# X0 = []
# X1 = []
# X2 = []
#
# for n in range(N_total):
#     X_ny = Xs[n, 1]
#     X_n = Xs[n]
#     if X_ny < 1.5:
#         t[n] = 0
#         X0.append(X_n)
#     elif X_ny < 2.5:
#         t[n] = 1
#         X1.append(X_n)
#     else:
#         t[n] = 2
#         X2.append(X_n)
#
# X0 = np.array(X0)
# X1 = np.array(X1)
# X2 = np.array(X2)
#
# # Data shift as a preprocessing step
# cutpoint_1 = np.mean([np.max(X1[:, 1]), np.min(X0[:, 1])])
#
# # Normalised cutpoint
# X0[:, 1] = np.subtract(X0[:, 1], cutpoint_1)
# X1[:, 1] = np.subtract(X1[:, 1], cutpoint_1)
# X2[:, 1] = np.subtract(X2[:, 1], cutpoint_1)
#
# # Prepare data
# Xt = np.c_[Xs, t]
#
# np.random.shuffle(Xt)
# X = Xt[:, :D]
# t = Xt[:, -1]
# y_true = Xt[:, 1]
#
# np.savez(path / "data.npz", X=X, t=t, Y=y_true, X0=X0,  X1=X1, X2=X2)

data = np.load(path / "data.npz")

X = data["X"]
y_true = data["Y"]
t = data["t"]
X0 = data["X0"]
X1 = data["X1"]
X2 = data["X2"]

plt.scatter(X0[:, 0], X0[:, 1], color=colors[0], label=r"$t={}$".format(0))
plt.scatter(X1[:, 0], X1[:, 1], color=colors[1], label=r"$t={}$".format(1))
plt.scatter(X2[:, 0], X2[:, 1], color=colors[2], label=r"$t={}$".format(2))
plt.legend()
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$y$", fontsize=16)
plt.show()

# This is the general kernel for a GP prior for the multi-class problem
varphi = 1.0
s = 1.0
sigma = 10e-6
tau = 10e-6
gamma_0 = np.array([np.NINF, 0.0, 1.0, np.inf])
kernel = SEIso(varphi, s, sigma=sigma, tau=tau)
gibbs_classifier = GibbsMultinomialOrderedGP(K, X, t, kernel)
steps_burn = 100
steps = 10000
m_0 = np.random.rand(N_total)  # shouldn't M_0 be (150, 3), not (50, 3)
y_0 = y_true.flatten()

# Burn in
m_samples, y_samples, gamma_samples = gibbs_classifier.sample_metropolis_within_gibbs(
    m_0, y_0, gamma_0, 1.0, steps_burn)
#m_samples, y_samples, gamma_samples = gibbs_classifier.sample(m_0, y_0, gamma_0, steps_burn)
m_0_burned = m_samples[-1]
y_0_burned = y_samples[-1]
gamma_0_burned = gamma_samples[-1]

# Sample
m_samples, y_samples, gamma_samples = gibbs_classifier.sample_metropolis_within_gibbs(
    m_0_burned, y_0_burned, gamma_0_burned, 0.2, steps)
#m_samples, y_samples, gamma_samples = gibbs_classifier.sample(m_0_burned, y_0_burned, gamma_0_burned, steps)
m_tilde = np.mean(m_samples, axis=0)
y_tilde = np.mean(y_samples, axis=0)
gamma_tilde = np.mean(gamma_samples, axis=0)
print(gamma_tilde)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
g_star = -1. * np.ones(3)
n0, g0, patches = ax[0].hist(gamma_samples[:, 1], 20, density="probability", histtype='stepfilled')
n1, g1, patches = ax[1].hist(gamma_samples[:, 2], 20, density="probability", histtype='stepfilled')
g_star[0] = g0[np.argmax(n0)]
g_star[1] = g1[np.argmax(n1)]
ax[0].axvline(g_star[0], color='k', label=r"Maximum likelihood $\gamma_1$")
ax[1].axvline(g_star[1], color='k', label=r"Maximum likelihood $\gamma_2$")
ax[0].set_xlabel(r"$\gamma_1$", fontsize=16)
ax[1].set_xlabel(r"$\gamma_2$", fontsize=16)
ax[0].legend()
ax[1].legend()
plt.title(r"$\gamma$ posterior samples")
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
m_star = -1. * np.ones(3)
n0, m00, patches = ax[0].hist(m_samples[:, 0], 20, density="probability", histtype='stepfilled')
n1, m01, patches = ax[1].hist(m_samples[:, 1], 20, density="probability", histtype='stepfilled')
n2, m20, patches = ax[2].hist(m_samples[:, 2], 20, density="probability", histtype='stepfilled')
m_star[0] = m00[np.argmax(n0)]
m_star[1] = m01[np.argmax(n1)]
m_star[2] = m20[np.argmax(n2)]
ax[0].axvline(m_star[0], color='k', label=r"Maximum $m_0$")
ax[1].axvline(m_star[1], color='k', label=r"Maximum $m_1$")
ax[2].axvline(m_star[2], color='k', label=r"Maximum $m_2$")
ax[0].set_xlabel(r"$m_0$", fontsize=16)
ax[1].set_xlabel(r"$m_1$", fontsize=16)
ax[2].set_xlabel(r"$m_2$", fontsize=16)
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.title(r"$m$ posterior samples")
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
y_star = -1. * np.ones(3)
n0, y00, patches = ax[0].hist(y_samples[:, 0], 20, density="probability", histtype='stepfilled')
n1, y01, patches = ax[1].hist(y_samples[:, 1], 20, density="probability", histtype='stepfilled')
n2, y20, patches = ax[2].hist(y_samples[:, 2], 20, density="probability", histtype='stepfilled')
y_star[0] = y00[np.argmax(n0)]
y_star[1] = y01[np.argmax(n1)]
y_star[2] = y20[np.argmax(n2)]
ax[0].axvline(y_star[0], color='k', label=r"Maximum $y_0$")
ax[1].axvline(y_star[1], color='k', label=r"Maximum $y_1$")
ax[2].axvline(y_star[2], color='k', label=r"Maximum $y_2$")
ax[0].set_xlabel(r"$y_0$", fontsize=16)
ax[1].set_xlabel(r"$y_1$", fontsize=16)
ax[2].set_xlabel(r"$y_2$", fontsize=16)
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.title(r"$y$ posterior samples")
plt.show()

plt.scatter(X[np.where(t == 0)], m_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(0))
plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(1))
plt.scatter(X[np.where(t == 2)], m_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(2))
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$\tilde{m}$", fontsize=16)
plt.title("GP regression posterior sample mean mbar, plotted against x")
plt.show()

plt.scatter(X[np.where(t == 0)], y_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(0))
plt.scatter(X[np.where(t == 1)], y_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(1))
plt.scatter(X[np.where(t == 2)], y_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(2))
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$\tilde{y}$", fontsize=16)
plt.title("Latent variable posterior sample mean ybar, plotted against x")
plt.show()

plt.plot(gamma_samples[:, 2], label="r$\gamma_2$")
plt.title(r"Mixing for $\gamma_2$")
plt.show()

N = 100
x = np.linspace(-0.1, 4.5, N)
X_new = x.reshape((N, D))
print(np.shape(gamma_samples), 'shape gamma')
Z = gibbs_classifier.predict(y_samples, gamma_samples, X_new, vectorised=True)
print(np.sum(Z, axis=1), 'sum')
#Z = np.moveaxis(Z, 2, 0)
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[-0.1, 4.5] and the data")
plt.stackplot(x, Z.T,
              labels=(
                  r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
              colors=(
                  colors[0], colors[1], colors[2])
              )
plt.legend()
val = 0.5  # this is the value where you want the data to appear on the y-axis.
plt.scatter(X[np.where(t == 0)], np.zeros_like(X[np.where(t == 0)]) + val, facecolors=colors[0], edgecolors='white')
plt.scatter(X[np.where(t == 1)], np.zeros_like(X[np.where(t == 1)]) + val, facecolors=colors[1], edgecolors='white')
plt.scatter(X[np.where(t == 2)], np.zeros_like(X[np.where(t == 2)]) + val, facecolors=colors[2], edgecolors='white')
plt.show()

# fig, ax = plt.subplots(5, 10, sharex='col', sharey='row')
# # axes are in a two-dimensional array, indexed by [row, col]
# for i in range(5):
#     for j in range(10):
#         ax[i, j].bar(np.array([0, 1, 2], dtype=np.intc), Z[i*5 + j])
#         ax[i, j].set_xticks(np.array([0, 1, 2], dtype=np.intc), minor=False)
# plt.show()

# plt.hist(X, bins=20)
# plt.xlabel(r"$y$", fontsize=16)
# plt.ylabel("counts")
# plt.show()

# for k in range(K):
#     _ = plt.subplots(1, figsize=(6, 6))
#     plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
#     plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
#     plt.scatter(X2[:, 0], X2[:, 1], color='g', label=r"$t=2$", zorder=10)
#     plt.contourf(x, y, Z[k], zorder=1)
#     plt.xlim(0, 2)
#     plt.ylim(0, 2)
#     plt.legend()
#     plt.xlabel(r"$x_1$", fontsize=16)
#     plt.ylabel(r"$x_2$", fontsize=16)
#     plt.title("Contour plot - Gibbs")
#     plt.show()
