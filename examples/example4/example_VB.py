"""Variational approximation."""
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal, expon
from utilities import (
    sample_U, function_u3, function_u2, sample_varphi)
from kernels import binary
import matplotlib.pyplot as plt

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



