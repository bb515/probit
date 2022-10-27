"""Compare the truncnorm implementation with the implicit.utilities
implementation"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from probit_jax.implicit.utilities import grad_log_probit_likelihood, hessian_log_probit_likelihood
from probit_jax.implicit.truncnorm import tnmean

f = jnp.linspace(4, 5, num=200)
fig, ax = plt.subplots()
sigma = 0.1
dZ1 = grad_log_probit_likelihood(f, 1, [sigma, [-jnp.inf, 0, 1, jnp.inf]])
Zhessi = hessian_log_probit_likelihood(f, 1, [sigma, [-jnp.inf, 0, 1, jnp.inf]])
# dZ2 = grad_log_probit_likelihood(f, 1, [sigma, [-jnp.inf, 0, 5, jnp.inf]])
ax.plot(f, dZ1, label="gradlog")
# ax.plot(f, Zhessi, label="hessian")
# ax.plot(f, dZ2)
# ax.plot(f, -f / sigma ** 2)

z1 = -f / sigma
z2 = (-f + 1) / sigma
dZtn = tnmean(z1, z2) / sigma
# ax.plot(f, dZtn, label="tnmean", alpha=0.5)

#linear region
ax.plot(f, -z2 / sigma, label="linear")

# xs = jnp.array([-1, -0.5])
# ys = grad_log_probit_likelihood(xs, 1, [sigma, [-jnp.inf, 0, 0.5, jnp.inf]])
# print(ys)
fig.legend()
fig.savefig("probit_jax/plots/temp_grad.png")
