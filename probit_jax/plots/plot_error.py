"""Plots the error between two subsequent approximations
for the grad log probit function, to find appropriate thresholds"""


import matplotlib.pyplot as plt
import jax.numpy as jnp

from probit_jax.implicit.utilities import grad_log_probit_likelihood, norm_cdf, norm_pdf, _Z_tails

likelihood_parameters = (
    1, 
    [-jnp.inf, 0, 10, jnp.inf]
)

# f = jnp.linspace(0, 20, num=500)
f = jnp.linspace(0, 20, num=500)
y=0

cutpoints_tplus1 = likelihood_parameters[1][y + 1]
cutpoints_t = likelihood_parameters[1][y]
noise_std = likelihood_parameters[0]
z2s = (cutpoints_tplus1 - f) / noise_std
z1s = (cutpoints_t - f) / noise_std

fig, ax = plt.subplots()


Z_1 = (norm_pdf(z2s) - norm_pdf(z1s)) / (norm_cdf(z2s) - norm_cdf(z1s))

Z_2 = _Z_tails(z1s, z2s)
Z_2 = (norm_pdf(z2s) - norm_pdf(z1s)) / Z_2


E_0 = grad_log_probit_likelihood(f, y, likelihood_parameters)
E_1 = grad_log_probit_likelihood(f, y, likelihood_parameters, 0.)
E_2 = grad_log_probit_likelihood(f, y, likelihood_parameters, 0., 0.)
E_3 = grad_log_probit_likelihood(f, y, likelihood_parameters, 0., 0., 0.)

# ax.plot(f, jnp.clip(Z_1, -1e10, 1e10))
# ax.plot(f, jnp.clip(Z_2, -1e10, 1e10))
e = jnp.abs(E_3 - E_2)
ax.plot(f, jnp.clip(e, 0, 10))

ax.set_xlabel('Posterior mean, f')
ax.set_ylabel('Absolute error $|Z_{computed}-Z_{approximated}|$')
# ax.set_ylim(1e-4, 0.2)
# ax.set_yscale("log")
fig.savefig("probit_jax/plots/error_optimisation.png")