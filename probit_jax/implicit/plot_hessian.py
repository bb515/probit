import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from probit_jax.implicit.utilities import hessian_log_probit_likelihood, log_probit_likelihood, _Z_tails, norm_cdf

# f=-1.2856698170478467
# y=2
# params=(0.31622777, [       -jnp.inf, -0.6850769 , -0.04564858, jnp.inf])

# H = jax.grad(log_probit_likelihood)(f, y, params)
# print(H)

fig, ax = plt.subplots()


z1 = 3.
z2 = jnp.linspace(3, 6, 50)
Z  = norm_cdf(z2) - norm_cdf(z1)
ax.plot(z2, _Z_tails(z1, z2))
ax.plot(z2, Z)
fig.savefig("Norm cdf sweep")