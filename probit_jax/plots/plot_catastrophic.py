import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from probit_jax.implicit.utilities import norm_pdf, norm_cdf, _safe_Z, hessian_log_probit_likelihood

def norm_cdf_jitter(z, jitter):
    return 0.5 * (1 + jax.lax.erf(z/jnp.sqrt(2))) * (1 - 2*jitter) + jitter


likelihood_parameters = (
    1., 
    [-jnp.inf, 0, 10, jnp.inf]
)

f = jnp.linspace(0, 10)
y=0

cutpoints_tplus1 = likelihood_parameters[1][y + 1]
cutpoints_t = likelihood_parameters[1][y]
noise_std = likelihood_parameters[0]
z2s = (cutpoints_tplus1 - f) / noise_std
z1s = (cutpoints_t - f) / noise_std

fig, ax = plt.subplots()

for jitter in (0., 1e-3):
    Z = (norm_pdf(z2s) - norm_pdf(z1s)) / (norm_cdf_jitter(z2s, jitter) - norm_cdf_jitter(z1s, jitter))
    ax.plot(f, jnp.clip(Z, -1e10, 1e10), label=f"jitter={jitter}")

Z, z1s, z2s = _safe_Z(f, y, likelihood_parameters)
print(jnp.min((Z)))
ax.twinx().plot(f, jnp.log(Z), label="Hmm")
Z = (norm_pdf(z2s) - norm_pdf(z1s)) / Z
ax.plot(f, Z, linestyle="--", label="Approximated Z")


ax.set_ylim((Z.min()-1, Z.max()+1))
fig.legend()
fig.savefig("probit_jax/plots/catastrophic.png")

print(
    hessian_log_probit_likelihood(10, y, likelihood_parameters)
)

# find linear approximation
