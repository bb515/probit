import matplotlib.pyplot as plt
import jax
from jax import grad, vmap, jit
import jax.numpy as jnp
from probit_jax.implicit.utilities import _safe_Z, hessian_log_probit_likelihood, norm_cdf, norm_pdf

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

# TODO: Plot real Z function phi - phi / Phi - Phi, include linear approximation.

BOUNDS = (5., 8., 10.)
BOUNDS = (1., 1., 1.)
XLIM = (-7, 7)

colours = {"upper_bound": "#A8F9FF", "upper_bound2": "#9AE5E6"}

likelihood_parameters = (
    0.1, 
    [-jnp.inf, 0., 1., jnp.inf]
)
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))
f = jnp.linspace(-1.01, 3., num=500)

for y, ax in zip(range(3), axs):
    cutpoints_tplus1 = likelihood_parameters[1][y + 1]
    cutpoints_t = likelihood_parameters[1][y]

    grad_hess = lambda f, y, params: grad(hessian_log_probit_likelihood, argnums=2)(f, y, params, *BOUNDS)
    foo = vmap(grad_hess, in_axes=(0, None, None), out_axes=(0))
    dV = foo(f, y, likelihood_parameters)
    # print("dV=", dV)
    Z, z1s, z2s = _safe_Z(f, y, likelihood_parameters)
    # Z_true = norm_cdf(z2s) - norm_cdf(z1s)
    # line_Z, = ax.plot(f, Z)

    V = hessian_log_probit_likelihood(f, y, likelihood_parameters, *BOUNDS)

    line_V, = ax.plot(f, V)
    line_gradV, = ax.plot(f, dV)

    # ax_error = ax.twinx()
    # line_err, = ax_error.plot(f, jnp.abs(E_truncnorm/0.1-E), color="red", linestyle="--")
    # ax_error.set_ylabel("Absolute approximation error")

    reg_ub = ax.fill_between(f, 0, 1, where=(z1s>BOUNDS[0])&(z1s<BOUNDS[1]) | 
                                            (z2s<-BOUNDS[0])&(z2s>-BOUNDS[1]), 
        alpha=0.4, transform=ax.get_xaxis_transform())

    reg_ub2 = ax.fill_between(f, 0, 1, where=(z1s>BOUNDS[1])&(z1s<BOUNDS[2]) | 
                                            (z2s<-BOUNDS[1])&(z2s>-BOUNDS[2]), 
        alpha=0.4, transform=ax.get_xaxis_transform())

    reg_ub3 = ax.fill_between(f, 0, 1, where=(z1s>BOUNDS[2]) | 
                                            (z2s<-BOUNDS[2]), 
        alpha=0.4, transform=ax.get_xaxis_transform())

    ax.set_ylabel('$Z$')
    ax.set_title(f'bin={y}')

axs[-1].set_xlabel('Posterior mean, $f$')
labels = (
    (line_V, "Hessian"),
    (line_gradV, "Autodiff hessian"),
    # (line_err, "Abs error"),
    (reg_ub, "'Tail' estimation region"),
    (reg_ub2, "'Far tail' estimation region"),
    (reg_ub3, "Linear region"),
)


fig.legend(*zip(*labels))
fig.savefig("probit_jax/plots/hessianlogprobit.png")
