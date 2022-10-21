import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from probit_jax.implicit.utilities import _safe_Z, norm_cdf

from jax.config import config


# f=-1.2856698170478467
# y=2
# params=(0.31622777, [       -jnp.inf, -0.6850769 , -0.04564858, jnp.inf])

# H = jax.grad(log_probit_likelihood)(f, y, params)
# print(H)

upper_bound = 3
upper_bound2 = 6
XLIM = (-7, 7)

colours = {"upper_bound": "#A8F9FF", "upper_bound2": "#9AE5E6"}
"""
fig, axs = plt.subplots(nrows=4, figsize=(10, 20), sharex=False)

for z1, ax in zip((-7, -4, 0, 6.5), axs):
    z2 = jnp.linspace(z1, XLIM[1], 50)

    # plot regions of estimation
    # if -3 > z2 > -6
    if -upper_bound2 < z2.min() < -upper_bound:
        ax.axvspan(-upper_bound, z2.min(), color=colours["upper_bound"])
    elif z2.min() < -upper_bound2:
        ax.axvspan(-upper_bound, -upper_bound2, color=colours["upper_bound"])
    # if 3 < z1 < 6
    if upper_bound < z1 < upper_bound2:
        ax.axvspan(z2.min(), z2.max(), color=colours["upper_bound"])
            
    # if z2 < -6
    if z2.min() < -upper_bound2:
        ax.axvspan(-upper_bound2, z2.min(), color=colours["upper_bound2"])
    # if z1 > 6
    if z1 > upper_bound2:
        ax.axvspan(z2.min(), z2.max(), color=colours["upper_bound2"])

    # clear all values below z1

    Z_true = norm_cdf(z2) - norm_cdf(z1)
    Z_approx = _safe_Z(z1, z2)

    ax.plot(z2, Z_true, label="True CDF diff")
    ax.plot(z2, Z_approx, linestyle="--", label="Approximated CDF diff")

    ax_error = ax.twinx()
    ax_error.plot(z2, jnp.abs(Z_true - Z_approx), color="red", label="Approximation error")
    ax_error.set_ylabel("Absolute approximation error")

    ax.set_title(f"z1={z1}")
    ax.set_xlabel("z2")
    ax.set_ylabel("Z")
    # ax.set_xlim(*XLIM)
    ax.legend()
"""
likelihood_parameters = (
    1., 
    [-jnp.inf, -1., 1., jnp.inf]
)
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(6, 12))
f = jnp.linspace(-6, 6, num=500)
for y, ax in zip(range(3), axs):
    cutpoints_tplus1 = likelihood_parameters[1][y + 1]
    cutpoints_t = likelihood_parameters[1][y]
    Z, z1s, z2s = _safe_Z(f, y, likelihood_parameters)
    Z_true = norm_cdf(z2s) - norm_cdf(z1s)

    line_Z, = ax.plot(f, Z)

    ax_error = ax.twinx()
    line_err, = ax_error.plot(f, jnp.abs(Z - Z_true), color="red", linestyle="--")
    ax_error.set_ylabel("Absolute approximation error")

    reg_ub = ax.fill_between(f, 0, 1, where=(z1s>3)&(z1s<6)|(z2s<-3)&(z2s>-6), 
        alpha=0.4, transform=ax.get_xaxis_transform())

    reg_ub2 = ax.fill_between(f, 0, 1, where=(z1s>=6)|(z2s<=-6), 
        alpha=0.4, transform=ax.get_xaxis_transform())

    ax.set_ylabel('$Z$')
    ax.set_title(f'bin={y}')

axs[-1].set_xlabel('Posterior mean, $f$')
fig.legend(
    (line_Z, line_err, reg_ub, reg_ub2), 
    ("Truncated norm CDF", "Approximation error","'Tail' estimation region", "'Far tail' estimation region")
)
fig.savefig("norm_cdf.pdf")