"""Experiment to try out the Numerical Recipes approach
Press, et al (2007) Numerical Recipes: The Art of Scientific Computing. 3rd Ed.
Section 6.2.2
https://github.com/cossio/TruncatedNormal.jl/blob/master/src/erf.jl"""

import jax.numpy as jnp
from jax import jit, grad
from jax.config import config
from probit_jax.implicit.utilities import grad_log_probit_likelihood

import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)

sqrt_2 = jnp.sqrt(2)
sqrt_2_over_pi = jnp.sqrt(2/jnp.pi)

erfcof = (-1.3026537197817094, 6.4196979235649026e-1,
    1.9476473204185836e-2,-9.561514786808631e-3,-9.46595344482036e-4,
    3.66839497852761e-4,4.2523324806907e-5,-2.0278578112534e-5,
    -1.624290004647e-6,1.303655835580e-6,1.5626441722e-8,-8.5238095915e-8,
    6.529054439e-9,5.059343495e-9,-9.91364156e-10,-2.27365122e-10,
    9.6467911e-11, 2.394038e-12,-6.886027e-12,8.94487e-13, 3.13092e-13,
    -1.12708e-13,3.81e-16,7.106e-15,-1.523e-15,-9.4e-17,1.21e-16,-2.8e-17)


def erf(x):
    return jnp.where(x >= 0, 1 - erfccheb(x), erfccheb(-x) - 1)

def erfc(x):
    return jnp.where(x >= 0, erfccheb(x), 2 - erfccheb(-x))

def erf_dif(x, y):
    #TODO: abs(x) > abs(y)
    return jnp.where((x < 0) & (0 <= y),
        erf(y) - erf(x),
        erfc(x) - erfc(y)
    )

def erfccheb(z):
    """Nonnegative z"""
    d = dd = 0.
    t = 2. / (2. + z)
    ty = 4. * t - 2.
    for c in erfcof[-1:0:-1]:
        d, dd = ty * d - dd + c, d
    return t * jnp.exp(-z**2 + 0.5 * (erfcof[0] + ty * d) - dd)

# function erfcx(x::Float64)
#     if x ≥ 0
#         return erfcxcheb(x)
#     else
#         return 2exp(x^2) - erfcxcheb(-x)
#     end
# end

def erfcx(x):
    return jnp.where(x >= 0, erfcxcheb(x), 2*jnp.exp(x**2) - erfcxcheb(-x)) 

def erfcxcheb(z):
    d = dd = 0
    t = 2 / (2 + z)
    ty = 4 * t - 2
    for c in erfcof[-1:0:-1]:
        d, dd = ty * d - dd + c, d

    return t * jnp.exp(0.5 * (erfcof[0] + ty * d) - dd)

def norm_cdf_dif(x, y):
    return erf_dif(x / sqrt_2, y / sqrt_2) / 2.


def tnmean(z1, z2):
    # handle a==b : return a
    #     if abs(a) > abs(b)
    #         return -tnmean(-b, -a)
    # if a is jnp.inf and b is jnp inf: return 0

    abs_gt = jnp.abs(z1) > jnp.abs(z2)
    mul = jnp.where(abs_gt, -1, 1)
    x = jnp.where(abs_gt, -z2, z1)
    y = jnp.where(abs_gt, -z1, z2)

    delta = (y - x) * (y + x) / 2

    # iszero(z) && return middle(a, b)

    z = jnp.exp(-delta) * erfcx(y/sqrt_2) - erfcx(x/sqrt_2)


    m = jnp.where((x <= 0) & (0 <= y),
        sqrt_2_over_pi * jnp.expm1(-delta) * jnp.exp(-x**2 / 2) / erf_dif(y/sqrt_2, x/sqrt_2),
        sqrt_2_over_pi * jnp.expm1(-delta) / z
    )

    return mul * jnp.clip(m, x, y)

def tnmom2(z1, z2):
    abs_gt = jnp.abs(z1) > jnp.abs(z2)

    x = jnp.where(abs_gt, -z2, z1)
    y = jnp.where(abs_gt, -z1, z2)
    

def tnvar(z1, z2):
    """Variance of a truncated standard normal distribution between [z1, z2]"""


# TODO: https://github.com/cossio/TruncatedNormal.jl/blob/master/src/tnmean.jl
# essentially just want to find this "mean" function


# print(norm_cdf_dif(jnp.array([5.]), jnp.array([7.])))


# print("---")
# print(grad_log_probit_likelihood(0, 1, [1, [-jnp.inf, 2, 2.01, jnp.inf]]))