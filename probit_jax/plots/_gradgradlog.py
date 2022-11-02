"""Trying to work out how to cause errors in the gradient of the objective function"""
import jax
from jax import custom_jvp, custom_vjp, grad, jit, value_and_grad
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from probit_jax.implicit.Laplace import objective_LA
from probit_jax.implicit.utilities import (hessian_log_probit_likelihood, 
grad_log_probit_likelihood, _safe_Z, norm_cdf)

f = jnp.array([0.78])
y = jnp.array([1])

@jax.custom_vjp
def print_grad(x):
    return x

def print_grad_fwd(x):
    jax.debug.print("grad1: {}", x)
    return x, None

def print_grad_bwd(_, x_grad):
    jax.debug.print("theta_grad: {}", x_grad)
    return (x_grad,)

print_grad.defvjp(print_grad_fwd, print_grad_bwd)

BOUNDS = (1., 1., 1.)
BOUNDS = (0., 0., 0.)




cutpoints_0 = [-jnp.inf, 0., 1., jnp.inf]

lp = [0.1, cutpoints_0]
f = 1
y = 0



# def too(x):
#     x = jnp.where()
#     return  (1. / x) * x

# print(jit(too)(0.))

# def func(x):
#   y = jnp.where(x == 0.0, 1.0, x)
#   return jnp.where(x == 0.0, 1.0, jnp.sin(y) / y)

# print(grad(func)(0.))

#---
def foo(sigma):
    b = jnp.inf
    _b = jnp.where(b == jnp.inf, 0., b)
    z = jnp.where(b == jnp.inf, jnp.inf, (-_b) / sigma)
    jax.debug.print("z={}", z)
    
    return norm_cdf(z)

#---

# print("gradfoo =", value_and_grad(foo)(0.1))

print("safe Z=", _safe_Z(f, y, lp, *BOUNDS))
print(grad(_safe_Z, argnums=2)(f, 1, lp, *BOUNDS))