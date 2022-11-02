"""Trying to work out how to cause errors in the gradient of the objective function"""
import jax
from jax import custom_jvp, grad
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from probit_jax.implicit.Laplace import objective_LA
from probit_jax.implicit.utilities import hessian_log_probit_likelihood

pm = jnp.array([0.78])
ys = jnp.array([0])

def f(theta):
        pp = theta[0]
        lp = theta[1]
        precision = -hessian_log_probit_likelihood(pm, ys, lp) + 1e-8
        jax.debug.print("precision={}", precision)
        lp = print_grad(lp)
        precision = print_grad(precision)
        return (
        # - B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters))
        # + 0.5 * posterior_mean.T @ grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)  # TODO minus before grad?
        # + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * jnp.sum(precision)
    )


def f_lp(lp):
        pp = jnp.sqrt(1./(2 * theta_0))
        precision = -hessian_log_probit_likelihood(pm, ys, lp) + 1e-8
        jax.debug.print("precision={}", precision)
        lp = print_grad(lp)
        precision = print_grad(precision)
        return (
        # - B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters))
        # + 0.5 * posterior_mean.T @ grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)  # TODO minus before grad?
        # + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * jnp.sum(precision)
        )

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


theta_0 = 0.01
theta = 0.01
cutpoints_0 = [-jnp.inf, 0., 1., jnp.inf]
theta = ((jnp.sqrt(1./(2 * theta_0))), [jnp.sqrt(theta), cutpoints_0])
print(grad(f)(theta))
# print(grad(f_lp)(theta[1]))