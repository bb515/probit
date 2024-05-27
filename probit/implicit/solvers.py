from jax import jacobian, custom_vjp, vjp
from functools import partial
import jax.numpy as jnp
from jaxopt import FixedPointIteration


def fwd_solver(f, z_init, tolerance):
    """
    Using jaxopt fix-point iteration
    (https://jaxopt.github.io/stable/fixed_point.html), return the latent
    variables at the fix-point.
    """
    fpi = FixedPointIteration(fixed_point_fun=f, tol=tolerance)
    z_star, _ = fpi.run(z_init)
    return z_star


def newton_solver(f, z_init, tolerance):
    """
    Using Newton's method, return the latent
    variables at the fix-point.
    """
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jacobian(f_root)(z), f_root(z))
    return fwd_solver(g, z_init, tolerance)


@partial(custom_vjp, nondiff_argnums=(2, 3))
def fixed_point_layer(z_init, tolerance, solver, f, params):
    """
    Following the tutorial, Chapter 2 of
    http://implicit-layers-tutorial.org/implicit_functions/

    A wrapper function for the parameterized fixed point solver.

    :arg solver: Root finding numerical solver
    :arg params: Parameters for the non-linear set of equations that must
        be satisfied
        .. math::
            z = f(a, z)
        where :math:`a` are parameters and :math:`z` are the latent
        variables, and :math:`f` is a non-linear function.
    """
    return solver(lambda z: f(params, z), z_init=z_init, tolerance=tolerance)


def fixed_point_layer_fwd(z_init, tolerance, solver, f, params):
    z_star = fixed_point_layer(z_init, tolerance, solver, f, params)
    return z_star, (z_init, tolerance, params, z_star)


def fixed_point_layer_bwd(solver, f, res, z_star_bar):
    z_init, tolerance, params, z_star = res
    _, vjp_a = vjp(lambda params: f(params, z_star), params)
    _, vjp_z = vjp(lambda z: f(params, z), z_star)
    return (
        None,
        None,
        *vjp_a(
            solver(
                lambda u: vjp_z(u)[0] + z_star_bar, z_init=z_init, tolerance=tolerance
            )
        ),
    )
