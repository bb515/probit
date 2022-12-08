import jax
from functools import partial
from jax import lax
import jax.numpy as jnp


def fwd_solver(f, z_init, tolerance):
    """
    Using fix point iteration, return the latent variables at the fix
    point.
    TODO isn't z_init always zero, can it be removed?
    """
    def cond_fun(carry):
        z_prev, z = carry
        return (jnp.linalg.norm(z_prev - z) > tolerance)  # TODO: This is a very unsafe implementation, can lead to infinite while loops!

    def body_fun(carry):
        _, z = carry
        return z, f(z)

    init_carry = (z_init, f(z_init))
    _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star


def newton_solver(f, z_init, tolerance):
    """
    Using Newton's method, return the latent variables at the fix point.
    """
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return fwd_solver(g, z_init, tolerance)


def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tolerance=1e-5, beta=1.0):
    """
    Using Anderson acceleration, return the latent  variables at the fix
    point.
    """
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate(
        [jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate(
        [jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    def step(n, k, X, F):
        G = F[:n] - X[:n]
        GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
        H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
                    [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
        alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]

        xk = beta * jnp.dot(alpha, F[:n])\
            + (1-beta) * jnp.dot(alpha, X[:n])
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        return X, F

    # unroll the first m steps
    for k in range(2, m):
        X, F = step(k, k, X, F)
        res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
        if res < tolerance or k + 1 >= max_iter:
            return X[k], k

    # run the remaining steps in a lax.while_loop
    def body_fun(carry):
        k, X, F = carry
        X, F = step(m, k, X, F)
        return k + 1, X, F

    def cond_fun(carry):
        k, X, F = carry
        kmod = (k - 1) % m
        res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
        return (k < max_iter) & (res >= tolerance)

    k, X, F = lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
    return X[(k - 1) % m]


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def fixed_point_layer(z_init, tolerance, solver, f, params):
    """
    Following the tutorial, Chapter 2 of
    http://implicit-layers-tutorial.org/implicit_functions/

    A wrapper function for the parameterized fixed point sovler.

    :arg solver: Root finding numerical solver
    :arg params: Parameters for the non-linear set of equations that must
        be satisfied
        .. math::
            z = f(a, z)
        where :math:`a` are parameters and :math:`z` are the latent
        variables, and :math:`f` is a non-linear function.
    """ 
    z_star = solver(
        lambda z: f(params, z), z_init=z_init, tolerance=tolerance)
    return z_star


def fixed_point_layer_fwd(z_init, tolerance, solver, f, params):
    z_star = fixed_point_layer(z_init, tolerance, solver, f, params)
    return z_star, (z_init, tolerance, params, z_star)


def fixed_point_layer_bwd(solver, f, res, z_star_bar):
    z_init, tolerance, params, z_star = res
    _, vjp_a = jax.vjp(lambda params: f(params, z_star), params)
    _, vjp_z = jax.vjp(lambda z: f(params, z), z_star)
    return (None, None, 
        *vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar, z_init=z_init, tolerance=tolerance))
        )


# @partial(jax.custom_vjp, nondiff_argnums=(0, 1))
# def fixed_point_layer(solver, f, params, x):
#   z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
#   return z_star

# def fixed_point_layer_fwd(solver, f, params, x):
#   z_star = fixed_point_layer(solver, f, params, x)
#   return z_star, (params, x, z_star)

# def fixed_point_layer_bwd(solver, f, res, z_star_bar):
#   params, x, z_star = res
#   _, vjp_a = jax.vjp(lambda params, x: f(params, x, z_star), params, x)
#   _, vjp_z = jax.vjp(lambda z: f(params, x, z), z_star)
#   return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar,
#                       z_init=jnp.zeros_like(z_star)))