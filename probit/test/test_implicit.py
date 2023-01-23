import pytest
import jax.numpy as jnp
from jax import grad, make_jaxpr, jit
import jax
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

from probit_jax.implicit.Laplace import objective_LA
from probit_jax.implicit.utilities import (grad_log_probit_likelihood, norm_cdf, _Z_tails, _Z_far_tails, _safe_Z,
 hessian_log_probit_likelihood, h)


def test_values_and_gradient_of_series_expansion():
    """Test the values and gradients of h(x) at typical and extreme values"""
    # TODO: should h(0) == -inf?
    h_jit = jit(h)
    assert jnp.isnan(h_jit(0.))
    assert h_jit(jnp.inf) == 0.
    assert h_jit(-jnp.inf) == 0.

    assert jnp.isclose(h_jit(1.), -65/6, rtol=1.0001)
    
    assert jnp.isnan(grad(h_jit)(0.))
    assert grad(h_jit)(jnp.inf) == 0.
    assert grad(h_jit)(-jnp.inf) == 0.


def test_values_and_gradient_of__Z_tails():
    """Test values and gradients of _Z_tails at extreme values.
    This function is not called for (z1, z2) \in [-3, 3], but these
    values should be tested anyway to ensure no nans are introduced."""
    Z_jit = jit(_Z_tails)
    assert Z_jit(0., 0.) == 0.
    assert Z_jit(jnp.inf, jnp.inf) == 0.
    assert Z_jit(-jnp.inf, -jnp.inf) == 0.
    assert Z_jit(-jnp.inf, jnp.inf) == 1.


@pytest.mark.parametrize("z1,z2", (
    (0., 0.),
    (jnp.inf, jnp.inf),
    (-jnp.inf, -jnp.inf),
    (-jnp.inf, jnp.inf),
    (-7., -3.),
    (1., 2.),
    (7., 8.),
))
def test_approximation_of_Z(z1, z2):
    """Test that the of the truncated CDF is accurate."""
    Z_true = norm_cdf(z2) - norm_cdf(z2)
    Z_approx = _safe_Z(z1, z2)
    assert jnp.isclose(Z_true, Z_approx, rtol=1, atol=1e-05)
