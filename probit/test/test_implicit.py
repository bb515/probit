import jax.numpy as jnp
from jax import grad, jit
import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

from probit.utilities import h


def test_values_and_gradient_of_series_expansion():
    """Test the values and gradients of h(x) at typical and extreme values"""
    h_jit = jit(h)
    assert jnp.isnan(h_jit(0.0))
    assert h_jit(jnp.inf) == 0.0
    assert h_jit(-jnp.inf) == 0.0

    assert jnp.isclose(h_jit(1.0), -65 / 6, rtol=1.0001)

    assert jnp.isnan(grad(h_jit)(0.0))
    assert grad(h_jit)(jnp.inf) == 0.0
    assert grad(h_jit)(-jnp.inf) == 0.0
