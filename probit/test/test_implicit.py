import pytest
import jax.numpy as jnp
from jax import grad, make_jaxpr, jit
import jax
from jax.config import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

from probit.implicit.Laplace import objective_LA
from probit.utilities import (grad_log_probit_likelihood, norm_cdf,
    _Z_tails, _Z_far_tails, _safe_Z,
    hessian_log_probit_likelihood, h)


def test_values_and_gradient_of_series_expansion():
    """Test the values and gradients of h(x) at typical and extreme values"""
    h_jit = jit(h)
    assert jnp.isnan(h_jit(0.))
    assert h_jit(jnp.inf) == 0.
    assert h_jit(-jnp.inf) == 0.

    assert jnp.isclose(h_jit(1.), -65/6, rtol=1.0001)
    
    assert jnp.isnan(grad(h_jit)(0.))
    assert grad(h_jit)(jnp.inf) == 0.
    assert grad(h_jit)(-jnp.inf) == 0.
