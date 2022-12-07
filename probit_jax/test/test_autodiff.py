"""Test the autodifferentiation of grad_log_probit_likelihood (glpl) with
respect to hyperparameters"""

import pytest
import jax.numpy as jnp
from jax import grad, make_jaxpr, jit
from jax.tree_util import tree_flatten
import jax
from jax.config import config

config.update("jax_enable_x64", True)

from probit_jax.implicit.Laplace import objective_LA
from probit_jax.implicit.utilities import (grad_log_probit_likelihood, norm_cdf, _Z_tails, _Z_far_tails, _safe_Z,
 hessian_log_probit_likelihood, h)


glpl = lambda f, y, lp: grad_log_probit_likelihood(f, y, lp, single_precision=False)

@pytest.mark.parametrize(
    "f,y,lp",
    [
        (0.5, 1, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
        (100., 1, (0.1, [-jnp.inf, 0., 1., jnp.inf]))
    ]
)
class TestGlplAutodiff:
    def test_glpldiff_wrt_likelihood_parameters(self, f, y, lp):
        g = grad(glpl, argnums=2,)(f, y, lp)
        print(g)
        # noise hyperparam
        assert not jnp.isnan(g[0])
        # cutpoint hyperparams
        assert g[1][0] == g[1][3] == 0
        assert not (jnp.isnan(g[1][1]) or jnp.isnan(g[1][2]))

