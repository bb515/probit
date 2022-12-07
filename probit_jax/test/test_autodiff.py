"""Test the autodifferentiation of grad_log_probit_likelihood (glpl) with
respect to hyperparameters"""

import pytest
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from jax import grad, make_jaxpr, jit, vmap
from jax.tree_util import tree_flatten
import jax
from jax.config import config
from mlkernels import EQ

config.update("jax_enable_x64", True)

from probit_jax.implicit.Laplace import objective_LA, f_LA
from probit_jax.implicit.utilities import (grad_log_probit_likelihood, norm_cdf, _Z_tails, _Z_far_tails, _safe_Z,
 hessian_log_probit_likelihood, h)
from probit_jax.solvers import fixed_point_layer, newton_solver


glpl = lambda f, y, lp: grad_log_probit_likelihood(f, y, lp, single_precision=False)
hlpl = lambda f, y, lp: hessian_log_probit_likelihood(f, y, lp, single_precision=False)

@pytest.fixture
def ordinal_data():
    data = jnp.load(Path("probit_jax", "test", "data_basic_ordinal.npz"))
    return data["X"], data["y"]

@pytest.mark.parametrize(
    "f,y,lp",
    [
        (0.5, 1, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
        (100., 1, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
        (-3., 0, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
        (2., 2, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
    ]
)
class TestGlplAutodiff:
    signal_variance_0 = 1.
    def prior(self):
        return lambda prior_parameters: self.signal_variance_0 * EQ().stretch(prior_parameters)


    def test_glpl_diff_wrt_likelihood_parameters(self, f, y, lp):
        g = grad(glpl, argnums=2,)(f, y, lp)
        # noise hyperparam
        assert not jnp.isnan(g[0])
        # cutpoint hyperparams
        assert g[1][0] == g[1][3] == 0
        assert not (jnp.isnan(g[1][1]) or jnp.isnan(g[1][2]))

    def test_fLA_diff_wrt_likelihood_parameters(self, f, y, lp, ordinal_data):
        """Test the fixed point iteration function defined in 
        `Approximator.construct`
        TODO: determine actual nature of gradient"""
        fixed_point_iteration = lambda parameters, posterior_mean: f_LA(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior(), 
            grad_log_likelihood=glpl,hessian_log_likelihood=hlpl,
            posterior_mean=posterior_mean, data=ordinal_data)

        stretch = 1.
        params = (stretch, lp)

        fpi_sum = lambda p, pm: jnp.sum(fixed_point_iteration(p, pm))
        g = grad(fpi_sum)(params, f)
        
        grads_flat, _ = tree_flatten(g)
        assert not np.isnan(grads_flat).any()

    # def test_objective_diff_wrt_likelihood_parameters(self, f, y, lp, ordinal_data):
    #     pp = [1., 1.]
    #     f = lambda theta: objective_LA(
    #             pp, theta[1],
    #             self.prior,
    #             self.log_likelihood,
    #             self.grad_log_likelihood,
    #             self.hessian_log_likelihood,
    #             fixed_point_layer(jnp.zeros(self.N), self.tolerance, newton_solver, self.construct(), theta),
    #             self.data)

