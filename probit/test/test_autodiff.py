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
from probit_jax.implicit.utilities import (log_probit_likelihood, 
    grad_log_probit_likelihood, hessian_log_probit_likelihood)
from probit_jax.solvers import (fixed_point_layer, newton_solver, 
    fixed_point_layer_fwd, fixed_point_layer_bwd)

lpl = log_probit_likelihood
glpl = lambda f, y, lp: grad_log_probit_likelihood(f, y, lp, single_precision=False)
hlpl = lambda f, y, lp: hessian_log_probit_likelihood(f, y, lp, single_precision=False)

fixed_point_layer.defvjp(
    fixed_point_layer_fwd, fixed_point_layer_bwd)

def pytree_has_nans(tree):
    flat, _ = tree_flatten(tree)
    any_nans = any(jnp.any(jnp.isnan(x)) for x in flat)
    return any_nans

@pytest.fixture
def ordinal_data():
    data = jnp.load(Path("probit_jax", "test", "data_basic_ordinal.npz"))
    return data["X"], data["y"]

@pytest.fixture
def lp():
    """Likelihood parameters"""
    return (0.1, [-jnp.inf, 0., 1., jnp.inf])


class TestGlplAutodiff:
    signal_variance_0 = 1.
    prior_parameters = 1.
    tolerance = 1e-6

    def prior(self):
        return lambda prior_parameters: self.signal_variance_0 * EQ().stretch(prior_parameters)

    @pytest.mark.parametrize(
        "f,y,lp",
        [
            (0.5, 1, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
            (100., 1, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
            (-3., 0, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
            (2., 2, (0.1, [-jnp.inf, 0., 1., jnp.inf])),
        ]
    )
    def test_glpl_diff_wrt_likelihood_parameters(self, f, y, lp):
        g = grad(glpl, argnums=2,)(f, y, lp)
        # noise hyperparam
        assert not jnp.isnan(g[0])
        # cutpoint hyperparams
        assert g[1][0] == g[1][3] == 0
        assert not (jnp.isnan(g[1][1]) or jnp.isnan(g[1][2]))

    @pytest.mark.parametrize(
        "f,",
        [
            (0.5),
        ]
    )
    def test_fLA_diff_wrt_likelihood_parameters(self, f, lp, ordinal_data):
        """Test the fixed point iteration function defined in 
        `Approximator.construct`
        TODO: determine expected values and shape of gradient"""
        fixed_point_iteration = lambda parameters, posterior_mean: f_LA(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior(), 
            grad_log_likelihood=glpl,hessian_log_likelihood=hlpl,
            posterior_mean=posterior_mean, data=ordinal_data)

        theta = (self.prior_parameters, lp)

        fpi_sum = lambda p, pm: jnp.sum(fixed_point_iteration(p, pm))
        g = grad(fpi_sum)(theta, f)
        
        assert not pytree_has_nans(g)

    def test_fixed_point_layer_diff_wrt_likelihood_parameters(self, lp, ordinal_data):
        """Test the fixed point layer"""
        N = jnp.shape(ordinal_data[1])

        fixed_point_iteration = lambda parameters, posterior_mean: f_LA(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior(), 
            grad_log_likelihood=glpl,hessian_log_likelihood=hlpl,
            posterior_mean=posterior_mean, data=ordinal_data)

        lp = (lp[0], jnp.array(lp[1]))
        theta = (self.prior_parameters, lp)

        fpl_sum = lambda z_init, tol, solver, fpi, params: jnp.sum(fixed_point_layer(z_init, tol, solver, fpi, params))
        g_fpl = grad(fpl_sum, argnums=4)(jnp.zeros(N), self.tolerance, newton_solver, fixed_point_iteration, theta)
        assert not pytree_has_nans(g_fpl)

    def test_objective_diff_wrt_likelihood_parameters(self, lp, ordinal_data):
        N = jnp.shape(ordinal_data[1])

        fixed_point_iteration = lambda parameters, posterior_mean: f_LA(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior(), 
            grad_log_likelihood=glpl,hessian_log_likelihood=hlpl,
            posterior_mean=posterior_mean, data=ordinal_data)
        
        obj = lambda theta: objective_LA(
            theta[0], theta[1],
            self.prior(),
            lpl, glpl, hlpl,
            fixed_point_layer(jnp.zeros(N), self.tolerance, newton_solver, fixed_point_iteration, theta),
            ordinal_data)

        lp = (lp[0], jnp.array(lp[1]))
        theta = (self.prior_parameters, lp)
        print(obj(theta))

        g = grad(obj)(theta)
        print(g)
        assert not pytree_has_nans(g)
