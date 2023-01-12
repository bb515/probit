import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jacrev, jacfwd
from jax import jit, vmap, grad
from probit_api.model import LaplaceM
import lab.jax as B
from probit_jax.implicit.solvers import (
    fwd_solver, newton_solver,
    fixed_point_layer, fixed_point_layer_fwd, fixed_point_layer_bwd)
from probit_jax.approximators import LaplaceGP


class RFM(LaplaceGP):

    def __init__(
            self, *args, **kwargs):
        """
        Create an :class:`VBGP` Approximator object.

        :returns: A :class:`VBGP` object.
        """
        super().__init__(*args, **kwargs)
        # Set up a JAX-transformable function for a custom VJP rule definition
        fixed_point_layer.defvjp(
            fixed_point_layer_fwd, fixed_point_layer_bwd)

    def get_grad_f(self, parameters):
        weight = fixed_point_layer(jnp.zeros(self.N), self.tolerance,
                newton_solver, self.construct(), parameters)
        def f(batch):
            batch = batch.reshape(1, -1)
            posterior_mean = B.dense(self.prior(parameters[0]))(batch, self.data[0]) @ weight
            return posterior_mean[0, 0]
        return jit(vmap(grad(f), in_axes=(0), out_axes=(0)))

    def SSget_grad_f(self, parameters):
        weight = fixed_point_layer(jnp.zeros(self.N), self.tolerance,
                newton_solver, self.construct(), parameters)
        def f(X):
            return B.dense(self.prior(parameters[0])(X, self.data[0])) @ weight
        return jacfwd(f)

    def get_grads(self, parameters):
        X, y = self.data
        N = self.N
        # solving GP regression using a likelihood
        # weight = fixed_point_layer(jnp.zeros(N), self.tolerance,
            # newton_solver, self.construct(), parameters)
        # adhoc
        weight = jnp.linalg.solve(B.dense(self.prior(parameters[0])(X, X)) + 1e-8 * jnp.eye(N), y)
        print(weight.shape)
        assert 0
        num_samples = 20000
        indices = np.random.randint(len(X), size=num_samples)
        if len(X) > len(indices):
            x = X[indices, :]
        else:
            x = X
        C = 1  # (N, C) is shape of weight, C=1 unless output is multidimensional
        M, D  = x.shape
        K = B.dense(self.prior(parameters[0])(X, x))
        dist = LaplaceM(parameters[0][0])._pw_dists2(X, x)
        dist = jnp.sqrt(dist)
        K = K / dist
        K = jnp.where(K==jnp.inf, 0.0, K)
        X1 = X @ parameters[0][0]
        X1 = X1.reshape(N, 1, D)
        weight1 = weight.reshape(N, C, 1)
        step1 = weight.reshape(N, C, 1) @ X1
        del weight1, X1
        step1 = step1.reshape(-1, C * D)
        step2 = K.T @ step1
        del step1
        step2 = step2.reshape(-1, C, D)
        a2 = weight
        step3 = (a2 @ K).T
        del K, a2
        step3 = step3.reshape(M, C, 1)
        x1 = (x @ parameters[0][0]).reshape(M, 1, D)
        step3 = step3 @ x1
        G = (step2 - step3) * -1. / parameters[0][1]
        return G

    def train(
            self, parameters, step_rng, num_epochs=10, batch_size=4):
        train_size = self.N
        batch_size = min(train_size, batch_size)
        steps_per_epoch = train_size // batch_size
        for i in range(num_epochs):
            rng, step_rng = random.split(step_rng)
            perms = random.permutation(step_rng, train_size)
            perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            M = jnp.zeros((self.D, self.D))
            # TODO: 24.5
            # g = self.get_grad_f(parameters)
            # G = g(self.data[0])
            # G = G.reshape(train_size, 1, self.D)
            # TODO: 64.5, even worse for fwd
            # jac doesn't scale with data
            # g = self.SSget_grad_f(parameters)
            # TODO: 8.4
            G = self.get_grads(parameters)
            for j, perm in enumerate(perms):
                grad = G[perm, : :]
                # grad = g(self.data[0][perm])
                gradT = grad.transpose(0, 2, 1)
                M += jnp.sum(gradT @ grad, axis=0)
                del grad, gradT
            M /= train_size
            parameters[0][0] = M
        return parameters

