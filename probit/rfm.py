import numpy as np
import jax.numpy as jnp
from jax import random
from jax import jacrev, jacfwd
from jax import jit, vmap, grad
# from probit_api.model import LaplaceM
import lab.jax as B
from probit.implicit.solvers import (
    fwd_solver, newton_solver,
    fixed_point_layer, fixed_point_layer_fwd, fixed_point_layer_bwd)
from probit.approximators import LaplaceGP
import matrix
from algebra.util import identical

from plum import dispatch
from mlkernels import Kernel, pairwise, elwise

class LaplaceM(Kernel):
    """Weighted Laplace kernel.

    Args:
        M (array): Weight matrix of the weighted Laplace kernel.
    """

    def __init__(self, M):
        self.M  = M

    def _compute(self, dist2):
        return B.exp(-B.sqrt(dist2))

    def _pw_dists2(self, a, b):
        a = B.uprank(a)
        b = B.uprank(b)

        # Optimise the one-dimensional case.
        if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
            return self.M * (a -B.transpose(b))**2

        a_norm = (a @ self.M) * a
        a_norm = B.sum(a_norm, axis=1, squeeze=False)

        if a is b:
            b_norm = a_norm
        else:
            b_norm = (b @ self.M) * b
            b_norm = B.sum(b_norm, axis=1, squeeze=False)
        b_norm = B.transpose(b_norm)
        # b_norm = B.reshape(b_norm, (1, -1))  # alternative

        distances = a @ (self.M @ B.transpose(b))
        distances = distances * -2
        distances = distances + a_norm
        distances = distances + b_norm
        distances = B.where(distances < 1e-10, 0.0, distances)
        return distances

    def _ew_dists2(self, a, b):
        a = B.uprank(a)
        b = B.uprank(b)

        # Optimise the one-dimensional case.
        if B.shape(a, -1) == 1 and B.shape(b, -1) == 1:
            return self.M * (a -b)**2

        a_norm = (a @ self.M) * a
        a_norm = B.sum(a_norm, axis=1, squeeze=True)

        if a is b:
            b_norm = a_norm
        else:
            b_norm = (b @ self.M) * b
            b_norm = B.sum(b_norm, axis=1, squeeze=True)

        distances = (a @ self.M) * b
        distances = B.sum(distances, axis=1, squeeze=True)
        distances = distances * -2
        distances = distances + a_norm
        distances = distances + b_norm
        distances = B.where(distances < 1e-10, 0.0, distances)
        return distances

    def render(self, formatter):
        # This method determines how the kernel is displayed.
        return "LaplaceM()"

    @property
    def _stationary(self):
        # This method can be defined to return `True` to indicate that the kernel is
        # stationary. By default, kernels are assumed to not be stationary.
        return True

    @dispatch
    def __eq__(self, other: "LaplaceM"):
        # If `other` is also a `EQWithLengthScale`, then this method checks whether
        # `self` and `other` can be treated as identical for the purpose of
        # algebraic simplifications. In this case, `self` and `other` are identical
        # for the purpose of algebraic simplification if `self.scale` and
        # `other.scale` are. We use `algebra.util.identical` to check this condition.
        return identical(self.M, other.M)


@pairwise.dispatch
def pairwise(k: LaplaceM, x: B.Numeric, y: B.Numeric):
    pw_dists2 = k._pw_dists2(x, y)
    return matrix.Dense(k._compute(pw_dists2))


@elwise.dispatch
def elwise(k: LaplaceM, x: B.Numeric, y: B.Numeric):
    ew_dists2 = k._ew_dists2(x, y)
    return k._compute(ew_dists2)


class CosineM(Kernel):
    """Sharpened Cosine kernel.

    Args:
        degree (int): Power to which the cosine similarity is raised.
        M (array): Weight matrix of the weighted Laplace kernel.
    """

    def __init__(self, M, degree=1):
        self.degree = degree
        self.M = M

    def _scs(self, dotproduct, norms):
        # This computes the kernel given squared distances. We use `B` to provide a
        # backend-agnostic implementation.
        cs = B.divide(dotproduct, norms + 1e-8)
        return B.sign(dotproduct) * B.abs(cs) ** self.degree

    def render(self, formatter):
        # This method determines how the kernel is displayed.
        return f"Polynomial({formatter(self.degree)})"

    @property
    def _stationary(self):
        # This method can be defined to return `True` to indicate that the kernel is
        # stationary. By default, kernels are assumed to not be stationary.
        return False

    @dispatch
    def __eq__(self, other: "CosineM"):
        # If `other` is also a `SharpenedCosine`, then this method checks whether 
        # `self` and `other` can be treated as identical for the purpose of 
        # algebraic simplifications. In this case, `self` and `other` are identical 
        # for the purpose of algebraic simplification if `self.scale` and
        # `other.scale` are. We use `algebra.util.identical` to check this condition.
        return identical(self.degree, other.degree)

def uprank(x):
    """Custom uprank - the lab uprank inserts dimension at the end,
    we want (N, D) matrices"""
    if B.rank(x) < 2:
        x = B.expand_dims(x, axis=0)
    return x

# It remains to implement pairwise and element-wise computation of the kernel.
@pairwise.dispatch
def pairwise(k: CosineM, x: B.Numeric, y: B.Numeric):
    x, y = uprank(x), uprank(y)
    L = B.cholesky(k.M)
    x, y = x @ L.T, y @ L.T
    dotproduct = B.matmul(x, y, tr_b=True)
    norms = B.sqrt(B.outer(B.sum(x ** 2, axis=1), B.sum(y ** 2, axis=1)))
    return B.Dense(k._scs(dotproduct, norms))


@elwise.dispatch
def elwise(k: CosineM, x: B.Numeric, y: B.Numeric):
    L = B.cholesky(k.M)
    x, y = x @ L.T, y @ L.T
    dotproduct = B.sum(B.multiply(x, y), axis=-1)
    norms = B.sqrt(B.multiply(B.sum(x ** 2, axis=1), B.sum(y ** 2, axis=1)))
    return B.expand_dims(k._scs(dotproduct, norms), axis=-1)


class RFM(LaplaceGP):

    def __init__(
            self, *args, **kwargs):
        """
        Create an :class:`RFM` Approximator object.

        :returns: A :class:`RFM` object.
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
        weight = jnp.linalg.solve(B.dense(self.prior(parameters[0])(X, X)) + 1e-3 * jnp.eye(N), y)
        num_samples = 20000
        indices = np.random.randint(len(X), size=num_samples)
        if len(X) > len(indices):
            x = X[indices, :]
        else:
            x = X
        if np.ndim(weight) == 1:
            C = 1
        else:
            C = weight.shape[1]  # (N, C) is shape of weight, C=1 unless output is multidimensional
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
        weight2 = weight.T
        step3 = (weight2 @ K).T  # ((C, N @ N, N))
        del K, weight2
        step3 = step3.reshape(M, C, 1)
        x1 = (x @ parameters[0][0]).reshape(M, 1, D)
        step3 = step3 @ x1
        G = (step2 - step3) * -1. / parameters[0][1]
        return G

    def get_grads_cosine(self, parameters):
        """Get the gradient for the cosine kernel"""
        M = parameters[0][0]
        X, y = self.data
        N = self.N

        # alpha is the `weights` vector for the GP
        # \alpha = K^-1 y
        K_train = self.prior(parameters[0])(X, X)
        alpha = jnp.linalg.solve(K_train + 1e-3 * jnp.eye(N), y)

        num_samples = 10_000
        indices = np.random.randint(len(X), size=num_samples)
        if len(X) > len(indices):
            x = X[indices, :]
        else:
            x = X

        # K taken at samples to reduce computation
        K = B.dense(self.prior(parameters[0])(X, x))

        # For the sake of readability, X and x are rewritten as
        # (D, N) matrices

        L = B.cholesky(M)
        # M = L @ L.T
        Lx = L @ x
        LX = L @ X
        normLx2 = B.sum(Lx ** 2, axis=0)
        norms = B.sqrt(B.outer(B.sum(LX ** 2, axis=0), normLx2))

        G1 = M @ X
        G2 = x.T @ M @ X @ M @ x
        G2 /= normLx2

        G = G1 - G2
        G /= norms
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
            print(i, M)
        return parameters
