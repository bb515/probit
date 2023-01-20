import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad, jacrev, jacfwd, jit, vmap
from probit_jax.utilities import (
    InvalidKernel, check_cutpoints,
    log_probit_likelihood, probit_predictive_distributions)
from probit_api.model import LaplaceM
import lab.jax as B
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from probit_jax.implicit.solvers import (
    fwd_solver, newton_solver,
    fixed_point_layer, fixed_point_layer_fwd, fixed_point_layer_bwd)
from probit_jax.approximators import Approximator


class RFM(Approximator):

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

    def get_grad_f(parameters):
        def f(X):
            return B.dense(self.prior(parameters[0])(X, self.data[0])) @ fixed_point_layer(jnp.zeros(N), self.tolerance,
                newton_solver, self.construct(), parameters)
        return jacrev(f)

def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def plot_samples(x, image_size=32, num_channels=3, fname="samples"):
    img = image_grid(x, image_size, num_channels)
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(fname)
    plt.close()


def train(data, num_epochs=10, name=None, batch_size=4, train_acc=False, loader=True, classif=True):
    """rfm"""
    (X, y) = data
    N, D = X.shape

    approximate_inference_method = "Variational Bayes"
    if approximate_inference_method=="Variational Bayes":
        from probit_jax.approximators import VBGP as Approximator
    elif approximate_inference_method=="Laplace":
        from probit_jax.approximators import LaplaceGP as Approximator

    # Initiate a misspecified model, using a kernel
    # other than the one used to generate data
    def prior(prior_parameters):
        # Here you can define the kernel that defines the Gaussian process
        M, lengthscale = prior_parameters
        return LaplaceM(M).stretch(lengthscale)

    classifier = Approximator(data=data, prior=prior,
        log_likelihood=log_probit_likelihood,
        tolerance=1e-5  # tolerance for the jaxopt fixed-point resolution
    )

    fixed_point_layer.defvjp(
        fixed_point_layer_fwd, fixed_point_layer_bwd)

    # TODO: for some reason this doesn't work
    # def SSStake_grad(classifier, parameters):
    #     weight, precision = classifier.approximate_posterior(parameters)
        # f = lambda X: classifier.predict(X, parameters, weight, precision)
        # return jacrev(f)


    def approximate_posterior(classifier, parameters, X, batch):
        weight = fixed_point_layer(jnp.zeros(N), classifier.tolerance,
            newton_solver, classifier.construct(), parameters)
        batch = batch.reshape(1, -1)
        K = B.dense(prior(parameters[0])(batch, X))
        posterior_mean = K @ weight
        return posterior_mean[0]

    def take_grad(classifier, parameters, X):
        return jit(
            vmap(
                grad(
                    lambda batch: approximate_posterior(
                classifier, parameters, X, batch)),
            in_axes=(0), out_axes=(0)))

    def update_step(g, batch, params):
        # doesn't feel good retaking grad at each step
        gs = g(batch)
        _M = B.einsum('ij, ik -> ijk', gs, gs)  # n_batch, D, D
        _M = B.sum(_M, axis=0) # D, D
        return _M

    def get_grads(classifier, parameters, X, y):
        N, D = X.shape
        weight = fixed_point_layer(jnp.zeros(N), classifier.tolerance,
            newton_solver, classifier.construct(), parameters)
        # print("here", weight)
        weight = jnp.linalg.solve(B.dense(prior(parameters[0])(X, X)) + 1e-8 * jnp.eye(N), y)
        # print("there", weight)
        #posterior_mean = K @ weight
        num_samples = 20000
        indices = np.random.randint(len(X), size=num_samples)
        if len(X) > len(indices):
            x = X[indices, :]
        else:
            x = X
        C = 1  # (N, C) is shape of weight, C=1 unless output is multidimensional
        M, D  = x.shape
        K = B.dense(prior(parameters[0])(X, x))
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

    def retrain_nn(
            classifer, update_step, num_epochs, step_rng, samples, y, params,
            batch_size):
        train_size, D = samples.shape
        batch_size = min(train_size, batch_size)
        steps_per_epoch = train_size // batch_size
        for i in range(num_epochs):
            rng, step_rng = random.split(step_rng)
            perms = random.permutation(step_rng, train_size)
            perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            M = jnp.zeros((D, D))
            # TODO: 7.67
            G = get_grads(classifier, params, samples, y)
            # TODO: 11.2
            # g = SStake_grad(classifer, params)
            # G = jnp.diagonal(g(samples))
            # G = G.reshape(train_size, 1, D)
            # TODO: 29.9
            # g = take_grad(classifer, params, X)
            # G = g(samples)
            # G = G.reshape(train_size, 1, D)
            for j, perm in enumerate(perms):
                grad = G[perm, : :]
                gradT = grad.transpose(0, 2, 1)
                M += jnp.sum(gradT @ grad, axis=0)
                del grad, gradT
            M /= train_size
            print("M", M)
            print(i)
            params[0][0] = M
        return M

    lengthscale = 1e1
    noise_std = 1e-1
    cutpoints = jnp.array([-jnp.inf, 0.0, jnp.inf])
    M = np.eye(D, dtype='float32')
    params = [[M, lengthscale], [noise_std, cutpoints]]

    rng = random.PRNGKey(2023)
    rng, step_rng = random.split(rng, 2)

    now = time.time()
    M = retrain_nn(classifier, update_step, num_epochs=num_epochs, step_rng=step_rng, samples=X, y=y, params=params, batch_size=batch_size)
    print(time.time() - now)
    print(M)
    w, v = np.linalg.eig(M)
    plt.plot(w)
    plt.savefig("eigs.png")
    plt.close()
    v = v.astype(np.float64)
    print(v[0])
    print(v.shape)
    plot_samples(v[:4], image_size=32, num_channels=1)
