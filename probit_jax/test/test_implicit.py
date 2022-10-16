import numpy as np
import jax.numpy as jnp
from jax import grad

from probit_jax.implicit.Laplace import objective_LA
from probit_jax.implicit.utilities import norm_cdf, _Z_tails


# def test_numerical_stability_of_LA():
#     dataset="EQ"

#     (X, y,
#         X_true, g_true,
#         cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
#         J, D, colors, Kernel) = load_data_synthetic(dataset, J=3)

#     N_train = np.shape(y)[0]
#     steps = np.max([2, N_train//1000])


def f(x):
    x_safe = jnp.where(x == 0., 1, x)
    return jnp.where( x > 0.5, jnp.arctan2(x_safe, x_safe), 0)

g = grad(f)(0.)
print(g)
    
print(norm_cdf(-1e6))

print('-'*9)

h = grad(_Z_tails)(-3., 1e-5)
print(h)