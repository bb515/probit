import gpflow
from gpflow.likelihoods import Ordinal
import numpy as np

from probit_jax.implicit.utilities import log_probit_likelihood

ord = Ordinal(np.array([0., 1.]))


lp = ord._scalar_log_prob([], [1], [1])

print(lp)