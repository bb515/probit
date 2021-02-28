This is an example to test convergence of the Gibbs sampler that aims to repeat
"Accelerating Monte Carlo Markov chain
convergence for cumulative-link generalized
linear models"
by Mary Kathryn Cowles (1996)

"The ordinal probit may present challeneges in obtaining satisfactory convergence." The purpose
of this example is to show the slow convergence of the Gibbs sampler in this scenario, and
to use the alternative Gibbs sampler.

The data has the following dimensions
x is on the real line
y is the latent response variable on the real line
N = 1792 = K * 256
The simulated orderd data has the model
y = \beta_0 + \beta_1 * x + \epsilon
where \epislon is a vector of Normal(0, 1) random variables.
y is then put into ORDER and split into septiles (K=7 bins) to get the
    "true" latent responses. The ordinal response variables are then t = {1,...,7}
The data is in the form of a ".npz" format, and can be read like:
>>> data = np.load("data.npz")
>>> X_k = data["X_k"]  # Contains (256, 7) array of binned x values
>>> Y_true_k = data["Y_k"]  # Costaints (256, 7) array of binned y values
>>> X = data["X"]  # Containts (1792,) array of x values
>>> t = data["t"]  # Contains (1792,) array of ordinal response variables, corresponding to Xs values
>>> Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)