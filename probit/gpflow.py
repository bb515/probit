"""GPFlow models. Probably no need to inherit Approximator here."""
from probit.approximators import Approximator
import enum
import pathlib
from tqdm import trange
import warnings
import matplotlib.pyplot as plt
from gpflow.ci_utils import reduce_in_tests
# from gpflow.monitor import (
#     ImageToTensorBoard,
#     ModelToTensorBoard,
#     Monitor,
#     MonitorTaskGroup,
#     ScalarToTensorBoard)
import time
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["figure.figsize"] = (12, 6)

# np.random.seed(123)  # for reproducibility

# # make a one-dimensional ordinal regression problem

# # This function generates a set of inputs X,
# # quantitative output f (latent) and ordinal values Y

# def generate_data(num_data):
#     # First generate random inputs
#     X = np.random.rand(num_data, 1)

#     # Now generate values of a latent GP
#     kern = gpflow.kernels.SquaredExponential(lengthscales=0.1)
#     K = kern(X)
#     f = np.random.multivariate_normal(mean=np.zeros(num_data), cov=K).reshape(-1, 1)

#     # Finally convert f values into ordinal values Y
#     Y = np.round((f + f.min()) * 3)
#     Y = Y - Y.min()
#     Y = np.asarray(Y, np.float64)

#     return X, f, Y

# np.random.seed(1)
# N = 20
# X, f, Y = generate_data(N)
# data = (X, Y)

# # plt.figure(figsize=(11, 6))
# # plt.plot(X, f, ".")
# # plt.ylabel("latent function value")
# # plt.show()

# # plt.twinx()
# # plt.plot(X, Y, "kx", mew=1.5)
# # plt.ylabel("observed data value")
# # plt.show()

# # construct ordinal likelihood - bin_edges is the same as unique(Y) but centered
# bin_edges = np.array(np.arange(np.unique(Y).size + 1), dtype=float)
# bin_edges = bin_edges - bin_edges.mean()
# likelihood = gpflow.likelihoods.Ordinal(bin_edges)


# kernel = gpflow.kernels.SquaredExponential()

# M = 20  # Number of inducing locations

# Z = X[:M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

# m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

# def plot(title=""):
#     Xtest = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#     ## to see the predictive density, try predicting every possible discrete value for Y.
#     def pred_log_density(m):
#         Xtest = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#         ys = np.arange(Y.max() + 1)
#         densities = []
#         for y in ys:
#             Ytest = np.full_like(Xtest, y)
#             # Predict the log density
#             densities.append(m.predict_log_density((Xtest, Ytest)))
#         return np.vstack(densities)

#     fig = plt.figure(figsize=(14, 6))
#     plt.imshow(
#         np.exp(pred_log_density(m)),
#         interpolation="nearest",
#         extent=[X.min(), X.max(), -0.5, Y.max() + 0.5],
#         origin="lower",
#         aspect="auto",
#         cmap=plt.cm.viridis,
#     )
#     plt.colorbar()
#     plt.plot(X, Y, "kx", mew=2, scalex=False, scaley=False)
#     plt.show()

#     # Predictive density for a single input x=0.5
#     x_new = 0.5
#     Y_new = np.arange(np.max(Y + 1)).reshape([-1, 1])
#     X_new = np.full_like(Y_new, x_new)
#     # for predict_log_density x and y need to have the same number of rows
#     dens_new = np.exp(m.predict_log_density((X_new, Y_new)))
#     fig = plt.figure(figsize=(8, 4))
#     plt.bar(x=Y_new.flatten(), height=dens_new.flatten())
#     plt.show()

#     # plt.figure(figsize=(12, 4))
#     # plt.title(title)
#     # pX = np.linspace(-1, 1, 100)[:, None]  # Test locations
#     # pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
#     # plt.plot(X, Y, "x", label="Training points", alpha=0.2)
#     # (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
#     # col = line.get_color()
#     # plt.fill_between(
#     #     pX[:, 0],
#     #     (pY - 2 * pYv ** 0.5)[:, 0],
#     #     (pY + 2 * pYv ** 0.5)[:, 0],
#     #     color=col,
#     #     alpha=0.6,
#     #     lw=1.5,
#     # )
#     # Z = m.inducing_variable.Z.numpy()
#     # plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
#     # plt.legend(loc="lower right")
#     # plt.show()

# plot(title="Predictions before training")

# elbo = tf.function(m.elbo)

# tensor_data = tuple(map(tf.convert_to_tensor, data))
# elbo(tensor_data)  # run it once to trace & compile

# print("Hi")
# elbo(tensor_data)
# print("Lo")

# minibatch_size = 20

# train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)

# train_iter = iter(train_dataset.batch(minibatch_size))

# ground_truth = elbo(tensor_data).numpy()

# print(ground_truth)

# # We turn off training for inducing point locations
# gpflow.set_trainable(m.inducing_variable, False)



# maxiter = reduce_in_tests(4000)

# logf = run_adam(m, maxiter)
# plt.plot(np.arange(maxiter)[::10], logf)
# plt.xlabel("iteration")
# _ = plt.ylabel("ELBO")
# plt.show()


# plot(title="Predictions after training")

# # We turn off training for inducing point locations
# gpflow.set_trainable(m.inducing_variable, True)


# maxiter = reduce_in_tests(20000)

# logf = run_adam(m, maxiter)
# plt.plot(np.arange(maxiter)[::10], logf)
# plt.xlabel("iteration")
# _ = plt.ylabel("ELBO")
# plt.show()

# plot(title="Predictions after training")



class VGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the variational
    Gauassian process by Hensman, using GPFlow.
 
    Inherits the Approximator ABC. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference. It is for the ordinal likelihood. For this a
    :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "OrdinalVGP"

    def __init__(
            self, cutpoints, noise_variance=1.0, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`OrdinalVGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: A :class:`OrdinalVGP` object.
        """
        super().__init__(*args, **kwargs)
        # Tends to work well in practice - should it be made smaller?
        # Just keep it consistent
        #self.jitter = 1e-6
        self.jitter = 1e-10
        # Initiate hyperparameters
        self._model_initiate(cutpoints)
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _update_prior(self):
        pass

    def _model_initiate(self, cutpoints):
        """Update prior covariances."""
        likelihood = gpflow.likelihoods.Ordinal(cutpoints[1:-1])
        warnings.warn(
            "Updating prior model using gpflow")
        self._model = gpflow.models.VGP(
            data=(self.X_train, self.t_train),
            kernel=self.kernel, likelihood=likelihood)
        def plot_prediction(fig, ax):
            Xnew = np.linspace(
                self.X_train.min() - 0.5,
                self.X_train.max() + 0.5, 100).reshape(-1, 1)
            Ypred = self._model.predict_f_samples(
                Xnew, full_cov=True, num_samples=20)
            ax.plot(Xnew.flatten(), np.squeeze(Ypred).T, "C1", alpha=0.2)
            ax.plot(self.X_train, self.t_train, "o")
        warnings.warn("Done updating prior using gpflow.")
        # Fix hyperparameters
        gpflow.set_trainable(self._model.kernel.lengthscales, False)
        gpflow.set_trainable(self._model.kernel.variance, False)
        # bin_edges are not trainable in GPFlow!
        # gpflow.set_trainable(self._model.likelihood.bin_edges, False)
        gpflow.set_trainable(self._model.likelihood.sigma, False)
        # Instantiate optimizer
        self._optimizer = gpflow.optimizers.Scipy()
        self._training_loss = self._model.training_loss_closure(
            compile=True
        )  # compile=True (default): compiles using tf.function
        # log_dir_scipy = "scipy"
        # model_task = ModelToTensorBoard(
        #     log_dir_scipy, self._model)
        # lml_task = ScalarToTensorBoard(
        #     log_dir_scipy, lambda: self._model.training_loss(),
        #     "training_objective")
        # image_task = ImageToTensorBoard(
        #     log_dir_scipy, plot_prediction, "image_samples")
        # self._monitor = Monitor(
        #     MonitorTaskGroup([model_task, lml_task], period=1),
        #     MonitorTaskGroup(image_task, period=5)
        # )
        self._monitor = None


    def hyperparameters_update(
        self, cutpoints=None, varphi=None, variance=None, noise_variance=None,
        varphi_hyperparameters=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg float noise_variance: The noise variance.
        :type noise_variance:
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters:
        """
        if cutpoints is not None:
            self.cutpoints = self._check_cutpoints(cutpoints)
            self.cutpoints_ts = self.cutpoints[self.t_train]
            self.cutpoints_tplus1s = self.cutpoints[self.t_train + 1]
            self._model.likelihood.bin_edges = self.cutpoints[1:-1]
        if varphi is not None or variance is not None:
            self._model.kernel.lengthscales.assign(varphi)
            self._model.kernel.variance.assign(variance)
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
            self._model.likelihood.sigma.assign(self.noise_std)
        # Handle a possible case where the kernel has hyperhyperparameters

    def grid_over_hyperparameters(
            self, domain, res, indices=None, posterior_mean_0=None,
            verbose=False, steps=100):
        """
        TODO: Can this be moved to a plot.py
        Return meshgrid values of fx and directions of gx over hyperparameter
        space.

        The particular hyperparameter space is inferred from the user inputs
        - the rule is that if any of the
        variables are None, then those are the variables to grid over. We can
        only visualise these surfaces for
        maximum of 2 variables, so the number of combinations is Mc2 + Mc1
        where M is the total no. of hyperparameters.

        Special cases are frequent: log and non log variables. 2 axis vs 1
        axis objective function, calculate
        new Gram matrix or not. So the simplest way is to combinate manually.
        """
        (
        x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        Phi_new,
        fxs, gxs, gx_0,
        intervals, indices_where) = self._grid_over_hyperparameters_initiate(
            res, domain, indices, self.cutpoints)
        error = np.inf
        fx_old = np.inf
        for i, phi in enumerate(Phi_new):
            self._grid_over_hyperparameters_update(
                phi, indices, self.cutpoints)
            # Reset error and posterior mean
            fx, gx = self.approximate_posterior(
                theta, indices, verbose=False)
            fxs[i] = fx
            gxs[i] = gx[indices_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print(
                    "cutpoints={}, varphi={}, noise_variance={}, variance={}, "
                    "fx={}, gx={}".format(
                    self.cutpoints, self.kernel.varphi, self.noise_variance,
                    self.kernel.variance, fx, gxs[i]))
        if x2s is not None:
            return (
                fxs.reshape((len(x1s), len(x2s))), gxs,
                xx, yy, xlabel, ylabel, xscale, yscale)
        else:
            return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale

    def approximate(
            self, steps=None, write=False):
        """
        Estimating the posterior means are a 3 step iteration over
        posterior_mean, varphi and psi Eq.(8), (9), (10), respectively or,
        optionally, just an iteration over posterior_mean.

        :arg int steps: The number of iterations the Approximator takes.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg int first_step: The first step. Useful for burn in algorithms.
        :arg bool write: Boolean variable to store and write arrays of
            interest. If set to "True", the method will output non-empty
            containers of evolution of the statistics over the steps. If
            set to "False", statistics will not be written and those
            containers will remain empty.
        :return: Approximate posterior means and covariances.
        :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate
            posterior means, other statistics and tuple of lists of per-step
            evolution of those statistics.
        """
        if write:
            self._optimizer.minimize(
                self._training_loss, self._model.trainable_variables,
                    callback=self._monitor)
        else:
            self._optimizer.minimize(
                training_loss, self._model.trainable_variables,
                    callback=monitor)

    def approximate_posterior(
            self, theta, indices, steps=None, first_step=None,
            return_reparameterised=False, verbose=False):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :arg indices:
        :arg first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx, gx
        :rtype: float, `:class:numpy.ndarray`
        """
        # Update prior covariance and get hyperparameters from theta
        (intervals, steps, error, iteration, indices_where,
        gx) = self._hyperparameter_training_step_initialise(
            theta, indices, steps)
        eval_func = self._optimizer.eval_func(
            self._model.training_loss, self._model.trainable_variables)
        x = self._optimizer.initial_parameters(
            tuple(self._model.trainable_variables))
        fx, gx = eval_func(x)
        print("before training ", fx)
        self.approximate()
        x = self._optimizer.initial_parameters(
            tuple(self._model.trainable_variables))
        fx, gx = eval_func(x)
        print("after training ", fx)

        posterior = self._model.posterior()
        # gpflow.set_trainable(self._model.inducing_variable, False)
        assert 0
        if return_reparameterised is True:
            return fx, gx, posterior.q_mu, (posterior.q_sqrt, False)
        elif return_reparameterised is False:
            posterior_mean, posterior_covariance = posterior.predict_f(
                self.X_train)
            return fx, gx, posterior_mean, (
                posterior_covariance, True)
        elif return_reparameterised is None:
            return fx, gx

    def predict(
            self, X_test, cached_posterior=None):
        """
        Return the posterior predictive distribution over classes.

        :arg X_test: The new data points, array like (N_test, D).
        :type X_test: :class:`numpy.ndarray`.
        :arg cov: The approximate
            covariance-posterior-inverse-covariance matrix. Array like (N, N).
        :type cov: :class:`numpy.ndarray`.
        :arg f: Array like (N,).
        :type f: :class:`numpy.ndarray`.
        :arg bool reparametrised: Boolean variable that is `True` if f is
            reparameterised, and `False` if not.
        :arg bool whitened: Boolean variable that is `True` if f is whitened,
            and `False` if not.
        :return: The ordinal class probabilities.
        """
        posterior.predict_f(X_test)
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Y_test = np.full_like(Xtest, j)
            predictive_distributions[:, j] = m.predict_log_density(
                (X_test, Y_test))
        return predictive_distributions 


class SVGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the sparse variational
    Gauassian process by Hensman, using GPFlow.
 
    Inherits the Approximator ABC. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference. It is for the ordinal likelihood. For this a
    :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SVGP"

    def __init__(
            self, cutpoints, noise_variance=1.0, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`SVGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: A :class:`SVGP` object.
        """
        super().__init__(*args, **kwargs)
        # if cutpoints_hyperparameters is not None:
        #     warnings.warn("cutpoints_hyperparameters set as {}".format(cutpoints_hyperparameters))
        #     self.cutpoints_hyperparameters = cutpoints_hyperparameters
        # else:
        #     self.cutpoints_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        if self.kernel._ARD:
            raise ValueError(
                "The kernel must not be ARD type (kernel._ARD=1),"
                " but ISO type (kernel._ARD=0). (got {}, expected)".format(
                    self.kernel._ARD, 0))
        if self.kernel._general:
            raise ValueError(
                "The kernel must not be general "
                "type (kernel._general=1), but simple type "
                "(kernel._general=0). (got {}, expected)".format(
                    self.kernel._general, 0))
        # Tends to work well in practice - should it be made smaller?
        # Just keep it consistent
        #self.jitter = 1e-6
        self.jitter = 1e-10
        # Initiate hyperparameters
        self.cutpoints = self._check_cutpoints(cutpoints)
        self.noise_variance = noise_variance
        self.noise_std = noise_std
        self._update_prior()
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _update_prior(self):
        pass
        
    def hyperparameters_update(
        self, cutpoints=None, varphi=None, variance=None, noise_variance=None,
        varphi_hyperparameters=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg float noise_variance: The noise variance.
        :type noise_variance:
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters:
        """
        if cutpoints is not None:
            self.cutpoints = self._check_cutpoints(cutpoints)
            self.cutpoints_ts = self.cutpoints[self.t_train]
            self.cutpoints_tplus1s = self.cutpoints[self.t_train + 1]
            self._model.likelihood.bin_edges.assign(self.cutpoints[1:-1])
        if varphi is not None or variance is not None:
            self._model.kernel.lengthscales.assign(varphi)
            self._model.kernel.variance.assign(variance)
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
            self._model.likelihood.variance.assign(self.noise_variance)
        # Handle a possible case where the kernel has hyperhyperparameters

    def grid_over_hyperparameters(
            self, domain, res, indices=None, posterior_mean_0=None,
            verbose=False, steps=100):
        """
        TODO: Can this be moved to a plot.py
        Return meshgrid values of fx and directions of gx over hyperparameter
        space.

        The particular hyperparameter space is inferred from the user inputs
        - the rule is that if any of the
        variables are None, then those are the variables to grid over. We can
        only visualise these surfaces for
        maximum of 2 variables, so the number of combinations is Mc2 + Mc1
        where M is the total no. of hyperparameters.

        Special cases are frequent: log and non log variables. 2 axis vs 1
        axis objective function, calculate
        new Gram matrix or not. So the simplest way is to combinate manually.
        """
        (
        x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        Phi_new,
        fxs, gxs, gx_0,
        intervals, indices_where) = self._grid_over_hyperparameters_initiate(
            res, domain, indices, self.cutpoints)
        error = np.inf
        fx_old = np.inf
        for i, phi in enumerate(Phi_new):
            self._grid_over_hyperparameters_update(
                phi, indices, self.cutpoints)
            # Reset error and posterior mean
            fx, gx = self.approximate_posterior(
                theta, indices, verbose=False)
            fxs[i] = fx
            gxs[i] = gx[indices_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print(
                    "cutpoints={}, varphi={}, noise_variance={}, variance={}, "
                    "fx={}, gx={}".format(
                    self.cutpoints, self.kernel.varphi, self.noise_variance,
                    self.kernel.variance, fx, gxs[i]))
        if x2s is not None:
            return (
                fxs.reshape((len(x1s), len(x2s))), gxs,
                xx, yy, xlabel, ylabel, xscale, yscale)
        else:
            return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale

    def approximate_posterior(
            self, theta, indices, steps=None, first_step=None,
            return_reparameterised=False, verbose=False):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :arg indices:
        :arg first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx, gx
        :rtype: float, `:class:numpy.ndarray`
        """
        # Update prior covariance and get hyperparameters from theta
        (intervals, steps, error, iteration, indices_where,
        gx) = self._hyperparameter_training_step_initialise(
            theta, indices, steps)
        fx_old = np.inf
        posterior_mean = None
        # Convergence is sometimes very fast so this may not be necessary
        posterior = self._model.posterior()
        opt = gpflow.optimizers.Scipy()
        fx, gx = opt.eval_func()
        if return_reparameterised is True:
            return fx, gx, posterior.q_mu, (posterior.q_sqrt, False)
        elif return_reparameterised is False:
            posterior_mean, posterior_covariance = posterior.predict_f(
                self.X_train)
            return fx, gx, posterior_mean, (
                posterior_covariance, True)
        elif return_reparameterised is None:
            return fx, gx
