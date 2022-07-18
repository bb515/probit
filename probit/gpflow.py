"""GPFlow models. Probably no need to inherit Approximator here."""
from probit.approximators import Approximator
from probit.utilities import check_cutpoints
from tqdm import trange
import warnings
import matplotlib.pyplot as plt
from gpflow.ci_utils import reduce_in_tests
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard)
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
        Return a string representation of this class, used to import the class
        from the string.
        """
        return "VGP"

    def __init__(
            self, cutpoints, noise_variance=1.0, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`VGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: A :class:`VGP` object.
        """
        super().__init__(*args, **kwargs)
        # Tends to work well in practice - should it be made smaller?
        # Just keep it consistent
        self.jitter = 1e-6
        # self.jitter = 1e-10  # may be too small for samples
        self.EPS = 1e-4  # perhaps not low enough.
        # self.EPS = 1e-8
        #self.EPS_2 = 1e-7
        self.EPS_2 = self.EPS**2
        # Initiate hyperparameters
        if cutpoints is not None:
            self.cutpoints = check_cutpoints(cutpoints, self.J)
            self.cutpoints_ts = self.cutpoints[self.y_train]
            self.cutpoints_tplus1s = self.cutpoints[self.y_train + 1]
            # self._model.likelihood.bin_edges = self.cutpoints[1:-1]
        self._model_initiate(self.cutpoints)
        self.hyperparameters_update(
            noise_variance=noise_variance)

    def _update_prior(self):
        """Only need to do this if using a sampling algorithm."""
        # TODO: tidy
        # self.K = self.kernel.K(self.X_train, self.X_train).numpy()

    def _model_initiate(self, cutpoints):
        """Update prior covariances."""
        likelihood = gpflow.likelihoods.Ordinal(cutpoints[1:-1])
        warnings.warn(
            "Initiating model using gpflow")
        self._model = gpflow.models.VGP(
            data=(self.X_train.astype(np.float64), self.y_train.reshape(-1, 1)),
            kernel=self.kernel, likelihood=likelihood)
        def plot_prediction(fig, ax):
            Xnew = np.linspace(
                self.X_train.min() - 0.5,
                self.X_train.max() + 0.5, 100).reshape(-1, 1)
            Ypred = self._model.predict_f_samples(
                Xnew, full_cov=True, num_samples=20)
            ax.plot(Xnew.flatten(), np.squeeze(Ypred).T, "C1", alpha=0.2)
            ax.plot(self.X_train, self.y_train, "o")
        warnings.warn("Done initiating model using gpflow.")
        # Fix hyperparameters - difficult to do without letting user do it
        gpflow.set_trainable(self._model.kernel.lengthscales, False)
        gpflow.set_trainable(self._model.kernel.variance, False)
        # bin_edges are not trainable in GPFlow!
        # gpflow.set_trainable(self._model.likelihood.bin_edges, False)
        gpflow.set_trainable(self._model.likelihood.sigma, False)
        # Instantiate optimizer
        self._optimizer = gpflow.optimizers.Scipy()
        self._training_loss = self._model.training_loss
        # self._training_loss = self._model.training_loss_closure(
        #     compile=True
        # )  # compile=True (default): compiles using tf.function
        log_dir_scipy = "scipy"
        model_task = ModelToTensorBoard(
            log_dir_scipy, self._model)
        lml_task = ScalarToTensorBoard(
            log_dir_scipy, lambda: self._model.training_loss(),
            "training_objective")
        image_task = ImageToTensorBoard(
            log_dir_scipy, plot_prediction, "image_samples")
        self._monitor = Monitor(
            MonitorTaskGroup([model_task, lml_task], period=1)
            # MonitorTaskGroup(image_task, period=5)
        )

    def get_phi(self, trainables):
        """
        Get the parameters (phi) for unconstrained optimization.

        :arg trainables: Indicator array of the hyperparameters to optimize over.
            TODO: it is not clear, unless reading the code from this method,
            that trainables[0] means noise_variance, etc. so need to change the
            interface to expect a dictionary with keys the hyperparameter
            names and values a bool that they are fixed?
        :type trainables: :class:`numpy.ndarray`
        :returns: The unconstrained parameters to optimize over, phi.
        :rtype: :class:`numpy.array`
        """
        phi = []
        if trainables[0]:
            phi.append(np.log(np.sqrt(self.noise_variance)))
        if trainables[1]:
            phi.append(self.cutpoints[1])
        for j in range(2, self.J):
            if trainables[j]:
                phi.append(np.log(self.cutpoints[j] - self.cutpoints[j - 1]))
        if trainables[self.J]:
            phi.append(np.log(np.sqrt(self.kernel.variance)))
        # TODO: replace this with kernel number of hyperparameters.
        if trainables[self.J + 1]:
            phi.append(np.log(self.kernel.lengthscales.numpy()))
        return np.array(phi)

    def hyperparameters_update(
            self, cutpoints=None, varphi=None, variance=None,
            noise_variance=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. 

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
        if varphi is not None:
            self._model.kernel.lengthscales.assign(varphi)
        if variance is not None:
            self._model.kernel.variance.assign(variance)
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
            self._model.likelihood.sigma.assign(self.noise_std)
        # Handle a possible case where the kernel has hyperhyperparameters

    def grid_over_hyperparameters(
            self, domain, res, trainables=None, posterior_mean_0=None,
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
        intervals, trainables_where) = self._grid_over_hyperparameters_initiate(
            res, domain, trainables, self.cutpoints)
        error = np.inf
        fx_old = np.inf
        for i, phi in enumerate(Phi_new):
            self._grid_over_hyperparameters_update(
                phi, trainables, self.cutpoints)
            # Reset error and posterior mean
            fx, gx = self.approximate_posterior(
                phi, trainables, verbose=False)
            fxs[i] = fx
            gxs[i] = gx[trainables_where]
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
                    callback=self._monitor, options=dict(max_iter=steps))
        else:
            self._optimizer.minimize(
                self._training_loss, self._model.trainable_variables,
                options=dict(maxiter=steps))

    def approximate_posterior(
            self, phi, trainables, steps=None, first_step=None,
            return_reparameterised=False, verbose=False):
        """
        Optimisation routine for hyperparameters.

        :arg phi: (log-)hyperparameters to be optimised.
        :arg trainables:
        :arg first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx, gx
        :rtype: float, `:class:numpy.ndarray`
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, error, iteration, trainables_where,
                gx) = self._hyperparameter_training_step_initialise(
            phi, trainables)
        self.approximate(steps=steps, write=False)
        fx = self._training_loss().numpy()
        gx = 0
        posterior = self._model.posterior()
        if return_reparameterised is True:
            return fx, gx, posterior.q_mu, (posterior.q_sqrt, False)
        elif return_reparameterised is False:
            posterior_mean, posterior_covariance = posterior.predict_f(
                self.X_train, full_cov=True)
            return fx, gx, posterior_mean.numpy().flatten(), (
                posterior_covariance.numpy()[0], False)
        elif return_reparameterised is None:
            return fx, gx

    def predict(
            self, X_test, cov, weights, cached_posterior=None):
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
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        posterior = self._model.posterior()
        mu, var = posterior.predict_f(
            X_test)
        posterior_std = np.sqrt(var.numpy()).flatten()
        posterior_pred_mean = mu.numpy().flatten()
        print(posterior_std)
        print(posterior_pred_mean)
        predictive_distributions = np.empty((np.shape(X_test)[0], self.J))
        for j in range(self.J):
            Y_test = np.full((np.shape(X_test)[0], 1), j)
            predictive_distributions[:, j] = self._model.predict_log_density(
                (X_test, Y_test))
        predictive_distributions = np.exp(predictive_distributions)
        return predictive_distributions, posterior_pred_mean, posterior_std


class SVGP(VGP):
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
        Return a string representation of this class, used to import the class
        from the string.
        """
        return "SVGP"

    def __init__(
             self, M, minibatch_size=40, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`SVGP` Approximator object.

        :returns: A :class:`SVGP` object.
        """
        self.M = M
        super().__init__(*args, **kwargs)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_train, self.y_train.reshape(-1, 1))).repeat().shuffle(
                self.N)
        train_iter = iter(train_dataset.batch(minibatch_size))
        # Create an Adam Optimizer action
        self._train_iter = iter(train_dataset.batch(minibatch_size))
        self._training_loss = self._model.training_loss_closure(
            train_iter, compile=True)
        self._optimizer = tf.optimizers.Adam()

    def _model_initiate(self, cutpoints):
        """Update prior covariances."""
        inducing_idx = np.random.choice(
            self.X_train.shape[0], size=self.M, replace=False)
        self.Z = self.X_train[inducing_idx, :]
        likelihood = gpflow.likelihoods.Ordinal(cutpoints[1:-1])
        warnings.warn(
            "Initiating model using gpflow")
        self._model = gpflow.models.SVGP(
            kernel=self.kernel, likelihood=likelihood,
            inducing_variable=self.Z, num_data=self.N)
        def plot_prediction(fig, ax):
            Xnew = np.linspace(
                self.X_train.min() - 0.5,
                self.X_train.max() + 0.5, 100).reshape(-1, 1)
            Ypred = self._model.predict_f_samples(
                Xnew, full_cov=True, num_samples=20)
            ax.plot(Xnew.flatten(), np.squeeze(Ypred).T, "C1", alpha=0.2)
            ax.plot(self.X_train, self.y_train, "o")
        warnings.warn("Done initiating model using gpflow.")
        # Fix inducing variables
        gpflow.set_trainable(self._model.inducing_variable, False)
        # Fix hyperparameters
        gpflow.set_trainable(self._model.kernel.lengthscales, False)
        gpflow.set_trainable(self._model.kernel.variance, False)
        # bin_edges are not trainable in GPFlow!
        # gpflow.set_trainable(self._model.likelihood.bin_edges, False)
        gpflow.set_trainable(self._model.likelihood.sigma, False)
        log_dir_scipy = "scipy"
        model_task = ModelToTensorBoard(
            log_dir_scipy, self._model)
        lml_task = ScalarToTensorBoard(
            log_dir_scipy, lambda: self._training_loss,
            "training_objective")
        image_task = ImageToTensorBoard(
            log_dir_scipy, plot_prediction, "image_samples")
        # Plotting tasks can be quite slow. We want to run them less
        # frequently.
        # We group them in a `MonitorTaskGroup` and set the period to 5.
        slow_tasks = MonitorTaskGroup(image_task, period=5)
        # The other tasks are fast. We run them at each iteration of the
        # optimisation.
        fast_tasks = MonitorTaskGroup([model_task, lml_task], period=1)
        # Both groups are passed to the monitor.
        # `slow_tasks` will be run five times less frequently than
        # `fast_tasks`.
        self._monitor = Monitor(fast_tasks)

    @tf.function
    def optimization_step(self):
            self._optimizer.minimize(
                self._training_loss, self._model.trainable_variables)

    def _approximate_initiate(self, steps):
        return reduce_in_tests(steps), []

    def approximate(
            self, steps=None, first_step=1, write=False):
        steps, fxs = self._approximate_initiate(steps)
        for step in trange(first_step, first_step + steps,
                    desc="SVGP approximator progress",
                    unit="iterations", disable=True):
            self.optimization_step()
            if write:
                self._monitor(step)
            if step % 10 == 0:
                fx = self._training_loss().numpy()
                fxs.append(fx)
        fxs = np.array(fxs)

        plt.plot(np.arange(steps)[::10], fxs)
        plt.xlabel("iteration")
        _ = plt.ylabel("ELBO")
        plt.savefig("fx")
        plt.close()
        plt.show()
        return fxs
