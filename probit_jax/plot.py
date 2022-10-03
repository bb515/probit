"""Useful plot functions for classifiers."""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from probit_jax.data.utilities import colors, datasets, load_data_paper
import warnings
import time
from scipy.optimize import minimize
import matplotlib.colors as mcolors
from matplotlib import rc


BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4

def grid(classifier, X_trains, y_trains,
        domain, res, steps, now, trainables=None, verbose=False):
    """
    Grid of (optimised and converged) variational lower bound across the chosen
    hyperparameters.

    :arg int J: The number of ordinal classes.
    :arg Kernel:
    :type Kernel: :class:`Kernel`
    :arg X_trains:
    :type X_trains: :class:`numpy.ndarray`
    :arg y_trains:
    :type y_trains: :class:`numpy.ndarray`
    :arg domain:
    :type domain: (tuple, tuple) or (tuple, None)
    :arg res:
    :type res: (tuple, tuple) or (tuple, None)
    :arg int steps:
    :arg string now: A string for plots, can be e.g., a time stamp, and is
        to distinguish between plots.
    :arg cutpoints:
    :type cutpoints: :class:`numpy.ndarray` or None
    :arg theta:
    :type theta: :class:`numpy.ndarray` or float or None
    :arg noise_variance:
    :type noise_variance: float or None
    :arg variance:
    :type variance: float or None:
    :arg trainables:
    :type trainables: :class:`numpy.ndarray`
    """
    Z_av = []
    grad_av = []
    for split in range(5):
        # Reinitiate classifier with new data
        # TODO: not sure if this is good code, but makes sense conceptually
        # as new data requires a new model.
        classifier.__init__(
            classifier.cutpoints, classifier.noise_variance,
            classifier.kernel, X_trains[split], y_trains[split], classifier.J)

        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        phis, fxs,
        gxs) = _grid_over_hyperparameters_initiate(
            res, domain, trainables, classifier.J)
        for i, phi in enumerate(phis):
            print(phi)
            fx, gx = classifier.approximate_posterior(phi, trainables, steps)
            fxs[i] = fx
            gxs[i] = gx
            if verbose:
                print(
                "\ncutpoints={}, theta={}, noise_variance={}, variance={},"
                "\nfunction_eval={}, \nfunction_grad={}".format(
                    classifier.cutpoints, classifier.kernel.theta,
                    classifier.noise_variance, classifier.kernel.variance,
                    fx, gxs[i]))
        # TODO: tidy up this code
        if x2s is not None:
            (Z, grad,
            x, y,
            xlabel, ylabel,
            xscale, yscale) = (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                    xlabel, ylabel, xscale, yscale)
        else:
            (Z, grad,
            x, y,
            xlabel, ylabel,
            xscale, yscale) = (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

        Z_av.append(Z)
        grad_av.append(grad)
        if ylabel is None:
            # norm = np.abs(np.max(grad))
            # grad = grad / norm  # normalise - all the same length
            index = np.argmin(Z)
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.scatter(
                x[index], Z[index],
                color='red', label=r"$\theta$ = {}".format(x[index]))
            ax.legend()
            ax.plot(x, Z)
            ax.set_xscale(xscale)
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$\mathcal{F}(\theta)$")
            fig.savefig(
                "bound_{}_{}.png".format(split, now),
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.plot(x, grad)
            ax.set_xscale(xscale)
            ax.set_xlabel(xlabel)
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"\frac{\partial \mathcal{F}}{\partial theta}")
            fig.savefig(
                "grad_{}_{}.png".format(split, now),
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()
        else:
            fig, axs = plt.subplots(1, figsize=(6, 6))
            ax = plt.axes(projection='3d')
            ax.plot_surface(x, y, Z, rstride=1, cstride=1, alpha=FG_ALPHA,
                            cmap='viridis', edgecolor='none')
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            fig.savefig("grid_over_hyperparameters_{}_{}.png".format(
                split, now), facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

            norm = np.linalg.norm(np.array((grad[:, 0], grad[:, 1])), axis=0)
            u = grad[:, 0] / norm
            v = grad[:, 1] / norm
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect(1)
            ax.contourf(x, y, Z, 100, cmap='viridis', zorder=1)
            ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(0.1, 10.0)
            ax.set_ylim(0.1, 10.0)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            fig.savefig("Contour plot - lower bound on the log "
                "likelihood_{}_{}.png".format(split, now),
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

    Z_av = np.mean(np.array(Z_av), axis=0)
    grad_av = np.mean(np.array(grad_av), axis=0)
    if ylabel is None:
        # norm = np.abs(np.max(grad))
        # u = grad / norm
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.plot(x, Z_av)
        ax.set_xscale(xscale)
        ax.set_ylabel(r"$\mathcal{F}(\theta)$")
        ax.set_xlabel(r"$\theta$")
        index = np.argmin(Z_av)
        ax.scatter(
            x[index], Z_av[index],
            color='red', label=r"\theta = {}".format(x[index]))
        ax.legend()
        fig.savefig("bound_av_{}.png".format(now),
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.plot(x, grad_av)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial theta}$")
        fig.savefig("grad_av_{}.png".format(now),
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
    else:
        fig, axs = plt.subplots(1, figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, Z_av, rstride=1, cstride=1, alpha=FG_ALPHA,
                        cmap='viridis', edgecolor='none')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        fig.savefig("grid_over_hyperparameters_av_{}.png".format(now),
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        norm = np.linalg.norm(np.array((grad_av[:, 0], grad_av[:, 1])), axis=0)
        u = grad_av[:, 0] / norm
        v = grad_av[:, 1] / norm
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.contourf(x, y, Z_av, 100, cmap='viridis', zorder=1)
        ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        ax.set_xlim(0.1, 10.0)
        ax.set_ylim(0.1, 10.0)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        fig.savefig("Contour plot - lower bound on the log "
            "likelihood_av_{}.png".format(now),
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()


def plot_contour(
    classifier, X_test, t_test, steps, cov, weights,
    domain):
    """
    Given classifier and approximate posterior mean and covariance,
    save a contour plot of predictive probabilities over the domain.
    """
    (x_lims, y_lims) = domain
    N = 75
    x1 = np.linspace(x_lims[0], x_lims[1], N)
    x2 = np.linspace(y_lims[0], y_lims[1], N)
    xx, yy = np.meshgrid(x1, x2)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, 2))
    X_new_ = np.zeros((N * N, classifier.D))
    X_new_[:, :2] = X_new
    (Z,
    posterior_predictive_m,
    posterior_std) = classifier.predict(
        X_new_, cov, weights)
    Z_new = Z.reshape((N, N, classifier.J))
    print(np.sum(Z, axis=1), 'sum')
    for j in range(classifier.J):
        fig, axs = plt.subplots(1, figsize=(6, 6))
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.contourf(x1, x2, Z_new[:, :, j], zorder=1)
        ax.scatter(
            classifier.X_train[np.where(classifier.y_train == j)][:, 0],
            classifier.X_train[np.where(classifier.y_train == j)][:, 1],
            color='red')
        ax.scatter(
            X_test[np.where(t_test == j)][:, 0],
            X_test[np.where(t_test == j)][:, 1], color='blue')
        # ax.scatter(
        #   X_train[np.where(t == j + 1)][:, 0],
        #   X_train[np.where(t == j + 1)][:, 1],
        #   color='blue')
        # ax.set_xlim(0, 2)
        # ax.set_ylim(0, 2)
        ax.set_xlabel(r"$x_1$", fontsize=16)
        ax.set_ylabel(r"$x_2$", fontsize=16)
        fig.savefig(
            "contour_{}.png".format(j),
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()


def calculate_metrics(y_test, t_test, Z, cutpoints):
    """Calculate nPlan metrics and return a big tuple containing them."""
    t_pred = np.argmax(Z, axis=1)
    print("t_pred")
    print(t_pred)
    print("t_test")
    print(t_test)
    grid = np.ogrid[0:len(t_test)]
    # Other error
    predictive_likelihood = Z[grid, t_test]
    mean_absolute_error = np.sum(np.abs(t_pred - t_test)) / len(t_test)
    root_mean_squared_error = np.sqrt(
        np.sum(pow(t_pred - t_test, 2)) / len(t_test))
    print("root_mean_squared_error", root_mean_squared_error)
    print("mean_absolute_error ", mean_absolute_error)
    log_predictive_probability = np.sum(np.log(predictive_likelihood))
    print("log_pred_probability ", log_predictive_probability)
    predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
    print("predictive_likelihood ", predictive_likelihood)
    print("av_prob_of_correct ", predictive_likelihood)
    print(np.sum(t_pred != t_test), "sum incorrect")
    mean_zero_one = np.sum(t_pred != t_test) / len(t_test)
    print("mean_zero_one_error", mean_zero_one)
    print(np.sum(t_pred == t_test), "sum correct")
    mean_zero_one = np.sum(t_pred == t_test) / len(t_test)
    print("mean_zero_one_correct", mean_zero_one)
    return (
        mean_zero_one,
        root_mean_squared_error,
        mean_absolute_error,
        log_predictive_probability,
        predictive_likelihood)


def test(classifier, X_test, t_test, y_test, steps):
    (fx, gx,
        weights, (cov, is_reparameterised)
        ) = classifier.approximate_posterior(
                None, None, steps,
                return_reparameterised=True, verbose=True)
    (Z, posterior_predictive_m, posterior_std) = classifier.predict(
        X_test, cov, weights)
    return calculate_metrics(y_test, t_test, Z, classifier.cutpoints)


def run_adam(classifier, iterations):
    """
    Utility function running the Adam optimizer

    :param classifier: GPflow model
    :param interations: number of iterations
    """
    import tensorflow as tf
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf


def plot(classifier, X_test, title=""):
    """Plot from GPFlow documentation for ordinal regression."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.imshow(
        classifier.predict(X_test),
        interpolation="nearest",
        extent=[X_test.min(), X_test.max(), -0.5, classifier.J + 0.5],
        origin="lower",
        aspect="auto",
        cmap=plt.cm.viridis,
    )
    ax.colorbar()
    ax.plot(
        classifier.X_train, classifier.y_train, "kx",
        mew=2, scalex=False, scaley=False)
    fig.savefig("test", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    plt.close()

    # Predictive density for a single input x=0.5
    x_new = 0.5
    Y_new = np.arange(np.max(Y + 1)).reshape([-1, 1])
    X_new = np.full_like(Y_new, x_new)
    # for predict_log_density x and y need to have the same number of rows
    dens_new = np.exp(classifier._model.predict_log_density((X_new, Y_new)))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    plt.bar(x=Y_new.flatten(), height=dens_new.flatten())
    fig.savefig("test2", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    plt.title(title)
    pY, pYv = classifier._model.predict_y(X_test)  # Predict Y values at test locations
    ax.plot(classifier.X_train, classifier.y_train, "x", label="Training points", alpha=MG_ALPHA)
    (line,) = ax.plot(X_test, pY, lw=1.5, label="Mean of predictive posterior")
    col = line.get_color()
    ax.fill_between(
        X_test[:, 0],
        (pY - 2 * pYv ** 0.5)[:, 0],
        (pY + 2 * pYv ** 0.5)[:, 0],
        color=col,
        alpha=0.6,
        lw=1.5,
    )
    Z = classifier._model.inducing_variable.Z.numpy()
    ax.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
    ax.legend(loc="lower right")
    plt.show()
    fig.savefig("test3", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()


def train(classifier, method, trainables, verbose=True, steps=None, max_sec=5000):
    """
    Hyperparameter training via gradient descent of the objective function, a
    negative log marginal likelihood (or bound thereof).

    :arg classifier:
    :type classifier: :class:`probit.estimators.Approximator`
        or :class:`probit.samplers.Sampler`
    :arg str method: "CG" or "L-BFGS-B" seem to be widely used and work well.
    :arg trainables: Binary array or list of trainables indicating which
        hyperparameters to fix, and which to optimize.
    :type trainables: :class:`numpy.ndarray` or list
    :arg bool verbose: Verbosity bool, default True
    :arg int steps: The number of steps to run the algorithm on the inner loop.
    :arg float max_sec: The max time to do optimization for, in seconds.
        TODO: test.
    :return: The hyperparameters after optimization.
    :rtype: tuple (
        :class:`numpy.ndarray`, float or :class:`numpy.ndarray`, float, float)
    """
    minimize_stopper = MinimizeStopper(max_sec=max_sec)
    phi = classifier.get_phi(trainables)
    args = (trainables, steps, verbose)
    res = minimize(
        classifier.approximate_posterior, phi,
        args, method=method, jac=True,
        callback=minimize_stopper.__call__)
    return classifier


def save_model(
    classifier, model_file, metadata_file, steps, optimizer_method_string):
    """
    Approximate posterior to obtain model weights and save model.

    :arg classifier:
    :arg model_file:
    :arg metadata_file:
    :arg steps: Included as an argument as not a class variable of classifier.
    :arg optimizer_method_string: Included as an argument as not a class variable of classifier.

    """
    (fx, gx,
        posterior_mean, (posterior_inv_cov, is_reparametrised)
        ) = classifier.approximate_posterior(
                None, None, steps,
                return_reparameterised=True, verbose=True)
    np.savez(
        model_file, fx=fx, gx=gx, posterior_mean=posterior_mean,
        posterior_inv_cov=posterior_inv_cov, is_reparametrised=is_reparametrised)

    np.savez(
            metadata_file, kernel_string=repr(classifier.kernel),
            approximator_string=repr(classifier),
            optimizer_method_string=optimizer_method_string,
            N_train=classifier.N,
            theta=classifier.kernel.theta,
            theta_hyperparameters=classifier.kernel.theta_hyperparameters,
            theta_hyperhyperparameters=classifier.kernel.theta_hyperhyperparameters,
            signal_variance=classifier.kernel.variance,
            cutpoints=classifier.cutpoints,
            noise_variance=classifier.noise_variance,
            num_ordinal_classes=classifier.J)


#SS
# def _test_given_posterior(classifier, X_test, t_test, y_test, steps, posterior):
#     """
#     Test the trained model.

#     :arg classifier:
#     :type classifier: :class:`probit.approximator.Approximator`
#         or :class:`probit.samplers.Sampler`
#     """
#     if posterior_mean is not None and posterior_inv_cov is not None:
#         (fx, gx,
#         posterior_mean, (posterior_inv_cov, is_reparametrised)
#         ) = classifier.approximate_posterior(
#                 None, None, steps,
#                 return_reparameterised=True, verbose=True)
#         # Test
#         (Z,
#         posterior_predictive_m,
#         posterior_std) = classifier.predict(
#             X_test, cov, weights)
#     return posterior_inv_cov, posterior_mean, calculate_metrics(y_test, t_test, Z, classifier.cutpoints)
 

def outer_loop_problem_size(
        test, Approximator, Kernel, method, X_trains, y_trains, X_tests, t_tests,
        y_tests, steps,
        cutpoints_0, theta_0, noise_variance_0, scale_0, J, D, size, num,
        string="VB"):
    """
    Plots outer loop for metrics and variational lower bound over N_train
    problem size.

    :arg test:
    :type test:
    :arg Approximator:
    :type Approximator:
    :arg Kernel:
    :type Kernel:
    :arg method:
    :type method:
    :arg X_trains:
    :type X_trains:
    :arg y_trains:
    :type y_trains:
    :arg X_tests:
    :type X_tests:
    :arg  t_tests:
    :type t_tests:
    :arg y_tests:
    :type y_tests:
    :arg steps:
    :arg cutpoints_0
    :type cutpoints_0:
    :arg theta_0:
    :type theta_0:
    :arg noise_variance_0:
    :type noise_variance_0:
    :arg variance_0:
    :type variance_0:
    :arg J:
    :type J:
    :arg D:
    :type D:
    :arg size:
    :type size:
    :arg num:
    :type num:
    :arg str string: Title for plots, default="client".
    """
    plot_N = []
    plot_mean_fx = []
    plot_mean_metrics = []
    plot_std_fx = []
    plot_std_metrics = []
    for iter, N in enumerate(np.logspace(1, size, num=num)):
        N = int(N)
        print("iter {}, N {}".format(iter, N))
        mean_fx, std_fx, mean_metrics, std_metrics= outer_loops(
            test, Approximator, Kernel, method,
            X_trains[:, :N, :], y_trains[:, :N],
            X_tests, t_tests, y_tests, steps,
            cutpoints_0, theta_0, noise_variance_0, variance_0, J, D)
        plot_N.append(N)
        plot_mean_fx.append(mean_fx)
        plot_std_fx.append(std_fx)
        plot_mean_metrics.append(mean_metrics)
        plot_std_metrics.append(std_metrics)
    plot_N = np.array(plot_N)
    plot_mean_fx = np.array(plot_mean_fx)
    plot_std_fx = np.array(plot_std_fx)
    plot_mean_metrics = np.array(plot_mean_metrics)
    plot_std_metrics = np.array(plot_std_metrics)

    print(plot_mean_metrics)
    print(np.shape(plot_mean_metrics))

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_fx, '-', color='gray',
        label="variational bound +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_fx - plot_std_fx, plot_mean_fx + plot_std_fx, 
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} fx.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 13], '-', color='gray',
        label="RMSE +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 13] - plot_std_metrics[:, 13],
        plot_mean_metrics[:, 13] + plot_std_metrics[:, 13],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} RMSE.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 14], '-', color='gray',
        label="MAE +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 14] - plot_std_metrics[:, 14],
        plot_mean_metrics[:, 14] + plot_std_metrics[:, 14],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} MAE.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 15], '-', color='gray',
        label="log predictive probability +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 15] - plot_std_metrics[:, 15],
        plot_mean_metrics[:, 15] + plot_std_metrics[:, 15],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} log predictive probability.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 4], '-', color='gray',
        label="mean zero-one accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 4] - plot_std_metrics[:, 4],
        plot_mean_metrics[:, 4] + plot_std_metrics[:, 4],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} mean zero-one accuracy.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 4], '-', color='black',
        label="in top 1 accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 4] - plot_std_metrics[:, 4],
        plot_mean_metrics[:, 4] + plot_std_metrics[:, 4],
        color='black', alpha=MG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 5], '-', color='gray',
        label="in top 3 accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 5] - plot_std_metrics[:, 5],
        plot_mean_metrics[:, 5] + plot_std_metrics[:, 5],
        color='gray', alpha=MG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 6], '-', color='lightgray',
        label="in top 5 accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 6] - plot_std_metrics[:, 6],
        plot_mean_metrics[:, 6] + plot_std_metrics[:, 6],
        color='lightgray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} in top accuracy.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 7], '-', color='black',
        label="distance 1 accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 7] - plot_std_metrics[:, 7],
        plot_mean_metrics[:, 7] + plot_std_metrics[:, 7],
        color='black', alpha=MG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 8], '-', color='gray',
        label="distance 3 accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 8] - plot_std_metrics[:, 8],
        plot_mean_metrics[:, 8] + plot_std_metrics[:, 8],
        color='gray', alpha=MG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 9], '-', color='lightgray',
        label="distance 5 accuracy +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 9] - plot_std_metrics[:, 9],
        plot_mean_metrics[:, 9] + plot_std_metrics[:, 9],
        color='lightgray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} distance accuracy.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 0], '-', color='gray',
        label="f1 score +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 0] - plot_std_metrics[:, 0],
        plot_mean_metrics[:, 0] + plot_std_metrics[:, 0],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} f1 score.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 1], '-', color='gray',
        label="uncertainty plus +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 1] - plot_std_metrics[:, 1],
        plot_mean_metrics[:, 1] + plot_std_metrics[:, 1],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} uncertainty plus.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.plot(plot_N, plot_mean_metrics[:, 2], '-', color='gray',
        label="uncertainty minus +/- 1 std")
    ax.fill_between(
        plot_N, plot_mean_metrics[:, 2] - plot_std_metrics[:, 2],
        plot_mean_metrics[:, 2] + plot_std_metrics[:, 2],
        color='gray', alpha=MG_ALPHA)
    ax.set_xscale("log")
    ax.legend()
    fig.savefig("{} uncertainty minus.png".format(string),
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    np.savez(
            "{} plot.npz".format(string),
            plot_N=plot_N, plot_mean_fx=plot_mean_fx, plot_std_fx=plot_std_fx,
            plot_mean_metrics=plot_mean_metrics,
            plot_std_metrics=plot_std_metrics)
    return 0


def outer_loops(
        test, Approximator, Kernel, method, X_trains, y_trains, X_tests, t_tests,
        y_tests, steps, cutpoints_0, theta_0, noise_variance_0, variance_0, J, D):
    moments_fx = []
    #moments_theta = []
    #moments_noise_variance = []
    #moments_cutpoints = []
    moments_metrics = []
    for split in range(1):
        # Reset kernel
        kernel = Kernel(theta=theta_0, variance=variance_0)
        # Build the classifier with the new training data
        classifier = Approximator(
            cutpoints_0, noise_variance_0, kernel, J,
            (X_trains[split, :, :], y_trains[split, :]))
        fx, metrics = test(
            classifier,
            X_tests[split, :, :], t_tests[split, :],
            y_tests[split, :],
            steps)
        moments_fx.append(fx / classifier.N)  # if divided by N it is average per datapoint
        moments_metrics.append(metrics)
        # moments_theta.append(classifier.theta)
        # moments_noise_variance.append(classifier.noise_variance)
        # moments_cutpoints.append(classifier.cutpoints[1:-1])
    moments_fx = np.array(moments_fx)
    moments_metrics = np.array(moments_metrics)
    mean_fx = np.average(moments_fx)
    mean_metrics = np.average(moments_metrics, axis=0)
    std_fx = np.std(moments_fx)
    std_metrics = np.std(moments_metrics, axis=0)
    return mean_fx, std_fx, mean_metrics, std_metrics


def outer_loops_Rogers(
        test, Approximator, Kernel, X_trains, y_trains, X_tests, t_tests,
        y_tests,
        cutpoints_0, theta_0, noise_variance_0, variance_0, J, D, plot=False):
    steps = 50
    grid = np.ogrid[0:len(X_tests[0, :, :])]
    moments_fx_Z = []
    moments_metrics_Z = []
    lower_x1 = -10
    lower_x2 = -1
    upper_x1 = 0
    upper_x2 = 6
    N = 20
    x1 = np.logspace(lower_x1, upper_x1, N)
    x2 = np.logspace(lower_x2, upper_x2, N)
    xx, yy = np.meshgrid(x1, x2)
    X_new = np.dstack((xx, yy))
    X_new = X_new.reshape((N * N, 2))
    for split in range(20):
        # Reset kernel
        kernel = Kernel(theta=theta_0, variance=variance_0)
        # Build the classifier with the new training data
        classifier = Approximator(
            cutpoints_0, noise_variance_0, kernel, J,
            (X_trains[split, :, :], y_trains[split, :, :]))
        X_test = X_tests[split, :, :]
        t_test = t_tests[split, :]
        y_test = y_tests[split, :]
        # Outer loop
        fx_Z = []
        metrics_Z = []
        for x_new in X_new:
            noise_std = x_new[0]
            noise_variance = noise_std**2
            theta = x_new[1]
            classifier.hyperparameters_update(
                noise_variance=noise_variance, theta=theta)
            (fx, metrics) = test(
            classifier,
            X_test, t_test, y_test, steps)
            fx_Z.append(fx)
            metrics_Z.append(metrics)
        moments_metrics_Z.append(metrics_Z)
        moments_fx_Z.append(fx_Z)
        fx_Z = np.array(fx_Z)
        metrics_Z = np.array(metrics_Z)
        max_fx = np.max(fx_Z)
        max_metrics = np.max(metrics, axis=1)
        min_metrics = np.min(metrics, axis=1)
        argmax_fx = np.argmax(fx_Z)
        argmax_metrics = np.argmax(metrics, axis=1)

        fx_Z = fx_Z.reshape((N, N))
        metrics_Z = metrics_Z.reshape((N, N, np.shape(metrics_Z)[1]))
        if plot==True:
            fig, ax = plt.subplots(1, figsize=(6, 6))
            ax.contourf(x1, x2, predictive_likelihood_Z)
            ax.scatter(
                X_new[argmax_predictive_likelihood, 0],
                X_new[argmax_predictive_likelihood, 1], c='r')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\log{\theta}$", fontsize=16)
            ax.set_ylabel(r"$\log{s}$", fontsize=16)
            fig.savefig("Contour plot - Predictive likelihood of test set.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

            fig, axs = plt.subplots(1, figsize=(6, 6))
            ax.contourf(x1, x2, bounds_Z)
            ax.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\log{\theta}$", fontsize=16)
            ax.set_ylabel(r"$\log{s}$", fontsize=16)
            fig.savefig("Contour plot - Variational lower bound.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

            fig, axs = plt.subplots(1, figsize=(6, 6))
            ax.contourf(x1, x2, zero_one_Z)
            ax.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\log{\theta}$", fontsize=16)
            ax.set_ylabel(r"$\log{s}$", fontsize=16)
            ax.savefig("Contour plot - mean zero-one accuracy.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

    moments_metrics_Z = np.array(moments_metrics_Z)
    moments_fx_Z = np.array(moments_fx_Z)
    mean_metrics_Z = np.average(moments_metrics_Z, axis=1)
    std_metrics_Z = np.std(moments_metrics_Z, axis=1)
    mean_fx_Z = np.average(moments_fx_Z)
    std_fx_Z = np.std(moments_fx_Z)
    mean_max_metrics = np.average(max_metrics)
    std_max_metrics = np.std(max_metrics)
    mean_max_fx = np.average(max_fx)
    std_max_fx = np.std(max_fx)
    argmax_fx = np.argmax(mean_fx_Z)
    argmax_metrics = np.argmax(mean_metrics_Z, axis=1)
    max_fx = np.max(mean_fx_Z)
    max_metrics = np.max(mean_metrics_Z)
    min_metrics = np.min(mean_metrics_Z)
    argmin_metrics = np.min(mean_metrics_Z, axis=1)

    mean_fx_Z = mean_fx_Z.reshape((N, N))
    mean_metrics_Z = mean_metrics_Z.reshape((N, N, np.shape(mean_metrics_Z[1])))

    # fig, axs = plt.subplots(1, figsize=(6, 6))
    # ax.contourf(x1, x2, mean_metrics_Z[0])
    # ax.scatter(X_new[argmax_metrics[0], 0], X_new[argmax_metrics[0], 1], c='r')
    # axs.set_xscale('log')
    # axs.set_yscale('log')
    # ax.set_xlabel(r"$\log{\theta}$", fontsize=16)
    # ax.set_ylabel(r"$\log{s}$", fontsize=16)
    # fig.savefig("Contour plot - Predictive likelihood of test set.png")
    # plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    ax.contourf(x1, x2, mean_fx_Z)
    ax.scatter(X_new[argmax_fx, 0], X_new[argmax_fx, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    ax.set_xlabel(r"$\log{\theta}$", fontsize=16)
    ax.set_ylabel(r"$\log{s}$", fontsize=16)
    fig.savefig("Contour plot - Variational lower bound.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def _grid_over_hyperparameters_initiate(
        classifier, res, domain, trainables):
    """
    Initiate metadata and hyperparameters for plotting the objective
    function surface over hyperparameters.

    :arg int res:
    :arg domain: ((2,)tuple, (2.)tuple) of the start/stop in the domain to
        grid over, for each axis, like: ((start, stop), (start, stop)).
    :type domain:
    :arg trainables: Indicator array of the hyperparameters to sample over.
    :type trainables: :class:`numpy.ndarray`
    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    """
    index = 0
    label = []
    axis_scale = []
    space = []
    phi_space = []
    phi_0 = classifier.get_phi(trainables)
    theta_0 = []
    if trainables[0]:
        # Grid over noise_std
        label.append(r"$\sigma$")
        axis_scale.append("log")
        theta = np.logspace(domain[index][0], domain[index][1], res[index])
        space.append(theta)
        phi_space.append(np.log(theta))
        theta_0.append(np.exp(phi_0[index]))
        index += 1
    if trainables[1]:
        # Grid over b_1, the first cutpoint
        label.append(r"$b_{}$".format(1))
        axis_scale.append("linear")
        theta = np.linspace(domain[index][0], domain[index][1], res[index])
        space.append(theta)
        phi_space.append(theta)
        theta_0.append(phi_0[index])
        index += 1
    for j in range(2, classifier.J):
        if trainables[j]:
            # Grid over b_j - b_{j-1}, the differences between cutpoints
            label.append(rf"$b_{ {j} } - b_{ {j-1} }$")
            axis_scale.append("log")
            theta = np.logspace(
                domain[index][0], domain[index][1], res[index])
            space.append(theta)
            phi_space.append(np.log(theta))
            theta_0.append(np.exp(phi_0[index]))
            index += 1
    if trainables[classifier.J]:
        # Grid over signal variance
        label.append(r"$\sigma_{\theta}^{ 2}$")
        axis_scale.append("log")
        theta = np.logspace(domain[index][0], domain[index][1], res[index])
        space.append(theta)
        phi_space.append(np.log(theta))
        theta_0.append(np.exp(phi_0[index]))
        index += 1
    if classifier.kernel._ARD is True:
        # In this case, then there is a scale parameter,
        #  the first cutpoint, the interval parameters,
        # and lengthvariances parameter for each dimension and class
        for d in range(classifier.D):
            if trainables[classifier.J + 1][d]:
                # Grid over kernel hyperparameter, theta in this dimension
                label.append(r"$\theta_{}$".format(d))
                axis_scale.append("log")
                theta = np.logspace(domain[index][0], domain[index][1], res[index])
                space.append(theta)
                phi_space.append(np.log(theta))
                theta_0.append(np.exp(phi_0[index]))
                index += 1
    else:
        if trainables[classifier.J + 1]:
            # Grid over only kernel hyperparameter, theta
            label.append(r"$\theta$")
            axis_scale.append("log")
            theta = np.logspace(domain[index][0], domain[index][1], res[index])
            space.append(theta)
            phi_space.append(np.log(theta))
            theta_0.append(np.exp(phi_0[index]))
            index +=1
    if index == 2:
        meshgrid_theta = np.meshgrid(space[0], space[1])
        meshgrid_phi = np.meshgrid(phi_space[0], phi_space[1])
        phis = np.dstack(meshgrid_phi)
        phis = phis.reshape((len(space[0]) * len(space[1]), 2))
        fxs = np.empty(len(phis))
        gxs = np.empty((len(phis), 2))
        theta_0 = np.array(theta_0)
    elif index == 1:
        meshgrid_theta = (space[0], None)
        space.append(None)
        phi_space.append(None)
        axis_scale.append(None)
        label.append(None)
        phis = phi_space[0].reshape(-1, 1)
        fxs = np.empty(len(phis))
        gxs = np.empty(len(phis))
        theta_0 = theta_0[0]
    else:
        raise ValueError(
            "Too few or too many independent variables to plot objective over!"
            " (got {}, expected {})".format(
            index, "1, or 2"))
    assert len(axis_scale) == 2
    assert len(meshgrid_theta) == 2
    assert len(space) ==  2
    assert len(label) == 2
    return (
        space[0], space[1],
        label[0], label[1],
        axis_scale[0], axis_scale[1],
        meshgrid_theta[0], meshgrid_theta[1],
        phis, fxs, gxs, theta_0, phi_0)


def grid_synthetic(
    classifier, domain, res, steps, trainables, show=False, verbose=False):
    """Grid of optimised lower bound across the hyperparameters with cutpoints set."""
    (x1s, x2s,
    xlabel, ylabel,
    xscale, yscale,
    xx, yy,
    phis, fxs,
    gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
        classifier, res, domain, trainables)
    for i, phi in enumerate(phis):
        fx, gx = classifier.approximate_posterior(
            phi, trainables, steps, verbose=verbose)
        fxs[i] = fx
        gxs[i] = gx

    if x2s is not None:
        (Z, grad,
        x, y,
        xlabel, ylabel,
        xscale, yscale) = (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                xlabel, ylabel, xscale, yscale)
    else:
        (Z, grad,
        x, y,
        xlabel, ylabel,
        xscale, yscale) = (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    if ylabel is None:
        #Normalization:
        #First derivatives: need to calculate them in the log domain if theta is in log domain
        if xscale == "log":
            log_x = np.log(x)
            dZ_ = np.gradient(Z, log_x)
            # Alternatively
            #dlog_x = np.diff(log_x)
            #dZ = np.diff(Z) / dlog_x
        elif xscale == "linear":
            dZ_ = np.gradient(Z, x)
            # Alternatively
            #dx = np.diff(x)
            #dZ = np.diff(Z) / dx
        idx_hat = np.argmin(Z)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(x, Z)
        fig.savefig("grid_over_hyperparameters.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(x, Z, 'b',  label=r"$\mathcal{F}}$")
        ylim = ax.get_ylim()
        ax.vlines(x[idx_hat], 0.99 * ylim[0], 0.99 * ylim[1], 'r',
            alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
        ax.vlines(theta_0, 0.99 * ylim[0], 0.99 * ylim[1], 'k',
            alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        ax.set_ylabel(r"$\mathcal{F}$")
        ax.legend()
        fig.savefig("bound.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(
            x, dZ_, 'r--',
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ numeric")
        #ax.set_ylim(ax.get_ylim())
        ax.plot(
            x, grad, 'b', alpha=0.7,
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic")
        # ax.vlines(theta_0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
        #     alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
        # ax.vlines(x[idx_hat], 0.9 * ylim[0], 0.9 * ylim[1], 'r',
        #     alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
        ax.legend()
        fig.savefig("grad.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()
 
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.plot(x, Z, 'b', label=r"$\mathcal{F}}$")
        # ax.plot(
        #     x, dZ_, 'r--',
        #     label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ numeric")
        #ax.set_ylim(ax.get_ylim())
        ax.set_xscale(xscale)
        ax.vlines(theta_0, 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k',
            alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
        ax.vlines(x[idx_hat], 0.9 * ylim[0], 0.9 * ylim[1], 'r',
            alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
        ax.plot(
            x, grad, 'r',
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic")
        ax.set_xlabel("log " + xlabel)
        ax.legend()
        ax.grid()
        plt.tight_layout()
        fig.savefig(
            "bound_grad.png", facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

    else:
        #First derivatives: need to calculate them in the log domain
        log_x = np.log(x1s)
        log_y = np.log(x2s)
        dlog_x = np.diff(log_x)
        dlog_y = np.diff(log_y)
        dZdlog_x = np.gradient(Z, log_x, axis=1)
        dZdlog_y = np.gradient(Z, log_y, axis=0)
        # d2Zlog_x2 = (Z[:, :-2] - 2 * Z[:, 1:-1] + Z[:, 2:]) / dlog_x ** 2
        # d2Zlog_y2 = (Z[:-2, :] - 2 * Z[1:-1, :] + Z[2:, :]) / dlog_y ** 2
        u_analytic_x = grad[:, 0].reshape((len(x1s), len(x2s)))
        u_numerical_x = dZdlog_x
        error_x = (u_numerical_x - u_analytic_x)
        u_analytic_y = grad[:, 1].reshape((len(x1s), len(x2s)))
        u_numerical_y = dZdlog_y
        error_y = (u_numerical_y - u_analytic_y)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, Z, rstride=1, cstride=1, alpha=FG_ALPHA,
                        cmap='viridis', edgecolor='none')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        plt.tight_layout()
        fig.savefig(
            "grid_over_hyperparameters.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        CS = ax.contourf(x, y, u_analytic_x, 100, cmap='viridis', zorder=1)
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        cbar = fig.colorbar(CS)
        fig.savefig(
            "grad analytic x.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        CS = ax.contourf(x, y, u_numerical_x, 100, cmap='viridis', zorder=1)
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        cbar = fig.colorbar(CS)
        fig.savefig(
            "grad numerical x.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        CS = ax.contourf(x, y, u_analytic_y, 100, cmap='viridis', zorder=1)
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        cbar = fig.colorbar(CS)
        fig.savefig(
            "grad analytic y.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        CS = ax.contourf(x, y, u_numerical_y, 100, cmap='viridis', zorder=1)
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        cbar = fig.colorbar(CS)
        fig.savefig(
            "grad numerical y.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        CS = ax.contourf(x, y, error_x, 100, cmap='viridis', zorder=1)
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        cbar = fig.colorbar(CS)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        fig.savefig(
            "grad error x.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        CS = ax.contourf(x, y, error_y, 100, cmap='viridis', zorder=1)
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        cbar = fig.colorbar(CS)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        fig.savefig(
            "grad error y.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        norm = np.linalg.norm(np.array((grad[:, 0], grad[:, 1])), axis=0)
        u = grad[:, 0] / norm
        v = grad[:, 1] / norm
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        ax.contourf(x, y, Z, 100, cmap='viridis', zorder=1)
        ax.quiver(x, y, u, v, units='xy', scale=0.1, color='red')
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_xlim((10**domain[0][0], 10**domain[0][1]))
        ax.set_ylim((10**domain[1][0], 10**domain[1][1]))
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        fig.savefig(
            "Contour plot - VB lower bound on the log likelihood.png",
            facecolor=fig.get_facecolor(), edgecolor='none')
        if show: plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.set_aspect(1)
        ax.contourf(x, y, Z, 100, cmap='viridis', zorder=1)
        ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        ax.scatter(theta_0[0], theta_0[1], c='k', s=45)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        plt.tight_layout()
        fig.savefig("Contour plot2 - VB lower bound on the log likelihood.png")
        if show: plt.show()
        plt.close()


def plot(classifier, steps, domain=None):
    """
    TODO: needs generalizing to other datasets other than Chu.
    Contour plot of the predictive probabilities over a chosen domain.
    """
    (fx, gx,
    weights, (cov, is_reparametrised)
    ) = classifier.approximate_posterior(
            None, None, steps, return_reparameterised=True,
            verbose=True)
    if domain is not None:
        (xlims, ylims) = domain
        N = 75
        x1 = np.linspace(xlims[0], xlims[1], N)
        x2 = np.linspace(ylims[0], ylims[1], N)
        xx, yy = np.meshgrid(x1, x2)
        X_new = np.dstack((xx, yy))
        X_new = X_new.reshape((N * N, 2))
        X_new_ = np.zeros((N * N, classifier.D))
        X_new_[:, :2] = X_new
        # Test
        (Z, posterior_predictive_mean, posterior_std) = classifier.predict(
            X_new_, cov, weights)
        Z_new = Z.reshape((N, N, classifier.J))
        print(np.sum(Z, axis=1), 'sum')
        for j in range(classifier.J):
            fig, ax = plt.subplots(1, figsize=(6, 6))
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.contourf(x1, x2, Z_new[:, :, j], zorder=1)
            ax.scatter(classifier.X_train[np.where(
                classifier.y_train == j)][:, 0],
                classifier.X_train[np.where(
                    classifier.y_train == j)][:, 1], color='red')
            ax.scatter(
                classifier.X_train[np.where(
                    classifier.y_train == j + 1)][:, 0],
                classifier.X_train[np.where(
                    classifier.y_train == j + 1)][:, 1], color='blue')
            ax.set_xlabel(r"$x_1$", fontsize=16)
            ax.set_ylabel(r"$x_2$", fontsize=16)
            plt.tight_layout()
            fig.savefig(
                "contour_{}.png".format(j),
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()
    return fx


def plot_synthetic(
    classifier, dataset, X_true, Y_true, steps, colors=colors):
    """
    Plots for synthetic data.

    TODO: needs generalizing to other datasets other than Chu.
    """
    (fx, gx,
    weights, (cov, is_reparametrised)
    ) = classifier.approximate_posterior(
            None, None, steps,
            return_reparameterised=True, verbose=True)

    if dataset in datasets["synthetic"]:
        if classifier.J == 3:
            if classifier.D == 1:
                x_lims = (-0.5, 1.5)
                N = 1000
                x = np.linspace(x_lims[0], x_lims[1], N)
                X_new = x.reshape((N, classifier.D))
                # Test
                (Z,
                posterior_predictive_m,
                posterior_std) = classifier.predict(
                    X_new, cov, weights)
                print(np.sum(Z, axis=1), 'sum')
                fig, ax = plt.subplots(1, 1)
                fig.patch.set_facecolor('white')
                fig.patch.set_alpha(BG_ALPHA)
                ax.set_xlim(x_lims)
                ax.set_ylim(0.0, 1.0)
                ax.set_xlabel(r"$x$", fontsize=16)
                ax.set_ylabel(r"$p(y|x, X, y)$", fontsize=16)
                ax.stackplot(x, Z.T,
                            labels=(
                                r"$p(y=0|x, X, y)$",
                                r"$p(y=1|x, X, y)$",
                                r"$p(y=2|x, X, y)$"),
                            colors=(
                                colors[0], colors[1], colors[2])
                            )
                val = 0.5  # where the data lies on the y-axis.
                for j in range(classifier.J):
                    ax.scatter(
                        classifier.X_train[np.where(classifier.y_train == j)],
                        np.zeros_like(classifier.X_train[np.where(
                            classifier.y_train == j)]) + val,
                        s=15, facecolors=colors[j], edgecolors='white')
                plt.tight_layout()
                fig.savefig(
                    "Cumulative distribution plot of ordinal class "
                    "distributions for x_new=[{}, {}].png".format(
                        x_lims[0], x_lims[1]),
                        facecolor=fig.get_facecolor(), edgecolor='none')
                plt.show()
                plt.close()

                np.savez(
                    "tertile.npz",
                    x=X_new, y=posterior_predictive_m, s=posterior_std)
                fig, ax = plt.subplots(1, 1)
                fig.patch.set_facecolor('white')
                fig.patch.set_alpha(BG_ALPHA)
                ax.plot(X_new, posterior_predictive_m, 'r')
                ax.fill_between(
                    X_new[:, 0], posterior_predictive_m - 2*posterior_std,
                    posterior_predictive_m + 2*posterior_std,
                    color='red', alpha=MG_ALPHA)
                ax.scatter(X_true, Y_true, color='b', s=4)
                ax.set_ylim(-2.2, 2.2)
                ax.set_xlim(-0.5, 1.5)
                for j in range(classifier.J):
                    ax.scatter(
                        classifier.X_train[np.where(classifier.y_train == j)],
                        np.zeros_like(classifier.X_train[
                            np.where(classifier.y_train == j)]),
                        s=15, facecolors=colors[j], edgecolors='white')
                plt.tight_layout()
                fig.savefig(
                    "Scatter plot of data compared to posterior mean.png",
                    facecolor=fig.get_facecolor(), edgecolor='none')
                plt.show()
                plt.close()

            elif classifier.D == 2:
                x_lims = (-4.0, 6.0)
                y_lims = (-4.0, 6.0)
                # x_lims = (-0.5, 1.5)
                # y_lims = (-0.5, 1.5)
                N = 200
                x = np.linspace(x_lims[0], x_lims[1], N)
                y = np.linspace(y_lims[0], y_lims[1], N)
                xx, yy = np.meshgrid(x, y)
                X_new = np.dstack([xx, yy]).reshape(-1, 2)
                # Test
                (Z,
                posterior_predictive_m,
                posterior_std) = classifier.predict(
                    X_new, cov, weights)
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(
                    X_new[:, 0].reshape(N, N),
                    X_new[:, 1].reshape(N, N),
                    Z.T[0, :].reshape(N, N), alpha=FG_ALPHA, color=colors[0],
                        label=r"$p(y=0|x, X, t)$")
                surf = ax.plot_surface(
                    X_new[:, 0].reshape(N, N),
                    X_new[:, 1].reshape(N, N),
                    Z.T[1, :].reshape(N, N), alpha=FG_ALPHA, color=colors[1],
                        label=r"$p(y=1|x, X, t)$")
                surf = ax.plot_surface(
                    X_new[:, 0].reshape(N, N),
                    X_new[:, 1].reshape(N, N),
                    Z.T[2, :].reshape(N, N), alpha=FG_ALPHA, color=colors[2],
                        label=r"$p(y=2|x, X, t)$")
                cmap = plt.cm.get_cmap('viridis', classifier.J)
                import matplotlib as mpl
                fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap))  # TODO: how to not normalize this
                for j in range(classifier.J):
                    ax.scatter3D(
                        classifier.X_train[np.where(classifier.y_train == j), 0],
                        classifier.X_train[np.where(classifier.y_train == j), 1],
                        Y_true[np.where(
                            classifier.y_train == j)],
                        s=15, facecolors=colors[j], edgecolors='white')
                plt.tight_layout()
                fig.savefig("surface.png",
                    facecolor=fig.get_facecolor(), edgecolor='none')
                plt.show()

                # ax.set_xlim(x_lims)
                # ax.set_ylim(y_lims)
                # ax.set_ylim(0.0, 1.0)
                # ax.set_xlabel(r"$x$", fontsize=16)
                # ax.set_ylabel(r"$p(y|x, X, y)$", fontsize=16)
                # ax.stackplot(x, Z.T,
                #             labels=(
                #                 r"$p(y=0|x, X, t)$",
                #                 r"$p(y=1|x, X, t)$",
                #                 r"$p(y=2|x, X, t)$"),
                #             colors=(
                #                 colors[0], colors[1], colors[2])
                #             )
                # plt.tight_layout()
                # fig.savefig(
                #     "Ordered Gibbs Cumulative distribution plot of class "
                #     "distributions for x_new=[{}, {}].png".format(
                #         x_lims[0], x_lims[1]))
                # plt.show()
                # plt.close()
                def colorTriangle(r,g,b):
                    image = np.stack([r,g,b],axis=2)
                    return image/image.max(axis=2)[:,:,None]
                ax.imshow(
                    colorTriangle(
                        Z.T[0, :].reshape(N, N), Z.T[1, :].reshape(N, N),
                        Z.T[2, :].reshape(N, N)),
                        origin='lower',extent=(-4.0,6.0,-4.0,6.0))
                fig.savefig("color_triangle.png",
                    facecolor=fig.get_facecolor(), edgecolor='none')
                plt.show()
                plt.close()
        else:
            x_lims = (-0.5, 1.5)
            N = 1000
            x = np.linspace(x_lims[0], x_lims[1], N)
            X_new = x.reshape((N, classifier.D))
            # Test
            (Z,
            posterior_predictive_m,
            posterior_std) = classifier.predict(
                X_new, cov, weights)
            print(np.sum(Z, axis=1), 'sum')
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.set_xlim(x_lims)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(r"$x$", fontsize=16)
            ax.set_ylabel(r"$p(y_{*}={}|x, X, y)$", fontsize=16)
            if classifier.J == 13:
                plt.stackplot(x, Z.T,
                                labels=(
                                r"$p(y_{*}=0|x, X, y)$",
                                r"$p(y_{*}=1|x, X, y)$",
                                r"$p(y_{*}=2|x, X, y)$",
                                r"$p(y_{*}=3|x, X, y)$",
                                r"$p(y_{*}=4|x, X, y)$",
                                r"$p(y_{*}=5|x, X, y)$",
                                r"$p(y_{*}=6|x, X, y)$",
                                r"$p(y_{*}=7|x, X, y)$",
                                r"$p(y_{*}=8|x, X, y)$",
                                r"$p(y_{*}=9|x, X, y)$",
                                r"$p(y_{*}=10|x, X, y)$",
                                r"$p(y_{*}=11|x, X, y)$",
                                r"$p(y_{*}=12|x, X, y)$"),
                            colors=colors
                            )
            else:
                ax.stackplot(x, Z.T, colors=colors)
            val = 0.5  # where the data lies on the y-axis.
            for j in range(classifier.J):
                ax.scatter(
                    classifier.X_train[np.where(classifier.y_train == j)],
                    np.zeros_like(
                        classifier.X_train[np.where(
                            classifier.y_train == j)]) + val,
                            s=15, facecolors=colors[j],
                        edgecolors='white')
            fig.savefig(
                "Ordered Gibbs Cumulative distribution plot of class "
                "distributions for x_new=[{}, {}].png"
                    .format(x_lims[0], x_lims[1]),
                    facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()
            plt.close()

            np.savez(
                "{}.npz".format(classifier.J),
                x=X_new, y=posterior_predictive_m, s=posterior_std)
            fig, ax = plt.subplots(1, 1)
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax.plot(X_new, posterior_predictive_m, 'r')
            ax.fill_between(
                X_new[:, 0], posterior_predictive_m - 2*posterior_std,
                posterior_predictive_m + 2*posterior_std,
                color='red', alpha=MG_ALPHA)
            ax.scatter(X_true, Y_true, color='b', s=4)
            ax.set_ylim(-2.2, 2.2)
            ax.set_xlim(-0.5, 1.5)
            for j in range(classifier.J):
                ax.scatter(
                    classifier.X_train[np.where(classifier.y_train == j)],
                    np.zeros_like(classifier.X_train[np.where(classifier.y_train == j)]),
                    s=15,
                    facecolors=colors[j],
                    edgecolors='white')
            ax.savefig("scatter_versus_posterior_mean.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()
            plt.close()
    return fx


def figure2(
        hyper_sampler, approximator, domain, res, trainables,
        num_importance_samples, steps=None,
        reparameterised=False, verbose=False, show=False, write=True):
    """
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
    thetas,
    *_) = approximator._grid_over_hyperparameters_initiate(
        res, domain, trainables, approximator.cutpoints)
    log_p_pseudo_marginalss = []
    log_p_priors = []
    if x2s is not None:
        raise ValueError("Multivariate plots are TODO")
    else:
        # if one dimension... find a better way of doing this
        theta_step = thetas[1:] - thetas[:-1]
        thetas = thetas[:-1]
        M = len(thetas)
    for i in trange(0, M,
                        desc="Posterior density mesh-grid progress",
                        unit=" mesh-grid points", disable=False):
        theta = thetas[i]
        # Need to update sampler hyperparameters
        approximator._grid_over_hyperparameters_update(
            theta, trainables, approximator.cutpoints)
        phi = approximator.get_phi(trainables)
        (log_p_pseudo_marginals,
                log_p_prior) = hyper_sampler.tmp_compute_marginal(
            phi, trainables, steps, reparameterised=reparameterised,
            num_importance_samples=num_importance_samples)
        log_p_pseudo_marginalss.append(log_p_pseudo_marginals)
        log_p_priors.append(log_p_prior)
        if verbose:
            print("{}/{}".format(i, len(thetas)))
            print("log_p_pseudo_marginal {}, log_p_prior {}".format(
                np.mean(log_p_pseudo_marginals), log_p_prior))
            print(
                "cutpoints={}, theta={}, noise_variance={},"
                " variance={}".format(approximator.cutpoints,
                approximator.kernel.theta, approximator.noise_variance,
                approximator.kernel.variance))
    if x2s is not None:
        raise ValueError("Multivariate plots are TODO")
    else:
        log_p_pseudo_marginalss = np.array(log_p_pseudo_marginalss)
        log_priors = np.array(log_p_priors)
        log_p_pseudo_marginals_ms = np.mean(log_p_pseudo_marginalss, axis=1)
        log_p_pseudo_marginals_std = np.std(log_p_pseudo_marginalss, axis=1)
        # Normalize prior distribution - but need to make sure domain is such that approx all posterior mass is covered
        log_prob = log_p_priors + np.log(theta_step)
        max_log_prob = np.max(log_prob)
        log_sum_exp = max_log_prob + np.log(np.sum(np.exp(log_prob - max_log_prob)))
        p_priors = np.exp(log_p_priors - log_sum_exp)
        # Can probably use a better quadrature rule for numerical integration
        # Normalize posterior distribution - but need to make sure domain is such that approx all posterior mass is covered
        log_prob = log_p_pseudo_marginals_ms + np.log(theta_step)
        max_log_prob = np.max(log_prob)
        log_sum_exp = max_log_prob + np.log(
            np.sum(np.exp(log_prob - max_log_prob)))
        # p_pseudo_marginals = np.exp(log_p_pseudo_marginals_ms - log_sum_exp)
        # remember that there is a jacobian involved in the coordinate transformation
        # log_p_pseudo_marginalss = log_p_pseudo_marginalss + np.log(theta_step).reshape(-1, 1)
        # max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginalss, axis=0)
        # log_sum_exp = np.tile(max_log_p_pseudo_marginals, (M, 1)) + np.tile(
        #     np.log(np.sum(np.exp(log_p_pseudo_marginalss - max_log_p_pseudo_marginals), axis=0)), (M, 1))
        p_pseudo_marginals = np.exp(log_p_pseudo_marginalss - log_sum_exp.reshape(-1, 1))
        p_pseudo_marginals_mean = np.mean(p_pseudo_marginals, axis=1)
        p_pseudo_marginals_lo = np.quantile(p_pseudo_marginals, 0.025, axis=1)
        p_pseudo_marginals_hi = np.quantile(p_pseudo_marginals, 0.975, axis=1)
        # p_pseudo_marginals_std = np.std(p_pseudo_marginals, axis=1)
        return thetas, theta_step, p_pseudo_marginals_mean, p_pseudo_marginals_lo, p_pseudo_marginals_hi, p_priors
        #return log_p_pseudo_marginalss, log_p_priors, x1s, None, xlabel, ylabel, xscale, yscale


def _potential_scale_reduction(
        state, independent_chain_ndims=1, split_chains=False):
    """
    Gelman and Rubin (1992)'s potential scale reduction for chain convergence.
    Given `N > 1` states from each of `C > 1` independent chains, the potential
    scale reduction factor, commonly referred to as R-hat, measures convergence
    of the chains (to the same target) by testing for equality of means.
    Specifically, R-hat measures the degree to which variance (of the means)
    between chains exceeds what one would expect if the chains were identically
    distributed. See [Gelman and Rubin (1992)][1];
    [Brooks and Gelman (1998)][2].
    Some guidelines:
    * The initial state of the chains should be drawn from a distribution
    overdispersed with respect to the target. (TODO: what about burn-in?)
    * If all chains converge to the target, then as `N --> infinity`,
    R-hat --> 1. Before that, R-hat > 1 (except in pathological cases, e.g. if
    the chain paths were identical).
    * The above holds for any number of chains `C > 1`.  Increasing `C` does
    improve effectiveness of the diagnostic.
    * Sometimes, R-hat < 1.2 is used to indicate approximate convergence, but
    of course this is problem-dependent. See [Brooks and Gelman (1998)][2].
    * R-hat only measures non-convergence of the mean. If higher moments, or
    other statistics are desired, a different diagnostic should be used. See
    [Brooks and Gelman (1998)][2].

    :arg state: (n_samples, n_chains, n_parameters)

    :arg split_chains: Python `bool`. If `True`, divide samples from each chain
    into first and second halves, treating these as separate chains.  This
    makes R-hat more robust to non-stationary chains, and is recommended in
    [3].
 
    :returns: `numpy.ndarray` structure parallel to `chains_states`
    representing the R-hat statistic for the state(s). Shape equal to
    `chains_state.shape[2:]`.

    To see why R-hat is reasonable, let `X` be a random variable drawn
    uniformly from the combined states (combined over all chains).  Then, in
    the limit `N, C --> infinity`, with `E`, `Var` denoting expectation and
    variance,
    ```R-hat = ( E[Var[X | chain]] + Var[E[X | chain]] ) / E[Var[X | chain]].```
    Using the law of total variance, the numerator is the variance of the
    combined states, and the denominator is the total variance minus the
    variance of the individual chain means.  If the chains are all drawing from
    the same distribution, they will have the same mean, and thus the ratio
    should be one.

    #### References
    [1]: Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
        Convergence of Iterative Simulations. _Journal of Computational and
        Graphical Statistics_, 7(4), 1998.
    [2]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
        Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.
    [3]: Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter,
        Paul-Christian Burkner. Rank-normalization, folding, and localization:
        An improved R-hat for assessing convergence of MCMC, 2019. Retrieved
        from http://arxiv.org/abs/1903.08008
    """
    if split_chains:
        # Split the sample dimension in half, doubling the number of
        # independent chains.

        # For odd number of samples, keep all but the last sample.
        state_shape = np.shape(state)
        n_samples = state_shape[0]
        state = state[:n_samples - n_samples % 2]  # (n_samples, n_chains, n_params)

        # Suppose state = [0, 1, 2, 3, 4, 5]
        # Step 1: reshape into [[0, 1, 2], [3, 4, 5]]
        # E.g. reshape states of shape [a, b] into [2, a//2, b].
        state = np.reshape(
            state,
            np.concatenate([[2, n_samples // 2], state_shape[1:]], axis=0)  # (2, n_samples_new, n_chains, n_params)
        )
        # Step 2: Put the size `2` dimension in the right place to be treated as a
        # chain, changing [[0, 1, 2], [3, 4, 5]] into [[0, 3], [1, 4], [2, 5]],
        state = state.transpose(1, 0, np.range(2, np.size(np.shape(state))))  # (n_after_burn, n_params, n_chains)  #  (n_chains, n_samples, n_params)

        # We're treating the new dim as indexing 2 chains, so increment.
        independent_chain_ndims += 1

    sample_axis = 0
    chain_axis = np.arange(1, 1 + independent_chain_ndims, dtype=int)
    sample_and_chain_axis = np.arange(0, 1 + independent_chain_ndims, dtype=int)

    # n = int(np.shape(state)[sample_axis])
    # m = int(np.prod(np.shape(state)[chain_axis]))

    # TODO: temp
    n = int(np.shape(state)[1])
    m = int(np.prod(np.shape(state)[0]))

    # In the language of Brooks and Gelman (1998),
    # b_div_n is the between chain variance, the variance of the chain means.
    # w is the within sequence variance, the mean of the chain variances.

    # b_div_n = np.var(np.mean(state, axis=sample_axis, keepdims=True), axis=sample_and_chain_axis, ddof=1)
    # w = np.mean(np.var(state, axis=sample_axis, keepdims=True, ddof=1), axis=sample_and_chain_axis)
    b_div_n = np.var(np.mean(state, axis=1, keepdims=True), axis=0, ddof=1)
    w = np.mean(np.var(state, axis=1, keepdims=True, ddof=1), axis=0)

    # sigma_2_plus is an estimate of the true variance, which would be unbiased if
    # each chain was drawn from the target.  c.f. "law of total variance."
    sigma_2_plus = ((n - 1) / n) * w + b_div_n
    R = ((m + 1.) / m) * sigma_2_plus / w - (n - 1.) / (m * n)
    return ((m + 1.) / m) * sigma_2_plus / w - (n - 1.) / (m * n)


def _effective_sample_size(
        states, filter_beyond_lag=None, filter_threshold=0.0,
        filter_beyond_positive_pairs=False):
    """
    Estimate the effective sample size (ESS) as described in Kass et al (1998) and Robert and Casella (2004; pg 500)
    Roughly speaking, ESS is the size of an iid sample with the same variance as the current sample.
    ESS = T / kappa, where kappa is the ``autocorrelation time" for the sample = 1 + 2 \sum lag_auto-correlations
    Here we use a version analogous to IMSE where we cut off correlations beyond a certain lag (to reduce noise).
    
    Estimate a lower bound on effective sample size for each independent chain.
    Roughly speaking, "effective sample size" (ESS) is the size of an iid sample
    with the same variance as `state`.
    More precisely, given a stationary sequence of possibly correlated random
    variables `X_1, X_2, ..., X_N`, identically distributed, ESS is the
    number such that
    ```
    Variance{ N**-1 * Sum{X_i} } = ESS**-1 * Variance{ X_1 }.
    ```
    If the sequence is uncorrelated, `ESS = N`.  If the sequence is positively
    auto-correlated, `ESS` will be less than `N`. If there are negative
    correlations, then `ESS` can exceed `N`.
    Some math shows that, with `R_k` the auto-correlation sequence,
    `R_k := Covariance{X_1, X_{1+k}} / Variance{X_1}`, we have
    ```
    ESS(N) =  N / [ 1 + 2 * ( (N - 1) / N * R_1 + ... + 1 / N * R_{N-1}  ) ]
    ```
    This function estimates the above by first estimating the auto-correlation.
    Since `R_k` must be estimated using only `N - k` samples, it becomes
    progressively noisier for larger `k`.  For this reason, the summation over
    `R_k` should be truncated at some number `filter_beyond_lag < N`. This
    function provides two methods to perform this truncation.
    * `filter_threshold` -- since many MCMC methods generate chains where `R_k >
        0`, a reasonable criterion is to truncate at the first index where the
        estimated auto-correlation becomes negative. This method does not estimate
        the `ESS` of super-efficient chains (where `ESS > N`) correctly.
    * `filter_beyond_positive_pairs` -- reversible MCMC chains produce
        an auto-correlation sequence with the property that pairwise sums of the
        elements of that sequence are positive [Geyer][1], i.e.
        `R_{2k} + R_{2k + 1} > 0` for `k in {0, ..., N/2}`. Deviations are only
        possible due to noise. This method truncates the auto-correlation sequence
        where the pairwise sums become non-positive.
    The arguments `filter_beyond_lag`, `filter_threshold` and
    `filter_beyond_positive_pairs` are filters intended to remove noisy tail terms
    from `R_k`.  You can combine `filter_beyond_lag` with `filter_threshold` or
    `filter_beyond_positive_pairs. E.g., combining `filter_beyond_lag` and
    `filter_beyond_positive_pairs` means that terms are removed if they were to be
    filtered under the `filter_beyond_lag` OR `filter_beyond_positive_pairs`
    criteria.

    :arg states: `Tensor` or Python structure of `Tensor` objects.  Dimension zero
        should index identically distributed states.
        filter_threshold: `Tensor` or Python structure of `Tensor` objects.  Must
        broadcast with `state`.  The sequence of auto-correlations is truncated
        after the first appearance of a term less than `filter_threshold`.
        Setting to `None` means we use no threshold filter.  Since `|R_k| <= 1`,
        setting to any number less than `-1` has the same effect. Ignored if
        `filter_beyond_positive_pairs` is `True`.
        filter_beyond_lag: `Tensor` or Python structure of `Tensor` objects.  Must
        be `int`-like and scalar valued.  The sequence of auto-correlations is
        truncated to this length.  Setting to `None` means we do not filter based
        on the size of lags.
    :arg filter_beyond_positive_pairs: Python boolean. If `True`, only consider the
        initial auto-correlation sequence where the pairwise sums are positive.
    Returns:
        ess: `numpy.ndarray` structure parallel to `states`.  The effective sample size of
        each component of `states`.  The shape is `np.shape(states)[1:]`.
    #### References
    [1]: Charles J. Geyer, Practical Markov chain Monte Carlo (with discussion).
        Statistical Science, 7:473-511, 1992.
    [2]: Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, Paul-Christian
        Burkner. Rank-normalization, folding, and localization: An improved R-hat
        for assessing convergence of MCMC, 2019. Retrieved from
        http://arxiv.org/abs/1903.08008
    """
    # Assume np.size(states) = (n_samples, n_hyperparameters)
    # filter_beyond_lag == None ==> auto_corr is the full sequence.
    # With R[k] := auto_corr[k, ...],
    # ESS = N / {1 + 2 * Sum_{k=1}^N R[k] * (N - k) / N}
    #     = N / {-1 + 2 * Sum_{k=0}^N R[k] * (N - k) / N} (since R[0] = 1)
    #     approx N / {-1 + 2 * Sum_{k=0}^M R[k] * (N - k) / N}
    # where M is the filter_beyond_lag truncation point chosen above.
    auto_corr = _auto_correlation(states, filter_beyond_lag)
    weighted_auto_cov = _weighted_auto_cov(states, filter_beyond_lag)
    # Autocorrelation
    n = np.shape(states)[0]
    weighted_auto_corr = weighted_auto_cov / weighted_auto_cov[:1]
    num_chains = 1
    if filter_beyond_positive_pairs:
        def _sum_pairs(x):
            x_len = np.shape(x)[0]
            # For odd sequences, we drop the final value.
            x = x[:x_len - x_len % 2]

            new_shape = np.concatenate([[x_len // 2, 2], np.shape(x)[1:]], axis=0)
            # new_shape = ps.concat([[x_len // 2, 2], ps.shape(x)[1:]], axis=0)
            return np.sum(x.reshape(new_shape), axis=1)
            # return tf.reduce_sum(tf.reshape(x, new_shape), 1)
        # Pairwise sums are all positive for auto-correlation spectra derived from
        # reversible MCMC chains.
        # E.g. imagine the pairwise sums are [0.2, 0.1, -0.1, -0.2]
        # Step 1: mask = [False, False, True, True]
        mask = _sum_pairs(weighted_auto_corr) < 0
        # Step 2: mask = [0, 0, 1, 1]
        mask =  mask.astype(int)
        # Step 3: mask = [0, 0, 1, 2]
        mask = np.cumsum(mask, axis=0)
        # Step 4: mask = [1, 1, 0, 0]
        mask = np.maximum(1. - mask, 0.)
        # N.B. this reduces the length of weighted_auto_corr by a factor of 2.
        # It still works fine in the formula below.
        weighted_auto_corr = _sum_pairs(weighted_auto_corr) * mask
    elif filter_threshold is not None:
        # Get a binary mask to zero out values of auto_corr below the threshold.
        #   mask[i, ...] = 1 if auto_corr[j, ...] > threshold for all j <= i,
        #   mask[i, ...] = 0, otherwise.
        # So, along dimension zero, the mask will look like [1, 1, ..., 0, 0,...]
        # Building step by step,
        #   Assume auto_corr = [1, 0.5, 0.0, 0.3], and filter_threshold = 0.2.
        # Step 1:  mask = [False, False, True, False]
        mask = auto_corr < filter_threshold
        # Step 2:  mask = [0, 0, 1, 0]
        mask = mask.astype(int)
        # Step 3:  mask = [0, 0, 1, 1]
        mask = np.cumsum(mask, axis=0)
        # Step 4:  mask = [1, 1, 0, 0]
        mask = np.maximum(1. - mask, 0.)
        weighted_auto_corr *= mask
    return num_chains * n / (-1 + 2 * np.sum(weighted_auto_corr, axis=0))


def _auto_correlation(state, max_lags):
    """
    auto correlation across one axis. Same result as
        tensorflow_probability.stats.auto_correlation(
            samples, axis=0, max_lags=filter_beyond_lag, normalize=False)
    np.shape(state) = (n_samples, n_hyperparameters)
    """
    n_samples, n_hyperparameters = np.shape(state)
    if max_lags is None:
        max_lags = n_samples - 1
    w = (state - np.mean(state, axis=0))    
    auto_cov = np.empty((max_lags + 1, n_hyperparameters))
    k = np.tile(np.arange(max_lags + 1), (n_hyperparameters, 1)).T
    for i in range(n_hyperparameters):
        auto_cov[:, i] = np.correlate(w[:, i], w[:, i], mode='full')[n_samples - 1:n_samples + max_lags]
    auto_cov = auto_cov / (n_samples - k)
    return auto_cov


def _weighted_auto_cov(x, max_lags):
    """
    Weighted, centred auto correlation across one axis: r[k] * (n + k) / n, k = 0, ..., max_lags
 
    np.shape(state) = (n_samples, n_hyperparameters)
    """
    n_samples, n_hyperparameters = np.shape(x)
    if max_lags is None:
        max_lags = n_samples - 1
    w = (x - np.mean(x, axis=0))    
    auto_cov = np.empty((max_lags + 1, n_hyperparameters))
    for i in range(n_hyperparameters):
        auto_cov[:, i] = np.correlate(w[:, i], w[:, i], mode='full')[n_samples - 1:n_samples + max_lags]
    auto_cov = auto_cov
    return auto_cov


def draw_mixing(states, state_0, write_path, file_name, logplot=True):
    """Plot mixing"""
    (Nsamp, Nparam) = np.shape(states)

    if not logplot:
        states = np.exp(states)
        state_0 = np.exp(state_0)
        label = "theta"
    else:
        label = "phi"
    for i in range(Nparam):
        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax.plot(states[:, i])
        mean = np.mean(states[:, i], axis=0)
        xs = np.linspace(0, Nsamp, 2)
        xs = np.tile(xs, (Nparam, 1)).T
        ys = mean * np.ones(np.shape(xs))
        trues = state_0[i] * np.ones(np.shape(xs))
        ax.plot(xs, ys, label="sample mean")
        ax.plot(xs, trues, color='k', label="true")
        ax.set_xlim(0, Nsamp)
        ax.legend()
        file_name_i = "{}_{}_".format(label, i) + file_name
        fig.savefig(
            write_path / file_name_i,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()


def draw_histogram(states, state_0, write_path, file_name,
        bins=50, logplot=True):
    """Plot histogram"""
    (Nsamp, Nparam) = np.shape(states)
    if not logplot:
        states = np.exp(states)
        state_0 = np.exp(state_0)
        label = "theta"
    else:
        label = "phi"
    for i in range(Nparam):
        # plot histogram
        font = {'family' : 'sans-serif',
        'size'   : 22}
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.hist(states[:, i], density=True, bins=bins)
        ax.vlines(state_0[i], 0.9 * ax.get_ylim()[0], 0.9 * ax.get_ylim()[1], 'k')
        ax.set_xlabel(label, **font)
        plt.tight_layout()
        file_name_i = "{}_{}_".format(label, i) + file_name
        fig.savefig(
            write_path / file_name_i,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()


def table1(
        read_path, n_samples, show=False, write=True):
    """
    Sample from the pseudo-marginal. Evaluate Acceptance rate, convergence
    diagnostics.

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
    approach = "PM"
    Nsamp = n_samples
    Nchain = 3
    Nhyperparameters = 2
    for N in [200]:
        for D in [2]:
            for J in [3, 13]:
                for approximation in ["EP", "LA"]:
                    for Nimp in [2, 64]:
                        for ARD in [False]:
                            try:
                                # initiate containers
                                acceptance_rate = np.empty(Nchain)
                                effective_sample_size = np.empty((Nchain, Nhyperparameters))
                                states = np.empty((Nchain, Nsamp, Nhyperparameters))
                                Rhat = np.empty((4, Nhyperparameters)) 
                                for chain in range(Nchain):
                                    # Get the data
                                    data_chain_theta = np.load(
                                        read_path/'theta_N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chain={}.npz'.format(
                                            N, D, J, Nimp, ARD, approach, approximation, chain))
                                    states_chain = data_chain_theta["X"]
                                    states[chain, :, :] = states_chain
                                    acceptance_rate[chain] = data_chain_theta["acceptance_rate"]
                                    effective_sample_size[chain, :] = _effective_sample_size(
                                        states_chain[:, :], filter_beyond_lag=None, filter_beyond_positive_pairs=True)
                                # Find
                                for i, N_samp in enumerate([1000, 2000, 5000, 10000]):
                                    # print(_potential_scale_reduction(
                                    #     states[:, :Nsamp, :], independent_chain_ndims=1, split_chains=False))
                                    Rhat[i] = _potential_scale_reduction(
                                        states[:, :N_samp, :], independent_chain_ndims=1, split_chains=False)
                                pr1 = np.mean(effective_sample_size, axis=0)
                                pr2 = np.std(effective_sample_size, axis=0)
                                pr30 = Rhat[0]
                                pr31 = Rhat[1]
                                pr32 = Rhat[2]
                                pr33 = Rhat[3]
                                pr4 = np.mean(acceptance_rate * 100, axis=0)
                                pr5 = np.std(acceptance_rate * 100, axis=0)
                                print("N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chains={}:".format(
                                    N, D, J, Nimp, ARD, approach, approximation, chain + 1))
                                print("ESS={}+/-{},  R0={}, R1={}, R2={}, R3={}, Acc={}+/-{}".format(
                                    pr1, pr2, pr30, pr31, pr32, pr33, pr4, pr5))
                            except:
                                print("N={}_D={}_J={}_Nimp={}_ARD={}_{}_{}_chains={}:".format(
                                    N, D, J, Nimp, ARD, approach, approximation, chain + 1))
                                print("pass")

    # rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size' : 12})
    # rc('text', usetex=True)


class TookTooLong(Warning):
    pass


class MinimizeStopper(object):
    def __init__(self, max_sec=100):
        self.max_sec = max_sec
        self.start   = time.time()

    def __call__(self, xk):
        # callback to terminate if max_sec exceeded
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)