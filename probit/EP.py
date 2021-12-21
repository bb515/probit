"""Approximate inference: EP approximation. Methods for inference and test."""
import numpy as np
import matplotlib.pyplot as plt
from probit.data.utilities import datasets, colors
from probit.data.utilities import calculate_metrics


def plot(classifier, domain=None):
    """
    TODO: needs generalizing to other datasets other than Chu.
    Contour plot of the predictive probabilities over a chosen domain.
    """
    steps = classifier.N
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    while error / steps > classifier.EPS ** 2:
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma,
        mean_EP, precision_EP, amplitude_EP, containers
        ) = classifier.estimate(
            steps, posterior_mean_0=posterior_mean, Sigma_0=Sigma,
            mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP,
            write=True)
        print("iteration {}, error={}".format(iteration, error / steps))
    (weights, precision_EP, Lambda_cholesky, Lambda
    ) = classifier.compute_EP_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
    t1, *_ = classifier.compute_integrals_vector(
        np.diag(Sigma), posterior_mean, classifier.noise_variance)
    fx = classifier.objective(
        precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
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
        Z, posterior_predictive_m, posterior_std = classifier.predict(
            classifier.gamma, Sigma, mean_EP, precision_EP,
            classifier.kernel.varphi,
            classifier.noise_variance, X_new_, Lambda, vectorised=True)
        Z_new = Z.reshape((N, N, classifier.J))
        print(np.sum(Z, axis=1), 'sum')
        for j in range(classifier.J):
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, Z_new[:, :, j], zorder=1)
            plt.scatter(classifier.X_train[np.where(
                classifier.t_train == j)][:, 0],
                classifier.X_train[np.where(
                    classifier.t_train == j)][:, 1], color='red')
            plt.scatter(
                classifier.X_train[np.where(
                    classifier.t_train == j + 1)][:, 0],
                classifier.X_train[np.where(
                    classifier.t_train == j + 1)][:, 1], color='blue')
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.savefig("contour_EP_{}.png".format(j))
            plt.close()
    return fx


def plot_synthetic(
    classifier, dataset, X_true, Y_true, colors=colors):
    """
    Plots for synthetic data.

    TODO: needs generalizing to other datasets other than Chu.
    """
    steps = classifier.N
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    while error / steps > classifier.EPS**2:
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma,
            mean_EP, precision_EP, amplitude_EP,
            containers) = classifier.estimate(
            steps, posterior_mean_0=posterior_mean, Sigma_0=Sigma,
            mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP,
            write=True)
        # plot for animations TODO: make an animation function
        # plt.scatter(X, posterior_mean)
        # plt.scatter(X_true, Y_true)
        # plt.ylim(-3, 3)
        # plt.savefig("scatter_versus_posterior_mean.png")
        # plt.close()
        print("iteration {}, error={}".format(iteration, error / steps))
    (weights, precision_EP,
    Lambda_cholesky, Lambda) = classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
    t1, *_ = classifier.compute_integrals_vector(
        np.diag(Sigma), posterior_mean, classifier.noise_variance)
    fx = classifier.objective(
        precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
    if dataset in datasets["synthetic"]:
        if classifier.J == 3:
            x_lims = (-0.5, 1.5)
            N = 1000
            x = np.linspace(x_lims[0], x_lims[1], N)
            X_new = x.reshape((N, classifier.D))
            (Z,
            posterior_predictive_m,
            posterior_std) = classifier.predict(
                classifier.gamma, Sigma, mean_EP, precision_EP,
                classifier.kernel.varphi, classifier.noise_variance,
                X_new, Lambda, vectorised=True)
            print(np.sum(Z, axis=1), 'sum')
            plt.xlim(x_lims)
            plt.ylim(0.0, 1.0)
            plt.xlabel(r"$x$", fontsize=16)
            plt.ylabel(r"$p(\omega|x, X, \omega)$", fontsize=16)
            plt.stackplot(x, Z.T,
                        labels=(
                            r"$p(\omega=0|x, X, t)$",
                            r"$p(\omega=1|x, X, t)$",
                            r"$p(\omega=2|x, X, t)$"),
                        colors=(
                            colors[0], colors[1], colors[2])
                        )
            val = 0.5  # where the data lies on the y-axis.
            for j in range(classifier.J):
                plt.scatter(
                    classifier.X[np.where(classifier.t == j)],
                    np.zeros_like(classifier.X[np.where(
                        classifier.t == j)]) + val,
                    s=15, facecolors=colors[j], edgecolors='white')
            plt.savefig(
                "Ordered Gibbs Cumulative distribution plot of class "
                "distributions for x_new=[{}, {}].png".format(
                    x_lims[0], x_lims[1]))
            plt.show()
            plt.close()
            np.savez(
                "EP_tertile.npz",
                x=X_new, y=posterior_predictive_m, s=posterior_std)
            plt.plot(X_new, posterior_predictive_m, 'r')
            plt.fill_between(
                X_new[:, 0], posterior_predictive_m - 2*posterior_std,
                posterior_predictive_m + 2*posterior_std,
                color='red', alpha=0.2)
            plt.scatter(X_true, Y_true, color='b', s=4)
            plt.ylim(-2.2, 2.2)
            plt.xlim(-0.5, 1.5)
            for j in range(classifier.J):
                plt.scatter(
                    classifier.X[np.where(classifier.t == j)],
                    np.zeros_like(classifier.X[np.where(classifier.t == j)]),
                    s=15, facecolors=colors[j], edgecolors='white')
            plt.savefig("scatter_versus_posterior_mean.png")
            plt.show()
            plt.close()
        elif classifier.J == 13:
            x_lims = (-0.5, 1.5)
            N = 1000
            x = np.linspace(x_lims[0], x_lims[1], N)
            X_new = x.reshape((N, classifier.D))
            (Z,
            posterior_predictive_m,
            posterior_std) = classifier.predict(
                classifier.gamma, Sigma, mean_EP, precision_EP,
                classifier.kernel.varphi, classifier.noise_variance,
                X_new, Lambda, vectorised=True)
            print(np.sum(Z, axis=1), 'sum')
            plt.xlim(x_lims)
            plt.ylim(0.0, 1.0)
            plt.xlabel(r"$x$", fontsize=16)
            plt.ylabel(r"$p(\omega_{*}={}|x, X, \omega)$", fontsize=16)
            plt.stackplot(x, Z.T,
                            labels=(
                            r"$p(\omega_{*}=0|x, X, \omega)$",
                            r"$p(\omega_{*}=1|x, X, \omega)$",
                            r"$p(\omega_{*}=2|x, X, \omega)$",
                            r"$p(\omega_{*}=3|x, X, \omega)$",
                            r"$p(\omega_{*}=4|x, X, \omega)$",
                            r"$p(\omega_{*}=5|x, X, \omega)$",
                            r"$p(\omega_{*}=6|x, X, \omega)$",
                            r"$p(\omega_{*}=7|x, X, \omega)$",
                            r"$p(\omega_{*}=8|x, X, \omega)$",
                            r"$p(\omega_{*}=9|x, X, \omega)$",
                            r"$p(\omega_{*}=10|x, X, \omega)$",
                            r"$p(\omega_{*}=11|x, X, \omega)$",
                            r"$p(\omega_{*}=12|x, X, \omega)$"),
                        colors=colors
                        )
            val = 0.5  # where the data lies on the y-axis.
            for j in range(classifier.J):
                plt.scatter(
                    classifier.X[np.where(classifier.t == j)],
                    np.zeros_like(
                        classifier.X[np.where(
                            classifier.t == j)]) + val,
                            s=15, facecolors=colors[j],
                        edgecolors='white')
            plt.savefig(
                "Ordered Gibbs Cumulative distribution plot of class "
                "distributions for x_new=[{}, {}].png"
                    .format(x_lims[0], x_lims[1]))
            plt.show()
            plt.close()
            np.savez(
                "EP_thirteen.npz",
                x=X_new, y=posterior_predictive_m, s=posterior_std)
            plt.plot(X_new, posterior_predictive_m, 'r')
            plt.fill_between(
                X_new[:, 0], posterior_predictive_m - 2*posterior_std,
                posterior_predictive_m + 2*posterior_std,
                color='red', alpha=0.2)
            plt.scatter(X_true, Y_true, color='b', s=4)
            plt.ylim(-2.2, 2.2)
            plt.xlim(-0.5, 1.5)
            for j in range(classifier.J):
                plt.scatter(
                    classifier.X[np.where(classifier.t == j)],
                    np.zeros_like(classifier.X[np.where(classifier.t == j)]),
                    s=15,
                    facecolors=colors[j],
                    edgecolors='white')
            plt.savefig("scatter_versus_posterior_mean.png")
            plt.show()
            plt.close()
    return fx


def test(classifier, X_test, t_test, y_test, L=None, Lambda=None, domain=None):
    """Test the trained model."""
    steps = classifier.N
    error = np.inf
    iteration = 0
    posterior_mean = None
    Sigma = None
    mean_EP = None
    precision_EP = None
    amplitude_EP = None
    while error / steps > classifier.EPS**2:
        iteration += 1
        (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
            precision_EP, amplitude_EP, *_) = classifier.estimate(
            steps, posterior_mean_0=posterior_mean, Sigma_0=Sigma,
            mean_EP_0=mean_EP, precision_EP_0=precision_EP,
            amplitude_EP_0=amplitude_EP, write=True)
        print("iteration {}, error={}".format(iteration, error / steps))
    (weights,
    precision_EP,
    L,
    Lambda) = classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean,
        L, Lambda)
    t1, *_ = classifier.compute_integrals_vector(
        np.diag(Sigma), posterior_mean, classifier.noise_variance)
    fx = classifier.objective(
        precision_EP, posterior_mean, t1, L, Lambda, weights)
    # Test
    (Z,
    posterior_predictive_m,
    posterior_std) = classifier.predict(
        classifier.gamma, Sigma, mean_EP, precision_EP,
        classifier.kernel.varphi, classifier.noise_variance, X_test,
        Lambda, vectorised=True)  # (N_test, J)
    # # TODO: Placeholder
    # fx = 0.0 
    # Z = np.zeros((len(y_test), J))
    # Z[:, 4] = 1.0
    if domain is not None:
        (x_lims, y_lims) = domain
        N = 75
        x1 = np.linspace(x_lims[0], x_lims[1], N)
        x2 = np.linspace(y_lims[0], y_lims[1], N)
        xx, yy = np.meshgrid(x1, x2)
        X_new = np.dstack((xx, yy))
        X_new = X_new.reshape((N * N, 2))
        X_new_ = np.zeros((N * N, D))
        X_new_[:, :2] = X_new
        (Z,
        posterior_predictive_m,
        posterior_std) = classifier.predict(
            classifier.gamma, Sigma, mean_EP, precision_EP,
            classifier.kernel.varphi, classifier.noise_variance,
            X_new_, Lambda, vectorised=True)
        Z_new = Z.reshape((N, N, J))
        print(np.sum(Z, axis=1), 'sum')
        for j in range(classifier.J):
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, Z_new[:, :, j], zorder=1)
            plt.scatter(
                classifier.X_train[np.where(classifier.t_train == j)][:, 0],
                classifier.X_train[np.where(classifier.t_train == j)][:, 1],
                color='red')
            plt.scatter(
                X_test[np.where(t_test == j)][:, 0],
                X_test[np.where(t_test == j)][:, 1], color='blue')
            # plt.scatter(
            #   X_train[np.where(t == j + 1)][:, 0],
            #   X_train[np.where(t == j + 1)][:, 1],
            #   color='blue')
            # plt.xlim(0, 2)
            # plt.ylim(0, 2)
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.savefig("contour_EP_{}.png".format(j))
            plt.close()
    return fx, calculate_metrics(y_test, t_test, Z, classifier.gamma)
