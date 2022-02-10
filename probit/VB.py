"""Approximate inference: VB approximation. Methods for inference and test."""
import numpy as np
import matplotlib.pyplot as plt
from probit.data.utilities import datasets, colors
from probit.data.utilities import calculate_metrics
from .utilities import truncated_norm_normalising_constant

def plot(classifier, posterior_mean_0, steps, J, D, domain=None):
    """
    TODO: needs generalizing to other datasets other than Chu.
    Contour plot of the predictive probabilities over a chosen domain.
    """
    iteration = 0
    error = np.inf
    while error / steps > classifier.EPS:
        iteration += 1
        posterior_mean_0, nu, y, *_ = classifier.approximate(
            steps, posterior_mean_0=posterior_mean_0, first_step=1,
            fix_hyperparameters=True, write=False)
        Z, *_ = truncated_norm_normalising_constant(
            classifier.gamma_ts, classifier.gamma_tplus1s,
            classifier.noise_std, posterior_mean_0,
            classifier.EPS)
        fx = classifier.objective(
            classifier.N, posterior_mean_0, nu, classifier.trace_cov,
            classifier.trace_posterior_cov_div_var,
            Z,
            classifier.noise_variance,
            classifier.log_det_K, classifier.log_det_cov)
        error = np.abs(fx_old - fx)  # TODO: redundant?
        fx_old = fx
        if 1:
            print("({}), error={}".format(iteration, error))
    if domain is not None:
        (xlims, ylims) = domain
        N = 75
        x1 = np.linspace(xlims[0], xlims[1], N)
        x2 = np.linspace(ylims[0], ylims[1], N)
        xx, yy = np.meshgrid(x1, x2)
        X_new = np.dstack((xx, yy))
        X_new = X_new.reshape((N * N, 2))
        X_new_ = np.zeros((N * N, D))
        X_new_[:, :2] = X_new
        # Test
        Z, posterior_predictive_m, posterior_std = classifier.predict(
            classifier.gamma, classifier.cov, y,
            classifier.noise_variance, X_new_)  # (N_test, J)
        Z_new = Z.reshape((N, N, J))
        print(np.sum(Z, axis=1), 'sum')
        for j in range(J):
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, Z_new[:, :, j], zorder=1)
            plt.scatter(
                classifier.X_train[np.where(classifier.t_train == j)][:, 0],
                classifier.X_train[np.where(classifier.t_train == j)][:, 1],
                color='red')
            # plt.scatter(
            #     classifier.X_train[
            #         np.where(classifier.t_train == j + 1)][:, 0],
            #     classifier.X_train[
            #         np.where(classifier.t_train == j + 1)][:, 1],
            #         color='blue')
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.savefig("contour_VB_{}.png".format(j))
            plt.close()
    return fx


def plot_synthetic(
        classifier,
        dataset, X_true, Y_true, posterior_mean_0, steps, colors=colors):
    """
    Plots for synthetic data.

    TODO: needs generalizing to other datasets other than Chu.
    """
    (posterior_mean, nu, y_tilde, p, containers) = classifier.approximate(
        steps, posterior_mean_0=posterior_mean_0, write=True)
    plt.scatter(classifier.X_train, posterior_mean, color='r')
    plt.scatter(X_true, Y_true, color='b')
    plt.show()
    (ms, ys, varphis, psis, fxs) = containers
    plt.plot(fxs)
    plt.title("Variational lower bound on the marginal likelihood")
    plt.show()
    Z, *_ = truncated_norm_normalising_constant(
        classifier.gamma_ts, classifier.gamma_tplus1s,
        classifier.noise_std, posterior_mean, classifier.EPS,
        upper_bound=classifier.upper_bound,
        upper_bound2=classifier.upper_bound2)
    J = classifier.J
    N = classifier.N
    fx = classifier.objective(
            classifier.N, posterior_mean, nu, classifier.trace_cov,
            classifier.trace_posterior_cov_div_var,
            Z,
            classifier.noise_variance,
            classifier.log_det_K, classifier.log_det_cov)
    if dataset in datasets["synthetic"]:
        if classifier.J == 3:
            x_lims = (-0.5, 1.5)
            N = 1000
            x = np.linspace(x_lims[0], x_lims[1], N)
            X_new = x.reshape((N, classifier.D))
            print("y", y_tilde)
            print("varphi", classifier.kernel.varphi)
            print("noisevar", classifier.noise_variance)
            (Z, posterior_predictive_m,
            posterior_std) = classifier.predict(
                classifier.gamma, classifier.cov, y_tilde,
                classifier.noise_variance, X_new)
            print(np.sum(Z, axis=1), 'sum')
            plt.xlim(x_lims)
            plt.ylim(0.0, 1.0)
            plt.xlabel(r"$x$", fontsize=16)
            plt.ylabel(r"$p(\omega_{*}|x, X, \omega)$", fontsize=16)
            plt.stackplot(x, Z.T,
                        labels=(
                            r"$p(\omega_{*}=0|x, X, \omega)$",
                            r"$p(\omega_{*}=1|x, X, \omega)$",
                            r"$p(\omega_{*}=2|x, X, \omega)$"),
                        colors=(
                            colors[0], colors[1], colors[2])
                        )
            val = 0.5  # the value where you want the data to appear on the y-axis
            for j in range(J):
                plt.scatter(classifier.X_train[np.where(classifier.t_train == j)], np.zeros_like(
                    classifier.X_train[np.where(classifier.t_train == j)]) + val, s=15, facecolors=colors[j],
                    edgecolors='white')
            plt.savefig(
                "Ordered Gibbs Cumulative distribution plot of class distributions"
                " for x_new=[{}, {}].png".format(x_lims[0], x_lims[1]))
            plt.show()
            plt.close()
            np.savez(
                "VB_tertile.npz",
                x=X_new, y=posterior_predictive_m, s=posterior_std)
            plt.plot(X_new, posterior_predictive_m, 'r')
            plt.fill_between(
                X_new[:, 0],
                posterior_predictive_m - 2*posterior_std,
                posterior_predictive_m + 2*posterior_std,
                color='red', alpha=0.2)
            plt.scatter(X_true, Y_true, color='b', s=4)
            plt.ylim(-2.2, 2.2)
            plt.xlim(-0.5, 1.5)
            for j in range(J):
                plt.scatter(
                    classifier.X_train[np.where(classifier.t_train == j)],
                    np.zeros_like(classifier.X_train[np.where(classifier.t_train == j)]) + val,
                    s=15, facecolors=colors[j], edgecolors='white')
            plt.savefig("scatter_versus_posterior_mean.png")
            plt.show()
            plt.close()

            # x_lims = (-0.5, 1.5)
            # N = 1000
            # x = np.linspace(x_lims[0], x_lims[1], N)
            # X_new = x.reshape((N, classifier.D))
            # (Z,
            # posterior_predictive_m,
            # posterior_std) = classifier.predict(
            #     classifier.gamma, posterior_cov, mean_EP, precision_EP,
            #     classifier.kernel.varphi, classifier.noise_variance,
            #     X_new, Lambda, vectorised=True)
            # print(np.sum(Z, axis=1), 'sum')
            # plt.xlim(x_lims)
            # plt.ylim(0.0, 1.0)
            # plt.xlabel(r"$x$", fontsize=16)
            # plt.ylabel(r"$p(\omega|x, X, \omega)$", fontsize=16)
            # plt.stackplot(x, Z.T,
            #             labels=(
            #                 r"$p(\omega=0|x, X, t)$",
            #                 r"$p(\omega=1|x, X, t)$",
            #                 r"$p(\omega=2|x, X, t)$"),
            #             colors=(
            #                 colors[0], colors[1], colors[2])
            #             )
            # val = 0.5  # where the data lies on the y-axis.
            # for j in range(classifier.J):
            #     plt.scatter(
            #         classifier.X_train[np.where(classifier.t_train == j)],
            #         np.zeros_like(classifier.X_train[np.where(
            #             classifier.t_train == j)]) + val,
            #         s=15, facecolors=colors[j], edgecolors='white')
            # plt.savefig(
            #     "Ordered Gibbs Cumulative distribution plot of class "
            #     "distributions for x_new=[{}, {}].png".format(
            #         x_lims[0], x_lims[1]))
            # plt.show()
            # plt.close()
            # np.savez(
            #     "EP_tertile.npz",
            #     x=X_new, y=posterior_predictive_m, s=posterior_std)
            # plt.plot(X_new, posterior_predictive_m, 'r')
            # plt.fill_between(
            #     X_new[:, 0], posterior_predictive_m - 2*posterior_std,
            #     posterior_predictive_m + 2*posterior_std,
            #     color='red', alpha=0.2)
            # plt.scatter(X_true, Y_true, color='b', s=4)
            # plt.ylim(-2.2, 2.2)
            # plt.xlim(-0.5, 1.5)
            # for j in range(classifier.J):
            #     plt.scatter(
            #         classifier.X_train[np.where(classifier.t_train == j)],
            #         np.zeros_like(classifier.X_train[np.where(classifier.t_train == j)]),
            #         s=15, facecolors=colors[j], edgecolors='white')
            # plt.savefig("scatter_versus_posterior_mean.png")
            # plt.show()
            # plt.close()
        
    # elif dataset == "septile":
    #     x_lims = (-0.5, 1.5)
    #     N = 1000
    #     x = np.linspace(x_lims[0], x_lims[1], N)
    #     X_new = x.reshape((N, classifier.D))
    #     (Z, posterior_predictive_m,
    #     posterior_std) = classifier.predict(
    #         classifier.gamma, classifier.cov, y_tilde, classifier.kernel.varphi,
    #         classifier.noise_variance, X_new)
    #     print(np.sum(Z, axis=1), 'sum')
    #     plt.xlim(x_lims)
    #     plt.ylim(0.0, 1.0)
    #     plt.xlabel(r"$x$", fontsize=16)
    #     plt.ylabel(r"$p(\omega_{*}={}|x, X, \omega)$", fontsize=16)
    #     plt.stackplot(x, Z.T,
    #         labels=(
    #             r"$p(\omega_{*}=0|x, X, \omega)$",
    #             r"$p(\omega_{*}=1|x, X, \omega)$",
    #             r"$p(\omega_{*}=2|x, X, \omega)$",
    #             r"$p(\omega_{*}=3|x, X, \omega)$",
    #             r"$p(\omega_{*}=4|x, X, \omega)$",
    #             r"$p(\omega_{*}=5|x, X, \omega)$",
    #             r"$p(\omega_{*}=6|x, X, \omega)$"),
    #         colors=(
    #             colors[0], colors[1], colors[2],
    #             colors[3], colors[4], colors[5], colors[6]))
    #     plt.legend()
    #     val = 0.5  # the value where the data appears on the y-axis
    #     for j in range(J):
    #         plt.scatter(
    #             classifier.X_train[np.where(classifier.t_train == j)],
    #             np.zeros_like(classifier.X_train[np.where(classifier.t_train == j)]) + val,
    #             s=15, facecolors=colors[j], edgecolors='white')
    #     plt.savefig(
    #         "Ordered Gibbs Cumulative distribution plot of\nclass "
    #         "distributions for x_new=[{}, {}].png".format(
    #             x_lims[1], x_lims[0]))
    #     plt.close()
    # elif dataset=="thirteen":
    #     x_lims = (-0.5, 1.5)
    #     N = 1000
    #     x = np.linspace(x_lims[0], x_lims[1], N)
    #     X_new = x.reshape((N, classifier.D))
    #     (Z,
    #     posterior_predictive_m,
    #     posterior_std) = classifier.predict(
    #         classifier.gamma, classifier.cov, y_tilde, classifier.kernel.varphi,
    #         classifier.noise_variance, X_new)
    #     print(np.sum(Z, axis=1), 'sum')
    #     plt.xlim(x_lims)
    #     plt.ylim(0.0, 1.0)
    #     plt.xlabel(r"$x$", fontsize=16)
    #     plt.ylabel(r"$p(\omega_{*}|x, X, \omega)$", fontsize=16)
    #     plt.stackplot(x, Z.T,
    #                   labels=(
    #                       r"$p(\omega_{*}=0|x, X, \omega)$",
    #                       r"$p(\omega_{*}=1|x, X, \omega)$",
    #                       r"$p(\omega_{*}=2|x, X, \omega)$",
    #                       r"$p(\omega_{*}=3|x, X, \omega)$",
    #                       r"$p(\omega_{*}=4|x, X, \omega)$",
    #                       r"$p(\omega_{*}=5|x, X, \omega)$",
    #                       r"$p(\omega_{*}=6|x, X, \omega)$",
    #                       r"$p(\omega_{*}=7|x, X, \omega)$",
    #                       r"$p(\omega_{*}=8|x, X, \omega)$",
    #                       r"$p(\omega_{*}=9|x, X, \omega)$",
    #                       r"$p(\omega_{*}=10|x, X, \omega)$",
    #                       r"$p(\omega_{*}=11|x, X, \omega)$",
    #                       r"$p(\omega_{*}=12|x, X, \omega)$"),
    #                   colors=colors
    #                   )
    #     val = 0.5  # where the data to appears on the y-axis
    #     for j in range(J):
    #         plt.scatter(
    #             classifier.X_train[np.where(classifier.t_train == j)],
    #             np.zeros_like(
    #                 classifier.X_train[np.where(
    #                     classifier.t_train == j)]) + val,
    #             s=15, facecolors=colors[j], edgecolors='white')
    #     plt.savefig(
    #         "Ordered Gibbs Cumulative distribution plot of class distributions"
    #         " for x_new=[{}, {}].png".format(x_lims[0], x_lims[1]))
    #     plt.show()
    #     plt.close()
    #     np.savez(
    #         "VB_thirteen.npz",
    #         x=X_new, y=posterior_predictive_m, s=posterior_std)
    #     plt.plot(X_new, posterior_predictive_m, 'r')
    #     plt.fill_between(
    #         X_new[:, 0],
    #         posterior_predictive_m - 2*posterior_std,
    #         posterior_predictive_m + 2*posterior_std, color='red', alpha=0.2)
    #     plt.plot(X_true, Y_true, 'b')
    #     plt.ylim(-2.2, 2.2)
    #     plt.xlim(-0.5, 1.5)
    #     for j in range(J):
    #         plt.scatter(
    #             classifier.X_train[np.where(classifier.t_train == j)],
    #             np.zeros_like(
    #                 classifier.X_train[np.where(classifier.t_train == j)]),
    #             s=15, facecolors=colors[j], edgecolors='white')
    #     plt.savefig("scatter_versus_posterior_mean.png")
    #     plt.show()
    #     plt.close()
    return fx


def test(
        classifier, X_test, t_test, y_test,
        steps, domain=None, verbose=True):
    """Test the trained model."""
    iteration = 0
    error = np.inf
    fx_old = np.inf
    posterior_mean_0 = None
    # TODO: replace with approximate_posterior
    while error / steps > classifier.EPS:
        iteration += 1
        (posterior_mean_0, nu, y, *_) = classifier.approximate(
            steps, posterior_mean_0=posterior_mean_0, first_step=1, write=False)
        Z, *_ = truncated_norm_normalising_constant(
            classifier.gamma_ts, classifier.gamma_tplus1s,
            classifier.noise_std, posterior_mean_0,
            classifier.EPS)
        fx = classifier.objective(
            classifier.N, posterior_mean_0, nu,
            classifier.trace_cov,
            classifier.trace_posterior_cov_div_var,
            Z, classifier.noise_variance,
            classifier.log_det_K,
            classifier.log_det_cov)
        error = np.abs(fx_old - fx)
        fx_old = fx
        if verbose:
            print("({}), error={}".format(iteration, error))
    # Test
    (Z, posterior_predictive_m,
    posterior_std) = classifier.predict(
        classifier.gamma, classifier.cov,
        y, classifier.noise_variance, X_test)  # (N_test, J)
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
        X_new_ = np.zeros((N * N, classifier.D))
        X_new_[:, :2] = X_new
        Z, posterior_predictive_m, posterior_std = classifier.predict(
            classifier.gamma, classifier.cov, y,
            classifier.kernel.varphi, classifier.noise_variance, X_new_)
        Z_new = Z.reshape((N, N, classifier.J))
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
                X_test[np.where(t_test == j)][:, 1],
                color='blue')
            # plt.scatter(
            #     X_train[np.where(t == j + 1)][:, 0],
            #     X_train[np.where(t == j + 1)][:, 1],
            #     color='blue')
            # plt.xlim(xlims)
            # plt.ylim(ylims)
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.savefig("contour_VB_{}.png".format(j))
            plt.close()
    return fx, calculate_metrics(y_test, t_test, Z, classifier.gamma)
