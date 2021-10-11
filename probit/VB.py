"""
Ordered probit regression concrete examples. Approximate inference:
VB approximation.
"""
import numpy as np
import matplotlib.pyplot as plt
from probit.data.utilities import colors
from probit.data.utilities import calculate_metrics


def grid(classifier, X_trains, t_trains, domain, res, now, indices=None):
    """
    TODO: merge with probit.probit.EP.<method_name> once classifiers take same
    args.
    Grid of (optimised and converged) variational lower bound across the chosen
    hyperparameters.

    :arg int J: The number of ordinal classes.
    :arg Kernel:
    :type Kernel: :class:`Kernel`
    :arg X_trains:
    :type X_trains: :class:`numpy.ndarray`
    :arg t_trains:
    :type t_trains: :class:`numpy.ndarray`
    :arg domain:
    :type domain: (tuple, tuple) or (tuple, None)
    :arg res:
    :type res: (tuple, tuple) or (tuple, None)
    :arg string now: A string for plots, can be e.g., a time stamp, and is
        to distinguish between plots.
    :arg gamma:
    :type gamma: :class:`numpy.ndarray` or None
    :arg varphi:
    :type varphi: :class:`numpy.ndarray` or float or None
    :arg noise_variance:
    :type noise_variance: float or None
    :arg scale:
    :type scale: float or None:
    :arg indices:
    :type indices: :class:`numpy.ndarray`
    """
    Z_av = []
    grad_av = []
    for split in range(20):
        # Reinitiate classifier with new data
        # TODO: not sure if this is good code, but makes sense conceptually
        # as new data requires a new model.
        classifier.__init__(
            classifier.gamma, classifier.noise_variance,
            classifier.kernel, X_trains[split], t_trains[split], classifier.J)
        (Z, grad,
        x, y,
        xlabel, ylabel,
        xscale, yscale) = classifier.grid_over_hyperparameters(
            domain, res, indices=indices, verbose=True)
        Z_av.append(Z)
        grad_av.append(grad)
        if ylabel is None:
            # norm = np.abs(np.max(grad))
            # grad = grad / norm  # normalise - all the same length
            index = np.argmin(Z)
            plt.scatter(
                x[index], Z[index],
                color='red', label=r"$\varphi$ = {}".format(x[index]))
            plt.legend()
            plt.plot(x, Z)
            plt.xscale(xscale)
            plt.xlabel(r"$\varphi$")
            plt.ylabel(r"$\mathcal{F}(\varphi)$")
            plt.savefig("bound_{}_{}.png".format(split, now))
            plt.close()
            plt.plot(x, grad)
            plt.xscale(xscale)
            plt.xlabel(xlabel)
            plt.xlabel(r"$\varphi$")
            plt.ylabel(r"\frac{\partial \mathcal{F}}{\partial varphi}")
            plt.savefig("grad_{}_{}.png".format(split, now))
            plt.close()
        else:
            fig, axs = plt.subplots(1, figsize=(6, 6))
            ax = plt.axes(projection='3d')
            ax.plot_surface(x, y, Z, rstride=1, cstride=1, alpha=0.4,
                            cmap='viridis', edgecolor='none')
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.savefig("grid_over_hyperparameters_{}_{}.png".format(
                split, now))
            plt.close()
            norm = np.linalg.norm(np.array((grad[:, 0], grad[:, 1])), axis=0)
            u = grad[:, 0] / norm
            v = grad[:, 1] / norm
            fig, ax = plt.subplots(1, 1)
            ax.set_aspect(1)
            ax.contourf(x, y, np.log(Z), 100, cmap='viridis', zorder=1)
            ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
            ax.plot(0.1, 30, 'm')
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            plt.savefig("Contour plot - EP lower bound on the log "
                "likelihood_{}_{}.png".format(split, now))
            plt.close()
    Z_av = np.mean(np.array(Z_av), axis=0)
    grad_av = np.mean(np.array(grad_av), axis=0)
    if ylabel is None:
        # norm = np.abs(np.max(grad))
        # u = grad / norm
        plt.plot(x, Z_av)
        plt.xscale(xscale)
        plt.ylabel(r"$\mathcal{F}(\varphi)$")
        plt.xlabel(r"$\varphi$")
        index = np.argmin(Z_av)
        plt.scatter(
            x[index], Z_av[index],
            color='red', label=r"\varphi = {}".format(x[index]))
        plt.legend()
        plt.savefig("bound_av_{}.png".format(now))
        plt.close()
        plt.plot(x, grad_av)
        plt.xscale(xscale)
        plt.xlabel(xlabel)
        plt.xlabel(r"$\varphi$")
        plt.ylabel(r"$\frac{\partial \mathcal{F}}{\partial varphi}$")
        plt.savefig("grad_av_{}.png".format(now))
        plt.close()
    else:
        fig, axs = plt.subplots(1, figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, Z_av, rstride=1, cstride=1, alpha=0.4,
                        cmap='viridis', edgecolor='none')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.savefig("grid_over_hyperparameters_av_{}.png".format(now))
        plt.close()
        norm = np.linalg.norm(np.array((grad_av[:, 0], grad_av[:, 1])), axis=0)
        u = grad_av[:, 0] / norm
        v = grad_av[:, 1] / norm
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.contourf(x, y, np.log(Z_av), 100, cmap='viridis', zorder=1)
        ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        ax.plot(0.1, 30, 'm')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.savefig("Contour plot - EP lower bound on the log "
            "likelihood_av_{}.png".format(now))
        plt.close()


def VB_plot(classifier, m_0, steps, J, D, domain=None):
    """
    TODO: needs generalizing to other datasets other than Chu.
    Contour plot of the predictive probabilities over a chosen domain.
    """
    iteration = 0
    error = np.inf
    while error / steps > classifier.EPS:
        iteration += 1
        (m_0, dm_0, Sigma, cov, C, y, p, *_) = classifier.estimate(
            steps, m_tilde_0=m_0, first_step=1,
            fix_hyperparameters=True, write=False)
        (calligraphic_Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s, *_) = classifier._calligraphic_Z(
            classifier.gamma, classifier.noise_std, m_0)
        fx, C_inv = classifier.objective(
            classifier.N,
            m_0, Sigma, C, calligraphic_Z, classifier.noise_variance,
            numerical_stability=True, verbose=False)
        error = np.abs(fx_old - fx)  # TODO: redundant?
        fx_old = fx
        if 1:
            print("({}), error={}".format(iteration, error))
    fx, C_inv= classifier.objective(
        classifier.N, m_0, Sigma, C,
        calligraphic_Z, classifier.noise_variance, verbose=True)
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
            classifier.gamma, cov, y, classifier.varphi,
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


def VB_plot_synthetic(
        classifier,
        dataset, X_true, Y_true, m_tilde_0, steps, colors=colors):
    """
    Plots for synthetic data.

    TODO: needs generalizing to other datasets other than Chu.
    """
    (m_tilde, dm_tilde,
    Sigma_tilde, cov, C, y_tilde, p, containers) = classifier.estimate(
        steps, m_tilde_0=m_tilde_0, fix_hyperparameters=True, write=True)
    plt.scatter(classifier.X, m_tilde)
    plt.plot(X_true, Y_true)
    plt.show()
    (ms, ys, varphis, psis, fxs) = containers
    plt.plot(fxs)
    plt.title("Variational lower bound on the marginal likelihood")
    plt.show()
    (calligraphic_Z, *_) = classifier._calligraphic_Z(
                    classifier.gamma, classifier.noise_std, m_tilde,
                    upper_bound=classifier.upper_bound,
                    upper_bound2=classifier.upper_bound2)
    J = classifier.J
    N = classifier.N
    fx, _ = classifier.objective(
        N, m_tilde, Sigma_tilde, C, calligraphic_Z, classifier.noise_variance)
    if dataset == "tertile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, classifier.D))
        print("y", y_tilde)
        print("varphi", classifier.kernel.varphi)
        print("noisevar", classifier.noise_variance)
        (Z, posterior_predictive_m,
        posterior_std) = classifier.predict(
            classifier.gamma, cov, y_tilde, classifier.kernel.varphi,
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
            plt.scatter(X[np.where(t == j)], np.zeros_like(
                X[np.where(t == j)]) + val, s=15, facecolors=colors[j],
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
        plt.plot(X_true, Y_true, 'b')
        plt.ylim(-2.2, 2.2)
        plt.xlim(-0.5, 1.5)
        for j in range(J):
            plt.scatter(
                X[np.where(t == j)],
                np.zeros_like(X[np.where(t == j)]) + val,
                s=15, facecolors=colors[j], edgecolors='white')
        plt.savefig("scatter_versus_posterior_mean.png")
        plt.show()
        plt.close()
    elif dataset == "septile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        (Z, posterior_predictive_m,
        posterior_std) = classifier.predict(
            classifier.gamma, cov, y_tilde, classifier.kernel.varphi,
            classifier.noise_variance, X_new)
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
                r"$p(\omega_{*}=6|x, X, \omega)$"),
            colors=(
                colors[0], colors[1], colors[2],
                colors[3], colors[4], colors[5], colors[6]))
        plt.legend()
        val = 0.5  # the value where the data appears on the y-axis
        for j in range(J):
            plt.scatter(
                X[np.where(t == j)],
                np.zeros_like(X[np.where(t == j)]) + val,
                s=15, facecolors=colors[j], edgecolors='white')
        plt.savefig(
            "Ordered Gibbs Cumulative distribution plot of\nclass "
            "distributions for x_new=[{}, {}].png".format(
                x_lims[1], x_lims[0]))
        plt.close()
    elif dataset=="thirteen":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        (Z,
        posterior_predictive_m,
        posterior_std) = classifier.predict(
            classifier.gamma, cov, y_tilde, classifier.kernel.varphi,
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
        val = 0.5  # where the data to appears on the y-axis
        for j in range(J):
            plt.scatter(
                classifier.X_train[np.where(classifier.t_train == j)],
                np.zeros_like(
                    classifier.X_train[np.where(
                        classifier.t_train == j)]) + val,
                s=15, facecolors=colors[j], edgecolors='white')
        plt.savefig(
            "Ordered Gibbs Cumulative distribution plot of class distributions"
            " for x_new=[{}, {}].png".format(x_lims[0], x_lims[1]))
        plt.show()
        plt.close()
        np.savez(
            "VB_thirteen.npz",
            x=X_new, y=posterior_predictive_m, s=posterior_std)
        plt.plot(X_new, posterior_predictive_m, 'r')
        plt.fill_between(
            X_new[:, 0],
            posterior_predictive_m - 2*posterior_std,
            posterior_predictive_m + 2*posterior_std, color='red', alpha=0.2)
        plt.plot(X_true, Y_true, 'b')
        plt.ylim(-2.2, 2.2)
        plt.xlim(-0.5, 1.5)
        for j in range(J):
            plt.scatter(
                classifier.X_train[np.where(classifier.t_train == j)],
                np.zeros_like(
                    classifier.X_train[np.where(classifier.t_train == j)]),
                s=15, facecolors=colors[j], edgecolors='white')
        plt.savefig("scatter_versus_posterior_mean.png")
        plt.show()
        plt.close()
    return fx


def test(
        classifier, X_test, t_test, y_test,
        steps, domain=None):
    iteration = 0
    error = np.inf
    fx_old = np.inf
    m_0 = None
    print(steps)
    while error / steps > classifier.EPS:
        iteration += 1
        (m_0, dm_0, y, p, *_) = classifier.estimate(
            steps, m_tilde_0=m_0, first_step=1, write=False)
        (calligraphic_Z,
        norm_pdf_z1s,
        norm_pdf_z2s,
        z1s,
        z2s,
        *_ )= classifier._calligraphic_Z(
            classifier.gamma, classifier.noise_std, m_0)
        fx = classifier.objective(
            classifier.N, m_0, y,
            classifier.Sigma_div_var,
            classifier.cov,
            classifier.K,
            calligraphic_Z, classifier.noise_variance,
            classifier.log_det_K,
            classifier.log_det_cov)
        error = np.abs(fx_old - fx)  # TODO: redundant?
        fx_old = fx
        if 1:
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



def outer_loop_problem_size(
        Classifier, Kernel, method, X_trains, t_trains, X_tests, t_tests,
        y_tests, steps,
        gamma_0, varphi_0, noise_variance_0, scale_0, J, D, size, num,
        string="client"):
    """
    Plots outer loop for metrics and variational lower bound over N_train
    problem size.

    :arg Classifier:
    :type Classifier:
    :arg Kernel:
    :type Kernel:
    :arg method:
    :type method:
    :arg X_trains:
    :type X_trains:
    :arg t_trains:
    :type t_trains:
    :arg X_tests:
    :type X_tests:
    :arg  t_tests:
    :type t_tests:
    :arg y_tests:
    :type y_tests:
    :arg steps:
    :arg gamma_0
    :type gamma_0:
    :arg varphi_0:
    :type varphi_0:
    :arg noise_variance_0:
    :type noise_variance_0:
    :arg scale_0:
    :type scale_0:
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
            Classifier, Kernel, method,
            X_trains[:, :N, :], t_trains[:, :N],
            X_tests, t_tests, y_tests, steps,
            gamma_0, varphi_0, noise_variance_0, scale_0, J, D)
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

    plt.plot(plot_N, plot_mean_fx, '-', color='gray',
        label="variational bound +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_fx - plot_std_fx, plot_mean_fx + plot_std_fx, 
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} fx.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 13], '-', color='gray',
        label="RMSE +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 13] - plot_std_metrics[:, 13],
        plot_mean_metrics[:, 13] + plot_std_metrics[:, 13],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} RMSE.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 14], '-', color='gray',
        label="MAE +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 14] - plot_std_metrics[:, 14],
        plot_mean_metrics[:, 14] + plot_std_metrics[:, 14],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} MAE.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 15], '-', color='gray',
        label="log predictive probability +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 15] - plot_std_metrics[:, 15],
        plot_mean_metrics[:, 15] + plot_std_metrics[:, 15],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} log predictive probability.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 4], '-', color='gray',
        label="mean zero-one accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 4] - plot_std_metrics[:, 4],
        plot_mean_metrics[:, 4] + plot_std_metrics[:, 4],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} mean zero-one accuracy.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 4], '-', color='black',
        label="in top 1 accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 4] - plot_std_metrics[:, 4],
        plot_mean_metrics[:, 4] + plot_std_metrics[:, 4],
        color='black', alpha=0.2)
    plt.plot(plot_N, plot_mean_metrics[:, 5], '-', color='gray',
        label="in top 3 accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 5] - plot_std_metrics[:, 5],
        plot_mean_metrics[:, 5] + plot_std_metrics[:, 5],
        color='gray', alpha=0.2)
    plt.plot(plot_N, plot_mean_metrics[:, 6], '-', color='lightgray',
        label="in top 5 accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 6] - plot_std_metrics[:, 6],
        plot_mean_metrics[:, 6] + plot_std_metrics[:, 6],
        color='lightgray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} in top accuracy.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 7], '-', color='black',
        label="distance 1 accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 7] - plot_std_metrics[:, 7],
        plot_mean_metrics[:, 7] + plot_std_metrics[:, 7],
        color='black', alpha=0.2)
    plt.plot(plot_N, plot_mean_metrics[:, 8], '-', color='gray',
        label="distance 3 accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 8] - plot_std_metrics[:, 8],
        plot_mean_metrics[:, 8] + plot_std_metrics[:, 8],
        color='gray', alpha=0.2)
    plt.plot(plot_N, plot_mean_metrics[:, 9], '-', color='lightgray',
        label="distance 5 accuracy +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 9] - plot_std_metrics[:, 9],
        plot_mean_metrics[:, 9] + plot_std_metrics[:, 9],
        color='lightgray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} distance accuracy.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 0], '-', color='gray',
        label="f1 score +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 0] - plot_std_metrics[:, 0],
        plot_mean_metrics[:, 0] + plot_std_metrics[:, 0],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} f1 score.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 1], '-', color='gray',
        label="uncertainty plus +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 1] - plot_std_metrics[:, 1],
        plot_mean_metrics[:, 1] + plot_std_metrics[:, 1],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} uncertainty plus.png".format(string))
    plt.close()

    plt.plot(plot_N, plot_mean_metrics[:, 2], '-', color='gray',
        label="uncertainty minus +/- 1 std")
    plt.fill_between(
        plot_N, plot_mean_metrics[:, 2] - plot_std_metrics[:, 2],
        plot_mean_metrics[:, 2] + plot_std_metrics[:, 2],
        color='gray', alpha=0.2)
    plt.xscale("log")
    plt.legend()
    plt.savefig("{} uncertainty minus.png".format(string))
    plt.close()

    np.savez(
            "{} plot.npz".format(string),
            plot_N=plot_N, plot_mean_fx=plot_mean_fx, plot_std_fx=plot_std_fx,
            plot_mean_metrics=plot_mean_metrics,
            plot_std_metrics=plot_std_metrics)

    return 0


def outer_loops(
        Classifier, Kernel, method, X_trains, t_trains, X_tests, t_tests,
        y_tests, steps, gamma_0, varphi_0, noise_variance_0, scale_0, J, D):
    moments_fx = []
    #moments_varphi = []
    #moments_noise_variance = []
    #moments_gamma = []
    moments_metrics = []
    for split in range(5):
        # Reset kernel
        kernel = Kernel(varphi=varphi_0, scale=scale_0)
        # Build the classifier with the new training data
        classifier = Classifier(
            gamma_0, noise_variance_0, kernel,
            X_trains[split, :, :], t_trains[split, :], J)
        fx, metrics = test(
            classifier,
            X_tests[split, :, :], t_tests[split, :],
            y_tests[split, :],
            steps)
        moments_fx.append(fx / classifier.N)  # if divided by N it is average per datapoint
        moments_metrics.append(metrics)
        # moments_varphi.append(classifier.varphi)
        # moments_noise_variance.append(classifier.noise_variance)
        # moments_gamma.append(classifier.gamma[1:-1])
    moments_fx = np.array(moments_fx)
    moments_metrics = np.array(moments_metrics)
    
    mean_fx = np.average(moments_fx)
    mean_metrics = np.average(moments_metrics, axis=0)
    std_fx = np.std(moments_fx)
    std_metrics = np.std(moments_metrics, axis=0)
    return mean_fx, std_fx, mean_metrics, std_metrics

def outer_loops_Rogers(
        Classifier, Kernel, X_trains, t_trains, X_tests, t_tests, y_tests,
        gamma_0, varphi_0, noise_variance_0, scale_0, J, D, plot=False):
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
        kernel = Kernel(varphi=varphi_0, scale=scale_0)
        # Build the classifier with the new training data
        classifier = Classifier(
            gamma_0, noise_variance_0, kernel,
            X_trains[split, :, :], t_trains[split, :, :], J)
        X_test = X_tests[split, :, :]
        t_test = t_tests[split, :]
        y_test = y_tests[split, :]
        # Outer loop
        fx_Z = []
        metrics_Z = []
        for x_new in X_new:
            noise_std = x_new[0]
            noise_variance = noise_std**2
            varphi = x_new[1]
            classifier.hyperparameters_update(
                noise_variance=noise_variance, varphi=varphi)
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
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, predictive_likelihood_Z)
            plt.scatter(X_new[argmax_predictive_likelihood, 0], X_new[argmax_predictive_likelihood, 1], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - Predictive likelihood of test set.png")
            plt.close()
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, bounds_Z)
            plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - Variational lower bound.png")
            plt.close()
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, zero_one_Z)
            plt.scatter(
                X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - mean zero-one accuracy.png")
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
    # plt.contourf(x1, x2, mean_metrics_Z[0])
    # plt.scatter(X_new[argmax_metrics[0], 0], X_new[argmax_metrics[0], 1], c='r')
    # axs.set_xscale('log')
    # axs.set_yscale('log')
    # plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    # plt.ylabel(r"$\log{s}$", fontsize=16)
    # plt.savefig("Contour plot - Predictive likelihood of test set.png")
    # plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, mean_fx_Z)
    plt.scatter(X_new[argmax_fx, 0], X_new[argmax_fx, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - Variational lower bound.png")
    plt.close()

def grid_synthetic(
    classifier, domain, res, indices, show=False):
    """Grid of optimised lower bound across the hyperparameters with cutpoints set."""
    (Z, grad,
    x, y,
    xlabel, ylabel,
    xscale, yscale) = classifier.grid_over_hyperparameters(
        domain=domain, res=res, indices=indices)
    print("xscale={}, yscale={}".format(xscale, yscale))
    if ylabel is None:
        plt.plot(x, Z)
        plt.savefig("grid_over_hyperparameters.png")
        if show: plt.show()
        plt.close()
        plt.plot(x, Z, 'b')
        plt.vlines(30.0, -80, 20, 'k', alpha=0.5, label=r"'true' $\varphi$")
        plt.xlabel(xlabel)
        plt.xscale(xscale)
        plt.ylabel(r"$\mathcal{F}$")
        plt.savefig("bound.png")
        if show: plt.show()
        plt.close()
        plt.plot(x, grad, 'r')
        plt.vlines(30.0, -20, 20, 'k', alpha=0.5, label=r"'true' $\varphi$")
        plt.xscale(xscale)
        plt.xlabel(xlabel)
        plt.ylabel(r"$\frac{\partial \mathcal{F}}{\partial \varphi}$")
        plt.savefig("grad.png")
        if show: plt.show()
        plt.close()
        #Normalization:
        #First derivatives: need to calculate them in the log domain
        log_x = np.log(x)
        dlog_x = np.diff(log_x)
        dZ_ = np.gradient(Z, log_x)
        dZ = np.diff(Z) / dlog_x
        plt.plot(
            log_x, grad, 'r',
            label=r"$\frac{\partial \mathcal{F}}{\partial \varphi}$ analytic")
        plt.vlines(
            np.log(30.0), -20, 20, 'k', alpha=0.5,
            label=r"'true' $\log \varphi$")
        plt.xlabel("log " + xlabel)
        plt.ylabel(
            r"$\frac{\partial \mathcal{F}}{\partial \varphi}$")
        plt.plot(
            log_x, dZ_, 'r--',
            label=r"$\frac{\partial \mathcal{F}}{\partial \varphi}$ numeric")
        plt.legend()
        plt.savefig("both.png")
        if show: plt.show()
        plt.close()
        plt.vlines(
            np.log(30.0), -80, 20, 'k', alpha=0.5,
            label=r"'true' $\log \varphi$")
        plt.plot(log_x, Z, 'b', label=r"$\mathcal{F}}$")
        plt.plot(
            log_x, grad, 'r',
            label=r"$\frac{\partial \mathcal{F}}{\partial \varphi}$ analytic")
        plt.plot(
            log_x, dZ_, 'r--',
            label=r"$\frac{\partial \mathcal{F}}{\partial \varphi}$ numeric")
        plt.xlabel("log " + xlabel)
        plt.legend()
        plt.savefig("bound_grad.png")
        if show: plt.show()
        plt.close()
    else:
        fig, axs = plt.subplots(1, figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, Z, rstride=1, cstride=1, alpha=0.4,
                        cmap='viridis', edgecolor='none')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.savefig("grid_over_hyperparameters.png")
        if show: plt.show()
        plt.close()
        norm = np.linalg.norm(np.array((grad[:, 0], grad[:, 1])), axis=0)
        u = grad[:, 0] / norm
        v = grad[:, 1] / norm
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)
        ax.contourf(x, y, Z, 100, cmap='viridis', zorder=1)
        ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        ax.plot(0.1, 30, 'm')
        plt.xscale(xscale)
        plt.xlim(1, 100.0)  #TODO: What is this for?
        plt.yscale(yscale)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.savefig("Contour plot - VB lower bound on the log likelihood.png")
        if show: plt.show()
        plt.close()
        # fig, ax = plt.subplots(1, 1)
        # ax.set_aspect(1)
        # ax.contourf(x, y, np.log(Z), 100, cmap='viridis', zorder=1)
        # ax.quiver(x, y, u, v, units='xy', scale=0.5, color='red')
        # ax.plot(0.1, 30, 'm')
        # plt.xscale(xscale)
        # plt.yscale(yscale)
        # plt.xlabel(xlabel, fontsize=16)
        # plt.ylabel(ylabel, fontsize=16)
        # plt.savefig("Contour plot - log VB lower bound on the log likelihood.png")
        # if show: plt.show()
        # plt.close()


# def test_plots(
#     dataset, classifier, X_test, t_test, Y_true):
#     """TODO: looks like it needs fixing for VB"""
#     grid = np.ogrid[0:len(X_test)]
#     steps = 50

#     *_ = classifier.estimate(
#         steps, gamma, varphi, noise_variance, fix_hyperparameters=False, write=True)

#     if dataset in datasets:
#         lower_x1 = 0.0
#         upper_x1 = 16.0
#         lower_x2 = -30
#         upper_x2 = 0
#         N = 60

#         x1 = np.linspace(lower_x1, upper_x1, N)
#         x2 = np.linspace(lower_x2, upper_x2, N)
#         xx, yy = np.meshgrid(x1, x2)
#         X_new = np.dstack((xx, yy))
#         X_new = X_new.reshape((N * N, 2))

#         # Test
#         Z = classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda,
#                                            vectorised=True)
#         predictive_likelihood = Z[grid, t_test]
#         predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
#         print("predictive_likelihood ", predictive_likelihood)

#         # Mean zero-one error
#         t_star = np.argmax(Z, axis=1)
#         print(t_star)
#         print(t_test)
#         zero_one = (t_star == t_test)
#         mean_zero_one = zero_one * 1
#         mean_zero_one = np.sum(mean_zero_one) / len(t_test)
#         print("mean_zero_one ", mean_zero_one)

#         Z = classifier.predict(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda,
#                                            vectorised=True)
#         Z_new = Z.reshape((N, N, J))
#         print(np.sum(Z, axis=1), 'sum')
#         for j in range(J):
#             fig, axs = plt.subplots(1, figsize=(6, 6))
#             plt.contourf(x1, x2, Z_new[:, :, j], zorder=1)
#             plt.scatter(X_train[np.where(t_train == j)][:, 0], X_train[np.where(t_train == j)][:, 1], color='red')
#             plt.scatter(X_test[np.where(t_test == j)][:, 0], X_test[np.where(t_test == j)][:, 1], color='blue')
#             #plt.scatter(X_train[np.where(t == j + 1)][:, 0], X_train[np.where(t == j + 1)][:, 1], color='blue')
#             # plt.xlim(0, 2)
#             # plt.ylim(0, 2)
#             plt.xlabel(r"$x_1$", fontsize=16)
#             plt.ylabel(r"$x_2$", fontsize=16)
#             plt.savefig("Contour plot - Variational.png")
#             plt.close()
