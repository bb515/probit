"""
Ordered probit regression concrete examples. Approximate inference:
VB approximation.
"""
import numpy as np
from probit.estimators import VBOrderedGP
import matplotlib.pyplot as plt
from probit.data.utilities import MinimizeStopper, colors, calculate_metrics


def grid(J, Kernel, X_trains, t_trains, domain, res, now, 
        gamma=None, varphi=None, noise_variance=None, scale=None,
        indices=None):
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
    varphi_0 = 1.0
    noise_variance_0 = 1.0
    kernel = Kernel(varphi=varphi_0, scale=scale)
    for split in range(20):
        # Initiate classifier
        classifier = VBOrderedGP(
            noise_variance_0, X_trains[split], t_trains[split], kernel, J)
        (Z, grad,
        x, y,
        xlabel, ylabel,
        xscale, yscale) = classifier.grid_over_hyperparameters(
            domain, res,
            gamma_0=gamma, varphi_0=varphi, noise_variance_0=noise_variance,
            scale_0=scale, indices=indices, verbose=True)
        Z_av.append(Z)
        grad_av.append(grad)
        if ylabel is None:
            plt.plot(x, Z)
            plt.savefig("grid_over_hyperparameters_{}_{}.png".format(
                split, now))
            plt.close()
            # norm = np.abs(np.max(grad))
            # grad = grad / norm  # normalise - all the same length
            plt.plot(x, Z)
            plt.xscale(xscale)
            plt.ylabel(r"\mathcal{F}(\varphi)")
            plt.savefig("bound_{}_{}.png".format(split, now))
            plt.close()
            plt.plot(x, grad)
            plt.xscale(xscale)
            plt.xlabel(xlabel)
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
        plt.plot(x, Z_av)
        plt.savefig("grid_over_hyperparameters_av_{}.png".format(now))
        plt.close()
        # norm = np.abs(np.max(grad))
        # u = grad / norm
        plt.plot(x, Z_av)
        plt.xscale(xscale)
        plt.ylabel(r"\mathcal{F}(\varphi)")
        plt.savefig("bound_av_{}.png".format(now))
        plt.close()
        plt.plot(x, grad_av)
        plt.xscale(xscale)
        plt.xlabel(xlabel)
        plt.ylabel(r"\frac{\partial \mathcal{F}}{\partial varphi}")
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


def training(
    method, classifier,
    gamma_0, varphi_0, noise_variance_0, scale_0, J,
    indices, max_sec=5000):
    """
    # TODO: Need this in EP and VB?
    An example ordinal training function.
    :arg classifier:
    :type classifier: :class:`probit.estimators.Estimator`
        or :class:`probit.samplers.Sampler`
    :arg gamma_0:
    :type gamma_0:
    :arg varphi_0:
    :type varphi_0:
    :arg noise_variance_0:
    :type noise_variance_0:
    :arg scale_0:
    :type scale_0:
    :arg J:
    :type J:
    :arg method:
    :type method:
    :arg int maxiter:
    :return:
    """
    minimize_stopper = MinimizeStopper(max_sec=max_sec)
    theta = []

    if indices[0]:
        theta.append(np.log(np.sqrt(noise_variance_0)))
    if indices[1]:
        theta.append(gamma_0[1])
    for j in range(2, J):
        if indices[j]:
            theta.append(np.log(gamma_0[j] - gamma_0[j - 1]))
    if indices[J]:
        theta.append(np.log(np.sqrt(scale_0)))
    if indices[J + 1]:  # TODO: replace this with kernel number of hyperparameters.
        theta.append(np.log(varphi_0))
    theta = np.array(theta)
    args = (
        gamma_0, varphi_0, noise_variance_0, scale_0, indices)
    res = minimize(
        classifier.hyperparameter_training_step, theta,
        args, method=method, jac=True,
        callback = minimize_stopper.__call__)
    theta = res.x
    scale = scale_0
    gamma = gamma_0
    varphi = varphi_0
    noise_variance = noise_variance_0
    index = 0
    if indices[0]:
        noise_std = np.exp(theta[index])
        noise_variance = noise_std**2
        scale = scale_0
        index += 1
    gamma = np.empty((J + 1,))
    gamma[0] = np.NINF
    gamma[-1] = np.inf
    if indices[1]:
        gamma[1] = theta[index]
        index += 1
    for j in range(2, J):
        if indices[j]:
            gamma[j] = gamma[j-1] + np.exp(theta[index])
            index += 1
    if indices[J]:
        scale_std = np.exp(theta[index])
        scale = scale_std**2
        index += 1
    if indices[J + 1]:  # TODO: replace this with kernel number of hyperparameters.
        varphi = np.exp(theta[index])
        index += 1
    return gamma, varphi, noise_variance, scale


def VB_plot(
    Kernel, X_train, t_train, X_true, Y_true, m_0, gamma,
    steps, varphi, noise_variance, scale, J, D, domain=None):
    """Plots for Chu data."""
    kernel = Kernel(varphi=varphi, scale=scale)
    # Initiate classifier
    classifier = VBOrderedGP(
        noise_variance, X_train, t_train, kernel)
    y_0 = Y_true.flatten()
    m_0 = y_0
    iteration = 0
    error = np.inf
    noise_std = np.sqrt(noise_variance)
    # TODO: check this works
    (m_tilde, dm_tilde,
    y_tilde, p, varphi_tilde, _) = classifier.estimate(
        steps, gamma=gamma, varphi=varphi, noise_variance=noise_variance,
        m_tilde_0=m_0, fix_hyperparameters=True, write=False)
    while error / steps > classifier.EPS:
        iteration += 1
        (m_0, dm_0, Sigma, cov, C, y, p, *_) = classifier.estimate(
            steps, gamma, varphi_tilde_0=varphi, noise_variance=noise_variance,
            m_tilde_0=m_0, first_step=1, fix_hyperparameters=True, write=False)
        (calligraphic_Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s, *_) = classifier._calligraphic_Z(
            gamma, noise_std, m_0)
        fx, C_inv = classifier.objective(
            classifier.N,
            m_0, Sigma, C, calligraphic_Z, noise_variance,
            numerical_stability=True, verbose=False)
        error = np.abs(fx_old - fx)  # TODO: redundant?
        fx_old = fx
        if 1:
            print("({}), error={}".format(iteration, error))
    fx, C_inv= classifier.objective(
        classifier.N, m_0, Sigma, C,
        calligraphic_Z, noise_variance, verbose=True)
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
            gamma, cov, y, varphi, noise_variance, X_new_)  # (N_test, J)
        Z_new = Z.reshape((N, N, J))
        print(np.sum(Z, axis=1), 'sum')
        for j in range(J):
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, Z_new[:, :, j], zorder=1)
            plt.scatter(X_train[np.where(t_train == j)][:, 0], X_train[np.where(t_train == j)][:, 1], color='red')
            # plt.scatter(X_train[np.where(t == j + 1)][:, 0], X_train[np.where(t == j + 1)][:, 1], color='blue')
            plt.xlabel(r"$x_1$", fontsize=16)
            plt.ylabel(r"$x_2$", fontsize=16)
            plt.savefig("contour_EP_{}.png".format(i))
            plt.close()
    return fx


def VB_plot_synthetic(
        dataset, Kernel, X, t, X_true, Y_true, m_tilde_0,
        gamma, steps, varphi, noise_variance, scale, J, D,
        colors=colors):
    """Plots for synthetic data."""
    kernel = Kernel(varphi=varphi, scale=scale)
    # Initiate classifier
    classifier = VBOrderedGP(noise_variance, X, t, kernel)
    (m_tilde, dm_tilde,
    Sigma_tilde, cov, C, y_tilde, p, varphi_tilde,
    containers) = classifier.estimate(
        steps, gamma, varphi, noise_variance=noise_variance,
        m_tilde_0=m_tilde_0, fix_hyperparameters=True, write=True)
    plt.scatter(X, m_tilde)
    plt.plot(X_true, Y_true)
    plt.show()
    (ms, ys, varphis, psis, fxs) = containers
    plt.plot(fxs)
    plt.title("Variational lower bound on the marginal likelihood")
    plt.show()
    noise_std = np.sqrt(noise_variance)
    (calligraphic_Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s, *_) = classifier._calligraphic_Z(
                    gamma, noise_std, m_tilde,
                    upper_bound=classifier.upper_bound,
                    upper_bound2=classifier.upper_bound2)
    N = classifier.N
    fx, _ = classifier.objective(
        N, m_tilde, Sigma_tilde, C, calligraphic_Z, noise_variance)
    if dataset == "tertile":
        x_lims = (-0.5, 1.5)
        N = 1000
        x = np.linspace(x_lims[0], x_lims[1], N)
        X_new = x.reshape((N, D))
        print("y", y_tilde)
        print("varphi", varphi_tilde)
        print("noisevar", noise_variance)
        (Z, posterior_predictive_m,
        posterior_std) = classifier.predict(
            gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new)
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
        val = 0.5  # this is the value where you want the data to appear on the y-axis.
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
            gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new)
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
            gamma, cov, y_tilde, varphi_tilde, noise_variance, X_new)
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
                X[np.where(t == j)],
                np.zeros_like(X[np.where(t == j)]) + val,
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
                X[np.where(t == j)],
                np.zeros_like(X[np.where(t == j)]),
                s=15, facecolors=colors[j], edgecolors='white')
        plt.savefig("scatter_versus_posterior_mean.png")
        plt.show()
        plt.close()
    return fx


def VB_testing(
        Kernel,
        X_train, t_train, X_test, t_test, y_test,
        steps, gamma, varphi, noise_variance, scale,
        J, D, domain=None):
    kernel = Kernel(varphi=varphi, scale=scale)
    noise_std = np.sqrt(noise_variance)
    # Initiate classifier
    classifier = VBOrderedGP(
        noise_variance, X_train, t_train, kernel, J)
    # Reset error and posterior mean
    iteration = 0
    error = np.inf
    fx_old = np.inf
    m_0 = None
    while error / steps > classifier.EPS:
        iteration += 1
        (m_0, dm_0, y, p, *_) = classifier.estimate(
            steps, gamma, varphi_tilde_0=varphi,
            noise_variance=noise_variance, m_tilde_0=m_0,
            first_step=1, fix_hyperparameters=True, write=False)
        (calligraphic_Z,
        norm_pdf_z1s,
        norm_pdf_z2s,
        z1s,
        z2s,
        *_ )= classifier._calligraphic_Z(
            gamma, noise_std, m_0)
        fx = classifier.objective(
            classifier.N, m_0, y,
            classifier.Sigma_div_var,
            classifier.cov,
            classifier.K,
            calligraphic_Z, noise_variance,
            classifier.log_det_K,
            classifier.log_det_cov)
        error = np.abs(fx_old - fx)  # TODO: redundant?
        fx_old = fx
        if 1:
            print("({}), error={}".format(iteration, error))
    # Test
    (Z, posterior_predictive_m,
    posterior_std) = classifier.predict(
        gamma, classifier.cov,
        y, varphi, noise_variance, X_test)  # (N_test, J)
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
        Z, posterior_predictive_m, posterior_std = classifier.predict(
            gamma, classifier.cov, y, varphi, noise_variance, X_new_)
        Z_new = Z.reshape((N, N, J))
        print(np.sum(Z, axis=1), 'sum')
        for j in range(J):
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, Z_new[:, :, j], zorder=1)
            plt.scatter(
                X_train[np.where(t_train == j)][:, 0],
                X_train[np.where(t_train == j)][:, 1],
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
    return fx, calculate_metrics(y_test, t_test, Z, gamma)


def test(
        Kernel, method, X_trains, t_trains, X_tests, t_tests,
        y_tests, split, steps, gamma_0, varphi_0, noise_variance_0, scale_0, J, D):
    X_train = X_trains[split, :, :]
    t_train = t_trains[split, :]
    X_test = X_tests[split, :, :]
    t_test = t_tests[split, :]
    y_test = y_tests[split, :]
    # kernel = Kernel(varphi=varphi_0, scale=scale)
    # # Initiate classifier
    # classifier = VBOrderedGP(
    #     noise_variance_0, X_train, t_train, kernel, J)
    # # Skip training
    # gamma, varphi, noise_variance = training(
    #     method, classifier,
    #     gamma_0, varphi_0, noise_variance_0, scale_0, J, indices)
    gamma = gamma_0
    varphi = varphi_0
    noise_variance = noise_variance_0
    scale = scale_0
    fx, zero_one, predictive_likelihood, mean_abs, mean_squared, log_pred_prob = VB_testing(
        Kernel, X_train, t_train, X_test, t_test, y_test, steps,
        gamma, varphi, noise_variance, scale, J, D)
    return (
        gamma, varphi, noise_variance, zero_one, predictive_likelihood,
        log_pred_prob, mean_abs, mean_squared, fx)


def outer_loop_problem_size(
        Kernel, method, X_trains, t_trains, X_tests, t_tests, steps,
        gamma_0, varphi_0, noise_variance_0, scale_0, J, D, size, num):
    """
    TODO: problem size. puts an outer loop to plot predictive performance
    for a given set of hyperparameters over the problem size.
    """
    spearman_ranks = []
    mean_zero_one_errors = []
    in_top_3s = []
    in_top_5s = []
    distance_accuracy_1s = []
    distance_accuracy_3s = []
    distance_accuracy_5s = []
    expected_opportunities_recalls = []
    weighted_pearson_ranks = []
    weighted_spearman_ranks = []
    root_mean_squared_errors = []
    mean_absolute_errors = []
    log_predictive_probabilities = []
    predictive_likelihoods = []  # but tells us something about confidence. Overconfidence will be penalised if this is much lower than accuracy
    bounds = []

    # for iter, N in enumerate(np.logspace(1, size, num=num)):
            # (
            #     avg_bound, avg_predictive_likelihood,
            #     avg_zero_one, avg_mean_abs),
            # (
            #     std_bound, std_predictive_likelihood,
            #     std_zero_one, std_mean_abs)
        #  VB_testing(
        #     Kernel, X_train, t_train, X_test, t_test, y_test,
        #     steps, gamma, varphi, noise_variance, scale,
        #     J, D)
    return 0


def outer_loops(
        Kernel, method, X_trains, t_trains, X_tests, t_tests,
        y_tests, steps, gamma_0, varphi_0, noise_variance_0, scale_0, J, D):
    bounds = []
    zero_ones = []
    mean_squares = []
    predictive_likelihoods = []
    mean_abss = []
    varphis = []
    noise_variances = []
    log_pred_probs = []
    gammas = []
    for split in range(20):
        (gamma, varphi, noise_variance, zero_one, predictive_likelihood,
        log_pred_prob, mean_abs, mean_square, fx) = test(
            Kernel, method, X_trains, t_trains, X_tests, t_tests,
            y_tests, split, steps,gamma_0=gamma_0, varphi_0=varphi_0,
            noise_variance_0=noise_variance_0, scale_0=scale_0, J=J, D=D)
        bounds.append(fx)
        zero_ones.append(zero_one)
        mean_squares.append(mean_square)
        predictive_likelihoods.append(predictive_likelihood)
        log_pred_probs.append(log_pred_prob)
        mean_abss.append(mean_abs)
        varphis.append(varphi)
        noise_variances.append(noise_variance)
        gammas.append(gamma[1:-1])
    bounds = np.array(bounds)
    zero_ones = np.array(zero_ones)
    predictive_likelihoods = np.array(predictive_likelihoods)
    log_pred_probs = np.array(log_pred_probs)
    mean_abss = np.array(mean_abss)
    mean_squares = np.array(mean_squares)
    varphis = np.array(varphis)
    noise_variances = np.array(noise_variances)
    gammas = np.array(gammas)
    avg_log_pred_prob = np.average(log_pred_probs)
    std_log_pred_prob = np.std(log_pred_probs)
    avg_bound = np.average(bounds)
    std_bound = np.std(bounds)
    avg_predictive_likelihood = np.average(predictive_likelihoods)
    std_predictive_likelihood = np.std(predictive_likelihoods)
    #print("zero_ones", zero_ones)
    avg_zero_one = np.average(zero_ones)
    std_zero_one = np.std(zero_ones)
    avg_mean_abs = np.average(mean_abss)
    std_mean_abs = np.std(mean_abss)
    avg_mean_squares = np.average(mean_squares)
    std_mean_squares = np.std(mean_squares)
    print(avg_bound, std_bound, "bound, std")
    print(
        avg_log_pred_prob, std_log_pred_prob,
        "log predictive probability, std")
    print(
        avg_predictive_likelihood,
        std_predictive_likelihood,
        "predictive likelihood, std")
    print(avg_zero_one, std_zero_one, "zero one, std")
    print(avg_mean_abs, std_mean_abs, "mean abs, std")
    print(
        avg_mean_squares,
        std_mean_squares, "mean squared error, std")
    avg_varphi = np.average(varphis)
    std_varphi = np.std(varphis)
    avg_noise_variances = np.average(noise_variances)
    std_noise_variances = np.std(noise_variances)
    avg_gammas = np.average(gammas, axis=0)
    std_gammas = np.std(gammas, axis=0)
    print(avg_varphi, std_varphi, "varphi, std")
    print(avg_noise_variances, std_noise_variances, "noise_variances, std")
    print(avg_gammas, std_gammas, "gammas, std")
    return (
        (avg_bound, avg_predictive_likelihood, avg_zero_one, avg_mean_abs),
        (std_bound, std_predictive_likelihood, std_zero_one, std_mean_abs))


def outer_loops_Rogers(
        J, Kernel, X_trains, t_trains, X_tests, t_tests, Y_true, gamma,
        plot=False):
    steps = 50
    grid = np.ogrid[0:len(X_tests[0, :, :])]
    avg_bounds_Z = []
    avg_zero_one_Z = []
    avg_predictive_likelihood_Z = []
    avg_mean_abs_Z = []
    max_bounds = []
    max_zero_ones = []
    max_predictive_likelihoods = []
    max_mean_abss = []
    for split in range(20):
        X_train = X_trains[split, :, :]
        t_train = t_trains[split, :]
        X_test = X_tests[split, :, :]
        t_test = t_tests[split, :]
        # Y_true = Y_trues[split, :]
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
        # Outer loop
        bounds_Z = []
        zero_one_Z = []
        mean_abs_Z = []
        predictive_likelihood_Z = []
        for x_new in X_new:
            noise_std = x_new[0]
            noise_variance = noise_std**2
            varphi = x_new[1]
            kernel = Kernel(varphi=varphi, scale=1.0)
            # Initiate classifier
            y_0 = Y_true.flatten()
            m_0 = y_0
            classifier = VBOrderedGP(
                noise_variance, X_train, t_train, kernel, J)
            (m_tilde, dm_tilde, Sigma_tilde, cov, C_tilde, calligraphic_Z,
            y_tilde, p, varphi_tilde, *_) = classifier.estimate(
                steps, gamma, varphi, noise_variance=noise_variance,
                m_tilde_0=m_0, fix_hyperparameters=True,
                write=False)
            fx = classifier.objective(
                m_tilde, Sigma_tilde, C_tilde, calligraphic_Z, verbose=True)
            bounds_Z.append(fx)
            # Test
            (Z,
            posterior_predictive_m,
            posterior_std) = classifier.predict(
                gamma, cov, y_tilde, varphi_tilde, noise_variance, X_test)
            predictive_likelihood = Z[grid, t_test]
            predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
            predictive_likelihood_Z.append(predictive_likelihood)
            # Mean zero-one error
            t_star = np.argmax(Z, axis=1)
            zero_one = (t_star == t_test)
            mean_zero_one = zero_one * 1
            mean_zero_one = np.sum(mean_zero_one) / len(t_test)
            zero_one_Z.append(mean_zero_one)
            # Other error
            mean_absolute_error = np.sum(np.abs(t_star - t_test)) / len(t_test)
            mean_abs_Z.append(mean_absolute_error)
        avg_bounds_Z.append(bounds_Z)
        avg_zero_one_Z.append(zero_one_Z)
        avg_predictive_likelihood_Z.append(predictive_likelihood_Z)
        avg_mean_abs_Z.append(mean_abs_Z)
        bounds_Z = np.array(bounds_Z)
        predictive_likelihood_Z = np.array(predictive_likelihood_Z)
        zero_one_Z = np.array(zero_one_Z)
        mean_abs_Z = np.array(mean_abs_Z)
        max_bound = np.max(bounds_Z)
        max_predictive_likelihood = np.max(predictive_likelihood_Z)
        max_zero_one = np.max(zero_one_Z)
        max_mean_abs = np.min(mean_abs_Z)
        max_bounds.append(max_bound)
        max_zero_ones.append(max_zero_one)
        max_mean_abss.append(max_mean_abs)
        max_predictive_likelihoods.append(max_predictive_likelihood)
        argmax_bound = np.argmax(bounds_Z)
        argmax_predictive_likelihood = np.argmax(predictive_likelihood_Z)
        argmax_zero_one = np.argmax(zero_one_Z)
        bounds_Z = bounds_Z.reshape((N, N))
        predictive_likelihood_Z = predictive_likelihood_Z.reshape((N, N))
        zero_one_Z = zero_one_Z.reshape((N, N))
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
            plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - mean zero-one accuracy.png")
            plt.close()
    avg_max_bound = np.average(np.array(max_bounds))
    std_max_bound = np.std(np.array(max_bounds))
    avg_max_predictive_likelihood = np.average(np.array(max_predictive_likelihoods))
    std_max_predictive_likelihood = np.std(np.array(max_predictive_likelihoods))
    avg_max_zero_one = np.average(np.array(max_zero_ones))
    std_max_zero_one = np.std(np.array(max_zero_ones))
    avg_max_mean_abs = np.average(np.array(max_mean_abss))
    std_max_mean_abs = np.std(np.array(max_mean_abss))
    print(avg_max_bound, std_max_bound, "average max bound, std")
    print(
        avg_max_predictive_likelihood,
        std_max_predictive_likelihood,
        "average max predictive likelihood, std")
    print(avg_max_zero_one, std_max_zero_one, "average max zero one, std")
    print(avg_max_mean_abs, std_max_mean_abs, "average max mean abs, std")
    avg_bounds_Z = np.array(avg_bounds_Z)
    avg_predictive_likelihood_Z = np.array(avg_predictive_likelihood_Z)
    avg_zero_one_Z = np.array(avg_zero_one_Z)
    avg_mean_abs_Z = np.array(avg_mean_abs_Z)
    avg_bounds_Z = np.average(avg_bounds_Z, axis=0)
    avg_predictive_likelihood_Z = np.average(avg_predictive_likelihood_Z, axis=0)
    avg_zero_one_Z = np.average(avg_zero_one_Z, axis=0)
    avg_mean_abs_Z = np.average(avg_mean_abs_Z, axis=0)
    std_max_bound = np.std(np.array(avg_bounds_Z))
    std_max_predictive_likelihood = np.std(np.array(avg_predictive_likelihood_Z))
    std_max_zero_one = np.std(np.array(avg_zero_one_Z))
    std_max_mean_abs = np.std(np.array(avg_mean_abs_Z))
    argmax_bound = np.argmax(avg_bounds_Z)
    argmax_predictive_likelihood = np.argmax(avg_predictive_likelihood_Z)
    argmax_zero_one = np.argmax(avg_zero_one_Z)
    argmax_mean_abs = np.argmax(avg_mean_abs_Z)
    max_bound = np.max(avg_bounds_Z)
    max_predictive_likelihood = np.max(avg_predictive_likelihood_Z)
    max_zero_one = np.max(avg_zero_one_Z)
    max_mean_abs = np.min(avg_mean_abs_Z)
    print(max_bound, std_max_bound, "Max avg bound, std", X_new[argmax_bound], "parameters")
    print(max_predictive_likelihood, std_max_predictive_likelihood, "Max avg predictive likelihood, std", X_new[argmax_predictive_likelihood], "parameters")
    print(max_zero_one, std_max_zero_one, "Max avg zero one, std", X_new[argmax_zero_one], "parameters")
    print(max_mean_abs, std_max_mean_abs, "Max avg mean abs, std", X_new[argmax_mean_abs], "parameters")
    avg_bounds_Z = avg_bounds_Z.reshape((N, N))
    avg_predictive_likelihood_Z = avg_predictive_likelihood_Z.reshape((N, N))
    avg_zero_one_Z = avg_zero_one_Z.reshape((N, N))
    avg_mean_abs_Z = avg_mean_abs_Z.reshape((N, N))
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_predictive_likelihood_Z)
    plt.scatter(X_new[argmax_predictive_likelihood, 0], X_new[argmax_predictive_likelihood, 1], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - Predictive likelihood of test set.png")
    plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_bounds_Z)
    plt.scatter(X_new[argmax_bound, 0], X_new[argmax_bound, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - Variational lower bound.png")
    plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_zero_one_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - mean zero-one accuracy.png")
    plt.close()
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_mean_abs_Z)
    plt.scatter(X_new[argmax_zero_one, 0], X_new[argmax_zero_one, 0], c='r')
    axs.set_xscale('log')
    axs.set_yscale('log')
    plt.xlabel(r"$\log{\varphi}$", fontsize=16)
    plt.ylabel(r"$\log{s}$", fontsize=16)
    plt.savefig("Contour plot - mean absolute error accuracy.png")
    plt.close()

def grid_synthetic(
    J, Kernel, X_train, t_train, range, res, indices,
    gamma=None, varphi=None, noise_variance=None, scale=1.0, show=False):
    """Grid of optimised lower bound across the hyperparameters with cutpoints set."""
    # Just for initiation
    kernel = Kernel(varphi=1.0, scale=scale)
    # Initiate classifier
    classifier = VBOrderedGP(
        noise_variance, X_train, t_train, kernel, J)
    (Z, grad,
    x, y,
    xlabel, ylabel,
    xscale, yscale) = classifier.grid_over_hyperparameters(
        range, res,
        gamma_0=gamma, varphi_0=varphi, noise_variance_0=noise_variance,
        scale_0=scale, indices=indices)
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


def VB_training(
        Kernel, method, X_train, t_train,
        gamma_0, varphi_0, noise_variance_0, scale_0, J):
    """
    An example ordinal training function.

    Returns the hyperparameters trained via gradient descent of the ELBO.

    :return: gamma, varphi, noise_variance
    """
    # Initiate kernel
    kernel = Kernel(varphi=varphi_0, scale=scale_0)
    # Initiate classifier
    classifier = VBOrderedGP(
        noise_variance_0, X_train, t_train, kernel, J)
    gamma, varphi, noise_variance, scale = training(
        method, classifier,
        gamma_0, varphi_0, noise_variance_0, scale_0, J)
    return gamma, varphi, noise_variance, scale


def test_synthetic(
        Kernel, method, X_train, t_train, X_true, Y_true,
        gamma_0, varphi_0, noise_variance_0, scale_0, J, D, colors=colors):
    """Test some particular hyperparameters."""
    gamma = gamma_0
    varphi = varphi_0
    noise_variance = noise_variance_0
    scale = scale_0
    steps = 1000
    # kernel = Kernel(varphi=varphi_0, scale=scale_0)
    # # Initiate classifier
    # classifier = VBOrderedGP(
    #     noise_variance_0, X_train, t_train, kernel, J)
    # gamma, varphi, noise_variance, scale = VB_training(
    #     Kernel, method, classifier, gamma_0, varphi_0, noise_variance_0, scale_0, J)
    # print("gamma = {}, gamma_0 = {}".format(gamma, gamma_0))
    # print("varphi = {}, varphi_0 = {}".format(varphi, varphi_0))
    # print("noise_variance = {}, noise_variance_0 = {}".format(noise_variance, noise_variance_0))
    # # print("gamma_0 = {}, varphi_0 = {}, noise_variance_0 = {}".format(gamma_0, varphi_0, noise_variance_0))
    fx = VB_plot_synthetic(Kernel, X_train, t_train, X_true, Y_true, None, gamma, steps,
        varphi, noise_variance, scale, J, D, colors=colors)
    print("fx={}".format(fx))
    return fx


# def test_plots(dataset, Kernel, X_test, X_train, t_test, t_train, Y_true, gamma, varphi, noise_variance, scale, J):
#     """TODO: looks like it needs fixing for VB"""
#     grid = np.ogrid[0:len(X_test)]
#     kernel = Kernel(varphi=varphi, scale=scale)
#     # Initiate classifier
#     classifier = VBOrderedGP(noise_variance, X_train, t_train, kernel)
#     steps = 50
#     y_0 = Y_true.flatten()
#     m_0 = y_0
#     (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma,
#      mean_EP, precision_EP, amplitude_EP, containers) = classifier.estimate(
#         steps, gamma, varphi, noise_variance, fix_hyperparameters=False, write=True)
#     weights, precision_EP, L, Lambda = classifier.compute_EP_weights(
#         precision_EP, mean_EP, grad_Z_wrt_cavity_mean)

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
