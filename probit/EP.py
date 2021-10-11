"""Approximate inference: EP approximation. Methods for inference and test."""
import numpy as np
from probit.estimators import EPOrderedGP
import matplotlib.pyplot as plt
from probit.data.utilities import datasets, colors
from probit.data.utilities import calculate_metrics


def plot(
        classifier, domain=None):
    """Plots for Chu data."""
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
    t1, *_ = classifier.compute_integrals(
        classifier.gamma, Sigma, precision_EP, posterior_mean,
        classifier.noise_variance)
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
    """Plots for synthetic data."""
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
    t1, *_ = classifier.compute_integrals(
        classifier.gamma, Sigma, precision_EP, posterior_mean,
        classifier.noise_variance)
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


def test(
        classifier, X_test, t_test, y_test, L=None, Lambda=None, domain=None):
    """Test the trained model."""
    grid = np.ogrid[0:len(X_test[:, :])]
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
            steps, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
            precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP,
            write=True)
        print("iteration {}, error={}".format(iteration, error / steps))
    (weights,
    precision_EP,
    L,
    Lambda) = classifier.compute_EP_weights(
        precision_EP, mean_EP, grad_Z_wrt_cavity_mean,
        L, Lambda)
    t1, *_ = classifier.compute_integrals(
        classifier.gamma, Sigma, precision_EP, posterior_mean,
        classifier.noise_variance)
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


def outer_loops(
        Kernel, method, X_trains, t_trains, X_tests, t_tests, y_tests,
        gamma_0, varphi_0, noise_variance_0, scale_0, J, D):
    bounds = []
    zero_ones = []
    predictive_likelihoods = []
    mean_abss = []
    varphis = []
    noise_variances = []
    log_pred_probs = []
    gammas = []
    for split in range(20):
        (
            gamma, varphi, noise_variance, zero_one, predictive_likelihood,
            log_pred_prob, mean_abs, fx) = test(
                Kernel, method, X_trains, t_trains, X_tests, t_tests,
                y_tests,
                split, gamma_0=gamma_0, varphi_0=varphi_0,
                noise_variance_0=noise_variance_0, scale_0=scale_0, J=J, D=D)
        bounds.append(fx)
        zero_ones.append(zero_one)
        predictive_likelihoods.append(predictive_likelihood)
        mean_abss.append(mean_abs)
        log_pred_probs.append(log_pred_prob)
        varphis.append(varphi)
        noise_variances.append(noise_variance)
        gammas.append(gamma[1:-1])
    bounds = np.array(bounds)
    zero_ones = np.array(zero_ones)
    predictive_likelihoods = np.array(predictive_likelihoods)
    log_pred_probs = np.array(log_pred_probs)
    mean_abss = np.array(mean_abss)
    varphis = np.array(varphis)
    noise_variances = np.array(noise_variances)
    gammas = np.array(gammas)
    avg_bound = np.average(bounds)
    std_bound = np.std(bounds)
    avg_log_pred_prob = np.average(log_pred_probs)
    std_log_pred_prob = np.std(log_pred_probs)
    avg_predictive_likelihood = np.average(predictive_likelihoods)
    std_predictive_likelihood = np.std(predictive_likelihoods)
    print("zero_ones", zero_ones)
    avg_zero_one = np.average(zero_ones)
    std_zero_one = np.std(zero_ones)
    avg_mean_abs = np.average(mean_abss)
    std_mean_abs = np.std(mean_abss)
    print(avg_bound, std_bound, "bound, std")
    print(
        avg_log_pred_prob, std_log_pred_prob,
        "log predictive probability, std")
    print(
        avg_predictive_likelihood, std_predictive_likelihood,
        "predictive likelihood, std")
    print(avg_zero_one, std_zero_one, "zero one, std")
    print(avg_mean_abs, std_mean_abs, "mean abs, std")
    avg_varphi = np.average(varphis)
    std_varphi = np.std(varphis)
    avg_noise_variances = np.average(noise_variances)
    std_noise_variances = np.std(noise_variances)
    avg_gammas = np.average(gammas, axis=0)
    std_gammas = np.std(gammas, axis=0)
    print(avg_varphi, std_varphi, "varphi, std")
    print(avg_noise_variances, std_noise_variances, "noise_variances, std")
    print(avg_gammas, std_gammas, "gammas, std")
    return 0


def outer_loops_test(
        Kernel, X_trains, t_trains, X_tests, t_tests,
        gamma_0, varphi_0, noise_variance_0, scale_0, J, D, verbose=False):
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
            (fx, zero_one, predictive_likelihood,
            mean_absolute_error) = EP_testing(
                X_train, t_train, X_test, t_test,
                gamma_0, varphi, noise_variance, J, D, plot=False)
            bounds_Z.append(fx)
            predictive_likelihood_Z.append(predictive_likelihood)
            zero_one_Z.append(zero_one)
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
        if verbose:
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, predictive_likelihood_Z)
            plt.scatter(
                X_new[argmax_predictive_likelihood, 0],
                X_new[argmax_predictive_likelihood, 1], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - Predictive likelihood of test set.png")
            plt.close()
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, bounds_Z)
            plt.scatter(
                X_new[argmax_bound, 0],
                X_new[argmax_bound, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - Variational lower bound.png")
            plt.close()
            fig, axs = plt.subplots(1, figsize=(6, 6))
            plt.contourf(x1, x2, zero_one_Z)
            plt.scatter(
                X_new[argmax_zero_one, 0],
                X_new[argmax_zero_one, 0], c='r')
            axs.set_xscale('log')
            axs.set_yscale('log')
            plt.xlabel(r"$\log{\varphi}$", fontsize=16)
            plt.ylabel(r"$\log{s}$", fontsize=16)
            plt.savefig("Contour plot - mean zero-one accuracy.png")
            plt.close()
    avg_max_bound = np.average(np.array(max_bounds))
    std_max_bound = np.std(np.array(max_bounds))
    avg_max_predictive_likelihood = np.average(
        np.array(max_predictive_likelihoods))
    std_max_predictive_likelihood = np.std(
        np.array(max_predictive_likelihoods))
    avg_max_zero_one = np.average(np.array(max_zero_ones))
    std_max_zero_one = np.std(np.array(max_zero_ones))
    avg_max_mean_abs = np.average(np.array(max_mean_abss))
    std_max_mean_abs = np.std(np.array(max_mean_abss))
    print(avg_max_bound, std_max_bound, "average max bound, std")
    print(
        avg_max_predictive_likelihood, std_max_predictive_likelihood,
        "average max predictive likelihood, std")
    print(avg_max_zero_one, std_max_zero_one, "average max zero one, std")
    print(avg_max_mean_abs, std_max_mean_abs, "average max mean abs, std")
    avg_bounds_Z = np.array(avg_bounds_Z)
    avg_predictive_likelihood_Z = np.array(avg_predictive_likelihood_Z)
    avg_zero_one_Z = np.array(avg_zero_one_Z)
    avg_mean_abs_Z = np.array(avg_mean_abs_Z)
    avg_bounds_Z = np.average(avg_bounds_Z, axis=0)
    avg_predictive_likelihood_Z = np.average(
        avg_predictive_likelihood_Z, axis=0)
    avg_zero_one_Z = np.average(avg_zero_one_Z, axis=0)
    avg_mean_abs_Z = np.average(avg_mean_abs_Z, axis=0)
    std_max_bound = np.std(np.array(avg_bounds_Z))
    std_max_predictive_likelihood = np.std(
        np.array(avg_predictive_likelihood_Z))
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
    print(
        max_bound,
        std_max_bound,
        "Max avg bound, std",
        X_new[argmax_bound],
        "parameters")
    print(
        max_predictive_likelihood,
        std_max_predictive_likelihood,
        "Max avg predictive likelihood, std",
    X_new[argmax_predictive_likelihood], "parameters")
    print(
        max_zero_one,
        std_max_zero_one,
        "Max avg zero one, std",
        X_new[argmax_zero_one],
        "parameters")
    print(
        max_mean_abs,
        std_max_mean_abs,
        "Max avg mean abs, std",
        X_new[argmax_mean_abs],
        "parameters")
    avg_bounds_Z = avg_bounds_Z.reshape((N, N))
    avg_predictive_likelihood_Z = avg_predictive_likelihood_Z.reshape((N, N))
    avg_zero_one_Z = avg_zero_one_Z.reshape((N, N))
    avg_mean_abs_Z = avg_mean_abs_Z.reshape((N, N))
    fig, axs = plt.subplots(1, figsize=(6, 6))
    plt.contourf(x1, x2, avg_predictive_likelihood_Z)
    plt.scatter(
        X_new[argmax_predictive_likelihood, 0],
        X_new[argmax_predictive_likelihood, 1], c='r')
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

def grid_synthetic(classifier, domain, res, indices,
                   gamma=None, varphi=None, noise_variance=None,
                   scale=None, show=False):
    """
    Grid of optimised lower bound across the hyperparameters.
    """
    (Z, grad, x, y,
    xlabel, ylabel,
    xscale, yscale) = classifier.grid_over_hyperparameters(
        domain, res, indices=indices)
    if ylabel is None:
        plt.plot(x, Z)
        plt.savefig("grid_over_hyperparameters.png")
        if show: plt.show()
        plt.close()
        plt.plot(x, Z, 'b')
        #plt.vlines(30.0, -80, 200, 'k', alpha=0.5, label=r"'true' $\varphi$")
        plt.xscale(xscale)
        plt.ylabel(r"$\mathcal{F}$")
        plt.xlabel(xlabel)
        plt.savefig("bound.png")
        if show: plt.show()
        plt.close()
        plt.plot(x, grad, 'r')
        #plt.vlines(30.0, -80, 20, 'k', alpha=0.5, label=r"'true' $\varphi$")
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
        # plt.vlines(
        #     np.log(30.0), -80, 20, 'k', alpha=0.5, label=r"'true' $\varphi$")
        plt.xlabel("log " + xlabel)
        plt.ylabel(r"$\frac{\partial \mathcal{F}}{\partial \varphi}$")
        plt.plot(
            log_x, dZ_, 'r--',
            label=r"$\frac{\partial \mathcal{F}}{\partial \varphi}$ numeric")
        plt.legend()
        plt.savefig("both.png")
        if show: plt.show()
        plt.close()
        # plt.vlines(
        #     np.log(30.0), -80, 200, 'k', alpha=0.5, label=r"'true' $\varphi$")
        plt.plot(log_x, Z, 'b', label=r"$\mathcal{F}}$")
        # plt.plot(
        #   log_x, grad, 'r',
        #   label=r"$\frac{\partial \mathcal{F}}{\partial \varphi}$ analytic")
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
        plt.yscale(yscale)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.savefig("Contour plot - EP lower bound on the log likelihood.png")
        if show: plt.show() 
        plt.close()


def grid(
        Kernel, X_trains, t_trains, domain, res, now, 
        gamma=None, varphi=None, noise_variance=None, scale=None,
        indices=None):
    """Grid of optimised lower bound across the chosen hyperparameters."""
    Z_av = []
    grad_av = []
    varphi_0 = 1.0
    kernel = Kernel(varphi=varphi_0, scale=scale)
    for split in range(20):
        # Initiate classifier
        classifier = EPOrderedGP(X_trains[split], t_trains[split], kernel)
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
            plt.savefig("grid_over_hyperparameters_{}_{}.png".format(split, now))
            plt.close()
            # norm = np.abs(np.max(grad))
            # u = grad / norm
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
            plt.savefig("grid_over_hyperparameters_{}_{}.png".format(split, now))
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
            plt.savefig("Contour plot - EP lower bound on the log likelihood_{}_{}.png".format(split, now))
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
        plt.savefig("Contour plot - EP lower bound on the log likelihood_av_{}.png".format(now))
        plt.close()
