"""
Ordinal regression, pseudo-marginal (PM) inference.

Fig. 1. Comparison of the posterior distribution $p(\theta|\mathbf{y})$ with
the posterior $p(\theta|\mathbf{f})$ in the SA parameterization, the posterior
$p(\theta|\mathbf{y}, \boldsymbol{\nu})$ in the AA parameterization, and the
parameterization used in the SURR method.
"""
# Make sure to limit CPU usage
import os
from re import A
os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = "6"
from numba import set_num_threads
set_num_threads(6)

import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
from scipy.stats import multivariate_normal
from probit.approximators import EPGP
from probit.samplers import (
    GibbsGP, EllipticalSliceGP,
    SufficientAugmentation, AncilliaryAugmentation, PseudoMarginal)
from probit.plot import outer_loops, grid_synthetic
from probit.Gibbs import plot
from probit.proposals import proposal_initiate
from probit.plot import figure2
import pathlib
from probit.data.utilities import (
    datasets, load_data, load_data_synthetic, load_data_paper)
import time
import matplotlib.pyplot as plt


now = time.ctime()
write_path = pathlib.Path()


def main():
    """Conduct Gibbs exact inference and hyperparameter sampling."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "bins", help="3, 13 or 52")
    parser.add_argument(
        "method", help="SA or AA")  # TODO: Surrogate
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    bins = int(args.bins)
    method = args.method
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets["benchmark"]:
        (X_trains, t_trains,
        X_tests, t_tests,
        X_true, g_tests,
        cutpoints_0, varphi_0, noise_variance_0, scale_0,
        J, D, Kernel) = load_data(
            dataset, bins)
        burn_steps = 2000
        steps = 1000
        f_0 = np.zeros(len(X_tests))
        g_0 = f_0.copy()
        # outer_loops(
        #     test, GibsGP, Kernel, X_trains, t_trains, X_tests,
        #     t_tests, burn_steps, steps,
        #     cutpoints_0, varphi_0, noise_variance_0, scale_0, J, D)
        # Initiate kernel
        kernel = Kernel(
            varphi=varphi_0, variance=scale_0)
        # Initiate classifier
        sampler = GibbsGP(
            cutpoints_0, noise_variance_0, kernel, J, (X_trains[2], t_trains[2]))
        plot(sampler, f_0, g_0, cutpoints_0, burn_steps, steps, J, D)
    elif dataset in datasets["paper"] or dataset in datasets["synthetic"]:
        if dataset in datasets["paper"]:
            (X, f_true, g_true, y,
                cutpoints_0, lengthscale_0, noise_variance_0, variance_0,
                J, D, colors, Kernel) = load_data_paper(dataset, plot=True)
        else:
            (X, y,
            X_true, g_true,
            cutpoints_0, varphi_0, noise_variance_0, scale_0,
            J, D, colors, Kernel) = load_data_synthetic(dataset, bins)

        # Set lengthscale hyperparameters
        lengthscale_hyperparameters = np.array([3.4, 2.0])  # [loc, scale] of a normal on np.exp(lengthscale)  # TODO check if this is true via the jacobian

        # Initiate kernel
        kernel = Kernel(
            varphi=lengthscale_0,
            variance=variance_0,
            varphi_hyperparameters=lengthscale_hyperparameters)
        # Initiate classifier
        burn_steps = 500
        steps = 1000
        # sampler = GibbsGP(cutpoints_0, noise_variance_0, kernel, J, (X, t))
        noise_std_hyperparameters = None
        cutpoints_hyperparameters = None
        sampler = EllipticalSliceGP(
            cutpoints_0, noise_variance_0,
            noise_std_hyperparameters,
            cutpoints_hyperparameters, kernel, J, (X, y))
        M = 100
        # varphis_step = varphis[1:] - varphis[:-1]
        # varphis = varphis[:-1]
        indices = np.ones(J + 2)
        # Fix noise_variance
        indices[0] = 0
        # Fix scale
        indices[J] = 0
        # Fix varphi
        #indices[-1] = 0
        # Fix cutpoints
        indices[1:J] = 0
        # Just varphi
        domain = ((-1.5, 0.33), None)
        res = (M + 1, None)

        # SS
        varphis = np.logspace(domain[0][0], domain[0][1], res[0])
        varphis_step = varphis[1:] - varphis[:-1]
        varphis = varphis[1:]

        theta = sampler.get_phi(indices)
        proposal_cov = 0.05
        proposal_L_cov = proposal_initiate(theta, indices, proposal_cov)

        # Pseudo Marginal approach - EP
        # Initiate hyper-parameter sampler
        approximator = EPGP(
            cutpoints_0, noise_variance_0,
            kernel, J, (X, y))

        # # Ancilliary Augmentation approach
        # hyper_sampler = AncilliaryAugmentation(sampler)
        # log_p_theta_giv_y_nus = []
        # for i, varphi in enumerate(varphis):
        #     print(i)
        #     # Need to update sampler hyperparameters
        #     sampler.hyperparameters_update(varphi=varphi)
        #     theta=sampler.get_phi(indices)
        #     log_p_theta_giv_y_nu = hyper_sampler.tmp_compute_marginal(
        #         f_true, theta, indices, proposal_L_cov, reparameterised=True)
        #     log_p_theta_giv_y_nus.append(log_p_theta_giv_y_nu)
        # plt.plot(varphis, log_p_theta_giv_y_nus)
        # plt.savefig("AAa {}.png".format(lengthscale_0))
        # plt.show()
        # plt.close()
        # max_log_p_theta_giv_y_nus = np.max(log_p_theta_giv_y_nus)
        # log_sum_exp = max_log_p_theta_giv_y_nus + np.log(np.sum(np.exp(log_p_theta_giv_y_nus - max_log_p_theta_giv_y_nus)))
        # p_theta_giv_y_nus = np.exp(log_p_theta_giv_y_nus - log_sum_exp - np.log(varphis_step))
        # plt.plot(varphis, p_theta_giv_y_nus)
        # plt.savefig("AAb {}.png".format(lengthscale_0))
        # plt.show()
        # plt.close()

        # # Sufficient Augmentation approach
        # hyper_sampler = SufficientAugmentation(sampler)
        # log_p_theta_giv_ms = []
        # for i, varphi in enumerate(varphis):
        #     print(i)
        #     # Need to update sampler hyperparameters
        #     sampler.hyperparameters_update(varphi=varphi)
        #     theta=sampler.get_phi(indices)
        #     log_p_theta_giv_ms.append(hyper_sampler.tmp_compute_marginal(
        #             f_true, theta, indices, proposal_L_cov, reparameterised=True))
        # plt.plot(log_p_theta_giv_ms, 'k')
        # plt.savefig("SAa.png")
        # plt.show()
        # plt.close()
        # max_log_p_theta_giv_ms = np.max(log_p_theta_giv_ms)
        # log_sum_exp = max_log_p_theta_giv_ms + np.log(np.sum(np.exp(log_p_theta_giv_ms - max_log_p_theta_giv_ms)))
        # p_theta_giv_ms = np.exp(log_p_theta_giv_ms - log_sum_exp - np.log(varphis_step))
        # plt.plot(varphis, p_theta_giv_ms)
        # plt.savefig("SAb.png")
        # plt.show()
        # plt.close()

        hyper_sampler = PseudoMarginal(approximator)

        for hyper_sampler, sampler, label, color in [
            #(AncilliaryAugmentation(sampler), sampler, "AA", 'r'),
            #(SufficientAugmentation(sampler), sampler, "SA", 'b'),
            (PseudoMarginal(approximator), approximator, "PM", 'k')]:

            # plot figures
            (theta, p_pseudo_marginals_mean, p_pseudo_marginals_lo,
                    p_pseudo_marginals_hi, p_priors) = figure2(
                hyper_sampler, approximator, domain, res, indices,
                num_importance_samples=64, steps=steps,
                reparameterised=False, show=True, write=True, verbose=False)
            plt.plot(theta, p_pseudo_marginals_mean, color,
                label=label)
            # axes = plt.gca()
            # y_min_0, y_max_0 = axes.get_ylim()
        plt.xlabel(r"Length-scale, $\ell$")
        plt.ylabel("Posterior density")
        plt.title("N = {}".format(len(y)))
        plt.savefig("fig1.png")
        plt.close()

        # max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginals_ms)
        # log_sum_exp = max_log_p_pseudo_marginals + np.log(np.sum(np.exp(log_p_pseudo_marginals_ms - max_log_p_pseudo_marginals)))
        # p_pseudo_marginals = np.exp(log_p_pseudo_marginals_ms - log_sum_exp - np.log(varphis_step))

        # plt.plot(varphis, p_pseudo_marginals)
        # # plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        # # plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r') 
        # plt.savefig("fig2b.png")
        # plt.show()
        # plt.close()

        # max_log_p_pseudo_marginals = np.max(log_p_pseudo_marginalss, axis=0)
        # print(np.shape(max_log_p_pseudo_marginals))
        # log_sum_exp = np.tile(max_log_p_pseudo_marginals, (M, 1)) + np.tile(np.log(np.sum(np.exp(log_p_pseudo_marginalss - max_log_p_pseudo_marginals), axis=0)), (M, 1))
        # p_pseudo_marginals = np.exp(log_p_pseudo_marginalss - log_sum_exp - np.log(varphis_step).reshape(-1, 1))

        # p_pseudo_marginals_mean = np.mean(p_pseudo_marginals, axis=1)
        # p_pseudo_marginals_std = np.std(p_pseudo_marginals, axis=1)

        # plt.plot(varphis, p_pseudo_marginals_mean)
        # plt.plot(varphis, p_pseudo_marginals_mean + p_pseudo_marginals_std, '--b')
        # plt.plot(varphis, p_pseudo_marginals_mean - p_pseudo_marginals_std, '--r') 
        # plt.savefig("fig2c.png")
        # plt.show()
        # plt.close() 

        # plt.plot(varphis, p_pseudo_marginals, label="PM")
        # plt.plot(varphis, p_theta_giv_ms, label="SA")
        # plt.plot(varphis, p_theta_giv_y_nus, label="AA")

        # plot(sampler, f_0, cutpoints_0, burn_steps, steps, J, D)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    #sys.stdout.close()


if __name__ == "__main__":
    main()


# # Burn in
# f_samples, g_samples, cutpoints_samples = gibbs_classifier.sample(f_0, g_0, cutpoints_0, steps_burn)
# #f_samples, g_samples, cutpoints_samples = gibbs_classifier.sample_metropolis_within_gibbs(f_0, g_0, cutpoints_0, 0.5, steps_burn)
# f_0_burned = f_samples[-1]
# g_0_burned = g_samples[-1]
# cutpoints_0_burned = cutpoints_samples[-1]

# # Sample
# f_samples, g_samples, cutpoints_samples = gibbs_classifier.sample(f_0_burned, g_0_burned, cutpoints_0_burned, steps)
# #f_samples, g_samples, cutpoints_samples = gibbs_classifier.sample_metropolis_within_gibbs(f_0, g_0, cutpoints_0, 0.5, steps)
# f_tilde = np.mean(f_samples, axis=0)
# g_tilde = np.mean(g_samples, axis=0)
# cutpoints_tilde = np.mean(cutpoints_samples, axis=0)

# if argument == "diabetes_quantile":
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].plot(cutpoints_samples[:, 1])
#     ax[0].set_ylabel(r"$\cutpoints_1$", fontsize=16)
#     ax[1].plot(cutpoints_samples[:, 2])
#     ax[1].set_ylabel(r"$\cutpoints_2$", fontsize=16)
#     plt.title('Mixing for cutpoint posterior samples $\cutpoints$')
#     plt.show()

#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].plot(cutpoints_samples[:, 3])
#     ax[0].set_ylabel(r"$\cutpoints_3$", fontsize=16)
#     ax[1].plot(cutpoints_samples[:, 4])
#     ax[1].set_ylabel(r"$\cutpoints_4$", fontsize=16)
#     plt.title('Mixing for cutpoint posterior samples $\cutpoints$')
#     plt.show()

#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].plot(cutpoints_samples[:, 5])
#     ax[0].set_ylabel(r"$\cutpoints_5$", fontsize=16)
#     ax[1].plot(cutpoints_samples[:, 6])
#     ax[1].set_ylabel(r"$\cutpoints_6$", fontsize=16)
#     plt.title('Mixing for cutpoint posterior samples $\cutpoints$')
#     plt.show()

#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     g_star = -1. * np.ones(3)
#     n0, g0, patches = ax[0].hist(cutpoints_samples[:, 1], 20, density="probability", histtype='stepfilled')
#     n1, g1, patches = ax[1].hist(cutpoints_samples[:, 2], 20, density="probability", histtype='stepfilled')
#     g_star[0] = g0[np.argmax(n0)]
#     g_star[1] = g1[np.argmax(n1)]
#     ax[0].axvline(g_star[0], color='k', label=r"Maximum $\cutpoints_1$")
#     ax[1].axvline(g_star[1], color='k', label=r"Maximum $\cutpoints_2$")
#     ax[0].set_xlabel(r"$\cutpoints_1$", fontsize=16)
#     ax[1].set_xlabel(r"$\cutpoints_2$", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()
#     plt.title(r"$\cutpoints$ posterior samples")
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     f_star = -1. * np.ones(3)
#     n0, m00, patches = ax[0].hist(f_samples[:, 0], 20, density="probability", histtype='stepfilled')
#     n1, m01, patches = ax[1].hist(f_samples[:, 1], 20, density="probability", histtype='stepfilled')
#     n2, m20, patches = ax[2].hist(f_samples[:, 2], 20, density="probability", histtype='stepfilled')
#     f_star[0] = m00[np.argmax(n0)]
#     f_star[1] = m01[np.argmax(n1)]
#     f_star[2] = m20[np.argmax(n2)]
#     ax[0].axvline(f_star[0], color='k', label=r"Maximum $f_0$")
#     ax[1].axvline(f_star[1], color='k', label=r"Maximum $m_1$")
#     ax[2].axvline(f_star[2], color='k', label=r"Maximum $m_2$")
#     ax[0].set_xlabel(r"$f_0$", fontsize=16)
#     ax[1].set_xlabel(r"$m_1$", fontsize=16)
#     ax[2].set_xlabel(r"$m_2$", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     plt.title(r"$f$ posterior samples")
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     g_star = -1. * np.ones(3)
#     n0, y00, patches = ax[0].hist(g_samples[:, 0], 20, density="probability", histtype='stepfilled')
#     n1, y01, patches = ax[1].hist(g_samples[:, 1], 20, density="probability", histtype='stepfilled')
#     n2, y20, patches = ax[2].hist(g_samples[:, 2], 20, density="probability", histtype='stepfilled')
#     g_star[0] = y00[np.argmax(n0)]
#     g_star[1] = y01[np.argmax(n1)]
#     g_star[2] = y20[np.argmax(n2)]
#     ax[0].axvline(g_star[0], color='k', label=r"Maximum $g_0$")
#     ax[1].axvline(g_star[1], color='k', label=r"Maximum $y_1$")
#     ax[2].axvline(g_star[2], color='k', label=r"Maximum $y_2$")
#     ax[0].set_xlabel(r"$g_0$", fontsize=16)
#     ax[1].set_xlabel(r"$y_1$", fontsize=16)
#     ax[2].set_xlabel(r"$y_2$", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     plt.title(r"$y$ posterior samples")
#     plt.show()

#     # plt.scatter(X[np.where(t == 0)], m_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
#     # plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
#     # plt.scatter(X[np.where(t == 2)], m_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
#     # plt.xlabel(r"$x$", fontsize=16)
#     # plt.ylabel(r"$\tilde{f}$", fontsize=16)
#     # plt.title("GP regression posterior sample mean mbar, plotted against x")
#     # plt.show()
#     #
#     # plt.scatter(X[np.where(t == 0)], y_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
#     # plt.scatter(X[np.where(t == 1)], y_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
#     # plt.scatter(X[np.where(t == 2)], y_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
#     # plt.xlabel(r"$x$", fontsize=16)
#     # plt.ylabel(r"$\tilde{g}$", fontsize=16)
#     # plt.title("Latent variable posterior sample mean ybar, plotted against x")
#     # plt.show()

#     lower_x1 = 0.0
#     upper_x1 = 16.0
#     lower_x2 = -30
#     upper_x2 = 0
#     N = 60
#     x1 = np.linspace(lower_x1, upper_x1, N)
#     x2 = np.linspace(lower_x2, upper_x2, N)
#     xx, yy = np.meshgrid(x1, x2)
#     X_new = np.dstack((xx, yy))
#     X_new = X_new.reshape((N * N, D))

#     # Test
#     Z = gibbs_classifier.predict(g_samples, cutpoints_samples, X_test, vectorised=True)  # (n_test, K)

#     # Mean zero-one error
#     t_star = np.argmax(Z, axis=1)
#     print(t_star)
#     print(t_test)
#     zero_one = np.logical_and(t_star, t_test)
#     mean_zero_one = zero_one * 1
#     mean_zero_one = np.sum(mean_zero_one) / len(t_test)
#     print(mean_zero_one)

#     # X_new = x.reshape((N, D))
#     print(np.shape(cutpoints_samples), 'shape cutpoints')
#     Z = gibbs_classifier.predict(g_samples, cutpoints_samples, X_new, vectorised=True)
#     Z_new = Z.reshape((N, N, K))
#     print(np.sum(Z, axis=1), 'sum')

#     for i in range(6):
#         fig, axs = plt.subplots(1, figsize=(6, 6))
#         plt.contourf(x1, x2, Z_new[:, :, i], zorder=1)
#         plt.scatter(X[np.where(t == i)][:, 0], X[np.where(t == i)][:, 1], color='red')
#         plt.scatter(X[np.where(t == i + 1)][:, 0], X[np.where(t == i + 1)][:, 1], color='blue')
#         # plt.scatter(X[np.where(t == 2)][:, 0], X[np.where(t == 2)][:, 1])
#         # plt.scatter(X[np.where(t == 3)][:, 0], X[np.where(t == 3)][:, 1])
#         # plt.scatter(X[np.where(t == 4)][:, 0], X[np.where(t == 4)][:, 1])
#         # plt.scatter(X[np.where(t == 5)][:, 0], X[np.where(t == 5)][:, 1])

#         # plt.xlim(0, 2)
#         # plt.ylim(0, 2)
#         plt.legend()
#         plt.xlabel(r"$x_1$", fontsize=16)
#         plt.ylabel(r"$x_2$", fontsize=16)
#         plt.title("Contour plot - Gibbs")
#         plt.show()

#     # plt.xlim(lower_x, upper_x)
#     # plt.ylim(0.0, 1.0)
#     # plt.xlabel(r"$x$", fontsize=16)
#     # plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
#     # plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
#     #           .format(lower_x, upper_x))
#     # plt.stackplot(x, Z.T,
#     #               labels=(
#     #                   r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
#     #               colors=(
#     #                   colors[0], colors[1], colors[2])
#     #               )
#     # plt.legend()
#     # val = 0.5  # this is the value where you want the data to appear on the y-axis.
#     # plt.scatter(X[np.where(t == 0)], np.zeros_like(X[np.where(t == 0)]) + val, facecolors=colors[0], edgecolors='white')
#     # plt.scatter(X[np.where(t == 1)], np.zeros_like(X[np.where(t == 1)]) + val, facecolors=colors[1], edgecolors='white')
#     # plt.scatter(X[np.where(t == 2)], np.zeros_like(X[np.where(t == 2)]) + val, facecolors=colors[2], edgecolors='white')
#     # plt.show()

# elif argument == "tertile":
#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     ax[0].plot(cutpoints_samples[:, 1])
#     ax[0].set_ylabel(r"$\cutpoints_1$", fontsize=16)
#     ax[1].plot(cutpoints_samples[:, 2])
#     ax[1].set_ylabel(r"$\cutpoints_1$", fontsize=16)
#     plt.title('Mixing for cutpoint posterior samples $\cutpoints$')
#     plt.show()

#     fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#     g_star = -1. * np.ones(3)
#     n0, g0, patches = ax[0].hist(cutpoints_samples[:, 1], 20, density="probability", histtype='stepfilled')
#     n1, g1, patches = ax[1].hist(cutpoints_samples[:, 2], 20, density="probability", histtype='stepfilled')
#     g_star[0] = g0[np.argmax(n0)]
#     g_star[1] = g1[np.argmax(n1)]
#     ax[0].axvline(g_star[0], color='k', label=r"Maximum $\cutpoints_1$")
#     ax[1].axvline(g_star[1], color='k', label=r"Maximum $\cutpoints_2$")
#     ax[0].set_xlabel(r"$\cutpoints_1$", fontsize=16)
#     ax[1].set_xlabel(r"$\cutpoints_2$", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()
#     plt.title(r"$\cutpoints$ posterior samples")
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     f_star = -1. * np.ones(3)
#     n0, m00, patches = ax[0].hist(f_samples[:, 0], 20, density="probability", histtype='stepfilled')
#     n1, m01, patches = ax[1].hist(f_samples[:, 1], 20, density="probability", histtype='stepfilled')
#     n2, m20, patches = ax[2].hist(f_samples[:, 2], 20, density="probability", histtype='stepfilled')
#     f_star[0] = m00[np.argmax(n0)]
#     f_star[1] = m01[np.argmax(n1)]
#     f_star[2] = m20[np.argmax(n2)]
#     ax[0].axvline(f_star[0], color='k', label=r"Maximum $f_0$")
#     ax[1].axvline(f_star[1], color='k', label=r"Maximum $m_1$")
#     ax[2].axvline(f_star[2], color='k', label=r"Maximum $m_2$")
#     ax[0].set_xlabel(r"$f_0$", fontsize=16)
#     ax[1].set_xlabel(r"$m_1$", fontsize=16)
#     ax[2].set_xlabel(r"$m_2$", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     plt.title(r"$m$ posterior samples")
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     g_star = -1. * np.ones(3)
#     n0, y00, patches = ax[0].hist(g_samples[:, 0], 20, density="probability", histtype='stepfilled')
#     n1, y01, patches = ax[1].hist(g_samples[:, 1], 20, density="probability", histtype='stepfilled')
#     n2, y20, patches = ax[2].hist(g_samples[:, 2], 20, density="probability", histtype='stepfilled')
#     g_star[0] = y00[np.argmax(n0)]
#     g_star[1] = y01[np.argmax(n1)]
#     g_star[2] = y20[np.argmax(n2)]
#     ax[0].axvline(g_star[0], color='k', label=r"Maximum $g_0$")
#     ax[1].axvline(g_star[1], color='k', label=r"Maximum $y_1$")
#     ax[2].axvline(g_star[2], color='k', label=r"Maximum $y_2$")
#     ax[0].set_xlabel(r"$g_0$", fontsize=16)
#     ax[1].set_xlabel(r"$y_1$", fontsize=16)
#     ax[2].set_xlabel(r"$y_2$", fontsize=16)
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     plt.title(r"$y$ posterior samples")
#     plt.show()

#     plt.scatter(X[np.where(t == 0)], m_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
#     plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
#     plt.scatter(X[np.where(t == 2)], m_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
#     plt.xlabel(r"$x$", fontsize=16)
#     plt.ylabel(r"$\tilde{f}$", fontsize=16)
#     plt.title("GP regression posterior sample mean mbar, plotted against x")
#     plt.show()

#     plt.scatter(X[np.where(t == 0)], y_tilde[np.where(t == 0)], color=colors[0], label=r"$t={}$".format(1))
#     plt.scatter(X[np.where(t == 1)], y_tilde[np.where(t == 1)], color=colors[1], label=r"$t={}$".format(2))
#     plt.scatter(X[np.where(t == 2)], y_tilde[np.where(t == 2)], color=colors[2], label=r"$t={}$".format(3))
#     plt.xlabel(r"$x$", fontsize=16)
#     plt.ylabel(r"$\tilde{g}$", fontsize=16)
#     plt.title("Latent variable posterior sample mean ybar, plotted against x")
#     plt.show()

#     lower_x = -0.5
#     upper_x = 1.5
#     N = 1000
#     x = np.linspace(lower_x, upper_x, N)
#     X_new = x.reshape((N, D))
#     print(np.shape(cutpoints_samples), 'shape cutpoints')
#     Z = gibbs_classifier.predict(g_samples, cutpoints_samples, X_new, vectorised=True)
#     print(np.sum(Z, axis=1), 'sum')
#     plt.xlim(lower_x, upper_x)
#     plt.ylim(0.0, 1.0)
#     plt.xlabel(r"$x$", fontsize=16)
#     plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
#     plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
#               .format(lower_x, upper_x))
#     plt.stackplot(x, Z.T,
#                   labels=(
#                       r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$"),
#                   colors=(
#                       colors[0], colors[1], colors[2])
#                   )
#     plt.legend()
#     val = 0.5  # this is the value where you want the data to appear on the y-axis.
#     plt.scatter(X[np.where(t == 0)], np.zeros_like(X[np.where(t == 0)]) + val, facecolors=colors[0], edgecolors='white')
#     plt.scatter(X[np.where(t == 1)], np.zeros_like(X[np.where(t == 1)]) + val, facecolors=colors[1], edgecolors='white')
#     plt.scatter(X[np.where(t == 2)], np.zeros_like(X[np.where(t == 2)]) + val, facecolors=colors[2], edgecolors='white')
#     plt.show()
# elif argument == "septile":
#     fig, ax = plt.subplots(1, 6, figsize=(30, 5))
#     for i in range(6):
#         ax[i].plot(cutpoints_samples[:, i + 1])
#         ax[i].set_ylabel(r"$\cutpoints_{}$".format(i + 1), fontsize=16)
#     plt.title('Mixing for cutpoint posterior samples $\cutpoints$')
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     for i in range(3):
#         ax[i].plot(f_samples[:, i])
#         ax[i].set_ylabel(r"$m{}$".format(i), fontsize=16)
#     plt.title('Mixing for GP posterior samples $m$')
#     plt.show()

#     fig, ax = plt.subplots(1, 6, figsize=(30, 5))
#     g_star = -1. * np.ones(6)
#     for i in range(6):
#         ni, gi, patches = ax[i].hist(cutpoints_samples[:, i + 1], 20, density="probability", histtype='stepfilled')
#         g_star[i] = gi[np.argmax(ni)]
#         ax[i].axvline(g_star[i], color='k', label=r"Maximum $\cutpoints_{}$".format(i+1))
#         ax[i].set_xlabel(r"$\cutpoints_{}$".format(i+1), fontsize=16)
#         ax[i].legend()
#     plt.title(r"$\cutpoints$ posterior samples")
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     f_star = -1. * np.ones(3)
#     for i in range(3):
#         ni, mi, patches = ax[i].hist(f_samples[:, i + 1], 20, density="probability", histtype='stepfilled')
#         f_star[i] = mi[np.argmax(ni)]
#         ax[i].axvline(f_star[i], color='k', label=r"Maximum $m_{}$".format(i + 1))
#         ax[i].set_xlabel(r"$m_{}$ posterior samples".format(i + 1), fontsize=16)
#         ax[i].legend()
#     plt.title(r"$m$ posterior samples")
#     plt.show()

#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#     g_star = -1. * np.ones(3)
#     for i in range(3):
#         ni, yi, patches = ax[i].hist(g_samples[:, i + 1], 20, density="probability", histtype='stepfilled')
#         g_star[i] = yi[np.argmax(ni)]
#         ax[i].axvline(g_star[i], color='k', label=r"Maximum $y_{}$".format(i + 1))
#         ax[i].set_xlabel(r"$y_{}$ posterior samples".format(i + 1), fontsize=16)
#         ax[i].legend()
#     plt.title(r"$y$ posterior samples")
#     plt.show()

#     #plt.scatter(X, m_tilde)
#     #plt.scatter(X[np.where(t == 1)], m_tilde[np.where(t == 1)])
#     for i in range(7):
#         plt.scatter(X[np.where(t == i)], m_tilde[np.where(t == i)], color=colors[i], label=r"$t={}$".format(i + 1))
#     plt.xlabel(r"$x$", fontsize=16)
#     plt.ylabel(r"$\tilde{f}$", fontsize=16)
#     plt.title("GP regression posterior sample mean mbar, plotted against x")
#     plt.legend()
#     plt.show()

#     for i in range(7):
#         plt.scatter(X[np.where(t == i)], y_tilde[np.where(t == i)], color=colors[i], label=r"$t={}$".format(i))
#     plt.xlabel(r"$x$", fontsize=16)
#     plt.ylabel(r"$\tilde{g}$", fontsize=16)
#     plt.title("Latent variable posterior sample mean ybar, plotted against x")
#     plt.legend()
#     plt.show()

#     lower_x = -0.5
#     upper_x = 1.5
#     N = 1000
#     x = np.linspace(lower_x, upper_x, N)
#     X_new = x.reshape((N, D))
#     print(np.shape(cutpoints_samples), 'shape cutpoints')
#     Z = gibbs_classifier.predict(g_samples, cutpoints_samples, X_new, vectorised=True)
#     print(np.sum(Z, axis=1), 'sum')
#     plt.xlim(lower_x, upper_x)
#     plt.ylim(0.0, 1.0)
#     plt.xlabel(r"$x$", fontsize=16)
#     plt.ylabel(r"$p(t={}|x, X, t)$", fontsize=16)
#     plt.title(" Ordered Gibbs Cumulative distribution plot of\nclass distributions for x_new=[{}, {}] and the data"
#               .format(lower_x, upper_x))
#     plt.stackplot(x, Z.T,
#                   labels=(
#                       r"$p(t=0|x, X, t)$", r"$p(t=1|x, X, t)$", r"$p(t=2|x, X, t)$", r"$p(t=3|x, X, t)$",
#                       r"$p(t=4|x, X, t)$", r"$p(t=5|x, X, t)$", r"$p(t=6|x, X, t)$"),
#                   colors=(
#                       colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6])
#                   )
#     plt.legend()
#     val = 0.5  # this is the value where you want the data to appear on the y-axis.
#     for i in range(7):
#         plt.scatter(X[np.where(t == i)], np.zeros_like(X[np.where(t == i)]) + val, facecolors=colors[i], edgecolors='white')
#     plt.show()



# # fig, ax = plt.subplots(5, 10, sharex='col', sharey='row')
# # # axes are in a two-dimensional array, indexed by [row, col]
# # for i in range(5):
# #     for j in range(10):
# #         ax[i, j].bar(np.array([0, 1, 2], dtype=np.intc), Z[i*5 + j])
# #         ax[i, j].set_xticks(np.array([0, 1, 2], dtype=np.intc), minor=False)
# # plt.show()

# #
# # plt.hist(X, bins=20)
# # plt.xlabel(r"$y$", fontsize=16)
# # plt.ylabel("counts")
# # plt.show()

# # for k in range(K):
# #     _ = plt.subplots(1, figsize=(6, 6))
# #     plt.scatter(X0[:, 0], X0[:, 1], color='b', label=r"$t=0$", zorder=10)
# #     plt.scatter(X1[:, 0], X1[:, 1], color='r', label=r"$t=1$", zorder=10)
# #     plt.scatter(X2[:, 0], X2[:, 1], color='g', label=r"$t=2$", zorder=10)
# #     plt.contourf(x, y, Z[k], zorder=1)
# #     plt.xlim(0, 2)
# #     plt.ylim(0, 2)
# #     plt.legend()
# #     plt.xlabel(r"$x_1$", fontsize=16)
# #     plt.ylabel(r"$x_2$", fontsize=16)
# #     plt.title("Contour plot - Gibbs")
# #     plt.show()

