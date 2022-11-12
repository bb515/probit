"""Ordinal regression concrete examples. Approximate inference."""
# Make sure to limit CPU usage
import os

# Enable double precision
from jax.config import config
config.update("jax_enable_x64", True)

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = "6"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION "] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import argparse
import cProfile
from io import StringIO
from pstats import Stats, SortKey
import numpy as np
import pathlib
from probit_jax.plot import (
    _grid_over_hyperparameters_initiate)
from probit.data.utilities import datasets as datasets_
from probit.data.utilities import load_data as load_data_
from probit.data.utilities import load_data_synthetic as load_data_synthetic_
from probit.data.utilities import load_data_paper as load_data_paper_
from probit.data.utilities import datasets as datasets_
from probit_jax.data.utilities import datasets, load_data, load_data_synthetic, load_data_paper
from mlkernels import Kernel as BaseKernel
from probit_jax.utilities import InvalidKernel, check_cutpoints
from probit_jax.implicit.utilities import (
    log_probit_likelihood, grad_log_probit_likelihood, hessian_log_probit_likelihood,
    posterior_covariance, predict_reparameterised, matrix_inverse, posterior_covariance)
import sys
import time
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt


BG_ALPHA = 1.0
MG_ALPHA = 0.2
FG_ALPHA = 0.4


now = time.ctime()
write_path = pathlib.Path()

#(get the probit_jax approximator)
def get_approximator(
        approximation, N_train):
    if approximation == "VB":
        from probit_jax.approximators import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//1000])
        print("steps: ", steps)
        Approximator = VBGP
    elif approximation == "LA":
        from probit_jax.approximators import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected VB, LA)".format(
                approximation))
    return Approximator, steps

# get the probit approximator (LA or VB)
def get_approximator_(
        approximation, Kernel, theta_0, signal_variance_0, N_train):
    # Initiate kernel
    kernel = Kernel(
        theta=theta_0, variance=signal_variance_0)
    M = None
    if approximation == "VB":
        from probit.approximators import VBGP
        # steps is the number of fix point iterations until check convergence
        steps = np.max([10, N_train//1000])
        Approximator = VBGP
    elif approximation == "LA":
        from probit.approximators import LaplaceGP
        # steps is the number of Newton steps until check convergence
        steps = np.max([2, N_train//1000])
        Approximator = LaplaceGP
    else:
        raise ValueError(
            "Approximator not found "
            "(got {}, expected VB or LA".format(
                approximation))
    return Approximator, steps, M, kernel


def main():
    """Conduct an approximation to the posterior, and optimise hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name", help="run example on a given dataset name")
    parser.add_argument(
        "J", help="e.g., 13, 53, 101, etc.")
    parser.add_argument(
        "D", help="number of classes")
    parser.add_argument(
        "method", help="L-BFGS-B or CG or Newton-CG or BFGS")
    parser.add_argument(
        "approximation", help="EP or VB or LA")
    # The --profile argument generates profiling information for the example
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()
    dataset = args.dataset_name
    
    J = int(args.J)
    D = int(args.D)
    method = args.method
    approximation = args.approximation
    write_path = pathlib.Path(__file__).parent.absolute()
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
 
    # Initiate data and classifier for probit_jax repo
    if dataset in datasets["benchmark"]:
        (X_trains, y_trains,
        X_tests, y_tests,
        X_true, g_tests,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, Kernel) = load_data(
            dataset, J)
        X = X_trains[2]
        y = y_trains[2]
    elif dataset in datasets["synthetic"]:
        (X, y,
        X_true, g_true,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_synthetic(dataset, J)
    elif dataset in datasets["paper"]:
        (X, f_, g_true, y,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_paper(
            dataset, J=J, D=D, ARD=False, plot=True)
    else:
        raise ValueError("Dataset {} not found.".format(dataset))

    from mlkernels import EQ
    N_train = np.shape(y)[0]
    Approximator, steps = get_approximator(approximation, N_train) # LA or VB

    # Initiate classifier
    def prior(prior_parameters):
        stretch = prior_parameters
        signal_variance = signal_variance_0
        # Here you can define the kernel that defines the Gaussian process
        kernel = signal_variance * EQ().stretch(stretch)
        # Make sure that model returns the kernel, cutpoints and noise_variance
        return kernel

    # Test prior
    if not (isinstance(prior(1.0), BaseKernel)):
        raise InvalidKernel(prior(1.0))

    # check that the cutpoints are in the correct format
    # for the number of classes, J
    cutpoints_0 = check_cutpoints(cutpoints_0, J)

    # passing the arguments needed for the LA or VB 
    classifier = Approximator(prior, log_probit_likelihood,
        grad_log_likelihood=grad_log_probit_likelihood,
        hessian_log_likelihood=hessian_log_probit_likelihood,
        data=(X, y), single_precision=False)

    # Initiate data and classifier for probit repo
    dataset = "SEIso"
    if args.profile:
        profile = cProfile.Profile()
        profile.enable()
    #sys.stdout = open("{}.txt".format(now), "w")
    if dataset in datasets_["benchmark"]:
        (X_trains, y_trains,
        X_tests, y_tests,
        X_true, g_tests,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, Kernel) = load_data_(
            dataset, J)
        X = X_trains[2]
        y = y_trains[2]
    elif dataset in datasets_["synthetic"]:
        (X, y,
        X_true, g_true,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_synthetic_(dataset, J)
    elif dataset in datasets_["paper"]:
        (X, f_, g_true, y,
        cutpoints_0, theta_0, noise_variance_0, signal_variance_0,
        J, D, colors, Kernel) = load_data_paper_(
            dataset, J=J, D=D, ARD=False, plot=True)
    else:
        raise ValueError("Dataset {} not found.".format(dataset))
    
    N_train = np.shape(y)[0]

    Approximator, steps, M, kernel = get_approximator_(
        approximation, Kernel, theta_0, signal_variance_0, N_train)
    if "S" in approximation:
        # Initiate sparse classifier
        _classifier = Approximator(
            M=M, cutpoints=cutpoints_0, noise_variance=noise_variance_0,
            kernel=kernel, J=J, data=(X, y))
    else:
        # Initiate classifier
        _classifier = Approximator(
            cutpoints=cutpoints_0, noise_variance=noise_variance_0,
            kernel=kernel, J=J, data=(X, y), )#single_precision=True)

    # Notes: fwd_solver, newton_solver work, anderson solver has bug with vmap ValueError

    g = classifier.take_grad()

    trainables = [1] * (J + 2) # J = 3 (3 bins)
    # Fix theta
    # trainables[-1] = 0
    # Fix noise standard deviation
    trainables[0] = 0
    # Fix signal standard deviation
    trainables[J] = 0
    # Fix cutpoints
    trainables[1:J] = [0] * (J - 1)
    print("trainables = {}".format(trainables))

    # hyperparameter domain and resolution
    domain = ((-1, 2), None) # x-axis domain range
    res = (30, None) # increments in domain

    (x1s, x2s,
    xlabel, ylabel,
    xscale, yscale,
    xx, yy,
    phis, fxs,
    gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
    _classifier, res, domain, trainables)

    # print("theta_0 outisde", theta_0)
    # # get test for probit
    # phi_test = phis[phis.shape[0]//2]
    # theta_test = jnp.exp(phi_test)[0]

    # # get parameters for probit_jax
    params = ((jnp.sqrt(1./(2 * theta_0))), (jnp.sqrt(noise_variance_0), cutpoints_0))
    # params_test = ((jnp.sqrt(1./(2 * (theta_test)))), (jnp.sqrt(noise_variance_0), cutpoints_0))    

    # # probit_jax - latent variables
    # fxp, gxp = g(params)
    # fxp_test, gxp = g(params_test)
    latent_jax = classifier.get_latents(params)
    #latent_jax_test = classifier.get_latents(params_test)

    # # probit - latent variables
    fx, gx, latent_probit, _ = _classifier.approximate_posterior(
    phi_0, trainables, steps, verbose=False, return_reparameterised = True)
    # # fx_test, gx_test, latent_probit_test, _ = _classifier.approximate_posterior(
    # phi_test, trainables, steps, verbose=False, return_reparameterised = True)
    
    #print("model evidence difference: ", fx - fxp)
    # probit - predictive distribution
    #cov = noise_variance_0**2*jnp.identity(N_train) + _classifier.prior_coveriance # sigma^2I + K)^{-1}
    #_, pred_mean, _ = _classifier.predict(
    #X, cov, f, reparameterised=True, whitened=False)    

    def _plot(title, probit, probit_jax):
        x = [_[0] for _ in X]

        #_x = jnp.linspace(0, len(latents)-1, len(latents))
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(x, probit, c='b', marker="X", markersize=8, alpha=1, linestyle = "None", label=r"analytic")
        ax.plot(x, probit_jax, c='g', marker ="o", alpha=0.7, linestyle = "None", label=r"autodiff")
        ax.hlines(sum(probit)/len(probit), min(x), max(x), 'r', alpha=0.5, label=r"analytic mean")
        ax.hlines(sum(probit_jax)/len(probit_jax), min(x), max(x), 'k',
            alpha=0.5, label=r"autodiff mean")
        ax.set_ylabel(r"Latent Variables")
        ax.legend()
        fig.savefig(title,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

    #_plot("steps=100",latent_probit, latent_jax)
    _plot("LA_newton_method_latents_new", latent_probit, latent_jax)

    def plot_diff(phis, plot=True):
        dfs = np.empty(res[0]) # list to store the differences of the likelihoods
        fis = []
        for i, phi in enumerate(phis):
            theta = jnp.exp(phi)[0]
            params = ((jnp.sqrt(1./(2 * theta))), (jnp.sqrt(noise_variance_0), cutpoints_0)) # params[0]-->prior. params[1]-->likelihood
            
            fxp, gxp, weight, _ = _classifier.approximate_posterior(
            phi, trainables, steps, verbose=False, return_reparameterised = True)
            fx, gx = g(params) # g is a function. passing the arguments for the solver to perform fixed_point_interation
            gs = gx[0] * (- 0.5 * (2 * theta)**(-1./2))  # multiply by the lengthscale Jacobian
            
            df = fxp - fx
            dfs[i] = df
            
            if df <= 5:
                fis.append(phi)
        
        if plot == True:
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(BG_ALPHA)
            ax = fig.add_subplot(111)
            ax.grid()
            ax.plot(x1s, dfs, 'b',  label=r"Evidence diff")
            ax.set_ylabel(r"Model evidence difference (probit-probit_jax)")
            ax.set_xlabel(xlabel)
            ax.set_xscale(xscale)
            ax.legend()
            fig.savefig("Difference_VB.png",
                facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

        return fis
    
    #plot_diff(phis)

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())

    def plot_bound_grad(title1="bound_LA_1", title2="grad_LA_1"):
        domain = ((-1, 2), None) # x-axis domain range
        res = (30, None) # increments in domain
        trainables = [0,0,0,0,1] # vary signal std 

        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        phis, fxs,
        gxs, theta_0, phi_0) = _grid_over_hyperparameters_initiate(
        _classifier, res, domain, trainables)
        
        print(phis)
        print("theta_inside", theta_0)
        print("phi_inside", phi_0)
        
        gs = np.empty(res[0])
        fs = np.empty(res[0])

        #outer loop for probit_jax
        for i, phi in enumerate(phis):
            theta = jnp.exp(phi)[0]
            
            params = ((jnp.sqrt(1./(2 * theta))), (jnp.sqrt(noise_variance_0), cutpoints_0)) # params[0]-->prior. params[1]-->likelihood
            fx, gx = g(params) # g is a function. passing the arguments for the solver to perform fixed_point_interation
            gs[i] = gx[0] * (- 0.5 * (2 * theta)**(-1./2))  # multiply by the lengthscale Jacobian
            fs[i] = fx
        
        # outer loop for probit
        for i, phi in enumerate(phis):
            fx, gx, weight, _ = _classifier.approximate_posterior(
                phi, trainables, steps, verbose=False, return_reparameterised = True)
            
            fxs[i] = fx
            gxs[i] = gx

        (fxs, gxs,
        x, y,
        xlabel, ylabel,
        xscale, yscale) = (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

        
        #Numerical derivatives: need to calculate them in the log domain if theta is in log domain
        if xscale == "log":
            log_x = np.log(x)
            dfxs_ = np.gradient(fxs, log_x)
            dfsxs = np.gradient(fs, log_x)
        elif xscale == "linear":
            dfxs_ = np.gradient(fxs, x)
            dfsxs = np.gradient(fs, x)

        idx_hat = np.argmin(fxs)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(x, fxs, 'b', marker="o", label=r"$\mathcal{F}}$ analytic")
        ax.plot(x, fs, 'g', label=r"$\mathcal{F}$ autodiff")
        ylim = ax.get_ylim()
        ax.vlines(x[idx_hat], 0.99 * ylim[0], 0.99 * ylim[1], 'r',
            alpha=0.5, label=r"$\hat\theta={:.2f}$".format(x[idx_hat]))
        ax.vlines(theta_0, 0.99 * ylim[0], 0.99 * ylim[1], 'k',
            alpha=0.5, label=r"'true' $\theta={:.2f}$".format(theta_0))
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        ax.set_ylabel(r"$\mathcal{F}$")
        ax.legend()
        fig.savefig(title1,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(BG_ALPHA)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(
            x, dfxs_, 'b--', marker="o", 
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic numeric")
        ax.set_ylim(ax.get_ylim())
        ax.plot(
            x, gxs, 'b', alpha=0.7, marker="o",
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ analytic")
        ax.plot(
            x, gs, 'g',
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff")
        ax.plot(
            x, dfsxs, 'g--',
            label=r"$\frac{\partial \mathcal{F}}{\partial \theta}$ autodiff numeric")
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\frac{\partial \mathcal{F}}{\partial \theta}$")
        ax.legend()
        fig.savefig(title2,
            facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
    
    #plot_bound_grad("bound_LA_fwd", "grad_LA_fwd")

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())
    sys.stdout.close()


if __name__ == "__main__":
    main()
