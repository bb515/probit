from scipy.optimize import minimize
import numpy as np

def optimisation_procedure(x0, callable_function, callable_jacobian, method=’L - BFGS - B’):
    """
    Run an optimisation procedure. TODO: needs class of different optimisers?

    Use the routine L-BFGS-B (Byrd et al., 1995) as the gradient-based optimisation package.

    Start from the initial values of the hyperparameters to infer the optimal values in the criterion of the
    variational lower bound for EP or the approximate evidence for MAP respectively.

    Initial values of the hyperparameters are chosen as \sigma^2 = 1., \varphi = 1./d for Gaussian kernel,
    the threshold b_1 = -1 and \Delta_l = 2./r. In practice, try several starting points and then choose the best
    model by the objective functional.

    :arg x0: Vector of initial guess of hyperparameters with shape (n,).
    :arg callable_function: The objective function to be minimized. `fun(x, *args) -> float`
                            where x is an 1-D array with shape (n,) and args is a tuple of the
                            fixed parameters needed to completely specify the function.
    :arg callable_jacobian: callable, Method for computing the gradient vector. Only for CG, BFGS, Newton-CG, L-BFGS-B,
                            TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr. If it is a
                            callable, it should be a function that returns the gradient vector:
                            `jac(x, *args) -> array_like, shape (n,)`
                            where x is an array with shape (n,) and args is a tuple with the fixed parameters.
                            If jac is a Boolean and is True, fun is assumed to return and objective and gradient as an
                            (f, g) tuple. Methods ‘Newton-CG’, ‘trust-ncg’, ‘dogleg’, ‘trust-exact’, and ‘trust-krylov‘
                            require that either a callable be supplied, or that fun return the objective and gradient.
                            If None or False, the gradient will be estimated using 2-point finite difference estimation
                            with an absolute step size. Alternatively, the keywords {‘2-point’, ‘3-point’, ‘cs’} can be
                            used to select a finite difference scheme for numerical estimation of the gradient with a
                            relative step size. These finite difference schemes obey any specified bounds.
    :arg method:str or callable, optional. Type of solver. Should be one of
        ‘Nelder-Mead’, ‘Powell’, ‘CG’, ‘BFGS’, ‘Newton-CG’, ‘L-BFGS-B’, ‘TNC’, ‘COBYLA’, ‘SLSQP’, ‘trust-constr’,
        ‘dogleg’, ‘trust-ncg’, ‘trust-exact’, ‘trust-krylov’.

    If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds
    """
    minimize(callable_function, x0, method=method, jac=callable_jacobian)


