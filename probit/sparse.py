"""Sparse GPs"""
import warnings
from approximators import VBOrdinalGP, EPOrdinalGP, LaplaceOrdinalGP
from scipy.linalg import cho_solve, cho_factor, solve_triangular
import numpy as np


class SparseVBOrdinalGP(VBOrdinalGP):
    """
    A sparse GP classifier for ordinal likelihood using the Variational Bayes
    (VB) approximation.
 
    Inherits the VBOrdinalGP class. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference and approximate prior using the Nystrom approximation. It is for
    the ordinal likelihood. For this a :class:`probit.kernels.Kernel` is
    required for the Gaussian Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SparseVBOrdinalGP"

    def __init__(
            self, M, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`SparseVBOrderedGP` Approximator object.

        :arg M: The number of basis functions.

        :returns: A :class:`SparseVBOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        # self.EPS = 1e-8
        # self.EPS_2 = self.EPS**2
        # self.jitter = 1e-10
        # Choose inducing points
        self.M = M
        inducing_idx = np.random.randint(self.X_train.shape[0], size=self.M)
        self.Z = self.X[inducing_idx, :]
        # Initiate hyperparameters
        self._update_nystrom_prior()

    def _update_nystrom_prior(self):
        """
        Update prior covariances with Nyström approximation.

        :arg M: Number of inducing inputs.

        """
        warnings.warn("Updating prior covariance with Nyström approximation")
        self.Kmm = self.kernel.kernel_matrix(self.Z, self.Z)
        self.Knn = self.kernel.kernel_diagonal(self.X, self.X)
        self.Knm = self.kernel.kernel_matrix(self.X, self.Z)
        warnings.warn("Done updating prior covariance with Nyström approximation")

        raise NotImplementedError()

    def hyperparameters_update(
        self, theta=None, cutpoints=None, varphi=None, variance=None, noise_variance=None,
        varphi_hyperparameters=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg float noise_variance: The noise variance.
        :type noise_variance:
        """
        self.hyperparameters_update(
            cutpoints=cutpoints, varphi=varphi,
            variance=variance, noise_variance=noise_variance)
        if varphi_hyperparameters is not None:
            self.kernel.update_hyperparameter(
                varphi_hyperparameters=varphi_hyperparameters)
        # Update posterior covariance
        warnings.warn("Updating posterior covariance.")
        self._update_nystrom_posterior()
        warnings.warn("Done updating posterior covariance.")

        def _update_posterior(self):
            """Update posterior covariances."""
            # TODO: Is this really the best cholesky to take. What are the eigenvalues?
            # are they bounded?
            # Note that this scipy implementation returns an upper triangular matrix
            # whereas numpy, tf, scipy.cholesky return a lower triangular,
            # then the position of the matrix transpose in the code would change.
            (self.L_cov, self.lower) = cho_factor(
                self.noise_variance * np.eye(self.N) + self.K)
            # Unfortunately, it is necessary to take this cho_factor,
            # only for log_det_K
            (L_K, lower) = cho_factor(self.K + self.jitter * np.eye(self.N))

            # tmp
            L_KT_inv = solve_triangular(
                L_K.T, np.eye(self.N), lower=True)
            self.K_inv = solve_triangular(L_K, L_KT_inv, lower=False)

            self.log_det_K = 2 * np.sum(np.log(np.diag(L_K)))
            self.log_det_cov = -2 * np.sum(np.log(np.diag(self.L_cov)))
            # TODO: If jax @jit works really well with the GPU for cho_solve,
            # it is worth not storing this matrix - due to storage cost, and it
            # will be faster. See alternative implementation on feature/cho_solve
            # For the CPU, storing self.cov saves solving for the gradient and the
            # fx. Maybe have it as part of a seperate method.
            # TODO: should be using  cho_solve and not solve_triangular, unless I used it because that is what is used
            # in tensorflow for whatever reason (maybe tensorflow has no cho_solve)
            # Note that Tensorflow uses tf.linalg.triangular_solve
            L_covT_inv = solve_triangular(
                self.L_cov.T, np.eye(self.N), lower=True)
            self.cov = solve_triangular(self.L_cov, L_covT_inv, lower=False)
            self.trace_cov = np.sum(np.diag(self.cov))
            self.trace_posterior_cov_div_var = np.einsum('ij, ij -> ', self.K, self.cov)
