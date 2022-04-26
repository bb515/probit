"""Sparse GPs"""
from approximators import VBOrdinalGP, EPOrdinalGP, LaplaceOrdinalGP


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
        # Initiate hyperparameters
        self._update_nystrom_prior(M)
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _update_nystrom_prior(self, M):
        """
        Update prior covariances with Nyström approximation.

        :arg M: Number of inducing inputs.

        """
        warnings.warn("Updating prior covariance with Nyström approximation")
        self.Kmm = self.kernel.kernel_matrix(self.Z, self.Z)
        self.Knn = self.kernel.kernel_diagonal(self.X, self.X)
        self.Knm = self.kernel.kernel_matrix(self.X, self.Z)
        self.M = M

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
        if cutpoints is not None:
            # Convert cutpoints to numpy array
            cutpoints = np.array(cutpoints)
            # Not including -\infty or \infty
            if np.shape(cutpoints)[0] == self.J - 1:
                cutpoints = np.append(cutpoints, np.inf)  # Append \infty
                cutpoints = np.insert(cutpoints, 0, np.NINF)  # Insert -\infty at index 0
                pass  # Correct format
            # Not including one cutpoints
            elif np.shape(cutpoints)[0] == self.J: 
                if cutpoints[-1] != np.inf:
                    if cutpoints[0] != np.NINF:
                        raise ValueError(
                            "Either the largest cutpoint parameter b_J is not "
                            "positive infinity, or the smallest cutpoint "
                            "parameter must b_0 is not negative infinity."
                            "(got {}, expected {})".format(
                            [cutpoints[0], cutpoints[-1]], [np.inf, np.NINF]))
                    else:  # cutpoints[0] is -\infty
                        cutpoints.append(np.inf)
                        pass  # correct format
                else:
                    cutpoints = np.insert(cutpoints, 0, np.NINF)
                    pass  # correct format
            # Including all the cutpoints
            elif np.shape(cutpoints)[0] == self.J + 1:
                if cutpoints[0] != np.NINF:
                    raise ValueError(
                        "The smallest cutpoint parameter b_0 must be negative "
                        "infinity (got {}, expected {})".format(
                            cutpoints[0], np.NINF))
                if cutpoints[-1] != np.inf:
                    raise ValueError(
                        "The largest cutpoint parameter b_J must be "
                        "positive infinity (got {}, expected {})".format(
                            cutpoints[-1], np.inf))
                pass  # correct format
            else:
                raise ValueError(
                    "Could not recognise cutpoints shape. "
                    "(np.shape(cutpoints) was {})".format(np.shape(cutpoints)))
            assert cutpoints[0] == np.NINF
            assert cutpoints[-1] == np.inf
            assert np.shape(cutpoints)[0] == self.J + 1
            if not all(
                    cutpoints[i] <= cutpoints[i + 1]
                    for i in range(self.J)):
                raise CutpointValueError(cutpoints)
            self.cutpoints = cutpoints
            self.cutpoints_ts = cutpoints[self.t_train]
            self.cutpoints_tplus1s = cutpoints[self.t_train + 1]
        if varphi is not None or variance is not None:
            self.kernel.update_hyperparameter(
                varphi=varphi, variance=variance)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior()
            warnings.warn("Done posterior covariance.")
        if varphi_hyperparameters is not None:
            self.kernel.update_hyperparameter(
                varphi_hyperparameters=varphi_hyperparameters)
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
        # Update posterior covariance
        warnings.warn("Updating posterior covariance.")
        self._update_posterior()
        warnings.warn("Done updating posterior covariance.")