"""Utility functions for probit."""
import jax.numpy as jnp
import warnings


def check_data(data):
    X_train, y_train = data
    if y_train.dtype not in [int, jnp.int32]:
        raise TypeError(
            "t must contain only integer values (got {})".format(
                y_train.dtype))
    else:
        y_train = y_train.astype(int)
    return X_train, y_train


def check_cutpoints(cutpoints, J):
    """
    Check that the cutpoints are compatible with this class.
    # TODO: convert to B lab code

    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    """
    warnings.warn("Checking cutpoints...")
    # Convert cutpoints to numpy array
    cutpoints = jnp.array(cutpoints)
    # Not including -\infty or \infty
    if jnp.shape(cutpoints)[0] == J - 1:
        # Append \infty
        cutpoints = jnp.append(cutpoints, jnp.inf)
        # Insert -\infty at index 0
        cutpoints = jnp.insert(cutpoints, 0, jnp.NINF)
        pass  # correct format
    # Not including one cutpoints
    elif jnp.shape(cutpoints)[0] == J:
        if cutpoints[-1] != jnp.inf:
            if cutpoints[0] != jnp.NINF:
                raise ValueError(
                    "Either the largest cutpoint parameter b_J is not "
                    "positive infinity, or the smallest cutpoint "
                    "parameter must b_0 is not negative infinity."
                    "(got {}, expected {})".format(
                    [cutpoints[0], cutpoints[-1]], [jnp.inf, jnp.NINF]))
            else:  #cutpoints[0] is -\infty
                cutpoints.append(jnp.inf)
                pass  # correct format
        else:
            cutpoints = jnp.insert(cutpoints, 0, jnp.NINF)
            pass  # correct format
    # Including all the cutpoints
    elif jnp.shape(cutpoints)[0] == J + 1:
        if cutpoints[0] != jnp.NINF:
            raise ValueError(
                "The smallest cutpoint parameter b_0 must be negative "
                "infinity (got {}, expected {})".format(
                    cutpoints[0], jnp.NINF))
        if cutpoints[-1] != jnp.inf:
            raise ValueError(
                "The largest cutpoint parameter b_J must be positive "
                "infinity (got {}, expected {})".format(
                    cutpoints[-1], jnp.inf))
        pass  # correct format
    else:
        raise ValueError(
            "Could not recognise cutpoints shape. "
            "(jnp.shape(cutpoints) was {})".format(jnp.shape(cutpoints)))
    assert cutpoints[0] == jnp.NINF
    assert cutpoints[-1] == jnp.inf
    assert jnp.shape(cutpoints)[0] == J + 1
    if not all(
            cutpoints[i] <= cutpoints[i + 1]
            for i in range(J)):
        raise CutpointValueError(cutpoints)
    return cutpoints


class CutpointValueError(Exception):
    """
    An invalid cutpoint argument was used to construct the classifier model.
    """

    def __init__(self, cutpoint):
        """
        Construct the exception.

        :arg cutpoint: The cutpoint parameters array.
        :type cutpoint: :class:`numpy.array` or list

        :rtype: :class:`CutpointValueError`
        """
        message = (
                "The cutpoint list or array "
                "must be in ascending order, "
                f" {cutpoint} was given."
                )

        super().__init__(message)


class InvalidKernel(Exception):
    """An invalid kernel has been passed to `Approximator` or `Sampler`"""

    def __init__(self, kernel):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`Approximator` or `Sampler`
            as the kernel argument.
        :rtype: :class:`InvalidKernel`
        """
        message = (
            f"{kernel} is not an instance of "
            "mlkernels.Kernel"
        )

        super().__init__(message)
