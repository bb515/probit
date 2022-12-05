"""Utility functions for probit."""
import jax.numpy as jnp
import warnings
import h5py


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


def write_array(write_path, dataset, array):
    """
    Write a :class: numpy.ndarray to a HDF5 file.
    :arg write_path: The path to which the HDF5 file is written.
    :type write_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :array: The array to be written to file.
    :type array: :class: numpy.ndarray
    :return: None
    :rtype: None type
    """
    with h5py.File(write_path, 'a') as hf:
        hf.create_dataset(dataset,  data=array)


def overwrite_array(write_path, dataset, array):
    """
    Write a :class: numpy.ndarray to a HDF5 file.
    :arg write_path: The path to which the HDF5 file is written.
    :type write_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :array: The array to be written to file.
    :type array: :class: numpy.ndarray
    :return: None
    :rtype: None type
    """
    with h5py.File(write_path, 'r+') as hf:
        data = hf[dataset]
        data = array


def read_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.
    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                array = hf[dataset][:]
                return array
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
                raise
    except OSError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        raise


def read_scalar(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.
    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                scalar = hf[dataset][()]
                return scalar 
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
                raise
    except OSError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        raise


def sample_g(g, f, y_train, cutpoints, noise_std, N):
    """TODO: Seems like this should be done in numpy or numba - have tried a
        lab implementation but not tested it yet
    """
    for i in range(N):
        # Target class index
        j_true = y_train[i]
        g_i = jnp.NINF  # this is a trick for the next line
        # Sample from the truncated Gaussian
        while g_i > cutpoints[j_true + 1] or g_i <= cutpoints[j_true]:
            # sample y
            g_i = f[i] + jnp.random.normal(loc=f[i], scale=noise_std)
        # Add sample to the Y vector
        g[i] = g_i
    return g


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
