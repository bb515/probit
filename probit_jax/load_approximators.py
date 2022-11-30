from probit_jax.approximators import LaplaceGP, VBGP
import enum


class ApproximatorLoader(enum.Enum):
    """Factory enum to load approximators.
    """
    LA = LaplaceGP
    VB = VBGP


def load_approximator(
    approximator_string,
    M=None,
    **kwargs):
    """
    Returns a brand new instance of the classifier manager for training.
    Observe that this instance is of no use until it has been trained.
    Input:
        approximator_string (str):       type of model to be loaded. Our interface can currently provide
                                trainable instances for: 'keras'
        M (int):    number of inducing points for the sparse models
        model_metadata (str):   absolute path to the file where the model's metadata is going to be
                                saved. This metadata file will contain all the information required
                                to re-load the model later.
        model_kwargs (kwargs):  hyperparameters required to initialise the classification model. For
                                details look at the desired model's constructor.
    Output:
        classifier (ClassifierManager): an instance of a classifier with a standard interface to
                                        be used in our pipeline.
    Raises:
        ValueError: if the classifier type provided is not supported by the interface.
    """
    if approximator_string in ["LA", "VB"]:
        return ApproximatorLoader[approximator_string].value(
            **kwargs)
    else:
        raise ValueError(
            "Approximator not found. (got {}, expected {})".format(
            approximator_string, "'LA' or 'VB'"))
