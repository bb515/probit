from probit.approximators import LaplaceGP, EPGP, VBGP
from probit.sparse import SparseLaplaceGP, SparseVBGP
from probit.gpflow import SVGP, VGP
import enum


class ApproximatorLoader(enum.Enum):
    """Factory enum to load approximators.
    """
    LA = LaplaceGP
    EP = EPGP
    VB = VBGP
    SLA = SparseLaplaceGP
    SVB = SparseVBGP
    V = VGP
    SV = SVGP


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
    if approximator_string in ["LA", "EP", "VB", "V"]:
        return ApproximatorLoader[approximator_string].value(
            **kwargs)
    elif approximator_string in ["SLA", "SVB", "SVGP"]:
        return ApproximatorLoader[approximator_string].value(
            M=M, **kwargs)
    else:
        raise ValueError(
            "Approximator not found. (got {}, expected {})".format(
            approximator_string, "LA, EP, VB, V, SLA, SVB or SV"))
