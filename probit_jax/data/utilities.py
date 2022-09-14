"""Utility functions for data."""
# Make sure to limit CPU usage
import os
from statistics import variance
nthreads = "6"
os.environ["OMP_NUM_THREADS"] = nthreads # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = nthreads # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = nthreads # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = nthreads # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_NUM_THREADS"] = nthreads
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib.resources as pkg_resources
from probit_jax.load_approximators import ApproximatorLoader
from mlkernels import EQ, Linear


# For plotting
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Dictionary of the names of the different datasets
datasets = {
    "benchmark" : [
        "abalone",
        "auto",
        "diabetes",
        "housing",
        "machine",
        "pyrim",
        "stocks",
        "triazines",
        "wpbc",
        ],
    "synthetic" : [
        "EQ",
        "Linear",
        ],
    "paper" : [
        "figure2og",
        "figure2",
        "13",
        "table1"
        ],
}


metadata = {
    "abalone": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (4177, 10),
        "init": None,
        "max_sec": 500,
    },
    "auto": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (392, 7),
        "init": None,
        "max_sec": 100,
    },
    "diabetes": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (43, 2),
        "init": None,
        "max_sec": 100,
    },
    "housing": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (506, 13),
        "init": None,
        "max_sec": 100,
    },
    "machine": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (209, 6),
        "init": None,
        "max_sec":100,
    },
    "pyrim": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (74, 27),
        "init": None,
        "max_sec":100,
    },
    "stocks": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (950, 9),
        "init": None,
        "max_sec":5000,
    },
    "triazines": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (186, 60),
        "init": None,
        "max_sec": 100,
    },
    "wpbc": {
        "plot_lims": ((15.0, 65.0), (15.0, 70.0)),
        "size": (194, 32),
        "init": None,
        "max_sec":100,
    },
    "tertile": {
        "plot_lims": (-0.5, 1.5),
        "size": (135, 3),  # (90, 3) if not new
        "init": None,
        "max_sec":500,
    },
    "thirteen": {
        "plot_lims": (-0.5, 1.5),
        "size": (585, 13),
        "init": None,
        "max_sec":500,
    },
    "septile": {
        "plot_lims": (-0.5, 1.5),
        "size": (256, 7),
        "init": None,
        "max_sec":500,
    },
    "linear": {
        "plot_lims": (-0.5, 1.5),
        "size": (585, 13),
        "init": None,
        "max_sec":500,
    }
}


def load_model(model_kwargs, data, J):
    """
    Loads an Ordinal GP classifer using the metadata file provided at
    construction. This provides the relevant Kernel (which defines the GP
    model), the data, and the Kernel hyperparameters.

    However, the model is not yet loaded in a state in which it can be used
    to make predictions or to further train.
    """
    approximator_string = model_kwargs['approximation_string']
    Approximator = ApproximatorLoader(approximator_string)

    kernel_string = model_kwargs['kernel_string']
    Kernel = KernelLoader(kernel_string)

    varphi_0 = model_kwargs['varphi_0']
    signal_variance_0 = model_kwargs['varphi_0']
    cutpoints_0 = model_kwargs['varphi_0']
    noise_variance_0 = model_kwargs['varphi_0']

    # Initiate kernel
    kernel = Kernel(
        varphi=varphi_0, variance=signal_variance_0)
    # Initiate the classifier with the training data
    classifier = Approximator(
        cutpoints_0, noise_variance_0, kernel,
        J, data)
    return classifier


def load_npz_data(file_path):
    """
    This is a minimum working example to load data as numpy arrays.

    :arg file_path: Location and name of the .npz file used as argument to
        `:meth:numpy.load`.
    :arg int N_train: Optional argument that returns only the first N examples
        in the training set.
    :returns: data tuple, number of ordinal classes, number of data dims,
        bin edges of the data.
    """
    with np.load(file_path) as data:
        log_cutpoints  = data["log_cutpoints"]
        text_trains = data["text_trains"]
        X = data["X"]  # (N_train, D)
        t = data["t"].astype(int)  # (N_train,)
        bin_edges = data["bin_edges"]
        J = data["J"]
        D = data["D"]
    return (
        (X, t), J, D, bin_edges)


def indices_initiate(
        self, cutpoints_0, varphi_0, noise_variance_0, scale_0,
        J, indices):
    """
    # TODO: include the kernel?
    # TODO: is calculate_all_gradients SS
    Evaluate container for gradient of the objective function.

    :arg cutpoints_0:
    :type cutpoints_0:
    :arg varphi_0:
    :type varphi_0:
    :arg noise_variance_0:
    :type noise_variance_0:
    :arg scale_0:
    :type scale_0:
    :arg bool calculate_all_gradients:
    """
    # noise_var, b_1, [\Delta^1, \Delta^2, ..., \Delta^(J-2)], scale(s),
    # kernel specific hyperparameters
    indices = np.zeros(
        (1 + 1 + (J - 2) + 1 + 1,))
    if calculate_all_gradients is True:
        # Optimize all hyperparameters
        indices[:] = 1
        if optimize_scale:
            indices[0] = 0
        else:
            indices[self.J] = 0
    elif (cutpoints_0 is not None
            and varphi_0 is None
            and noise_variance_0 is None
            and scale_0 is not None):
        # Optimize only varphi and noise variance
        indices[0] = 1
        indices[-self.kernel.num_hyperparameters:] = 1
    elif (cutpoints_0 is not None
            and varphi_0 is None
            and noise_variance_0 is not None
            and scale_0 is None):
        # Optimize only varphi and scale
        indices[self.J] = 1
        indices[-self.kernel.num_hyperparameters:] = 1
    elif (cutpoints_0 is not None
            and noise_variance_0 is not None
            and varphi_0 is None
            and scale_0 is not None):
        # Optimize only varphi
        indices[-self.kernel.num_hyperparameters:] = 1
    elif (cutpoints_0 is not None
            and noise_variance_0 is None
            and varphi_0 is not None
            and scale_0 is not None
            ):
        # Optimize only noise variance
        indices[0] = 1
    elif (cutpoints_0 is not None
            and noise_variance_0 is not None
            and varphi_0 is not None
            and scale_0 is None
            ):
        # Optimize only scale
        indices[self.J] = 1
    elif (cutpoints_0 is not None
            and noise_variance_0 is not None
            and varphi_0 is not None
            and scale_0 is not None
            ):
        # Optimize only first two threshold parameters
        indices[1] = 1
        indices[2] = 1
    indices = np.where(indices != 0)
    return indices

def get_Y_trues(X_trains, X_true, Y_true):
    """Get Y_trues (N/J, J) from full array of true y values."""
    Y_trues = []
    for j in range(19):
        y = []
        for i in range(len(X_trains[-1, :, :])):
            for j, two in enumerate(X_true):
                one = X_trains[j, i]
                if np.allclose(one, two):
                    y.append(Y_true[j])
        Y_trues.append(y)
    Y_trues = np.array(Y_trues)
    return Y_trues


def load_data(dataset, J):
    if dataset == "abalone":
        from probit_jax.data import abalone
        with pkg_resources.path(abalone, 'abalone.npz') as path:
            data_continuous = np.load(path)
        D = 10
        if J == 5:
            hyperparameters = {
                "init": (  # Unstable
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    100.0,
                    10.0
                ),
                "1085.1": (
                    [-np.inf, -0.93798221, 1.0093883, 1.5621378, 1.99464624, np.inf],
                    2.0,
                    0.2,
                ),
                "1085.0": (
                    np.array([-np.inf, -0.72645941,  0.2052068 ,  0.99282042,  1.79754696, np.inf]) ,
                    0.07043020535401254 ,
                    0.23622657165969294 ,
                ),
                "1073.0": (
                    np.array([-np.inf, -0.60729699, -0.3207209 , -0.2210021 , -0.10743028, np.inf]) ,
                    0.135,
                    0.005,
                ),
                "1046.0": (
                    np.array([-np.inf, -0.60729699, -0.3207209 , -0.2210021 , -0.10743028, np.inf]) ,
                    0.11,
                    0.005,
                ),
            }
            linear_hyperperameters = {
                "init": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf],
                    0.5 / D,
                    np.ones((10,))
                ),
                "init_alt": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf],
                    100.0,
                    np.ones((10,))
                ),
            }
            polynomial_hyperparameters = {
                "init": (  # Unstable
                    [-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf],
                    0.5 / D,
                    np.ones((10,))
                ),
                "init_alt": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf],
                    100.0,
                    np.ones((10,))
                ),
            }
            ARD_hyperparameters = {
                "init": (  # Unstable
                    [-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf],
                    0.5 / D,
                    np.ones((10,))
                ),
                "init_alt": (
                    [-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf],
                    100.0,
                    np.ones((10,))
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["1073.0"]
            from probit_jax.data.abalone import quantile
            with pkg_resources.path(quantile, 'abalone.npz') as path:
                data = np.load(path)
        elif J == 10:
            from probit_jax.data.abalone import decile
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            with pkg_resources.path(decile, 'abalone.npz') as path:
                data = np.load(path)
    elif dataset == "auto":
        from probit_jax.data import auto
        with pkg_resources.path(auto, 'auto.DATA.npz') as path:
            data_continuous = np.load(path)
        D = 7
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (  # Trying lower varphi to find local minima there - this worked
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.0001,
                    0.15
                ),
                "390.7": (
                    [-np.inf, -0.75623025, 0.17624028, 0.97368228, 1.76188594, np.inf],
                    0.0709125443999348,
                    0.20713212477690768
                ),
                "389.0": (  # gives a zo of 0.64 and ma of 0.822, varphi tended downwards. pl 0.25
                    [-np.inf, -0.75623025, 0.17624028, 0.97368228, 1.76188594, np.inf],
                    0.065,
                    0.15
                ),
                "300.9": (
                    np.array([-np.inf, -0.01715729,  0.99812162,  1.42527213,  1.80608152, np.inf]),
                    1.0e-05,
                    0.01,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["300.9"]
            from probit_jax.data.auto import quantile
            with pkg_resources.path(quantile, 'auto.data.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.auto import decile
            with pkg_resources.path(decile, 'auto.data.npz') as path:
                data = np.load(path)
        cutpoints_0 = np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf])
    elif dataset == "diabetes":
        D = 2
        from probit_jax.data import diabetes
        with pkg_resources.path(diabetes, 'diabetes.DATA.npz') as path:
            data_continuous = np.load(path)
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    100.0,
                    10.0
                ),
                "57.66": (
                    [-np.inf, -0.96965513, -0.59439608, 0.10485131, 0.55336265, np.inf],
                    7.05301883339537e-06,
                    0.33582851890990895
                ),
                "53.07": (
                    [-np.inf, -0.39296099, -0.34374783, -0.26326698, -0.20514771, np.inf],
                    1.8099468519640467e-05,
                    0.00813540519298387
                ),
                "52.66": (
                    [-np.inf, -0.99914905, -0.49661647, 0.84539859, 1.66267616, np.inf],
                    0.6800987545547965,
                    0.1415239624095029
                ),
                 "52.32": (
                    [-np.inf, -0.94, -0.47, 0.79, 1.55, np.inf],
                    0.378154023606708,
                    0.103
                ),
            }
            (cutpoints_0, varphi_0, noise_variance_0) = hyperparameters["52.32"]
            from probit_jax.data.diabetes import quantile
            with pkg_resources.path(quantile, 'diabetes.data.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.diabetes import decile
            with pkg_resources.path(decile, 'diabetes.data.npz') as path:
                data = np.load(path)
    elif dataset == "housing":
        D = 13
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        from probit_jax.data import bostonhousing
        with pkg_resources.path(bostonhousing, 'housing.npz') as path:
            data_continuous = np.load(path)
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "538.6": (
                    [-np.inf, -0.9, 0.4, 0.7, 1.2, np.inf],
                    0.01,
                    0.2
                ),
                "462.9": (  # Tried to find local minima here but its not good m0 0.40, ma 0.519
                    [-np.inf, -0.65824624, 0.71570933, 1.2696072, 1.65280723, np.inf],
                    0.0015,
                    0.1,
                ),
                "init_alt": (
                    [-np.inf, -0.65824624, 0.71570933, 1.2696072, 1.65280723, np.inf],
                    10.0,
                    0.1,
                )
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit_jax.data.bostonhousing import quantile
            with pkg_resources.path(quantile, 'housing.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.bostonhousing import decile
            with pkg_resources.path(decile, 'housing.npz') as path:
                data = np.load(path)
    elif dataset == "machine":
        D = 6
        varphi_0 = 2.0 / D
        noise_variance_0 = 2.0
        from probit_jax.data import machinecpu
        with pkg_resources.path(machinecpu, 'machine.npz') as path:
            data_continuous = np.load(path)
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, 0.82762696, 1.3234133 , 1.69382192, 2.02491639, np.inf]) ,
                    0.000008,
                    0.6,
                ),
                "211.6": (
                    [-np.inf, 0.86220899, 1.38172233, 1.76874495, 2.11477391, np.inf], 
                    0.08212108678729564,
                    0.6297364232519436,
                ),
                "199.7": (
                    np.array([-np.inf, 0.82481368, 1.31287655, 1.67778397, 2.00289883, np.inf]),
                    0.0007963123721287592,
                    0.20036340095411048,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit_jax.data.machinecpu import quantile
            with pkg_resources.path(quantile, 'machine.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.machinecpu import decile
            with pkg_resources.path(decile, 'machine.npz') as path:
                data = np.load(path)
    elif dataset == "pyrim":
        from probit_jax.data import pyrimidines
        with pkg_resources.path(pyrimidines, 'pyrim.npz') as path:
            data_continuous = np.load(path)
        D = 27
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    100.0,
                    10.0
                ),
                "99.0": (
                    np.array([-np.inf, -0.96921449, -0.50848339, -0.18839114,  0.13407544, np.inf]),
                    0.01852678001715496,
                    0.13877975100366893
                ),
                "101.0": (
                    [-np.inf, -1.11405451, -0.21082173, 0.36749149, 0.88030159, np.inf],
                    10.0,
                    0.12213407601036991,
                ),
                "92.4": (
                    np.array([-np.inf, -0.41399957, -0.25391163, -0.15800952, -0.0700965, np.inf]) ,
                    0.018,
                    0.006 ,
                ),
                "90.9": (
                    np.array([-np.inf, -0.41399957, -0.25391163, -0.15800952, -0.0700965, np.inf]) ,
                    0.005,
                    0.001 ,
                ),
                "89.9": (
                    np.array([-np.inf, -0.41399957, -0.25391163, 0.0800952, 0.1700965, np.inf]) ,
                    0.5,
                    0.001 ,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["89.9"]
            from probit_jax.data.pyrimidines import quantile
            with pkg_resources.path(quantile, 'pyrim.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.pyrimidines import decile
            with pkg_resources.path(decile, 'pyrim.npz') as path:
                data = np.load(path)
    elif dataset == "stocks":
        from probit_jax.data import stocksdomain
        with pkg_resources.path(stocksdomain, 'stock.npz') as path:
            data_continuous = np.load(path)
        D = 9
        if J == 5:
            hyperparameters = {
                "720.0": (
                    np.array([-np.inf, -0.5, -0.02, 0.43, 0.96, np.inf]),
                    0.03,
                    0.0001
                ),
                "565.0" : (
                    [-np.inf, -1.17119928, -0.65961478, 0.1277627, 0.64710874, np.inf],
                    0.00045,
                    0.01
                ),
                "561.3" : (
                    [-np.inf, -1.15383995, -0.77984497, -0.45804968, -0.0881168, np.inf],
                    0.0004859826674075834,
                    0.009998200271564093,
                ),
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "1245.4": (
                     [-np.inf, -0.91005114, -0.28471923, 0.45182957, 1.14570708, np.inf],
                    0.192573878772786,
                    9.989576742047504
                ),
                "init_alt" : (
                    [-np.inf, -1.15383995, -0.77984497, -0.45804968, -0.0881168, np.inf],
                    0.06,
                    0.01,
                ),
                "540.8" : (
                    np.array([-np.inf, -0.85938512, -0.11336137,  0.88349296,  1.62309025, np.inf]) ,
                    0.035268828486755506,
                    0.010540656172175512,
                ),
                "537.0" : (
                    np.array([-np.inf, -1.3952776 , -0.71642281,  0.1649921 ,  0.78461035, np.inf]) ,
                    0.003896416837153817,
                    0.01569970567257151,
                ),
                "init_alt" : (
                    np.array([-np.inf, -0.9 , -0.8,  0.0 ,  0.3, np.inf]) ,
                    0.001,
                    0.0125,
                ),
                 "556.8" : (
                    np.array([-np.inf, -0.84123951, -0.50561128, -0.14481583, 0.20429528, np.inf]) ,
                    0.000947,
                    0.008413,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["556.8"]
            from probit_jax.data.stocksdomain import quantile
            with pkg_resources.path(quantile, 'stock.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.stocksdomain import decile
            with pkg_resources.path(decile, 'stock.npz') as path:
                data = np.load(path)
    elif dataset == "triazines":
        from probit_jax.data import triazines
        with pkg_resources.path(triazines, 'triazines.npz') as path:
            data_continuous = np.load(path)
        D = 60
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    [-np.inf, -1.0157062, -0.408891, 0.08201205, 0.45461363, np.inf],
                    0.8,
                    0.0004
                ),
                "192.0": (
                    [-np.inf, -1.0157062, -0.908891, -0.78201205, -0.45461363, np.inf],
                    0.0008,
                    0.0004
                ),
                "179.1": (
                    [-np.inf, -1.01688376, -0.90419162, -0.76962449, -0.34357796, np.inf],
                    0.008334663676431068,
                    0.04633175816789728,
                ),
                "175.2": (  # m0 -0.02, ma -0.05
                    [-np.inf, -1.0157062, -0.908891, -0.78201205, -0.45461363, np.inf],
                    0.00833,
                    0.0475,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init_alt"]
            from probit_jax.data.triazines import quantile
            with pkg_resources.path(quantile, 'triazines.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.triazines import decile
            with pkg_resources.path(decile, 'triazines.npz') as path:
                data = np.load(path)
    elif dataset == "wpbc":
        D = 32
        varphi_0 = 2.0/D
        noise_variance_0 = 2.0
        from probit_jax.data import wisconsin
        with pkg_resources.path(wisconsin, 'wpbc.npz') as path:
            data_continuous = np.load(path)
        if J == 5:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J, np.inf]),
                    0.001,
                    0.02
                ),
                "266.4": (  #m0 0.64 ma1.35 nearly there need ma1.0
                    [-np.inf, -0.32841066, 0.17593563, 0.76336227, 1.21093938, np.inf],
                    0.0155,
                    0.2,
                ),
                "284.4": (  #m0 0.64 ma1.35 nearly there need ma1.0
                    [-np.inf, -0.01693337, 0.44281864, 0.88893125, 1.27677076, np.inf],
                    0.0009968872628072388,
                    0.021513219342523964,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["284.4"]
            from probit_jax.data.wisconsin import quantile
            with pkg_resources.path(quantile, 'wpbc.npz') as path:
                data = np.load(path)
        elif J == 10:
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, -1.0 + 2. * 2. / J, -1.0 + 3. * 2. / J,
                              -1.0 + 4. * 2. / J, -1.0 + 5. * 2. / J, -1.0 + 6. * 2. / J, -1.0 + 7. * 2. / J,
                              -1.0 + 8. * 2. / J, np.inf]), 
                    100.0,
                    10.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0 = hyperparameters["init"]
            from probit_jax.data.wisconsin import decile
            with pkg_resources.path(decile, 'wpbc.npz') as path:
                data = np.load(path)
    X_trains = data["X_train"]
    t_trains = data["t_train"]
    X_tests = data["X_test"]
    t_tests = data["t_test"]
    # Python indexing - this is only for the benchmark data
    t_tests = t_tests - 1
    t_trains = t_trains - 1
    t_tests = t_tests.astype(int)
    t_trains = t_trains.astype(int)
    # Number of splits
    N_splits = len(X_trains)
    assert len(X_trains) == len(X_tests)
    X_true = data_continuous["X"]
    Y_true = data_continuous["y"]  # this is not going to be the correct one(?) - see next line
    # Y_trues = get_Y_trues(X_trains, X_true, Y_true)
    Kernel = EQ
    scale_0 = 1.0
    cutpoints_0 = np.array(cutpoints_0)
    return (
        X_trains, t_trains,
        X_tests, t_tests,
        X_true, Y_true,
        cutpoints_0, varphi_0, noise_variance_0, scale_0,
        J, D, Kernel)


def load_data_synthetic(dataset, J, plot=False):
    """Load synthetic data. TODO SS"""
    print(dataset)
    if dataset == "EQ":
        D = 1
        Kernel = EQ
        if J == 3:
            from probit_jax.data.SEIso import tertile
            with pkg_resources.path(tertile, 'tertile_prior_s=1_sigma2=0.1_varphi=30_new.npz') as path:
            #with pkg_resources.path(tertile, 'tertile_prior_s=1_sigma2=0.1_varphi=30.npz') as path:  # works for varphi
            #with pkg_resources.path(tertile, 'tertile_prior_s=1_sigma2=1_varphi=30.npz') as path:  # is varphi actually 1?
            #with pkg_resources.path(tertile, 'tertile_prior_s=30_sigma2=10_varphi=30.npz') as path:
                data = np.load(path)
            # X_show = data["X_show"]
            # Z_show = data["Z_show"]
            X_j = data["X_k"]  # Contains (90, 3) array of binned x values
            Y_true_j = data["Y_k"]  # Contains (90, 3) array of binned y values
            X = data["X"]  # Contains (90,) array of x values
            t = data["t"]  # Contains (90,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (1792,) array of y values, corresponding to Xs values (not in order)
            X_true = X
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0,
                    1.0,
                ),
                "true": (
                    data["gamma"],
                    data["varphi"],
                    1.0,  # data["noise_variance"],  #  correct value is 1.0, not this: data["noise_variance"],  # np.sqrt(0.1) = 0.316 
                    data["scale"],
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, np.inf]),
                    100.0,
                    10.0,
                    1.0
                ),
                "cosine": (
                    data["gamma"],
                    [0.0, 5.0],
                    1.0,  # data["noise_variance"],  #  correct value is 1.0, not this: data["noise_variance"],  # np.sqrt(0.1) = 0.316 
                    1.0,
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0, scale_0 = hyperparameters["true"]
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        elif J == 13:
            from probit_jax.data.SEIso import thirteen
            with pkg_resources.path(
                thirteen,
                'thirteen_prior_s=1_sigma2=0.1_varphi=30_new.npz') as path:
                data = np.load(path)
            X_show = data["X_show"]
            Z_show = data["Z_show"]
            X_j = data["X_k"]  # Contains (13, 45) array of binned x values. Note, old variable name of 'k'.
            Y_true_j = data["Y_k"]  # Contains (13, 45) array of binned y values
            X = data["X"]  # Contains (585,) array of x values
            t = data["t"]  # Contains (585,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"]  # Contains (585,) array of y values, corresponding to Xs values (not in order)
            colors = data["colors"]
            X_true = X
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0,
                    1.0,
                ),
                "true": (
                    data["gamma"],
                    data["varphi"],
                    1.0,  # data["noise_variance"],  # np.sqrt(0.1) = 0.316 # Should be 1.0 not 0.1
                    data["scale"],
                ),
                "init_alt": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, np.inf]),
                    100.0,
                    10.0,
                    1.0
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0, scale_0 = hyperparameters["true"]
        elif J==52:
            from probit_jax.data.SEIso import sparse52
            with pkg_resources.path(
                sparse52,
                'J=52_kernel_string=SEIso_scale=1.0_noisevar=0.1_lengthscale=30.0.npz') as path:
                data = np.load(path)
            N = 10000
            idx = np.random.choice(
                data["X"].shape[0], size=N, replace=False)
            X_show = data["X_show"]
            Z_show = data["Z_show"]
            X_j = data["X_js"]  # Contains (13, 45) array of binned x values. Note, old variable name of 'k'.
            Y_true_j = data["Y_js"]  # Contains (13, 45) array of binned y values
            X = data["X"][idx]  # Contains (585,) array of x values
            t = data["t"][idx]  # Contains (585,) array of ordinal response variables, corresponding to Xs values
            Y_true = data["Y"][idx]  # Contains (585,) array of y values, corresponding to Xs values (not in order)
            colors = data["colors"]
            X_true = X
            hyperparameters = {
                "init": (
                    data["cutpoints"],
                    data["varphi"],
                    0.001,
                    data["scale"]
                ),
                "true": (
                    data["cutpoints"],
                    data["varphi"],
                    data["noise_variance"],
                    data["scale"],
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0, scale_0 = hyperparameters["init"]
    elif dataset == "Linear":
        D = 1
        Kernel = Linear
        if J == 3:
            from probit_jax.data.Linear import tertile
            with pkg_resources.path(tertile, 'data.npz') as path:
                data = np.load(path)
            X_show = data["X_show"]
            Z_show = data["Z_show"]
            X_j = data["X_k"] 
            Y_true_j = data["Y_k"]
            X = data["X"]
            t = data["t"]
            Y_true = data["Y"]
            colors = data["colors"]
            X_true = X
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0,
                    1.0,
                ),
                "true": (
                    data["gamma"],
                    [data["constant_variance"], data["c"]],
                    data["noise_variance"],
                    data["scale"],
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0, scale_0 = hyperparameters["true"]
        elif J == 13:
            from probit_jax.data.Linear import thirteen
            with pkg_resources.path(thirteen, 'data.npz') as path:
                data = np.load(path)
            X_show = data["X_show"]
            Z_show = data["Z_show"]
            X_j = data["X_k"]
            Y_true_j = data["Y_k"]
            X = data["X"]
            t = data["t"]
            Y_true = data["Y"]
            colors = data["colors"]
            X_true = X
            hyperparameters = {
                "init": (
                    np.array([-np.inf, -1.0, -1.0 + 1. * 2. / J, np.inf]),
                    0.5 / D,
                    1.0,
                    1.0,
                ),
                "true": (
                    data["gamma"],
                    [data["constant_variance"], data["c"]],
                    data["noise_variance"],
                    data["scale"]
                ),
            }
            cutpoints_0, varphi_0, noise_variance_0, scale_0 = hyperparameters[
                "true"]
    if plot:
        plt.scatter(X, Y_true)
        plt.savefig("plot.png")
        plt.close()
    cutpoints_0 = np.array(cutpoints_0)
    if plot:
        plot_ordinal(X, t, X_j, Y_true_j, J, D)
    return (
        X, t,
        X_true, Y_true,
        cutpoints_0, varphi_0, noise_variance_0, scale_0,
        J, D, colors, Kernel)


def load_data_paper(dataset, J=None, D=None, ARD=None, plot=False):
    """Load synthetic data."""
    if dataset == "figure2":
        Kernel = EQ
        from probit_jax.data.paper import figure2
        with pkg_resources.path(figure2, 'J=3_kernel_string=SquaredExponential_var=1.0_noisevar=0.1_lengthscale=0.35.npz') as path:
            data = np.load(path)
        if plot:
            N_show = data["N_show"]
            X_show = data["X_show"]
            f_show = data["f_show"]
            X_js = data["X_js"]
            g_js = data["g_js"]
        X = data["X"]
        g = data["g"]
        f = data["f"]
        y = data["y"]
        # N = data["N"]
        colors = data["colors"]
        J = data["J"]
        D = data["D"]
        hyperparameters = {
            "true" : (
                data["cutpoints"],
                data["lengthscale"],
                data["noise_variance"],
                data["variance"]
            )
        }
    elif dataset == "figure2og":
        Kernel = EQ
        from probit_jax.data.paper import figure2
        with pkg_resources.path(figure2, 'J=2_kernel_string=SquaredExponential_var=1.0_noisevar=4.3264_lengthscale=0.35.npz') as path:
            data = np.load(path)
        if plot:
            N_show = data["N_show"]
            X_show = data["X_show"]
            f_show = data["f_show"]
            X_js = data["X_js"]
            g_js = data["g_js"]
        X = data["X"]
        g = data["g"]
        f = data["f"]
        y = data["y"]
        # N = data["N"]
        colors = data["colors"]
        J = data["J"]
        D = data["D"]
        hyperparameters = {
            "true" : (
                [np.NINF, 0.0, np.inf], #  data["cutpoints"],
                data["lengthscale"],
                data["noise_variance"],
                data["variance"]
            )
        }
    elif dataset == "13":
        Kernel = EQ
        from probit_jax.data.paper import figure2
        with pkg_resources.path(figure2, '13_SEIso_scale=1.0_noisevar=0.1_lengthscale=30.0.npz') as path:
            data = np.load(path)
        if plot:
            N_show = data["N_show"]
            X_show = data["X_show"]
            Z_show = data["Z_show"]
            X_js = data["X_js"]
            Y_js = data["Y_js"]
        X = data["X"]
        Y = data["Y"]
        t = data["t"]
        # N = data["N"]
        colors = data["colors"]
        J = data["J"]
        D = data["D"]
        hyperparameters = {
            "true" : (
                data["gamma"],
                data["varphi"],
                data["noise_variance"],
                data["scale"]
            )
        }
    elif dataset == "table1":
        # TODO: Superceded, needs to be regenerated
        if D is None:
            raise ValueError(
                "Please supply the input dimensions argument, D. expected {}, got {}".format("2, 10", None))
        if J is None:
            raise ValueError(
                "Please supply the number of classes argument, J. expected {}, got {}".format("3, 11", None))
        if ARD is None:
            raise ValueError(
                "Please supply argument, ARD. expected {}, got {}".format("True, False", None))
        from probit_jax.data.paper import table1
        if J == 3:
            if ARD is True:
                Kernel = EQ
                if D == 2:
                    with pkg_resources.path(table1, 'D=2_J=3_kernel_string=SquaredExponential_var=1.0_noisevar=4.3264_lengthscale=0.35.npz') as path:
                        data = np.load(path)
                    if plot:
                        N_show = data["N_show"]
                        X_show = data["X_show"]
                        f_show = data["f_show"]
                        X_js = data["X_js"]
                        g_js = data["g_js"]
                    X = data["X"]
                    g = data["g"]
                    f = data["f"]
                    y = data["y"]
                    # N = data["N"]
                    colors = data["colors"]
                    J = data["J"]
                    D = data["D"]
                    hyperparameters = {
                        "true" : (
                            data["cutpoints"],
                            np.array([data["lengthscale"], data["lengthscale"]]),
                            data["noise_variance"],
                            data["variance"]
                        )
                    }
                elif D == 10:
                    assert 0
            else:
                Kernel = EQ
                if D == 2:
                    with pkg_resources.path(table1, 'D=2_J=3_kernel_string=SquaredExponential_var=1.0_noisevar=4.3264_lengthscale=0.35.npz') as path:
                        data = np.load(path)
                    if plot:
                        N_show = data["N_show"]
                        X_show = data["X_show"]
                        f_show = data["f_show"]
                        X_js = data["X_js"]
                        g_js = data["g_js"]
                    X = data["X"]
                    g = data["g"]
                    f = data["f"]
                    y = data["y"]
                    # N = data["N"]
                    colors = data["colors"]
                    J = data["J"]
                    D = data["D"]
                    hyperparameters = {
                        "true" : (
                            data["cutpoints"],
                            data["lengthscale"],
                            data["noise_variance"],
                            data["variance"]
                        )
                    }
                elif D == 10:
                    with pkg_resources.path(table1, 'D=10_J=3_kernel_string=SquaredExponential_var=1.0_noisevar=4.3264_lengthscale=0.35.npz') as path:
                        data = np.load(path)
                    if plot:
                        N_show = data["N_show"]
                        X_show = data["X_show"]
                        f_show = data["f_show"]
                        X_js = data["X_js"]
                        g_js = data["g_js"]
                    X = data["X"]
                    g = data["g"]
                    f = data["f"]
                    y = data["y"]
                    # N = data["N"]
                    colors = data["colors"]
                    J = data["J"]
                    D = data["D"]
                    hyperparameters = {
                        "true" : (
                            data["cutpoints"],
                            data["lengthscale"],
                            data["noise_variance"],
                            data["variance"]
                        )
                    }
        elif J == 13:
            if ARD is True:
                Kernel = EQ
                if D == 2:
                    assert 0
                elif D == 10:
                    assert 0
            else:
                Kernel = EQ
                if D == 2:
                    with pkg_resources.path(table1, 'D=2_J=13_kernel_string=SquaredExponential_var=1.0_noisevar=4.3264_lengthscale=0.35.npz') as path:
                        data = np.load(path)
                    if plot:
                        N_show = data["N_show"]
                        X_show = data["X_show"]
                        f_show = data["f_show"]
                        X_js = data["X_js"]
                        g_js = data["g_js"]
                    X = data["X"]
                    g = data["g"]
                    f = data["f"]
                    y = data["y"]
                    # N = data["N"]
                    colors = data["colors"]
                    J = data["J"]
                    D = data["D"]
                    hyperparameters = {
                        "true" : (
                            data["cutpoints"],
                            data["lengthscale"],
                            data["noise_variance"],
                            data["variance"]
                        )
                    }
                elif D == 10:
                    with pkg_resources.path(table1, 'D=10_J=13_kernel_string=SquaredExponential_var=1.0_noisevar=4.3264_lengthscale=0.35.npz') as path:
                        data = np.load(path)
                    if plot:
                        N_show = data["N_show"]
                        X_show = data["X_show"]
                        f_show = data["f_show"]
                        X_js = data["X_js"]
                        g_js = data["g_js"]
                    X = data["X"]
                    g = data["g"]
                    f = data["f"]
                    y = data["y"]
                    # N = data["N"]
                    colors = data["colors"]
                    J = data["J"]
                    D = data["D"]
                    hyperparameters = {
                        "true" : (
                            data["cutpoints"],
                            data["lengthscale"],
                            data["noise_variance"],
                            data["variance"]
                        )
                    }
    cutpoints_0, varphi_0, noise_variance_0, variance_0 = hyperparameters["true"]
    if plot:
        plot_ordinal(X, y, g, X_show, f_show, J, D, colors, plt.cm.get_cmap('viridis', J), N_show=N_show)
    cutpoints_0 = np.array(cutpoints_0)
    return (
        X, f, g, y,
        cutpoints_0, varphi_0, noise_variance_0, variance_0,
        J, D, colors, Kernel)


def plot_kernel(kernel, N_total=500, n_samples=10):
    for _ in range(n_samples):
        X = np.linspace(0., 1., N_total)  # 500 points evenly spaced over [0,1]
        X = X[:, None]  # reshape X to make it n*D
        mu = np.zeros((N_total))  # vector of the means
        K = kernel.kernel_matrix(X, X)
        Z = np.random.multivariate_normal(mu, K)
        plt.plot(X[:], Z[:])
    plt.show()


def plot_ordinal(X, y, g, X_show, f_show, J, D, colors, cmap, N_show=None):
    """TODO: generalise to 3D, move to plot.py"""
    colors_ = [colors[i] for i in y]
    if D==1:
            plt.scatter(X, g, color=colors_)
            plt.plot(X_show, f_show, color='k', alpha=0.4)
            plt.show()
            plt.savefig("scatter.png")
            # plot_ordinal(X, t, X_js, Y_js, J, D, colors=colors)
            # SS
            # N_total = len(t)
            # colors_ = [colors[i] for i in t]
            # fig, ax = plt.subplots()
            # plt.scatter(X[:, 0], t, color=colors_)
            # plt.title("N_total={}, J={}, D={} Ordinal response data".format(N_total, J, D))
            # plt.xlabel(r"$x$", fontsize=16)
            # ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
            # plt.ylabel(r"$t$", fontsize=16)
            # plt.show()
            # plt.savefig("N_total={}, J={}, D={} Ordinal response data.png".format(N_total, J, D))
            # plt.close()
            # # Plot from the binned arrays
            # for j in range(J):
            #     plt.scatter(X_j[j][:, 0], Y_j[j], color=colors[j], label=r"$t={}$".format(j))
            # plt.title("N_total={}, J={}, D={} Ordinal response data".format(N_total, J, D))
            # plt.legend()
            # plt.xlabel(r"$x$", fontsize=16)
            # plt.ylabel(r"$y$", fontsize=16)
            # plt.show()
            # plt.savefig("N_total={}, J={}, D={} Ordinal response data_.png".format(N_total, J, D))
            # plt.close()
    elif D==2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter3D(X[:, 0], X[:, 1], g[:], color=colors_)
        surf = ax.plot_surface(
            X_show[:, 0].reshape(N_show, N_show),
            X_show[:, 1].reshape(N_show, N_show),
            f_show.reshape(N_show, N_show), alpha=0.4)
        fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap))  # TODO: how to not normalize this
        plt.savefig("surface.png")
        plt.show()
        plt.close()


def calculate_metrics(y_test, t_test, Z, cutpoints):
    """Calculate nPlan metrics and return a big tuple containing them."""
    t_pred = np.argmax(Z, axis=1)
    print("t_pred")
    print(t_pred)
    print("t_test")
    print(t_test)
    grid = np.ogrid[0:len(t_test)]
    # Other error
    predictive_likelihood = Z[grid, t_test]
    mean_absolute_error = np.sum(np.abs(t_pred - t_test)) / len(t_test)
    root_mean_squared_error = np.sqrt(
        np.sum(pow(t_pred - t_test, 2)) / len(t_test))
    print("root_mean_squared_error", root_mean_squared_error)
    print("mean_absolute_error ", mean_absolute_error)
    log_predictive_probability = np.sum(np.log(predictive_likelihood))
    print("log_pred_probability ", log_predictive_probability)
    predictive_likelihood = np.sum(predictive_likelihood) / len(t_test)
    print("predictive_likelihood ", predictive_likelihood)
    print("av_prob_of_correct ", predictive_likelihood)
    print(np.sum(t_pred != t_test), "sum incorrect")
    mean_zero_one = np.sum(t_pred != t_test) / len(t_test)
    print("mean_zero_one_error", mean_zero_one)
    print(np.sum(t_pred == t_test), "sum correct")
    mean_zero_one = np.sum(t_pred == t_test) / len(t_test)
    print("mean_zero_one_correct", mean_zero_one)
    return (
        mean_zero_one,
        root_mean_squared_error,
        mean_absolute_error,
        log_predictive_probability,
        predictive_likelihood)


if __name__ == "__main__":
    J = 3
    cmap = plt.cm.get_cmap('viridis', J)    # J discrete colors
    colors = []
    for j in range(J):
        colors.append(cmap((j + 0.5)/J))
