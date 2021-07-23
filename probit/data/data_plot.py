import numpy as np
import matplotlib.pyplot as plt
from probit.data.utilities import (
    generate_prior_data, generate_synthetic_data, get_Y_trues, colors,
    datasets, metadata, load_data, load_data_synthetic, training, training_varphi)

argument = "septile"

X, t, X_true, Y_true, gamma_0, varphi_0, noise_variance_0, K, D = load_data_synthetic(argument, False, plot=True)
