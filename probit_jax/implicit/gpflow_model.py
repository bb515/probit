import gpflow
from gpflow.likelihoods import Ordinal
import numpy as np
import matplotlib.pyplot as plt


ord = Ordinal(np.array([0., 1.]))

def plot_model(model: gpflow.models.GPModel) -> None:
    X, Y = model.data
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    gpflow.utilities.print_summary(model, "notebook")

    Xplot = np.linspace(0.0, 1.0, 200)[:, None]

    y_mean, y_var = model.predict_y(Xplot, full_cov=False)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(X, Y, "kx", mew=2)
    (mean_line,) = ax.plot(Xplot, y_mean, "-")
    color = mean_line.get_color()
    ax.plot(Xplot, y_lower, lw=0.1, color=color)
    ax.plot(Xplot, y_upper, lw=0.1, color=color)
    ax.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color=color, alpha=0.1
    )

    fig.savefig("probit_jax/implicit/gpflow.png")

X = np.array(
    [
        [0.177], [0.183], [0.428], [0.838], [0.827], [0.293], [0.270], [0.593],
        [0.031], [0.650],
    ]
)
Y = np.array(
    [
        [1.22], [1.17], [1.99], [2.29], [2.29], [1.28], [1.20], [1.82], [1.01],
        [1.93],
    ]
)



model = gpflow.models.VGP(
    (X, Y),
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=ord,
)
plot_model(model)
