import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Sigmoid activation functions
g = lambda z: 1.0 / (1.0 + np.exp(-z))


def error_function(prediction, target=1):
    """Squared error function (f(x) - y)**2"""
    return (prediction - target) ** 2


# single layer ANN
def single_layer_network_predict(w1, w2, target_value=1):
    return g(w1 * target_value)


def plotSurfaceError_01(
    ax,
    k,
    layer_network_predict=single_layer_network_predict,
    set_xlabel_bool=False,
    view_angle=[20, 85],
):
    # Grid of allowed parameter values
    parameter_range = np.linspace(-k, k, 50)
    w1, w2 = np.meshgrid(parameter_range, parameter_range)

    single_layer_network_output = layer_network_predict(w1, w2)
    single_layer_network_error = error_function(single_layer_network_output)

    ax.plot_surface(w1, w2, single_layer_network_error, cmap="RdBu_r")
    ax.view_init(*view_angle)
    ax.set_xlabel("$w_1$")
    if set_xlabel_bool:
        ax.set_ylabel("$w_2$")
    else:
        plt.yticks([])
    ax.set_zlabel("$E(w)$")
    ax.set_zlim(0, 1)
