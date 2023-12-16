import numpy as np
import plotly.graph_objs as go


# Sigmoid activation functions
g = lambda z: 1.0 / (1.0 + np.exp(-z))


def error_function(prediction, target=1):
    """Squared error function (f(x) - y)**2"""
    return (prediction - target) ** 2


# single layer ANN
def single_layer_network_predict(w1, w2, target_value=1):
    return g(w1 * target_value)


def plotSurfaceErrorInteractive(k, layer_network_predict=single_layer_network_predict):
    # Grid of allowed parameter values
    parameter_range = np.linspace(-k, k, 50)
    w1, w2 = np.meshgrid(parameter_range, parameter_range)

    single_layer_network_output = layer_network_predict(w1, w2)
    single_layer_network_error = error_function(single_layer_network_output)

    surface = go.Surface(x=w1, y=w2, z=single_layer_network_error, colorscale="RdBu")
    return surface


def plotSurfaceError_02():
    parameter_range = 10  # Adjust the parameter range as needed

    surface_plot = plotSurfaceErrorInteractive(parameter_range)

    layout = go.Layout(
        title="Single-layer Network Error Surface",
        scene=dict(
            xaxis=dict(title="w_1"),
            yaxis=dict(title="w_2"),
            zaxis=dict(title="E(w)", range=[0, 1]),
        ),
    )

    fig = go.Figure(data=[surface_plot], layout=layout)
    fig.show()
