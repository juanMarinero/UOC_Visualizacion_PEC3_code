import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html

g = lambda z: 1.0 / (1.0 + np.exp(-z))


def error_function(prediction, target=1):
    """Squared error function (f(x) - y)**2"""
    return (prediction - target) ** 2


# single layer ANN
def single_layer_network_predict(w1, w2, target_value=1):
    return g(w1 * target_value)


def plotSurfaceErrorInteractiveSlider(
    parameter_range, layer_network_predict=single_layer_network_predict
):
    x_range = np.linspace(-parameter_range, parameter_range, 50)
    w1, w2 = np.meshgrid(x_range, x_range)

    single_layer_network_output = layer_network_predict(w1, w2)
    single_layer_network_error = error_function(single_layer_network_output)

    surface = go.Surface(x=w1, y=w2, z=single_layer_network_error, colorscale="RdBu")
    return surface


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id="surface-plot"),
        html.Label("X-axis Range"),
        dcc.Slider(
            id="x-axis-slider",
            min=1,
            max=20,
            step=1,
            value=10,
            marks={i: str(i) for i in range(0, 21, 5)},
        ),
    ]
)


@app.callback(
    dash.dependencies.Output("surface-plot", "figure"),
    [dash.dependencies.Input("x-axis-slider", "value")],
)
def update_figure(parameter_range):
    surface_plot = plotSurfaceErrorInteractiveSlider(parameter_range)
    layout = go.Layout(
        title="Single-layer Network Error Surface",
        scene=dict(
            xaxis=dict(title="w_1$"),
            yaxis=dict(title="w_2"),
            zaxis=dict(title="E(w)", range=[0, 1]),
            camera=dict(
                eye=dict(
                    x=0, y=2, z=0.3
                ),  # Adjust the camera position to be parallel to x-axis
            ),
        ),
        autosize=False,
        width=800,  # Set the width of the figure
        height=600,  # Set the height of the figure
    )
    return {"data": [surface_plot], "layout": layout}


def plotSurfaceError_03():
    app.run_server(debug=True)
