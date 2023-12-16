import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import OrderedDict

# https://plotly.com/python/hover-text-and-formatting/
# https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scattermapbox.html
# https://stackoverflow.com/a/77668124/9391770


def display_htmlFcn_plots_activation_fcn_interactive():

    # Define a few common activation functions
    g_linear = lambda z: z
    g_sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    g_tanh = lambda z: np.tanh(z)

    # ...and their analytic derivatives
    g_prime_linear = lambda z: np.ones(len(z))
    g_prime_sigmoid = lambda z: 1.0 / (1 + np.exp(-z)) * (1 - 1.0 / (1 + np.exp(-z)))
    g_prime_tanh = lambda z: 1 - np.tanh(z) ** 2

    # Visualize each g_*(z)
    activation_functions = OrderedDict(
        [
            ("linear", (g_linear, g_prime_linear, "red")),
            ("sigmoid", (g_sigmoid, g_prime_sigmoid, "blue")),
            ("tanh", (g_tanh, g_prime_tanh, "green")),
        ]
    )

    xs = np.linspace(-5, 5, 100)

    # Create subplots: 1 row, 2 cols
    fig = go.FigureWidget(
        make_subplots(
            rows=1, cols=2, subplot_titles=("Activation Functions", "Derivatives")
        )
    )

    for name, params in activation_functions.items():
        # Compute y-values for activation functions and derivatives
        y_values = params[0](xs)  # For activation function
        y_prime_values = params[1](xs)  # For derivative
        color = params[-1]
        # kwargs=dict(x=xs, mode='lines', line=dict(color=color),text=f"{name}(z)", hoverinfo='x+y+text',
        #           hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>%{name:s}",)
        kwargs = dict(
            x=xs,
            mode="lines",
            line=dict(color=color),
            text=[f"{name}(z)"] * len(xs),
            hovertemplate="<br>".join(["y: %{y:.2f}&nbsp;&nbsp;%{text}"])
            + "<extra></extra>",  # hide the secondary box completely
        )
        # hovertemplate="<br>".join(["y: %{y:.2f}&nbsp;&nbsp;%{text}","y: %{y:.2f}"])

        # Activation functions
        fig.add_trace(
            go.Scatter(y=y_values, name=f"$g_{{\mathrm{{{name}}}}}(z)$", **kwargs),
            row=1,
            col=1,
        )

        fig.update_layout(hovermode="x")

        # Derivatives
        fig.add_trace(
            go.Scatter(
                y=y_prime_values, name=f"$g_{{\mathrm{{{name}}}}}'(z)$", **kwargs
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=400,
        width=800,
        title_text="Interactive Activation Functions and Derivatives",
    )
    fig.show()
