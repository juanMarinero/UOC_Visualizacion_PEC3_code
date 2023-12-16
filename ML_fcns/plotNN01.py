#!/usr/bin/env python3

#  vim: set foldmethod=indent foldcolumn=4 :


import numpy as np
import matplotlib.pyplot as plt
import string


def plotNN01aux02(x, y, thresh=0.02, shift=0.01):
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(y, tuple):
        y = list(y)
    for i in range(len(x)):
        while abs(x[i] - y[i]) < thresh:
            if x[i] > y[i]:
                x[i] = x[i] + shift
            else:
                x[i] = x[i] - shift
    return tuple(x), tuple(x)


def plotNN01aux01(
    layer1,
    layer2,
    weights,
    shiftx=1 / 3,
    ax=None,
    i1="i",
    i2="j",
    showVals=True,
    showFormulas=True,
    radialBool=True,
    thresh=0.02,
    latexBool=True,
    formatVal=".2f",
    arrow_color_fcn=lambda x: "black" if x > 0 else "black",
    arrow_opacity_fcn=lambda x: 0.3 if x > 0 else 0.3,
    arrow_width=0.001
    * 10,  # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.arrow.html
    arrow_width_variable=False,
    fontsize=10,
    conector="$=$",
    debugBool=False,
):
    label_position_last = [np.Inf, np.Inf]
    conector = conector if (showVals and showFormulas) else ""
    weights_in_layer = [
        np.abs(weights[i, j]) for i in range(len(layer1)) for j in range(len(layer2))
    ]
    sum_weights_in_layer = sum(weights_in_layer)
    mean_weights_in_layer = np.mean(weights_in_layer)
    for i in range(len(layer1)):
        for j in range(len(layer2)):

            # weight text
            w_ij = weights[i, j]
            weight_value = f"{w_ij:{formatVal}}" if showVals else ""
            weight_formula = f"$w_{{{i1}{i + 1}{i2}{j + 1}}}$" if showFormulas else ""
            weight_label = weight_formula + conector + weight_value
            if not latexBool:
                weight_label = weight_label.replace("$", "")

            if radialBool:
                arrow_length = np.sqrt(
                    (layer2[j, 0] - layer1[i, 0]) ** 2
                    + (layer2[j, 1] - layer1[i, 1]) ** 2
                )
            else:
                arrow_length = 1

            label_position = (
                layer1[i, 0] + (shiftx / arrow_length) * (layer2[j, 0] - layer1[i, 0]),
                layer1[i, 1] + (shiftx / arrow_length) * (layer2[j, 1] - layer1[i, 1]),
            )
            label_position, label_position_last = plotNN01aux02(
                label_position, label_position_last, thresh=thresh
            )  # optimize label position
            ax.text(
                label_position[0],
                label_position[1],
                weight_label,
                ha="center",
                va="center",
                fontsize=fontsize,
            )

            # arrow
            # e.g. arrow_color_fcn = lambda x: "blue" if x > 0 else "black"
            # e.g. arrow_opacity_fcn=lambda x: 0.3 if x > 0 else 0.3,
            arrow_width_ij = arrow_width
            if arrow_width_variable:
                #  arrow_width_ij = max(0.0001, arrow_width * np.abs(w_ij) / sum_weights_in_layer )
                #  arrow_width_ij = max(0.0001, arrow_width * np.log(1 + np.abs(w_ij)) / sum_weights_in_layer)
                arrow_width_ij = arrow_width * np.log(
                    np.abs(w_ij) / mean_weights_in_layer
                )
                arrow_width_ij = max(arrow_width_ij, 0.0001)
                arrow_width_ij = min(arrow_width_ij, 0.0001 * 1000)
                if debugBool:
                    print(i, j, f"{w_ij:+.2f}", arrow_width_ij)
            ax.arrow(
                layer1[i, 0],
                layer1[i, 1],
                layer2[j, 0] - layer1[i, 0],
                layer2[j, 1] - layer1[i, 1],
                alpha=arrow_opacity_fcn(w_ij),
                color=arrow_color_fcn(w_ij),
                width=arrow_width_ij,
                head_length=0,
            )


def plotNN01(
    cells=[2, 4, 2],
    weights=None,
    biases=None,
    figsize=(10, 5),
    showVals=True,
    showFormulas=True,
    showVals_neuron=False,
    showFormulas_neuron=True,
    radialBool=True,
    thresh=0.02,
    latexBool=True,
    formatVal=".2f",
    layer_names=None,
    layer_colors=None,
    neuron_names=None,
    neuron_values=None,
    arrow_color_fcn=lambda x: "black" if x > 0 else "black",
    arrow_opacity_fcn=lambda x: 0.1,  # lambda x: 0.3 if x > 0 else 0.3,
    arrow_width=0.001,  # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.arrow.html
    arrow_width_variable=False,
    formulas_subindexes=None,
    conector="$=$",
    showLegend=True,
    fontsize=10,
    ax=None,
    debugBool=False,
):
    cells_max = max(cells)  # max nr. neurons in any layer
    n = len(cells)  # "n" is number of layers

    biasesPopulate = False
    if biases is None:
        biases = []
        biasesPopulate = True

    weightsPopulate = False
    if weights is None:
        weights = []
        weightsPopulate = True

    neuronsPopulate = False
    if neuron_values is None:
        neuron_values = []
        neuronsPopulate = True

    layer_positions = []
    for i, k in enumerate(cells):

        # populate layers positions from top to bottom
        if k > 1:
            aux = np.linspace(0, cells_max - 1, k)[::-1]
        else:
            aux = [(cells_max - 1) / 2]
        aux = np.array([[i, val] for val in aux])
        layer_positions.append(aux)

        # populate weights and biases
        if weightsPopulate and (i < (n - 1)):
            weights.append(np.random.randn(k, cells[i + 1]))
        if biasesPopulate and (i > 0):
            biases.append(np.random.randn(k))
        if neuronsPopulate:
            neuron_values.append(np.random.randn(k))
    if debugBool:
        print(f"weights: {weights}")
        print(f"biases: {biases}")
        print(f"neuron_values: {neuron_values}")

    if formulas_subindexes is None:
        alphabet = string.ascii_lowercase
        formulas_subindexes = [letter for letter in alphabet[alphabet.index("i") :]]
    if neuron_names is None:
        # layer super indexes
        if n > len(formulas_subindexes):
            formulas_subindexes = np.full(n, "")
        neuron_names = []
        for i in range(n):  # for each layer...
            layer1 = layer_positions[i]
            textArr = [f"{formulas_subindexes[i]}{k+1}" for k in range(len(layer1))]
            neuron_names.append(textArr)
    if debugBool:
        print(f"neuron_names: {neuron_names}")

    # layer names
    if layer_names is None:
        layer_names = [f"Hidden layer {k}" for k in range(n)]
        layer_names[0] = "Input Neurons"
        layer_names[-1] = "Output Neurons"
    if debugBool:
        print(f"layer_names: {layer_names}")

    # layer color
    if layer_colors is None:
        layer_colors = np.full(n, "brown")
        layer_colors[0] = "blue"
        layer_colors[-1] = "gold"
    if debugBool:
        print(f"layer_colors: {layer_colors}")

    # show weights-biases
    if not isinstance(showVals, list):
        showVals = np.full((n - 1,), showVals)
    if not isinstance(showFormulas, list):
        showFormulas = np.full((n - 1,), showFormulas)
    if debugBool:
        print(f"showVals: {showVals}")
        print(f"showFormulas: {showFormulas}")

    # show neurons
    if not isinstance(showVals_neuron, list):
        showVals_neuron = np.full((n,), showVals_neuron)
    if not isinstance(showFormulas_neuron, list):
        showFormulas_neuron = np.full((n,), showFormulas_neuron)
    if debugBool:
        print(f"showVals_neuron: {showVals_neuron}")
        print(f"showFormulas_neuron: {showFormulas_neuron}")

    # -------------------
    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        #  fig = plt.figure(figsize=figsize)
        fig = ax

    for i in range(n):  # for each layer...

        # scatter neurons
        layer1 = layer_positions[i]
        ax.scatter(
            layer1[:, 0],
            layer1[:, 1],
            label=layer_names[i],
            color=layer_colors[i],
            s=200,
            alpha=0.3,
        )

        # text neurons names and values
        conector_new = (
            conector if (showVals_neuron[i] and showFormulas_neuron[i]) else ""
        )
        for j in range(len(layer1)):
            neuron_value = (
                f"{neuron_values[i][j]:{formatVal}}" if showVals_neuron[i] else ""
            )
            neuron_formula = neuron_names[i][j] if showFormulas_neuron[i] else ""
            neuron_label = neuron_formula + conector_new + neuron_value
            if not latexBool:
                neuron_label = neuron_label.replace("$", "")
            ax.text(
                layer1[j, 0],
                layer1[j, 1],
                neuron_label,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

        # bias text
        if i > 0:
            conector_new = conector if (showVals[i - 1] and showFormulas[i - 1]) else ""
            for j in range(len(layer1)):
                bias_value = f"{biases[i-1][j]:{formatVal}}" if showVals[i - 1] else ""
                bias_formula = (
                    f"$b_{{{formulas_subindexes[i]}{j+1}}}$"
                    if showFormulas[i - 1]
                    else ""
                )
                bias_label = bias_formula + conector_new + bias_value
                if not latexBool:
                    bias_label = bias_label.replace("$", "")
                ax.text(
                    layer1[j, 0] - 0.1,
                    layer1[j, 1] + 0.2,
                    bias_label,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                )

        # weight text, arrows
        shiftx = 2 / 3
        if i < (n - 1):
            if cells[i] < cells[i + 1]:
                shiftx = 1 / 3
            layer2 = layer_positions[i + 1]
            # print(f"weights[i]: {weights[i]}")
            plotNN01aux01(
                layer1,
                layer2,
                weights[i],
                shiftx,
                ax,
                i1=formulas_subindexes[i],
                i2=formulas_subindexes[i + 1],
                showVals=showVals[i],
                showFormulas=showFormulas[i],
                radialBool=radialBool,
                thresh=thresh,
                latexBool=latexBool,
                formatVal=formatVal,
                arrow_color_fcn=arrow_color_fcn,
                arrow_opacity_fcn=arrow_opacity_fcn,
                arrow_width=arrow_width,
                arrow_width_variable=arrow_width_variable,
                fontsize=fontsize,
                debugBool=debugBool,
            )
    ax.axis("off")
    if showLegend:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(-0.3, cells_max - 1 + 0.3)

    return fig


if __name__ == "__main__":

    # plotNN01([2,3,4,5], figsize=(12,15))
    # plotNN01();
    # plotNN01([2,4,1],showVals=0, radialBool=0)
    # plotNN01_2_4_1_no_latex = plotNN01([2,4,1],showVals=0, radialBool=0, latexBool=0)

    if 0:
        plotNN01([2, 5, 4, 8, 2], figsize=(12, 25), thresh=0.03, radialBool=0)
    else:
        cells = [4, 2, 4]
        w = [
            np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]),
            np.array([[1, 1, 1, -1], [-1, 2, -5, -1]]),
        ]
        b = [np.zeros(cells[-2]), np.zeros(cells[-1])]
        neuron_names = [
            ["p_1", "p_2", "p_3", "p_4"],
            ["q_1", "q_2"],
            ["r_1", "r_2", "r_3", "r_4"],
        ]
        neuron_values = [[0] * len(sublist) for sublist in neuron_names]
        layer_names = ["INPUTS", "Hidden LAYER", "OUTPUTS"]
        layer_colors = ["orange", "brown", "gold"]
        alphabet = string.ascii_lowercase
        formulas_subindexes = [letter for letter in alphabet[alphabet.index("p") :]]
        plotNN01(
            cells,
            weights=w,
            biases=b,
            layer_names=layer_names,
            layer_colors=layer_colors,
            radialBool=0,
            formatVal=".0f",
            figsize=(10, 3),
            debugBool=True,
            showVals=[0, 1],
            showFormulas=[0, 1],
            showVals_neuron=[0, 1, 0],
            showFormulas_neuron=[0, 1, 1],
            neuron_names=neuron_names,
            neuron_values=neuron_values,
            arrow_color_fcn=lambda x: "green" if x > 0 else "red",
            arrow_opacity_fcn=lambda x: 0.3 if np.abs(x) > 0 else 0.1,
            arrow_width=0.001 * 50,
            arrow_width_variable=True,
            formulas_subindexes=formulas_subindexes,
            conector=r"$\rightarrow$",
            fontsize=14,
            showLegend=True,
        )

    plt.show()
