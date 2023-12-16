import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML



PREDICTION_SURFACE_RESOLUTION = 20
PREDICTION_COLORMAP = 'spring'

def visualize_classification_learning(generate_classification_data,plotNN02, problem_type, loss_history, prediction_history,
                                         weights_history, biases_history, outfile=None):
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))


    prediction_surface_range = np.linspace(-.5, 1.5, PREDICTION_SURFACE_RESOLUTION)
    prediction_surface_x, prediction_surface_y = np.meshgrid(prediction_surface_range, prediction_surface_range)

    xx, yy, cls = generate_classification_data(problem_type=problem_type)

    # Initialize plots
    contour = axs[0].contourf(
        prediction_surface_x,
        prediction_surface_y,
        prediction_history[0]
    )
    points = axs[0].scatter(xx, yy, c=cls, cmap='gray_r')
    axs[0].set_title("Prediction Surface")
    # line, = axs[1].plot([], [], 'r-', linewidth=2)
    line, = axs[1].plot(loss_history, 'r-', linewidth=2)
    axs[1].set_title("Loss Function")

    kwargs=dict(arrow_color_fcn=lambda x: "green" if x > 0 else "red",
                arrow_width_variable = True, arrow_width = 0.001*50,
                showFormulas=False, ax=axs[2])
    plotNN02(weights_history, biases_history, kwargs=kwargs)

    # https://stackoverflow.com/questions/66938713
    suptitle = plt.suptitle("", fontsize=16)

    def animate(ii):
        # plt.suptitle("Iteration: {}".format(ii + 1), fontsize=16)
        suptitle.set_text(f'Iteration {ii}')

        # axs[0].clear()
        contour = axs[0].contourf(
            prediction_surface_x,
            prediction_surface_y,
            prediction_history[ii]
        )
        axs[0].scatter(xx, yy, c=cls, cmap='gray_r')
        axs[0].set_title(f"Prediction Surface")

        # Update the loss function plot
        if ii>0:
            line.set_data(range(ii), loss_history[:ii])

            axs[2].clear()
            plotNN02(weights_history[:ii], biases_history[:ii], kwargs=kwargs)

        return axs, contour, line

    anim = FuncAnimation(
        fig,
        animate,
        frames=np.arange(len(loss_history)),
        interval=50, repeat=False
    )

    plt.close()  # Close the plot to prevent double display

    if outfile:
        # anim.save requires imagemagick library to be installed
        anim.save(outfile, dpi=80, writer='imagemagick')

    return HTML(anim.to_jshtml())

def ann_main_classif(generate_classification_data,run_ann_training_simulation,plotNN02, PROBLEM_TYPE, N_HIDDEN_UNITS, n_iterations=100, learning_rate=2):
    loss_history, prediction_history, weights_history, biases_history = run_ann_training_simulation(
        problem_type=PROBLEM_TYPE,
        n_hidden_units=N_HIDDEN_UNITS,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
    )

    return visualize_classification_learning(generate_classification_data,plotNN02,
        PROBLEM_TYPE,
        np.array(loss_history).flatten(), # loss_history,
        prediction_history,
        list(weights_history.values()), # weights_history,
        list(biases_history.values()) # biases_history
    )
