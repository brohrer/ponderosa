import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ponderosa.toolbox as ptb
from ponderosa.optimizers import EvoPowell

plt.switch_backend("agg")


def main():
    # Define your conditions - your discrete search space - by creating
    # a dictionary of key, value pairs
    # where each key is a parameter name, and each value
    # is the list of values that parameter can take.
    # Those values can be any object: numbers or strings, lists or dicts,
    # even functions of classes.
    conditions = {
        "x": list(np.linspace(0, np.pi, 10)),
        "y": list(np.linspace(0, np.pi, 10)),
    }

    # Choose your optimization algorithm and run its optimize() method.
    optimizer = EvoPowell()
    lowest_error, best_condition, results_logfile = optimizer.optimize(
        evaluate, conditions, verbose=False)

    print(
        "All done! The data on each condition evaluated, and its error\n"
        + f"are stored in {results_logfile}.")

    # Optionally, when you're done you can turn the results into an image.
    visualize(results_logfile)


def evaluate(x=0, y=0):
    """
    The objective function is a 2D variant of the sinc function.
    """
    x0 = 1
    y0 = 1.5
    return - np.sinc(x - x0) * np.sinc(y - y0)


def visualize(results_logfile):
    """
    The error is multiplied by -1 here, so that it looks like the
    algorithm is trying to climb the mountain, rather than find its
    way to the bottom of a well. It's easier to visualize well and
    a bit more cheerful.
    """
    results = ptb.results_csv_to_dict_list(results_logfile)
    x = []
    y = []
    z = []
    best_so_far = []
    bsf = 1e-10
    for result in results:
        x.append(float(result["x"]))
        y.append(float(result["y"]))
        zval = -1 * float(result["error"])
        z.append(zval)
        if zval > bsf:
            bsf = zval
        best_so_far.append(bsf)

    fig = plt.figure()

    # The upper left plot shows a 3D surface of the objective function.
    ax_surf = fig.add_subplot(221, projection='3d')
    x_all_hi = np.linspace(0, np.pi, 100)
    y_all_hi = np.linspace(0, np.pi, 100)
    X, Y = np.meshgrid(x_all_hi, y_all_hi)
    Z = -1 * evaluate(x=X, y=Y)

    ax_surf.plot_surface(
        X, Y, Z,
        cmap=cm.inferno,
        linewidth=0,
        antialiased=False)
    ax_surf.set_xlabel("x")
    ax_surf.set_ylabel("y")
    ax_surf.set_xlim(0, np.pi)
    ax_surf.set_ylim(0, np.pi)
    ax_surf.set_zlim(-.2, 1)

    # The upper right plot shows a 3D representation of the points
    # in the search space that were evaluated.
    ax_eval = fig.add_subplot(222, projection='3d')
    ax_eval.scatter(
        x, y, z,
        c=z,
        cmap=cm.inferno,
        vmax=1,
        vmin=-.2,
        s=10,
    )

    for i in range(len(x)):
        ax_eval.plot(
            [x[i], x[i]],
            [y[i], y[i]],
            [z[i], 0],
            linewidth=.5,
            color="blue",
        )
    ax_eval.set_xlabel("x")
    ax_eval.set_ylabel("y")
    ax_eval.set_xlim(0, np.pi)
    ax_eval.set_ylim(0, np.pi)
    ax_eval.set_zlim(-.2, 1)

    # The lower left function shows the best error score found so far as
    # more points are evaluated.
    ax_bsf = fig.add_subplot(223)
    ax_bsf.plot(
        np.arange(len(best_so_far)) + 1,
        best_so_far,
        color="blue",
    )
    ax_bsf.set_xlabel("Points evaluated")
    ax_bsf.set_ylabel("Best value so far")
    ax_bsf.set_xlim(0, len(best_so_far) + 1)
    ax_bsf.set_ylim(-.01, 1.01)

    # A 2D version of the plot in the upper right, showing the points
    # evaluated so far and the error associated with them.
    ax_cover = fig.add_subplot(224)
    ax_cover.scatter(
        x, y,
        c=z,
        vmax=1,
        vmin=-.2,
        cmap=cm.inferno,
        s=20,
    )
    ax_cover.set_xlabel("x")
    ax_cover.set_ylabel("y")
    ax_cover.set_xlim(-.1, np.pi + .1)
    ax_cover.set_ylim(-.1, np.pi + .1)

    viz_file = "demo_visualization.png"
    fig.savefig(viz_file, dpi=300)
    plt.close()

    print(f"There's also a 3D visualization of it in {viz_file}.")

main()
