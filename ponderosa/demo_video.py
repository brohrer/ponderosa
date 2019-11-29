"""
This is the file for re-creating the demo video.
It requires FFMEPG (ffmpeg.org) and takes a few minutes to run.
If you'd just like to run the basic demo, try 

python3 demo.py
"""
import os
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ponderosa.toolbox as ptb
from ponderosa.optimizers import EvoPowell
from lodgepole import animation_tools as at

plt.switch_backend("agg")

FRAMES_DIR = "frames"


def evaluate(x=0, y=0):
    x0 = 1
    y0 = 1.5
    return - np.sinc(x - x0) * np.sinc(y - y0)


def create_frames():
    try:
        os.mkdir(FRAMES_DIR)
    except Exception:
        pass
    previous_frames = os.listdir(FRAMES_DIR)
    for frame in previous_frames:
        if frame[-4:] == ".png":
            os.remove(os.path.join(FRAMES_DIR, frame))

    results = ptb.results_csv_to_dict_list(results_logfile)

    xx = []
    yy = []
    zz = []
    best_so_far = []
    bsf = 1e-10
    for result in results:
        xx.append(float(result["x"]))
        yy.append(float(result["y"]))
        zval = -1 * float(result["error"])
        zz.append(zval)
        if zval > bsf:
            bsf = zval
        best_so_far.append(bsf)

    i_frame = 10000
    for j in range(len(best_so_far)):
        x = xx[:j]
        y = yy[:j]
        z = zz[:j]

        fig = plt.figure()

        ax_surf = fig.add_subplot(221, projection='3d')
        x_all_hi = np.linspace(0, np.pi, 100)
        y_all_hi = np.linspace(0, np.pi, 100)
        X, Y = np.meshgrid(x_all_hi, y_all_hi)
        Z = -1 * evaluate(x=X, y=Y)

        surf = ax_surf.plot_surface(
            X, Y, Z,
            cmap=cm.inferno,
            linewidth=0,
            antialiased=False)
        ax_surf.set_xlabel("x")
        ax_surf.set_ylabel("y")
        ax_surf.set_xlim(0, np.pi)
        ax_surf.set_ylim(0, np.pi)
        ax_surf.set_zlim(-.2, 1)

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

        ax_bsf = fig.add_subplot(223)
        ax_bsf.plot(
            np.arange(j) + 1,
            best_so_far[:j],
            color="blue",
        )
        ax_bsf.set_xlabel("Points evaluated")
        ax_bsf.set_ylabel("Best value so far")
        ax_bsf.set_xlim(0, len(best_so_far) + 1)
        ax_bsf.set_ylim(-.01, 1.01)

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

        # The effective frame rate will be 30 / n_repeats.
        n_repeats = 3
        for _ in range(n_repeats):
            figname = f"f_{i_frame}.png"
            fig.savefig(os.path.join("frames", figname), dpi=300)
            i_frame += 1

        plt.close()


optimizer = EvoPowell()
x_all = np.linspace(0, np.pi, 10)
y_all = np.linspace(0, np.pi, 10)
conditions = {
    "x": list(x_all),
    "y": list(y_all),
}
lowest_error, best_condition, results_logfile = optimizer.optimize(
    evaluate, conditions, verbose=False)

create_frames()

video_filename = "ponderosa_demo.mp4"
at.render_movie(
    filename=video_filename,
    frame_dirname="frames",
    output_dirname=".",
)
at.convert_to_gif(
    filename=video_filename,
    dirname=".",
)
