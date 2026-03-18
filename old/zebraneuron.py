import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make the bounding box of all axes have the same size to achieve equal aspect ratio.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def draw_cell_pos_network(cell_pos, save_file_name):
    # Scaling factors
    scaling_factors = np.array([0.8, 0.8, 2])

    # Apply scaling to cell_pos
    scaled_cell_pos = cell_pos.copy()
    scaled_cell_pos[:, 0:3] *= scaling_factors

    fig = plt.figure(figsize=(2.75 * 2, 2.11 * 2))
    ax = fig.add_subplot(111, projection="3d")

    cc = np.random.rand(72, 3)

    for i in range(72):
        qq = np.where(scaled_cell_pos[:, 3] == i + 1)
        if len(qq[0]) > 1:
            ax.scatter(
                scaled_cell_pos[qq, 0],
                scaled_cell_pos[qq, 1],
                scaled_cell_pos[qq, 2],
                c=[cc[i]],
                s=0.5,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    set_axes_equal(ax)  # Ensure equal scale

    ax.set_frame_on(False)  # Disable the frame
    ax.grid(False)  # Disable the grid

    plt.savefig(save_file_name, format="svg", transparent=True, dpi=600)
    plt.show()


# Enable interactive mode
plt.ion()

cell_pos = np.load("./zebrafish_cell_xyz.npy")

qq = np.where(cell_pos[:, 3] == 1)
draw_cell_pos_network(cell_pos, "./fig10.svg")

# Keep the plot open
plt.ioff()
plt.show()
