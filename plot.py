from typing import *

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from constants import *

plt.rcParams.update({"font.size": 9})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["lines.linewidth"] = 1.0
plt.rc("legend", fontsize="small")
mpl.rcParams.update({"axes.grid": True, "grid.linewidth": 0.2})

def plot_orbits(xs: Sequence[np.ndarray], ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.add_patch(patches.Circle((0, 0), R_E, color="b"))
    ax.axis("equal")
    ax.set_xlabel("$x_1$")
    ax.set_xlabel("$x_3$")
    return ax, [ax.plot(x[:, 0], x[:, 2])[0] for x in xs]


def plot_states(
    xs: Sequence[np.ndarray],
    t: Optional[np.ndarray] = None,
    ylabels: Sequence[str] = [],
    xlabel: str = "",
    legend_labels: Sequence[str] = [],
    kwargs: Sequence[dict[str, Any]] = [],
    zoom_region_and_inset_bounds = None,
    axs=None,
):
    if axs is None:
        fig, axs = plt.subplots(xs[0].shape[1], sharex=True)

    if t is None:
        t = np.arange(xs[0].shape[1])

    lines = np.empty([len(axs), len(xs)], dtype=object)
    for i, ax in enumerate(axs):
        inset_ax = None
        if zoom_region_and_inset_bounds is not None:
            zoom_region, inset_bounds = zoom_region_and_inset_bounds[i]
            inset_ax = add_zoom_inset(ax, zoom_region, inset_bounds)

        for j, x in enumerate(xs):
            lines[i, j] = axs[i].plot(t, x[:, i], **(kwargs[j] if kwargs else {}))[0]
            if inset_ax is not None:
                inset_ax.plot(t, x[:, i], **(kwargs[j] if kwargs else {}))[0]

        if ylabels:
            ax.set_ylabel(ylabels[i])

    if xlabel:
        axs[-1].set_xlabel(xlabel)

    if legend_labels:
        axs[0].legend(
            lines[0, :],
            legend_labels,
            bbox_to_anchor=(0, 1.02, 1, 0.102),
            loc="lower center",
            ncols=6,
        )

    return axs, lines


def plot_measurements(
    y: np.ndarray,
    t: Optional[np.ndarray] = None,
    ylabels: Sequence[str] = [],
    xlabel: str = "",
):
    n_stations = y.shape[1] // 3
    ys = [y[:, i * 3 : (i + 1) * 3] for i in range(n_stations)]
    cmap = plt.get_cmap("tab20")
    kwargs = [
        {"markersize": 4, "marker": "x", "color": cmap(i), "linestyle": ""} for i in range(n_stations)
    ]
    return plot_states(
        ys,
        t,
        ylabels,
        xlabel=xlabel,
        legend_labels=map(str, range(1, n_stations + 1)),
        kwargs=kwargs,
    )

def add_zoom_inset(ax, zoom_region, inset_bounds, **kwargs):
    """
    Adds a zoomed-in inset to a given Matplotlib axis.
    
    Args:
        ax (matplotlib.axes.Axes): The main axis to which the inset will be added.
        zoom_region (tuple): A tuple (x_min, x_max, y_min, y_max) defining the zoomed-in region.
        inset_bounds (tuple): A tuple (x0, y0, width, height) in figure-relative coordinates [0, 1].
        **kwargs: Additional keyword arguments for inset customization (e.g., frame_on).
    
    Returns:
        matplotlib.axes.Axes: The inset axis object.
    """
    # Create an inset axis
    inset_ax = ax.inset_axes(inset_bounds, **kwargs)
    
    # Set the zoomed region for the inset
    x_min, x_max, y_min, y_max = zoom_region
    inset_ax.set_xlim(x_min, x_max)
    inset_ax.set_ylim(y_min, y_max)
    
    # Copy the style of the main plot (optional)
    inset_ax.tick_params(labelsize=8)
    
    # Indicate the zoomed region on the main plot
    ax.indicate_inset_zoom(inset_ax, edgecolor="k", linewidth=1)
    
    return inset_ax
