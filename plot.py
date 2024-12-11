from typing import *

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from constants import *


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
    ylables: Sequence[str] = [],
    xlabel: str = "",
    legend_labels: Sequence[str] = [],
    kwargs: Sequence[dict[str, Any]] = [],
    axs=None,
):
    if axs is None:
        fig, axs = plt.subplots(xs[0].shape[1], sharex=True)

    if t is None:
        t = np.arange(xs[0].shape[1])

    lines = np.empty([len(axs), len(xs)], dtype=object)
    for i, ax in enumerate(axs):
        for j, x in enumerate(xs):
            lines[i, j] = axs[i].plot(t, x[:, i], **(kwargs[j] if kwargs else {}))[0]

        if ylables:
            ax.set_ylabel(ylables[i])

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
    ylables: Sequence[str] = [],
    xlabel: str = "",
):
    n_stations = y.shape[1] // 3
    ys = [y[:, i * 3 : (i + 1) * 3] for i in range(n_stations)]
    cmap = plt.get_cmap("tab20")
    kwargs = [
        {"marker": "o", "color": cmap(i), "linestyle": ""} for i in range(n_stations)
    ]
    return plot_states(
        ys,
        t,
        ylables,
        xlabel=xlabel,
        legend_labels=map(str, range(1, n_stations + 1)),
        kwargs=kwargs,
    )
