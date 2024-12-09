from typing import *

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from dynamics import *

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
        fig, axs = plt.subplots(xs[0].shape[1])

    if t is None:
        t = np.arange(xs[0].shape[1])

    lines = np.empty([len(axs), len(xs)], dtype=object)
    for i, ax in enumerate(axs):
        for j, x in enumerate(xs):
            lines[i, j] = axs[i].plot(t, x[:, i], **(kwargs[j] if kwargs else {}))[0]

        if ylables:
            ax.set_ylabel(ylables[i])

        if ax is not axs[-1]:
            ax.set_xticklabels([])

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
    kwargs = [{"marker": "o", "color": cmap(i)} for i in range(n_stations)]
    return plot_states(
        ys,
        t,
        ylables,
        xlabel=xlabel,
        legend_labels=map(str, range(1, n_stations + 1)),
        kwargs=kwargs,
    )


def test_part_1():
    ts = np.arange(0, 1400 * dt, step=dt)

    x_nom = np.array([nominal_trajectory.state_at(t) for t in ts])
    y_nom = np.array(
        [measurements(x, station_trajectories, t) for t, x in zip(ts, x_nom)]
    )

    x_hat = np.zeros_like(x_nom)
    x_hat[0, :] = [0, 0.075, 0, -0.021]  # initial perturbation
    for i in range(x_hat.shape[0] - 1):
        F, G = lt.F_G(ts[i], dt)
        x_hat[i + 1, :] = F @ x_hat[i, :]

    y_hat = np.zeros_like(y_nom)
    for i in range(x_hat.shape[0]):
        y_hat[i, :] = lt.H(ts[i]) @ x_hat[i, :]

    x_pert_nonlinear = EllipticalTrajectory(x_nom[0, :] + x_hat[0, :]).propagate(ts)
    y_pert_nonlinear = np.array(
        [measurements(x, station_trajectories, t) for t, x in zip(ts, x_pert_nonlinear)]
    )

    x_pert_linear = x_nom + x_hat
    y_pert_linear = y_nom + y_hat

    nonlinear_control_points = {7250: [-3557, -6.481, 5651, -4.169]}
    linear_pert_control_points = {7410: [-82.61, 0.1607, -72.49, -0.1582]}
    linear_control_points = {9620: [1371, 7.685, -6462, 1.542]}

    for control_points, to_check, atol in [
        (nonlinear_control_points, x_pert_nonlinear, [1, 0.001, 1, 0.001]),
        (linear_pert_control_points, x_hat, [0.01, 0.001, 0.01, 0.001]),
        (linear_control_points, x_pert_linear, [1, 0.001, 1, 0.001]),
    ]:
        for t, truth in control_points.items():
            i = np.where(ts == t)[0][0]
            for j in range(4):
                npt.assert_allclose(to_check[i, j], truth[j], atol=atol[j], err_msg=j)


def part_1_ex_3():
    ts = np.arange(0, 14000, step=dt)

    x_nom = np.array([nominal_trajectory.state_at(t) for t in ts])
    y_nom = np.array(
        [measurements(x, station_trajectories, t) for t, x in zip(ts, x_nom)]
    )

    x_hat = np.zeros_like(x_nom)
    x_hat[0, :] = [0, 0.075, 0, -0.021]  # initial perturbation
    for i in range(x_hat.shape[0] - 1):
        F, G = lt.F_G(ts[i], dt)
        x_hat[i + 1, :] = F @ x_hat[i, :]

    y_hat = np.zeros_like(y_nom)
    for i in range(x_hat.shape[0]):
        y_hat[i, :] = lt.H(ts[i]) @ x_hat[i, :]

    x_pert_nonlinear = EllipticalTrajectory(x_nom[0, :] + x_hat[0, :]).propagate(ts)
    y_pert_nonlinear = np.array(
        [measurements(x, station_trajectories, t) for t, x in zip(ts, x_pert_nonlinear)]
    )

    x_pert_linear = x_nom + x_hat
    y_pert_linear = y_nom + y_hat

    xs = [x_nom, x_pert_nonlinear, x_pert_linear]
    xs_labels = ["Nominal", "Perturbed (Non-Lin)", "Perturbed (Lin)"]
    ys_1 = [y_nom[:, :3], y_pert_nonlinear[:, :3], y_pert_linear[:, :3]]

    # ax, lines = plot_orbits(xs)
    # ax.legend(lines, xs_labels )

    plot_states(
        [x_hat],
        ts,
        ylables=[f"$x_{i}$" for i in range(1, 5)],
        xlabel="Time [s]",
        legend_labels=[r"$\bar{x}$"],
    )

    plot_states(
        xs,
        ts,
        ylables=[f"$x_{i}$" for i in range(1, 5)],
        xlabel="Time [s]",
        legend_labels=xs_labels,
        kwargs=[{"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": ":"}],
    )

    plot_states(
        ys_1,
        ts,
        ylables=[f"$y^{{s=1}}_{i}$" for i in range(1, 4)],
        xlabel="Time [s]",
        legend_labels=xs_labels,
        kwargs=[{"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": ":"}],
    )

    axs, _ = plot_states(
        [(x_pert_nonlinear - x_pert_linear)],
        ts,
        ylables=[f"$x_{i}$" for i in range(1, 5)],
        xlabel="Time [s]",
    )
    axs[0].figure.suptitle(r"$(x_{nonlin} - x_{lin})$")

    axs, _ = plot_states(
        [(y_pert_nonlinear[:, :3] - y_pert_linear[:, :3])],
        ts,
        ylables=[f"$y^{{s=1}}_{i}$" for i in range(1, 4)],
        xlabel="Time [s]",
    )
    axs[0].figure.suptitle(r"$(x_{nonlin} - x_{lin})$")

    masked_y_pert_nonlinear = mask_non_visible(
        ts, x_pert_nonlinear, y_pert_nonlinear, station_trajectories
    )
    plot_measurements(
        masked_y_pert_nonlinear,
        ts,
        ylables=[f"$y^{{s=1}}_{i}$" for i in range(1, 4)],
        xlabel="Time [s]",
    )

    plt.show()


if __name__ == "__main__":
    # test_part_1()
    part_1_ex_3()
