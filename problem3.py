from typing import *

from constants import *
from dynamics import *
from plot import *

import numpy.testing as npt


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
