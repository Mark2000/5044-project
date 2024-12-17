import os

import matplotlib.pyplot as plt
from constants import *
from dynamics import *
from filters import *
from plot import *
from stats import *

FIGS_FOLDER = "figs"
os.makedirs(FIGS_FOLDER, exist_ok=True)

state_labels = ["$x$ [km]", r"$\dot{x}$ [km/s]", "y [km]", r"$\dot{y}$ [km/s]"]
delta_state_labels = [rf"$\Delta$ {i}" for i in state_labels]
measurement_labels = [r"$\rho$ [km]", r"$\dot{\rho}$ [km/s]", r"$\Phi$ [rad]"]


def part_1_ex_3():
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

    axs, _ = plot_states(
        [x_pert_nonlinear - x_nom, x_hat],
        ts,
        ylabels=delta_state_labels,
        xlabel="Time [s]",
        legend_labels=["Non-linear", "Linear"],
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:orange"},
        ],
    )
    fig = axs[0].figure
    fig.suptitle(
        "State Perturbation vs. Time, Nonlinear and Linearized Dynamics Simulation"
    )
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/part_1_perturbations.pdf")

    axs, _ = plot_states(
        [x_pert_nonlinear - x_nom - x_hat],
        ts,
        ylabels=delta_state_labels,
        xlabel="Time [s]",
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:orange"},
        ],
    )
    fig = axs[0].figure
    fig.suptitle(
        "Linearization Error vs. Time, Nonlinear - Linearized Dynamics Simulation"
    )
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/part_1_linearization_error.pdf")

    masked_y_pert_nonlinear = mask_non_visible(
        ts, x_pert_nonlinear, y_pert_nonlinear, station_trajectories
    )

    axs, _ = plot_measurements(
        masked_y_pert_nonlinear,
        ts,
        ylabels=measurement_labels,
        xlabel="Time [s]",
    )
    fig = axs[0].figure
    fig.suptitle("Measurements vs. Time, Full Nonlinear Dynamics Simulation")
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/part_1_nonlinear_measurements.pdf")

    masked_y_pert_linear = mask_non_visible(
        ts, x_pert_nonlinear, y_pert_linear, station_trajectories
    )

    axs, _ = plot_measurements(
        masked_y_pert_linear,
        ts,
        ylabels=measurement_labels,
        xlabel="Time [s]",
    )
    fig = axs[0].figure
    fig.suptitle("Measurements vs. Time, Linearized Dynamics Simulation")
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/part_1_linear_measurements.pdf")

    # plt.show()


def typical_simulation(filter):
    ts = np.arange(0, 1400 * dt, step=dt)
    x_pert = np.array([0, 0.075, 0, -0.021])

    x_nom = np.array([nominal_trajectory.state_at(t) for t in ts])
    x_pert_nonlinear = EllipticalTrajectory(x_nom[0, :] + x_pert).propagate(
        ts, use_process_noise=True
    )

    visible_stations = [
        [
            i
            for i, st in enumerate(station_trajectories)
            if is_station_visible(x, st.state_at(t))
        ]
        for t, x in zip(ts, x_pert_nonlinear)
    ]

    all_measurements = [
        measurements(x, [station_trajectories[i] for i in vs], t, R=R)
        for t, x, vs in zip(ts, x_pert_nonlinear, visible_stations)
    ]

    if filter == "lkf":
        lkf = LKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            delta_x_hat_zero=dx0_bar_true,
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=Q_tuned_lkf,
            dt=dt,
        )
        x_hat, P, _, _ = lkf.solve()
        x_tot = x_hat + x_nom

    elif filter == "ekf":
        ekf = EKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=dx0_bar_true + nominal_trajectory.state_at(0),
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=Q_tuned_ekf,
            dt=dt,
        )
        x_tot, P, _, _ = ekf.solve()
        x_hat = x_tot - x_nom

    elif filter == "ukf":
        ukf = UKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=dx0_bar_true + nominal_trajectory.state_at(0),
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=Q_tuned_ukf,
            dt=dt,
        )
        x_tot, P, _, _ = ukf.solve()
        x_hat = x_tot - x_nom

    sigma = np.sqrt(np.array([np.diag(p) for p in P]))

    axs, _ = plot_states(
        [x_pert_nonlinear],
        ts,
        ylabels=state_labels,
        xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
    )
    fig = axs[0].figure
    fig.suptitle(
        f"{filter.upper()}, States vs. Time, Full Nonlinear Dynamics Simulation"
    )
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/{filter}_nonlinear_states.pdf")

    axs, _ = plot_states(
        [x_pert_nonlinear - x_nom],
        ts,
        ylabels=delta_state_labels,
        xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
    )
    fig = axs[0].figure
    fig.suptitle(
        f"{filter.upper()}, (Perturbed - Nominal) vs. Time, Full Nonlinear Dynamics Simulation"
    )
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/{filter}_nonlinear_delta_states.pdf")

    y_pert_nonlinear = np.array(
        [measurements(x, station_trajectories, t) for t, x in zip(ts, x_pert_nonlinear)]
    )
    masked_y_pert_nonlinear = mask_non_visible(
        ts, x_pert_nonlinear, y_pert_nonlinear, station_trajectories
    )

    axs, _ = plot_measurements(
        masked_y_pert_nonlinear,
        ts,
        ylabels=measurement_labels,
        xlabel="Time [s]",
    )
    fig = axs[0].figure
    fig.suptitle(
        f"{filter.upper()}, Measurements vs. Time, Full Nonlinear Dynamics Simulation"
    )
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/{filter}_nonlinear_measurements.pdf")

    zoom_region_and_inset_bounds = None
    if filter != "lkf":
        i_min = len(ts) // 3
        i_max = len(ts) // 2
        zoom_region_and_inset_bounds = [
            (
                (
                    ts[i_min],
                    ts[i_max],
                    -2.5 * sigma[i_min:i_max, j].max(),
                    2.5 * sigma[i_min:i_max, j].max(),
                ),
                (0.6, 0.6, 0.3, 0.3),
            )
            for j in range(4)
        ]

    axs, _ = plot_states(
        [x_pert_nonlinear - x_tot]
        + (
            [
                2 * sigma,
                -2 * sigma,
            ]
            if filter != "lkf"
            else []
        ),
        ts,
        ylabels=delta_state_labels,
        xlabel="Time [s]",
        # legend_labels=xs_labels,
        zoom_region_and_inset_bounds=zoom_region_and_inset_bounds,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
    )

    fig = axs[0].figure
    fig.suptitle(f"{filter.upper()}, Estate Estimation Error vs. Time")
    fig.tight_layout()
    fig.savefig(f"{FIGS_FOLDER}/{filter}_estimation_error.pdf")

    # plt.show()


def filter_on_truth_data(filter, states_axs, twosigma_axs):
    ts = np.arange(0, len(y_truth) * dt, step=dt)
    x_nom = np.array([nominal_trajectory.state_at(t) for t in ts])

    if filter == "lkf":
        lkf = LKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            delta_x_hat_zero=dx0_bar_true,
            P_zero=P0_true,
            y_truth=y_truth,
            visible_stations=visible_stations_truth,
            R=R,
            Q=Q_tuned_lkf,
            dt=dt,
        )
        x_hat, P, S, e_NIS = lkf.solve()
        x_tot = x_hat + x_nom

    elif filter == "ekf":
        ekf = EKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=dx0_bar_true + nominal_trajectory.state_at(0),
            P_zero=P0_true,
            y_truth=y_truth,
            visible_stations=visible_stations_truth,
            R=R,
            Q=Q_tuned_ekf,
            dt=dt,
        )
        x_tot, P, S, e_NIS = ekf.solve()
        x_hat = x_tot - x_nom

    elif filter == "ukf":
        ukf = UKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=dx0_bar_true + nominal_trajectory.state_at(0),
            P_zero=P0_true,
            y_truth=y_truth,
            visible_stations=visible_stations_truth,
            R=R,
            Q=Q_tuned_ukf,
            dt=dt,
        )
        x_tot, P, S, e_NIS = ukf.solve()
        x_hat = x_tot - x_nom

    sigma = np.sqrt(np.array([np.diag(p) for p in P]))

    plot_states(
        [x_tot],
        ts,
        ylabels=state_labels,
        xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
        axs=states_axs,
    )

    zoom_region_and_inset_bounds = None
    if filter != "lkf":
        i_min = len(ts) // 3
        i_max = len(ts) // 2
        zoom_region_and_inset_bounds = [
            (
                (ts[i_min], ts[i_max], 0, 2.5 * sigma[i_min:i_max, j].max()),
                (0.4, 0.4, 0.5, 0.5),
            )
            for j in range(4)
        ]

    plot_states(
        [2 * sigma],
        ts,
        ylabels=state_labels,
        xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
        zoom_region_and_inset_bounds=zoom_region_and_inset_bounds,
        axs=twosigma_axs,
    )

    eps_NIS = stat_eps(e_NIS[1:], S[1:])
    fig, ax = plot_test(ts[1:], eps_NIS, test_name="NIS", n=3, N=1, alpha=0.05)
    # plt.show()
    fig.savefig(f"{FIGS_FOLDER}/{filter}_NIS_on_truth.pdf")


def compare_filters_on_truth(filter_a, filter_b):
    states_fig, states_axs = plt.subplots(4, 2)
    twosigma_fig, twosigma_axs = plt.subplots(4, 2)

    for i, filter in enumerate([filter_a, filter_b]):
        filter_on_truth_data(filter, states_axs[:, i], twosigma_axs[:, i])

    states_fig.suptitle(f"{filter_a.upper()} vs. {filter_b.upper()}, Estimated States")
    twosigma_fig.suptitle(
        rf"{filter_a.upper()} vs. {filter_b.upper()}, $2\sigma$ State Uncertainty"
    )

    states_fig.tight_layout()
    twosigma_fig.tight_layout()

    states_fig.savefig(f"{FIGS_FOLDER}/{filter_a}_vs_{filter_b}_states.pdf")
    twosigma_fig.savefig(f"{FIGS_FOLDER}/{filter_a}_vs_{filter_b}_twosigma.pdf")


def optimization_plot(path):
    stats = np.load(path, allow_pickle=True).item()
    qs = stats["qs"]
    kl_NEESs = stats["kl_NEESs"]
    stat_NEESs = stats["stat_NEESs"]
    kl_NISs = stats["kl_NISs"]
    stat_NISs = stats["stat_NISs"]
    filter = stats["filter"]
    alpha = stats["alpha"]

    fig, axs = plt.subplots(2, 1, sharex=True)
    ax = axs[0]
    ax.plot(qs, kl_NEESs, label="NEES", marker="o")
    ax.plot(qs, kl_NISs, label="NIS", marker="o")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_ylabel("KL Divergence")

    ax = axs[1]
    ax.plot(qs, 100 * np.array(stat_NEESs), label="NEES", marker="o")
    ax.plot(qs, 100 * np.array(stat_NISs), label="NIS", marker="o")
    ax.axhline(100 * (1 - alpha), color="k", linestyle="--")
    ax.set_ylabel("Consistency [%]")
    ax.set_ylim(bottom=0, top=100)

    ax.set_xlabel("$q$")
    ax.set_xscale("log")

    fig.savefig(f"{FIGS_FOLDER}/optimization_plot_{filter}.pdf")


def consistency_plot(filter, N):
    if filter == "lkf":
        Q_best = Q_tuned_lkf
    elif filter == "ekf":
        Q_best = Q_tuned_ekf
    elif filter == "ukf":
        Q_best = Q_tuned_ukf
    eps_NEES, eps_NIS = run_tests(Q=Q_best, filter=filter, N=N, ts=ts)
    alpha = 0.05
    fig, ax = plot_test(ts, eps_NEES, test_name="NEES", n=4, N=N, alpha=alpha)
    fig.savefig(f"{FIGS_FOLDER}/{filter}_NEES.pdf")
    fig, ax = plot_test(ts, eps_NIS, test_name="NIS", n=4, N=N, alpha=alpha)
    fig.savefig(f"{FIGS_FOLDER}/{filter}_NIS.pdf")


if __name__ == "__main__":
    part_1_ex_3()
    typical_simulation("lkf")
    typical_simulation("ekf")
    typical_simulation("ukf")
    compare_filters_on_truth("lkf", "ekf")
    compare_filters_on_truth("ekf", "ukf")
    optimization_plot(Path(__file__).resolve().parent / "dat" / f"stats_lkf.npy")
    optimization_plot(Path(__file__).resolve().parent / "dat" / f"stats_ekf.npy")
    # optimization_plot(Path(__file__).resolve().parent / "dat" / f"stats_ukf.npy")
    N = 48  # More for final
    consistency_plot("lkf", N)
    consistency_plot("ekf", N)
    # consistency_plot("ukf", N)
