import numpy as np
from dynamics import *
from scipy.stats.distributions import chi2
from tqdm import tqdm


def stat_eps(error, cov) -> np.ndarray:
    """Inputs for different test types:
    - NEES: x-x_true, P
    - NIS: y-y_true, S
    """
    error = list(error)
    cov = list(cov)

    eps = []
    for i in range(len(error)):
        eps.append(error[i] @ np.linalg.inv(cov[i]) @ error[i])

    return np.array(eps)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from constants import (  # visible_stations_truth,; y_truth,
        R,
        dt,
        lt,
        nominal_trajectory,
        station_trajectories,
    )
    from filters import EKF
    from problem3 import plot_states

    dx0_true = np.zeros(4)
    P0_true = np.diag([10, 0.5, 10, 0.5])

    ts = np.arange(0, 14000, step=dt)
    x_nom = np.array([nominal_trajectory.state_at(t) for t in ts])

    N = 10  # sim runs
    eps_all = []
    for _ in tqdm(range(N)):
        x_pert = np.random.multivariate_normal(dx0_true, P0_true)
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

        ekf = EKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=nominal_trajectory.state_at(0) + dx0_true,
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=np.eye(2) * 1e2,
            dt=dt,
        )
        x_tot, P, S = ekf.solve()

        eps = stat_eps(x_tot - x_pert_nonlinear, P)
        eps_all.append(eps)

    eps_all = np.array(eps_all)
    eps_mean = np.mean(eps_all, axis=0)

    alpha = 0.05
    n = 4  # dimension

    r1 = chi2.ppf(alpha / 2, df=N * n) / N
    r2 = chi2.ppf(1 - alpha / 2, df=N * n) / N

    statistic = np.sum(np.logical_and(r1 < eps_mean, eps_mean < r2)) / len(eps_mean)
    print(statistic)

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    ax.axhline(r1, color="r", linestyle="--")
    ax.axhline(r2, color="r", linestyle="--")
    ax.set_title(f"NEES statistic: {statistic*100:.2f}% (want: {(1-alpha)*100:.2f} %)")

    ax.plot(ts, eps_mean, ".")
    ax.set_ylim([0, r2 * 2])

    bins = np.linspace(0, r2 * 5, 100)

    ax = axs[1]
    _, bins, _ = ax.hist(eps_mean.flatten(), bins=bins, density=True)

    ax.plot(bins, chi2.pdf(bins * N, df=N * n) * N)

    plt.show()
