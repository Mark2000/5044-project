from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from constants import (  # visible_stations_truth,; y_truth,
    P0_true,
    R,
    dt,
    dx0_bar_true,
    nominal_trajectory,
    station_trajectories,
    ts,
)
from dynamics import (
    EllipticalTrajectory,
    LinearizedSystem,
    is_station_visible,
    measurements,
)
from filters import EKF, LKF, UKF
from joblib import Parallel, delayed
from scipy.stats import entropy, gaussian_kde
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


def run_test(x_pert, Q, filter="ekf", ts=ts):
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

    if filter == "ekf":
        ekf = EKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=nominal_trajectory.state_at(0) + dx0_bar_true,
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=Q,
            dt=dt,
        )
        x_tot, P, S, e_NIS = ekf.solve()

    if filter == "ukf":
        ukf = UKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            x_hat_zero=nominal_trajectory.state_at(0) + dx0_bar_true,
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=Q,
            dt=dt,
        )
        x_tot, P, S, e_NIS = ukf.solve()

    if filter == "lkf":
        lkf = LKF(
            lt=LinearizedSystem(nominal_trajectory, station_trajectories),
            delta_x_hat_zero=dx0_bar_true,
            P_zero=P0_true,
            y_truth=all_measurements,
            visible_stations=visible_stations,
            R=R,
            Q=Q,
            dt=dt,
        )
        x_hat, P, S, e_NIS = lkf.solve()
        x_tot = x_nom + x_hat

    eps_NEES = stat_eps(x_tot - x_pert_nonlinear, P)
    eps_NIS = stat_eps(e_NIS, S)

    return eps_NEES, eps_NIS


def run_tests(Q, filter="ekf", dx0_bar_true=dx0_bar_true, P0_true=P0_true, N=25, ts=ts):
    x_perts = [np.random.multivariate_normal(dx0_bar_true, P0_true) for _ in range(N)]
    results = Parallel(n_jobs=-1)(
        delayed(run_test)(x_pert, Q, filter, ts) for x_pert in x_perts
    )
    eps_NEES = np.mean(np.array([eps[0] for eps in results]), axis=0)
    eps_NIS = np.mean(np.array([eps[1] for eps in results]), axis=0)
    return eps_NEES, eps_NIS


def evaluate_test(eps, N=1, n=4):
    max_eps = chi2.ppf(1 - 1e-9, df=N * n) / N * 5

    bins = np.linspace(0, max_eps, 2000)
    counts, _ = np.histogram(eps, bins=bins, density=True)
    # counts[-1] += sum(eps > max_eps)
    counts[-sum(eps > max_eps)] += 1
    # counts = np.concatenate([counts, np.ones(sum(eps > max_eps))])
    sample_points = bins[:-1] + np.diff(bins) / 2
    theoretical_pdf = chi2.pdf(sample_points * N, df=N * n) * N

    # if sum(eps > max_eps) / len(eps) > 0.2:
    #     print("WARNING: KL Divergence may be inaccurate due to truncation")

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(sample_points, counts)
    # ax.plot(sample_points, theoretical_pdf)

    kl_divergence = entropy(counts, theoretical_pdf)
    # ax.set_title(f"KL Divergence: {kl_divergence}")
    return kl_divergence


def plot_test(ts, eps, alpha=0.05, N=1, n=4, test_name=None):
    r1 = chi2.ppf(alpha / 2, df=N * n) / N
    r2 = chi2.ppf(1 - alpha / 2, df=N * n) / N

    statistic = np.sum(np.logical_and(r1 < eps, eps < r2)) / len(eps)

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    ax.axhline(r1, color="r", linestyle="--")
    ax.axhline(r2, color="r", linestyle="--")
    ax.set_title(
        f"{test_name} statistic: {statistic*100:.2f}% (want: {(1-alpha)*100:.2f} %)"
    )

    ax.plot(ts, eps, ".")
    ax.set_ylim([0, r2 * 2])

    bins = np.linspace(0, r2 * 5, 100)

    ax = axs[1]
    _, bins, _ = ax.hist(eps.flatten(), bins=bins, density=True)
    sample_points = bins[:-1] + np.diff(bins) / 2
    theoretical_pdf = chi2.pdf(sample_points * N, df=N * n) * N
    ax.plot(sample_points, theoretical_pdf)
    ax.set_xlabel(f"KL: {evaluate_test(eps, N, n)}")

    return ax


if __name__ == "__main__":
    from constants import (
        P0_true,
        Q_truth,
        Q_tuned_ekf,
        Q_tuned_lkf,
        Q_tuned_ukf,
        dx0_bar_true,
        ts,
    )

    N = 48
    qs = np.logspace(-12, 8, 11)
    print(qs)
    filter = "lkf"
    Q_best = Q_tuned_ekf

    kl_NEESs = []
    kl_NISs = []
    for q in tqdm(qs):
        Q = np.eye(2) * q
        eps_NEES, eps_NIS = run_tests(Q=Q, filter=filter, N=N, ts=ts)
        kl_NEESs.append(evaluate_test(eps_NEES, N=N))
        kl_NISs.append(evaluate_test(eps_NIS, N=N))

    data = {
        "filter": filter,
        "N": N,
        "ts": ts,
        "qs": qs,
        "kl_NEESs": kl_NEESs,
        "kl_NISs": kl_NISs,
    }
    np.save(Path(__file__).resolve().parent / "dat" / f"stats_{filter}.npy", data)

    fig, ax = plt.subplots(1, 1)
    ax.plot(qs, kl_NEESs, label="NEES", marker="o")
    ax.plot(qs, kl_NISs, label="NIS", marker="o")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("$Q$")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("KL Divergence")

    eps_NEES, eps_NIS = run_tests(Q=Q_best, filter=filter, N=N, ts=ts)
    alpha = 0.05
    plot_test(ts, eps_NEES, test_name="NEES", n=4, N=N, alpha=alpha)
    plot_test(ts, eps_NIS, test_name="NIS", n=4, N=N, alpha=alpha)

    plt.show()
