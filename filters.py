from dataclasses import dataclass
from typing import Sequence

import numpy as np
from dynamics import (
    EllipticalTrajectory,
    LinearizedSystem,
    is_station_visible,
    measurements,
)


@dataclass
class LKF:
    lt: LinearizedSystem
    delta_x_hat_zero: np.ndarray
    P_zero: np.ndarray
    y_truth: list[np.ndarray]
    visible_stations: Sequence[Sequence[int]]
    R: np.ndarray
    Q: np.ndarray
    dt: float

    def __post_init__(self, *a, **kw):
        assert len(self.y_truth) == len(self.visible_stations)

    def solve(self):
        delta_x_hat = np.zeros([len(self.y_truth), 4])
        P = np.zeros([len(self.y_truth), 4, 4])
        S = []
        e_NIS = []

        delta_x_hat[0, :] = self.delta_x_hat_zero
        P[0, :, :] = self.P_zero

        H_0 = self.Hk(0)
        R_0 = self.Rk(0)
        S.append(H_0 @ self.P_zero @ H_0.T + R_0)
        e_NIS.append(np.zeros_like(self.y_truth[0]))

        for k in range(len(self.y_truth) - 1):
            delta_x_hat[k + 1, :], P[k + 1, :, :], S_k_plus_1, delta_y_k_plus_1 = (
                self.step(k, delta_x_hat[k, :], P[k, :, :])
            )
            S.append(S_k_plus_1)
            e_NIS.append(delta_y_k_plus_1)

        return delta_x_hat, P, S, e_NIS

    def Hk(self, k: int):
        ind = np.zeros(len(self.lt.stations) * 3, dtype=bool)
        for st in self.visible_stations[k]:
            ind[st * 3 : (st + 1) * 3] = True
        return self.lt.H_tilde(k * self.dt)[ind, :]

    def Rk(self, k: int):
        return np.kron(np.eye(len(self.visible_stations[k])), self.R)

    def step(self, k: int, delta_x_hat_k: np.ndarray, P_k: np.ndarray):
        y_k_plus_1 = self.y_truth[k + 1]
        t_k = k * self.dt
        t_k_plus_1 = (k + 1) * self.dt

        F_k = self.lt.F_tilde(t_k, self.dt)
        Omega_k = self.lt.Omega_tilde(t_k, self.dt)

        H_k_plus_1 = self.Hk(k + 1)

        y_k_plus_1_star = measurements(
            self.lt.nominal.state_at(t_k_plus_1),
            [self.lt.stations[i] for i in self.visible_stations[k + 1]],
            t_k_plus_1,
        )

        delta_x_k_plus_1_pre = F_k @ delta_x_hat_k
        P_x_k_plus_1_pre = F_k @ P_k @ F_k.T + Omega_k @ self.Q @ Omega_k.T

        R_k_plus_1 = self.Rk(k + 1)
        S_k_plus_1 = H_k_plus_1 @ P_x_k_plus_1_pre @ H_k_plus_1.T + R_k_plus_1
        K_k_plus_1 = P_x_k_plus_1_pre @ H_k_plus_1.T @ np.linalg.inv(S_k_plus_1)
        delta_y_k_plus_1 = y_k_plus_1 - y_k_plus_1_star
        delta_y_k_plus_1[2::3] = (delta_y_k_plus_1[2::3] + np.pi) % (2 * np.pi) - np.pi

        delta_x_k_plus_1_post = delta_x_k_plus_1_pre + K_k_plus_1 @ (
            delta_y_k_plus_1 - H_k_plus_1 @ delta_x_k_plus_1_pre
        )
        P_x_k_plus_1_post = (np.eye(4) - K_k_plus_1 @ H_k_plus_1) @ P_x_k_plus_1_pre
        return delta_x_k_plus_1_post, P_x_k_plus_1_post, S_k_plus_1, delta_y_k_plus_1


@dataclass
class EKF:
    lt: LinearizedSystem
    x_hat_zero: np.ndarray
    P_zero: np.ndarray
    y_truth: list[np.ndarray]
    visible_stations: Sequence[Sequence[int]]
    R: np.ndarray
    Q: np.ndarray
    dt: float

    def solve(self):
        x_hat = np.zeros([len(self.y_truth), 4])
        P = np.zeros([len(self.y_truth), 4, 4])
        S = []
        e_NIS = []

        x_hat[0, :] = self.x_hat_zero
        P[0, :, :] = self.P_zero

        H_0 = self.Hk(0, x_hat[0, :])
        R_0 = self.Rk(0)
        S.append(H_0 @ self.P_zero @ H_0.T + R_0)
        e_NIS.append(self.y_truth[0] - H_0 @ x_hat[0, :])

        for k in range(len(self.y_truth) - 1):
            x_hat[k + 1, :], P[k + 1, :, :], S_k_plus_1, e_k_plus_1 = self.step(
                k, x_hat[k, :], P[k, :, :]
            )
            S.append(S_k_plus_1)
            e_NIS.append(e_k_plus_1)

        return x_hat, P, S, e_NIS

    def Hk(self, k: int, x_k: np.ndarray):
        ind = np.zeros(len(self.lt.stations) * 3, dtype=bool)
        for st in self.visible_stations[k]:
            ind[st * 3 : (st + 1) * 3] = True
        return self.lt.H_tilde_at_state(k * self.dt, x_k)[ind, :]

    def Rk(self, k: int):
        return np.kron(np.eye(len(self.visible_stations[k])), self.R)

    def step(self, k: int, x_hat_k: np.ndarray, P_k: np.ndarray):
        y_k_plus_1 = self.y_truth[k + 1]
        t_k = k * self.dt
        t_k_plus_1 = (k + 1) * self.dt

        x_k_plus_1_pre = EllipticalTrajectory(x_hat_k).propagate([0, self.dt])[1, :]

        F_k = self.lt.F_tilde_at_state(x_hat_k, self.dt)
        Omega_k = self.lt.Omega_tilde(t_k, self.dt)

        P_x_k_plus_1_pre = F_k @ P_k @ F_k.T + Omega_k @ self.Q @ Omega_k.T

        H_k_plus_1 = self.Hk(k + 1, x_k_plus_1_pre)

        y_k_plus_1_pre = measurements(
            x_k_plus_1_pre,
            [self.lt.stations[i] for i in self.visible_stations[k + 1]],
            t_k_plus_1,
        )

        e_k_plus_1 = y_k_plus_1 - y_k_plus_1_pre
        e_k_plus_1[2::3] = (e_k_plus_1[2::3] + np.pi) % (2 * np.pi) - np.pi

        R_k_plus_1 = self.Rk(k + 1)
        S_k_plus_1 = H_k_plus_1 @ P_x_k_plus_1_pre @ H_k_plus_1.T + R_k_plus_1
        K_k_plus_1 = P_x_k_plus_1_pre @ H_k_plus_1.T @ np.linalg.inv(S_k_plus_1)

        x_k_plus_1_post = x_k_plus_1_pre + K_k_plus_1 @ e_k_plus_1
        P_x_k_plus_1_post = (np.eye(4) - K_k_plus_1 @ H_k_plus_1) @ P_x_k_plus_1_pre

        return x_k_plus_1_post, P_x_k_plus_1_post, S_k_plus_1, e_k_plus_1


@dataclass
class UKF:
    lt: LinearizedSystem
    x_hat_zero: np.ndarray
    P_zero: np.ndarray
    y_truth: list[np.ndarray]
    visible_stations: Sequence[Sequence[int]]
    R: np.ndarray
    Q: np.ndarray
    dt: float
    kappa: float = 0
    alpha: float = 1
    beta: float = 2

    def solve(self):
        x_hat = np.zeros([len(self.y_truth), 4])
        P = np.zeros([len(self.y_truth), 4, 4])
        S = []
        e_NIS = []

        x_hat[0, :] = self.x_hat_zero
        P[0, :, :] = self.P_zero

        S.append(np.eye(3))
        e_NIS.append(np.zeros(3))

        for k in range(len(self.y_truth) - 1):
            x_hat[k + 1, :], P[k + 1, :, :], S_k_plus_1, e_k_plus_1 = self.step(
                k, x_hat[k, :], P[k, :, :]
            )
            S.append(S_k_plus_1)
            e_NIS.append(e_k_plus_1)

        return x_hat, P, S, e_NIS

    def Rk(self, k: int):
        return np.kron(np.eye(len(self.visible_stations[k])), self.R)

    def step(self, k: int, x_hat_k: np.ndarray, P_k: np.ndarray):
        d = x_hat_k.size
        lambda_ = self.alpha**2 * (d + self.kappa) - d

        y_k_plus_1 = self.y_truth[k + 1]
        t_k = k * self.dt
        t_k_plus_1 = (k + 1) * self.dt

        Omega_k = self.lt.Omega_tilde(t_k, self.dt)
        R_k_plus_1 = self.Rk(k + 1)

        weights_c = np.zeros(2 * d + 1)
        weights_m = np.zeros(2 * d + 1)

        for i in range(2 * d + 1):
            if i == 0:
                weights_m[i] = lambda_ / (d + lambda_)
                weights_c[i] = weights_m[i] + 1 - self.alpha**2 + self.beta
            else:
                weights_m[i] = 1 / (2 * (d + lambda_))
                weights_c[i] = weights_m[i]

        def sigmaProp(x, P, fun):
            chol_P_k = np.linalg.cholesky(P)

            samples = []
            for i in range(2 * d + 1):
                if i == 0:
                    sample_pre = x
                else:
                    j = i
                    if j >= d + 1:
                        j -= d
                    S_k_j_T = chol_P_k[:, j - 1]

                    if i <= d:
                        sample_pre = x + np.sqrt(d + lambda_) * S_k_j_T
                    else:
                        sample_pre = x - np.sqrt(d + lambda_) * S_k_j_T

                samples.append(fun(sample_pre))

            return samples

        def getMean(samples):
            mean = np.zeros_like(samples[0])
            for i in range(2 * d + 1):
                mean += weights_m[i] * samples[i]
            return mean

        def getCov(ini, samples_a, mean_a, samples_b, mean_b):
            cov = np.copy(ini)
            for i in range(2 * d + 1):
                cov += weights_c[i] * np.outer(
                    samples_a[i] - mean_a, samples_b[i] - mean_b
                )

            return cov

        x_samples = sigmaProp(
            x_hat_k,
            P_k,
            lambda x: EllipticalTrajectory(x).propagate([0, self.dt])[1, :],
        )
        x_k_plus_1_pre = getMean(x_samples)
        P_x_k_plus_1_pre = getCov(
            Omega_k @ self.Q @ Omega_k.T,
            x_samples,
            x_k_plus_1_pre,
            x_samples,
            x_k_plus_1_pre,
        )

        y_samples = sigmaProp(
            x_k_plus_1_pre,
            P_x_k_plus_1_pre,
            lambda x: measurements(
                x,
                [self.lt.stations[i] for i in self.visible_stations[k + 1]],
                t_k_plus_1,
            ),
        )
        y_k_plus_1_pre = getMean(y_samples)
        P_y_k_plus_1 = getCov(
            R_k_plus_1, y_samples, y_k_plus_1_pre, y_samples, y_k_plus_1_pre
        )

        P_xy_k_plus_1 = getCov(
            np.zeros([d, R_k_plus_1.shape[0]]),
            x_samples,
            x_k_plus_1_pre,
            y_samples,
            y_k_plus_1_pre,
        )

        e_k_plus_1 = y_k_plus_1 - y_k_plus_1_pre
        e_k_plus_1[2::3] = (e_k_plus_1[2::3] + np.pi) % (2 * np.pi) - np.pi

        K_k_plus_1 = P_xy_k_plus_1 @ np.linalg.inv(P_y_k_plus_1)

        x_k_plus_1_post = x_k_plus_1_pre + K_k_plus_1 @ e_k_plus_1
        P_x_k_plus_1_post = P_x_k_plus_1_pre - K_k_plus_1 @ P_y_k_plus_1 @ K_k_plus_1.T

        return x_k_plus_1_post, P_x_k_plus_1_post, P_y_k_plus_1, e_k_plus_1


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    from constants import (  # visible_stations_truth,; y_truth,
        P0_true,
        Q_truth,
        Q_tuned_ekf,
        Q_tuned_lkf,
        R,
        dt,
        dx0_bar_true,
        nominal_trajectory,
        station_trajectories,
        ts,
    )
    from problem3 import plot_states

    x_nom = np.array([nominal_trajectory.state_at(t) for t in ts])
    x_pert = np.array([0, 0.075, 0, -0.021])
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

    filter = sys.argv[1] if len(sys.argv) > 1 else "lkf"

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
            Q=Q_tuned_ekf,
            dt=dt,
        )
        x_tot, P, _, _ = ukf.solve()
        x_hat = x_tot - x_nom

    sigma = np.sqrt(np.array([np.diag(p) for p in P]))

    plot_states(
        [x_hat, x_hat + 2 * sigma, x_hat - 2 * sigma, x_pert_nonlinear - x_nom],
        ts,
        # ylables=[f"$y^{{s=1}}_{i}$" for i in range(1, 4)],
        # xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
    )

    dx = x_pert_nonlinear - x_nom

    plot_states(
        [
            x_hat - dx,
            2 * sigma,
            -2 * sigma,
        ],
        ts,
        # ylables=[f"$y^{{s=1}}_{i}$" for i in range(1, 4)],
        # xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:purple"},
            {"linestyle": "--", "color": "tab:purple"},
            {"color": "tab:orange"},
        ],
    )

    plt.show()
