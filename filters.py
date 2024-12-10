from dataclasses import dataclass

import numpy as np
from dynamics import *


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

        delta_x_hat[0, :] = self.delta_x_hat_zero
        P[0, :, :] = self.P_zero

        for k in range(len(self.y_truth) - 1):
            delta_x_hat[k + 1, :], P[k + 1, :, :] = self.lkf_step(k, delta_x_hat[k, :], P[k, :, :])

        return delta_x_hat, P

    def lkf_step(self, k: int, delta_x_hat_k: np.ndarray, P_k: np.ndarray):
        y_k_plus_1 = self.y_truth[k + 1]
        t_k = k * self.dt
        visible_stations = self.visible_stations[k + 1]
        assert len(visible_stations) == y_k_plus_1.size // 3

        F_k = self.lt.F_tilde(t_k, self.dt)
        Omega_k = self.lt.Omega_tilde(t_k, self.dt)

        ind = np.zeros(len(self.lt.stations) * 3, dtype=bool)
        for st in visible_stations:
            ind[st * 3 : (st + 1) * 3] = True
        H_k = self.lt.H_tilde(t_k)[ind, :]

        y_k_plus_1_star = measurements(
            self.lt.nominal.state_at(t_k + self.dt),
            [self.lt.stations[i] for i in visible_stations],
            t_k + self.dt,
        )

        delta_x_k_plus_1_pre = F_k @ delta_x_hat_k
        P_x_k_plus_1_pre = F_k @ P_k @ F_k.T + Omega_k @ self.Q @ Omega_k.T

        R_block = np.kron(np.eye(len(visible_stations)), self.R)
        K = P_k @ H_k.T @ np.linalg.inv(H_k @ P_x_k_plus_1_pre @ H_k.T + R_block)
        delta_y_k_plus_1 = y_k_plus_1 - y_k_plus_1_star

        delta_x_k_plus_1_post = delta_x_k_plus_1_pre + K @ (
            delta_y_k_plus_1 - H_k @ delta_x_k_plus_1_pre
        )
        P_x_k_plus_1_post = (np.eye(4) - K @ H_k) @ P_x_k_plus_1_pre
        return delta_x_k_plus_1_post, P_x_k_plus_1_post


if __name__ == "__main__":
    from constants import R, dt, nominal_trajectory, station_trajectories

    ts = np.arange(0, 14000, step=dt)

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
        measurements(x, [station_trajectories[i] for i in vs], t)
        for t, x, vs in zip(ts, x_pert_nonlinear, visible_stations)
    ]

    lkf = LKF(
        lt=LinearizedSystem(nominal_trajectory, station_trajectories),
        delta_x_hat_zero=x_pert,
        P_zero=np.eye(4) * 1000,
        y_truth=all_measurements,
        visible_stations=visible_stations,
        R=R,
        Q=np.eye(2),
        dt=dt,
    )

    print(lkf.solve())
