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
        S = []

        delta_x_hat[0, :] = self.delta_x_hat_zero
        P[0, :, :] = self.P_zero
        
        H_0 = self.Hk(0)
        R_0 = self.Rk(0)
        S.append( H_0 @ self.P_zero @ H_0.T + R_0 )

        for k in range(len(self.y_truth) - 1):
            delta_x_hat[k + 1, :], P[k + 1, :, :], S_k_plus_1 = self.lkf_step(
                k, delta_x_hat[k, :], P[k, :, :]
            )
            S.append(S_k_plus_1)

        return delta_x_hat, P, S

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
        t_k_plus_1 = (k+1)*dt

        F_k = self.lt.F_tilde(t_k, self.dt)
        Omega_k = self.lt.Omega_tilde(t_k, self.dt)

        H_k_plus_1 = self.Hk(k+1)

        y_k_plus_1_star = measurements(
            self.lt.nominal.state_at(t_k_plus_1),
            [self.lt.stations[i] for i in self.visible_stations[k+1]],
            t_k_plus_1,
        )

        delta_x_k_plus_1_pre = F_k @ delta_x_hat_k
        P_x_k_plus_1_pre = F_k @ P_k @ F_k.T + Omega_k @ self.Q @ Omega_k.T

        R_k_plus_1 = self.Rk(k+1)
        S_k_plus_1 = H_k_plus_1 @ P_x_k_plus_1_pre @ H_k_plus_1.T + R_k_plus_1
        K_k_plus_1 = P_x_k_plus_1_pre @ H_k_plus_1.T @ np.linalg.inv(S_k_plus_1)
        delta_y_k_plus_1 = y_k_plus_1 - y_k_plus_1_star

        delta_x_k_plus_1_post = delta_x_k_plus_1_pre + K_k_plus_1 @ (
            delta_y_k_plus_1 - H_k_plus_1 @ delta_x_k_plus_1_pre
        )
        P_x_k_plus_1_post = (np.eye(4) - K_k_plus_1 @ H_k_plus_1) @ P_x_k_plus_1_pre
        return delta_x_k_plus_1_post, P_x_k_plus_1_post, S_k_plus_1

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

        x_hat[0, :] = self.x_hat_zero
        P[0, :, :] = self.P_zero
        
        H_0 = self.Hk(0)
        R_0 = self.Rk(0)
        S.append( H_0 @ self.P_zero @ H_0.T + R_0 )

        for k in range(len(self.y_truth) - 1):
            x_hat[k + 1, :], P[k + 1, :, :], S_k_plus_1 = self.step(k, x_hat[k, :], P[k, :, :])
            S.append(S_k_plus_1)

        return x_hat, P, S

    def Hk(self, k: int, x_k: np.ndarray):
        ind = np.zeros(len(self.lt.stations) * 3, dtype=bool)
        for st in visible_stations:
            ind[st * 3 : (st + 1) * 3] = True
        return self.lt.H_tilde_at_state(k * self.dt, x_k)[ind, :]

    def Rk(self, k: int):
        return np.kron(np.eye(len(self.visible_stations[k])), self.R)
    
    def step(self, k: int, x_hat_k: np.ndarray, P_k: np.ndarray):
        y_k_plus_1 = self.y_truth[k + 1]
        t_k = k * self.dt
        t_k_plus_1 = (k+1)*dt

        x_k_plus_1_pre = EllipticalTrajectory(x_hat_k).propagate([0, self.dt])[1,:]

        F_k = self.lt.F_tilde_at_state(x_hat_k, self.dt)
        Omega_k = self.lt.Omega_tilde(t_k, self.dt)

        P_x_k_plus_1_pre = F_k @ P_k @ F_k.T + Omega_k @ self.Q @ Omega_k.T
        
        H_k_plus_1 = self.Hk(k+1, x_k_plus_1_pre)
        
        y_k_plus_1_pre = measurements(
            x_k_plus_1_pre,
            [self.lt.stations[i] for i in visible_stations[k+1]],
            t_k_plus_1,
        )

        e_k_plus_1 = y_k_plus_1 - y_k_plus_1_pre

        R_k_plus_1 = self.Rk(k+1)
        S_k_plus_1 = H_k_plus_1 @ P_x_k_plus_1_pre @ H_k_plus_1.T + R_k_plus_1
        K_k_plus_1 = P_x_k_plus_1_pre @ H_k_plus_1.T @ np.linalg.inv(S_k_plus_1)
        
        x_k_plus_1_post = x_k_plus_1_pre + K_k_plus_1 @ e_k_plus_1
        P_x_k_plus_1_post = (np.eye(4) - K_k_plus_1 @ H_k_plus_1) @ P_x_k_plus_1_pre
        
        return x_k_plus_1_post, P_x_k_plus_1_post, S_k_plus_1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from constants import R, dt, nominal_trajectory, station_trajectories
    from problem3 import plot_states

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
        Q=np.eye(2) * 1e1,
        dt=dt,
    )

    x_hat, P = lkf.solve()

    sigma = np.sqrt(np.array([np.diag(p) for p in P]))

    plot_states(
        [x_hat, x_hat + 2 * sigma, x_hat - 2 * sigma, x_pert_nonlinear - x_nom],
        ts,
        # ylables=[f"$y^{{s=1}}_{i}$" for i in range(1, 4)],
        # xlabel="Time [s]",
        # legend_labels=xs_labels,
        kwargs=[
            {"linestyle": "-", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:blue"},
            {"linestyle": "--", "color": "tab:blue"},
            {"color": "tab:orange"},
        ],
    )

    plt.show()
