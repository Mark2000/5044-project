from dataclasses import dataclass

from dynamics import *

import numpy as np

@dataclass
class LKF:
    lt: LinearizedSystem
    x_hat_zero: np.ndarray
    P_zero: np.ndarray
    y_truth: np.ndarray
    visible_stations: Sequence[Sequence[int]]
    dt: float

    def __post_init__(self, *a, **kw):
        assert self.y_truth.shape[0] == len(self.visible_stations)

    def solve(self):
        x_hat = np.zeros([self.y_truth.shape[0], 4])
        P = np.zeros([self.y_truth.shape[0], 4, 4])
    
        x_hat[0,:] = self.x_hat_zero
        P[0,:,:] = self.P_zero

        for k in range(self.y_truth.shape[0]):
            x_hat[k+1,:], P[k+1,:,:] = self.lkf_step(k, x_hat[k,:], P[k,:,:])

        return x_hat, P

    def lkf_step(self, k: int, x_hat_k: np.ndarray, P_k: np.ndarray):
        y_k_plus_1 = self.y_truth[k+1,:]
        t_k = k*self.dt
        visible_stations = self.visible_stations[k]
        assert len(visible_stations) == y_k_plus_1.size // 3
        
        F_k = self.lt.F_tilde(t_k, self.dt)

        ind = np.zeros(len(self.lt.stations)*3, dtype=bool)
        for st in visible_stations:
            ind[st*3:(st+1)*3] = True
        H_k = self.lt.H_tilde(t_k)[ind,:]

        y_k_plus_1_star = measurements(
            self.lt.nominal.state_at(t_k + self.dt),
            [self.lt.stations[i].state_at(t_k + self.dt) for i in visible_stations],
            t_k
        )

        delta_x_k_plus_1_pre = F_k @ x_hat_k
        P_x_k_plus_1_pre = F_k @ P_k @ F_k.T

        K = P_k @ H_k.T @ np.linalg.inv( H_k@P_x_k_plus_1_pre@H_k.T + self.lt.R(t_k, len(visible_stations)) )
        delta_y_k_plus_1 = y_k_plus_1 - y_k_plus_1_star

        delta_x_k_plus_1_post = delta_x_k_plus_1_pre + K @ (delta_y_k_plus_1 - H_k@delta_x_k_plus_1_pre)
        P_x_k_plus_1_post = (np.eye(4) - K@H_k) @ P_x_k_plus_1_pre
        return delta_x_k_plus_1_post, P_x_k_plus_1_post
