from dynamics import *

import numpy as np

def lkf_step(lt: LinearizedSystem, t: float, dt: float, x_hat_k: np.ndarray, P_k: np.ndarray, y_k_plus_1: np.ndarray, visible_stations: list[int]):
    assert len(visible_stations) == y_k_plus_1.size // 3
    
    F_k = lt.F_tilde(t, dt)

    ind = np.zeros(len(lt.stations)*3, dtype=bool)
    for st in visible_stations:
        ind[st*3:(st+1)*3] = True
    H_k = lt.H_tilde(t)[ind,:]

    y_k_plus_1_star = measurements(
        lt.nominal.state_at(t),
        [lt.stations[i].state_at(t) for i in visible_stations],
        t
    )

    delta_x_k_plus_1_pre = F_k @ x_hat_k
    P_x_k_plus_1_pre = F_k @ P_k @ F_k.T

    K = P_k @ H_k.T @ np.linalg.inv( H_k@P_x_k_plus_1_pre@H_k.T + lt.R(t, len(visible_stations)) )
    delta_y_k_plus_1 = y_k_plus_1 - y_k_plus_1_star

    delta_x_k_plus_1_post = delta_x_k_plus_1_pre + K @ (delta_y_k_plus_1 - H_k@delta_x_k_plus_1_pre)
    P_x_k_plus_1_post = (np.eye(4) - K@H_k) @ P_x_k_plus_1_pre
    return delta_x_k_plus_1_post, P_x_k_plus_1_post
