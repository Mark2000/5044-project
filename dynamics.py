from dataclasses import dataclass
from itertools import product
from typing import *

import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

mu = 398600
R_E = 6378
omega_E = 2 * np.pi / 86400


@dataclass
class CircularTrajectory:
    radius: float
    omega: float
    theta_zero: float

    @classmethod
    def from_orbit(cls, radius: float, true_anomaly_zero: float):
        return cls(
            radius=radius, omega=np.sqrt(mu / radius**3), theta_zero=true_anomaly_zero
        )

    @property
    def period(self):
        return 2 * np.pi / self.omega

    def state_at(self, t: float):
        return [
            self.radius * np.cos(self.omega * t + self.theta_zero),
            -self.radius * self.omega * np.sin(self.omega * t + self.theta_zero),
            self.radius * np.sin(self.omega * t + self.theta_zero),
            self.radius * self.omega * np.cos(self.omega * t + self.theta_zero),
        ]


@dataclass
class EllipticalTrajectory:
    x0: Sequence[float]

    def propagate(self, t: np.ndarray):
        def orbit_dynamics(t, state):
            x, vx, y, vy = state
            r = np.sqrt(x**2 + y**2)
            ax = -mu * x / r**3
            ay = -mu * y / r**3
            return [vx, ax, vy, ay]

        return solve_ivp(
            orbit_dynamics, [t[0], t[-1]], self.x0, t_eval=t, rtol=1e-9
        ).y.T


@dataclass
class LinearizedSystem:
    nominal: CircularTrajectory
    stations: Sequence[CircularTrajectory]

    def A(self, t: float):
        x = self.nominal.state_at(t)
        r = np.sqrt(x[0] ** 2 + x[2] ** 2)
        r5 = r**5

        A = np.zeros([4, 4])
        A[0, 1] = 1
        A[2, 3] = 1

        A[1, 0] = (2 * x[0] ** 2 - x[2] ** 2) * mu / r5
        A[3, 2] = (2 * x[2] ** 2 - x[0] ** 2) * mu / r5
        A[1, 2] = A[3, 0] = 3 * mu * x[0] * x[2] / r5

        return A

    def B(self, t: float):
        B = np.zeros([4, 2])
        B[1, 0] = 1
        B[3, 1] = 1
        return B

    def C_i(self, i: int, t: float):
        C = np.zeros([3, 4])

        x = self.nominal.state_at(t)
        x_s = self.stations[i].state_at(t)
        rho = np.sqrt((x[0] - x_s[0]) ** 2 + (x[2] - x_s[2]) ** 2)
        N = (x[0] - x_s[0]) * (x[1] - x_s[1]) + (x[2] - x_s[2]) * (x[3] - x_s[3])

        C[0, 0] = C[1, 1] = (x[0] - x_s[0]) / rho
        C[0, 2] = C[1, 3] = (x[2] - x_s[2]) / rho

        C[1, 0] = ((x[1] - x_s[1]) * rho - N * C[0, 0]) / rho**2
        C[1, 2] = ((x[3] - x_s[3]) * rho - N * C[0, 2]) / rho**2

        det = 1 + ((x[2] - x_s[2]) / (x[0] - x_s[0])) ** 2
        C[2, 0] = -(x[2] - x_s[2]) / (x[0] - x_s[0]) ** 2 / det
        C[2, 2] = 1 / (x[0] - x_s[0]) / det

        return C

    def D_i(self, i: int, t: float):
        return np.zeros([3, 2])

    def C(self, t: float):
        return np.vstack([self.C_i(i, t) for i in range(len(self.stations))])

    def D(self, t: float):
        return np.zeros([3 * len(self.stations), 2])

    def F_G(self, t: float, dt: float):
        A = self.A(t)
        B = self.B(t)
        A_hat = np.zeros([A.shape[1] + B.shape[1]] * 2)
        A_hat[: A.shape[1], : A.shape[1]] = A
        A_hat[: A.shape[1], A.shape[1] :] = B
        Z = scipy.linalg.expm(A_hat * dt)
        return Z[: A.shape[1], : A.shape[1]], Z[A.shape[1] :, : A.shape[1]]

    def H(self, t: float):
        return self.C(t)

    def M(self, t: float):
        return self.D(t)

    def F_tilde(self, t: float, dt: float):
        return np.eye(4) + dt*self.A(t)

    def G_tilde(self, t: float, dt: float):
        return dt * self.B(t)

    def Omega(self, t: float):
        return self.B(t)

    def Omega_tilde(self, t: float, dt: float):
        B = np.zeros([4,2])
        B[1,0] = 1
        B[3,1] = 1 
        return B

    def H_tilde(self, t: float):
        return self.C(t)

    def R(self, t: float, n_stations: int):
        R = np.eye(3)
        R[0,0] = R[2,2] = 0.01
        return np.block([R] * n_stations)

    def Q(self, t: float):
        return 1e-10*np.eye(2)

def measurement(x: Sequence[float], x_s: Sequence[float]):
    rho = np.sqrt((x[0] - x_s[0]) ** 2 + (x[2] - x_s[2]) ** 2)
    N = (x[0] - x_s[0]) * (x[1] - x_s[1]) + (x[2] - x_s[2]) * (x[3] - x_s[3])
    psi = np.arctan2((x[2] - x_s[2]), (x[0] - x_s[0]))
    return [rho, N / rho, psi]


def measurements(x: Sequence[float], stations: Sequence[CircularTrajectory], t: float):
    return np.concatenate([measurement(x, st.state_at(t)) for st in stations])


def is_station_visible(x: Sequence[float], x_s: Sequence[float]):
    psi = measurement(x, x_s)[2]
    theta = np.arctan2(x_s[2], x_s[0])
    return -np.pi / 2 + theta < psi < np.pi / 2 + theta


def mask_non_visible(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, stations: Sequence[CircularTrajectory]
):
    y_copy = np.copy(y)
    for i, j in product(range(x.shape[0]), range(len(stations))):
        if not is_station_visible(x[i, :], stations[j].state_at(t[i])):
            y_copy[i, j * 3 : (j + 1) * 3] = np.nan
    return y_copy
