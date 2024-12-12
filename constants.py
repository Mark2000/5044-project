import numpy as np
from dynamics import CircularTrajectory, LinearizedSystem

# Physical
R_E = 6378
omega_E = 2 * np.pi / 86400


# Problem Setting
dt = 10
ts = np.arange(0, 14000, step=dt)
nominal_trajectory = CircularTrajectory.from_orbit(radius=6678, true_anomaly_zero=0)
station_trajectories = [
    CircularTrajectory(radius=R_E, omega=omega_E, theta_zero=i * np.pi / 6)
    for i in range(12)
]
lt = LinearizedSystem(nominal_trajectory, station_trajectories)
R = np.diag([0.01, 1, 0.01])


# Data
# raw_ydata = np.loadtxt("ydata.csv", delimiter=",")
# raw_ydata[:, 0] = 0
# y_truth = raw_ydata.T[:, :3]
# visible_stations_truth = [[int(i - 1)] for i in raw_ydata[3, :]]

# Filters
Q_tuned_lkf = np.eye(2) * 1e0
Q_tuned_ekf = np.eye(2) * 1e-8

# Initial Distributions
dx0_bar_true = np.zeros(4)
P0_true = np.diag([10, 0.5, 10, 0.5])
