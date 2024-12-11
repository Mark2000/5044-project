from dynamics import *

dt = 10

nominal_trajectory = CircularTrajectory.from_orbit(radius=6678, true_anomaly_zero=0)

station_trajectories = [
    CircularTrajectory(radius=R_E, omega=omega_E, theta_zero=i * np.pi / 6)
    for i in range(12)
]

lt = LinearizedSystem(nominal_trajectory, station_trajectories)

# raw_ydata = np.loadtxt("ydata.csv", delimiter=",")
# raw_ydata[:, 0] = 0
# y_truth = raw_ydata.T[:, :3]
# visible_stations_truth = [[int(i - 1)] for i in raw_ydata[3, :]]

R = np.diag([0.01, 1, 0.01])
