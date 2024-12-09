from dynamics import *

dt = 10

nominal_trajectory = CircularTrajectory.from_orbit(radius=6678, true_anomaly_zero=0)

station_trajectories = [
    CircularTrajectory(radius=R_E, omega=omega_E, theta_zero=i * np.pi / 6)
    for i in range(12)
]

lt = LinearizedSystem(nominal_trajectory, station_trajectories)

ydata = np.loadtxt("ydata.csv", delimiter=",")
