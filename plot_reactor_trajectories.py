import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_trajectory(x, u, dt):
    
    ncols = 5
    nrows = 3

    k = 1.0

    time_u = np.arange(u.shape[0]) * dt
    time_x = np.arange(x.shape[0]) * dt

    figsize = (6 * ncols * k, 4 * nrows * k)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize, constrained_layout = True)


    # States
    ax[0, 0].plot(time_x, x[:, 0] / 1e3)
    ax[0, 0].set_ylabel("Mass of water [t]")
    ax[0, 0].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 0].legend()

    ax[0, 1].plot(time_x, x[:, 1] / 1e3)
    ax[0, 1].set_ylabel("Mass of monomer [t]")
    ax[0, 1].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 1].legend()

    ax[0, 2].plot(time_x, x[:, 2] / 1e3)
    ax[0, 2].set_ylabel("Mass of product [t]")
    ax[0, 2].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 2].legend()

    ax[0, 3].plot(time_x, x[:, 8] / 1e3)
    ax[0, 3].set_ylabel("Accum. monomer [t]")
    ax[0, 3].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 3].plot(time_x, [30]*time_x.shape[0], 'r--', label = "Max mass")
    ax[0, 3].legend()

    ax[1, 0].plot(time_x, x[:, 3] - 273.15)
    ax[1, 0].set_ylabel("Temperature (reactor mixture) [°C]")
    ax[1, 0].plot(time_x, [90 + 2]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 0].plot(time_x, [90 - 2]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 0].legend()

    ax[1, 1].plot(time_x, x[:, 4] - 273.15)
    ax[1, 1].set_ylabel("Temperature (steel of vessel) [°C]")
    ax[1, 1].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 1].plot(time_x, [20]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 1].legend()

    ax[1, 2].plot(time_x, x[:, 5] - 273.15)
    ax[1, 2].set_ylabel("Temperature (jacket coolant) [°C]")
    ax[1, 2].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 2].plot(time_x, [20]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[1, 3].plot(time_x, x[:, 6] - 273.15)
    ax[1, 3].set_ylabel("Temperature (mixture external) [°C]")
    ax[1, 3].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 3].plot(time_x, [15]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 0].plot(time_x, x[:, 7] - 273.15)
    ax[2, 0].set_ylabel("Temperature (coolant external heat exchanger) [°C]")
    ax[2, 0].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[2, 0].plot(time_x, [15]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 1].plot(time_x, x[:, 9] - 273.15)
    ax[2, 1].set_ylabel("Adiabatic temperature [°C]")
    ax[2, 1].plot(time_x, [109]*time_x.shape[0], 'r--', label = "Max temperature")

    # Control inputs
    ax[0, -1].step(time_u, u[:, 0] / 1e3, where = "post")
    ax[0, -1].plot(time_u, [30]*time_u.shape[0], 'r--', label = "Max flowrate")
    ax[0, -1].plot(time_u, [0]*time_u.shape[0], 'r--', label = "Min flowrate")
    ax[0, -1].set_ylabel("Feed flowrate [t/s]")

    ax[1, -1].step(time_u, u[:, 1] - 273.15, where = "post")
    ax[1, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[1, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[1, -1].set_ylabel("Temperatur (jacket in) [°C]")

    ax[2, -1].step(time_u, u[:, 2] - 273.15, where = "post")
    ax[2, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[2, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[2, -1].set_ylabel("Temperature (external heat exchanger in) [°C]")

    plt.show()
    # plt.savefig(path)
    # plt.close("all")
    return

def plot_trajectory_save(x, u, dt, savepath: str):
    mpl.use("Agg")
    ncols = 5
    nrows = 3

    k = 1.0

    time_u = np.arange(u.shape[0]) * dt
    time_x = np.arange(x.shape[0]) * dt

    figsize = (6 * ncols * k, 4 * nrows * k)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize, constrained_layout = True)


    # States
    ax[0, 0].plot(time_x, x[:, 0] / 1e3)
    ax[0, 0].set_ylabel("Mass of water [t]")
    ax[0, 0].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 0].legend()

    ax[0, 1].plot(time_x, x[:, 1] / 1e3)
    ax[0, 1].set_ylabel("Mass of monomer [t]")
    ax[0, 1].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 1].legend()

    ax[0, 2].plot(time_x, x[:, 2] / 1e3)
    ax[0, 2].set_ylabel("Mass of product [t]")
    ax[0, 2].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 2].legend()

    ax[0, 3].plot(time_x, x[:, 8] / 1e3)
    ax[0, 3].set_ylabel("Accum. monomer [t]")
    ax[0, 3].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 3].plot(time_x, [30]*time_x.shape[0], 'r--', label = "Max mass")
    ax[0, 3].legend()

    ax[1, 0].plot(time_x, x[:, 3] - 273.15)
    ax[1, 0].set_ylabel("Temperature (reactor mixture) [°C]")
    ax[1, 0].plot(time_x, [90 + 2]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 0].plot(time_x, [90 - 2]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 0].legend()

    ax[1, 1].plot(time_x, x[:, 4] - 273.15)
    ax[1, 1].set_ylabel("Temperature (steel of vessel) [°C]")
    ax[1, 1].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 1].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 1].legend()

    ax[1, 2].plot(time_x, x[:, 5] - 273.15)
    ax[1, 2].set_ylabel("Temperature (jacket coolant) [°C]")
    ax[1, 2].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 2].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[1, 3].plot(time_x, x[:, 6] - 273.15)
    ax[1, 3].set_ylabel("Temperature (mixture external) [°C]")
    ax[1, 3].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 3].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 0].plot(time_x, x[:, 7] - 273.15)
    ax[2, 0].set_ylabel("Temperature (coolant external heat exchanger) [°C]")
    ax[2, 0].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[2, 0].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 1].plot(time_x, x[:, 9] - 273.15)
    ax[2, 1].set_ylabel("Adiabatic temperature [°C]")
    ax[2, 1].plot(time_x, [109]*time_x.shape[0], 'r--', label = "Max temperature")

    # Control inputs
    ax[0, -1].step(time_u, u[:, 0] / 1e3, where = "post")
    ax[0, -1].plot(time_u, [30]*time_u.shape[0], 'r--', label = "Max flowrate")
    ax[0, -1].plot(time_u, [0]*time_u.shape[0], 'r--', label = "Min flowrate")
    ax[0, -1].set_ylabel("Feed flowrate [t/s]")

    ax[1, -1].step(time_u, u[:, 1] - 273.15, where = "post")
    ax[1, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[1, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[1, -1].set_ylabel("Temperatur (jacket in) [°C]")

    ax[2, -1].step(time_u, u[:, 2] - 273.15, where = "post")
    ax[2, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[2, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[2, -1].set_ylabel("Temperature (external heat exchanger in) [°C]")

    # plt.show()
    
    plt.savefig(savepath)
    plt.close("all")
    return

def plot_trajectory_with_controller_setpoint(x, u, s, dt):
    
    ncols = 5
    nrows = 3

    k = 1.0

    time_u = np.arange(u.shape[0]) * dt
    time_x = np.arange(x.shape[0]) * dt

    figsize = (6 * ncols * k, 4 * nrows * k)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize, constrained_layout = True)


    # States
    ax[0, 0].plot(time_x, x[:, 0] / 1e3)
    ax[0, 0].set_ylabel("Mass of water [t]")
    ax[0, 0].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 0].legend()

    ax[0, 1].plot(time_x, x[:, 1] / 1e3)
    ax[0, 1].set_ylabel("Mass of monomer [t]")
    ax[0, 1].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 1].legend()

    ax[0, 2].plot(time_x, x[:, 2] / 1e3)
    ax[0, 2].set_ylabel("Mass of product [t]")
    ax[0, 2].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 2].legend()

    ax[0, 3].plot(time_x, x[:, 8] / 1e3)
    ax[0, 3].set_ylabel("Accum. monomer [t]")
    ax[0, 3].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 3].plot(time_x, [30]*time_x.shape[0], 'r--', label = "Max mass")
    ax[0, 3].legend()

    ax[1, 0].plot(time_x, x[:, 3] - 273.15)
    ax[1, 0].plot(time_x, s[:, 0] - 273.15, label = "Setpoint")
    ax[1, 0].set_ylabel("Temperature (reactor mixture) [°C]")
    ax[1, 0].plot(time_x, [90 + 2]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 0].plot(time_x, [90 - 2]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 0].legend()

    ax[1, 1].plot(time_x, x[:, 4] - 273.15)
    ax[1, 1].set_ylabel("Temperature (steel of vessel) [°C]")
    ax[1, 1].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 1].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 1].legend()

    ax[1, 2].plot(time_x, x[:, 5] - 273.15)
    ax[1, 2].plot(time_x, s[:, 1] - 273.15, label = "Setpoint")
    ax[1, 2].set_ylabel("Temperature (jacket coolant) [°C]")
    ax[1, 2].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 2].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[1, 3].plot(time_x, x[:, 6] - 273.15)
    ax[1, 3].plot(time_x, s[:, 2] - 273.15, label = "Setpoint")
    ax[1, 3].set_ylabel("Temperature (mixture external) [°C]")
    ax[1, 3].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 3].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 0].plot(time_x, x[:, 7] - 273.15)
    ax[2, 0].set_ylabel("Temperature (coolant external heat exchanger) [°C]")
    ax[2, 0].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[2, 0].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 1].plot(time_x, x[:, 9] - 273.15)
    ax[2, 1].set_ylabel("Adiabatic temperature [°C]")
    ax[2, 1].plot(time_x, [109]*time_x.shape[0], 'r--', label = "Max temperature")

    # Control inputs
    ax[0, -1].step(time_u, u[:, 0] / 1e3, where = "post")
    ax[0, -1].plot(time_u, [30]*time_u.shape[0], 'r--', label = "Max flowrate")
    ax[0, -1].plot(time_u, [0]*time_u.shape[0], 'r--', label = "Min flowrate")
    ax[0, -1].set_ylabel("Feed flowrate [t/s]")

    ax[1, -1].step(time_u, u[:, 1] - 273.15, where = "post")
    ax[1, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[1, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[1, -1].set_ylabel("Temperatur (jacket in) [°C]")

    ax[2, -1].step(time_u, u[:, 2] - 273.15, where = "post")
    ax[2, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[2, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[2, -1].set_ylabel("Temperature (external heat exchanger in) [°C]")

    plt.show()
    # plt.savefig(path)
    # plt.close("all")
    return

def plot_trajectory_with_controller_setpoint_save(x, u, s, dt, savepath: str):
    mpl.use("Agg")
    ncols = 5
    nrows = 3

    k = 1.0

    time_u = np.arange(u.shape[0]) * dt
    time_x = np.arange(x.shape[0]) * dt

    figsize = (6 * ncols * k, 4 * nrows * k)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize, constrained_layout = True)


    # States
    ax[0, 0].plot(time_x, x[:, 0] / 1e3)
    ax[0, 0].set_ylabel("Mass of water [t]")
    ax[0, 0].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 0].legend()

    ax[0, 1].plot(time_x, x[:, 1] / 1e3)
    ax[0, 1].set_ylabel("Mass of monomer [t]")
    ax[0, 1].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 1].legend()

    ax[0, 2].plot(time_x, x[:, 2] / 1e3)
    ax[0, 2].set_ylabel("Mass of product [t]")
    ax[0, 2].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 2].legend()

    ax[0, 3].plot(time_x, x[:, 8] / 1e3)
    ax[0, 3].set_ylabel("Accum. monomer [t]")
    ax[0, 3].plot(time_x, [0]*time_x.shape[0], 'r--', label = "Min mass")
    ax[0, 3].plot(time_x, [30]*time_x.shape[0], 'r--', label = "Max mass")
    ax[0, 3].legend()

    ax[1, 0].plot(time_x, x[:, 3] - 273.15)
    ax[1, 0].plot(time_x, s[:, 0] - 273.15, label = "Setpoint")
    ax[1, 0].set_ylabel("Temperature (reactor mixture) [°C]")
    ax[1, 0].plot(time_x, [90 + 2]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 0].plot(time_x, [90 - 2]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 0].legend()

    ax[1, 1].plot(time_x, x[:, 4] - 273.15)
    ax[1, 1].set_ylabel("Temperature (steel of vessel) [°C]")
    ax[1, 1].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 1].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")
    ax[1, 1].legend()

    ax[1, 2].plot(time_x, x[:, 5] - 273.15)
    ax[1, 2].plot(time_x, s[:, 1] - 273.15, label = "Setpoint")
    ax[1, 2].set_ylabel("Temperature (jacket coolant) [°C]")
    ax[1, 2].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 2].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[1, 3].plot(time_x, x[:, 6] - 273.15)
    ax[1, 3].plot(time_x, s[:, 2] - 273.15, label = "Setpoint")
    ax[1, 3].set_ylabel("Temperature (mixture external) [°C]")
    ax[1, 3].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[1, 3].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 0].plot(time_x, x[:, 7] - 273.15)
    ax[2, 0].set_ylabel("Temperature (coolant external heat exchanger) [°C]")
    ax[2, 0].plot(time_x, [100]*time_x.shape[0], 'r--', label = "Max temperature")
    ax[2, 0].plot(time_x, [60]*time_x.shape[0], 'r--', label = "Min temperature")

    ax[2, 1].plot(time_x, x[:, 9] - 273.15)
    ax[2, 1].set_ylabel("Adiabatic temperature [°C]")
    ax[2, 1].plot(time_x, [109]*time_x.shape[0], 'r--', label = "Max temperature")

    # Control inputs
    ax[0, -1].step(time_u, u[:, 0] / 1e3, where = "post")
    ax[0, -1].plot(time_u, [30]*time_u.shape[0], 'r--', label = "Max flowrate")
    ax[0, -1].plot(time_u, [0]*time_u.shape[0], 'r--', label = "Min flowrate")
    ax[0, -1].set_ylabel("Feed flowrate [t/s]")

    ax[1, -1].step(time_u, u[:, 1] - 273.15, where = "post")
    ax[1, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[1, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[1, -1].set_ylabel("Temperatur (jacket in) [°C]")

    ax[2, -1].step(time_u, u[:, 2] - 273.15, where = "post")
    ax[2, -1].plot(time_u, [100]*time_u.shape[0], 'r--', label = "Max temperature")
    ax[2, -1].plot(time_u, [60]*time_u.shape[0], 'r--', label = "Min temperature")
    ax[2, -1].set_ylabel("Temperature (external heat exchanger in) [°C]")

    # plt.show()
    plt.savefig(savepath)
    plt.close("all")
    return