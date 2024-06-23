
import os
import pickle

from tqdm import tqdm 
import casadi as cd
import numpy as np
from plot_reactor_trajectories import plot_trajectory_save as plot_trajectory_save
from environments_RL import Poly_reactor_SB3_direct as Environment

from stable_baselines3 import SAC


def main():
    path = f"D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_1_direct\\"

    n_ic = 50
    env = Environment()

    agent = SAC.load(path + "best_model.zip")

    clc_list = []
    batch_time_list = []
    for i in range(n_ic):

        print(f"Initial condition: {i + 1} / {n_ic}")
        state, info = env.reset()

        clc = 0
        tmax = 2

        unscaled_state = env.observation_unscaling_func(state)

        m_total_old = m_total = round(float(unscaled_state[2]))
        m_total_possible = round(float(unscaled_state[1] + unscaled_state[2] + env.x_upper_bounds[-2] * env.w_AF))
        tol = 1e-1

        tqdm.write("Running closed loop simulation")
        pbar = tqdm(total = 1)

        while m_total < m_total_possible - tol:
            action = agent.predict(state, deterministic = True)[0]
            
            state, reward, terminated, truncated, info = env.step(action)

            unscaled_state = env.observation_unscaling_func(state)

            m_total_old = m_total
            m_total = float(unscaled_state[2]) 
            clc += reward

            pbar.update(float(((m_total - m_total_old) / m_total_possible)))

        pbar.close()
        clc_list.append(clc)
        print(f"Closed loop cost: {clc:.2f}")

        batchtime= env.time
        batch_time_list.append(batchtime)
        print(f"Batch time: {batchtime:.2f}")


        x_data = env.x_data
        u_data = env.u_data
        if not path.endswith("\\"):
            path += "\\"
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = path + "eval\\"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        figpath = save_path + f"trajectory_{i}.png"
        # plot_trajectory_save(x_data, u_data, env.integration_settings.dt, path)
        plot_trajectory_save(x_data, u_data, env.integration_settings.dt, figpath)

        
        with open(save_path + f"x_data_{i}.pkl", "wb") as f:
            pickle.dump(x_data, f)
        with open(save_path + f"u_data_{i}.pkl", "wb") as f:
            pickle.dump(u_data, f)
        with open(save_path + f"clc_{i}.pkl", "wb") as f:
            pickle.dump(clc, f)

    clc_mean = np.mean(clc_list)
    clc_std = np.std(clc_list)
    print(f"Average closed loop cost: {clc_mean:.2f} +- {clc_std:.2f}")
    with open(save_path + "clc_list.pkl", "wb") as f:
        pickle.dump(clc_list, f)
    
    batch_time_mean = np.mean(batch_time_list)
    batch_time_std = np.std(batch_time_list)
    print(f"Average batch time: {batch_time_mean:.2f} +- {batch_time_std:.2f}")
    with open(save_path + "batch_time_list.pkl", "wb") as f:
        pickle.dump(batch_time_list, f)
    return

if __name__ == "__main__":
    main()

