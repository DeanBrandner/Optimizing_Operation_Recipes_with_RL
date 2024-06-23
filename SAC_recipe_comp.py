
import os
import pickle
from stable_baselines3 import SAC

from tqdm import tqdm 
import casadi as cd
import numpy as np
from plot_reactor_trajectories import plot_trajectory_with_controller_setpoint_save as plot_trajectory_save
from environments_RL import Poly_reactor_SB3_cascade as Environment


def main():
    # path = f"D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_2_time\\"
    path = f"D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_2_hybrid\\"
    # path = "data\\poly_reactor\\agent\\SB3_SAC_7_det\\"

    agent = SAC.load(path + "best_model.zip")
    n_ic = 50
    env = Environment()


    clc_list = []
    batch_time_list = []
    n_violations_list = []
    n_transitions_list = []
    for i in range(n_ic):

        print(f"Initial condition: {i + 1} / {n_ic}")
        state, info = env.reset()
        clc = 0
        tmax = 2

        unscaled_state = env.observation_unscaling_func(state)
        unscaled_state, _, _ = env.rev_s(unscaled_state)
        m_total_old = m_total = round(float(unscaled_state[2]))
        m_total_possible = round(float(unscaled_state[1] * env.w_AF + unscaled_state[2] + env.x_upper_bounds[-2] * env.w_AF))
        tol = 1e-3

        n_thetas = env.theta.shape[0]

        transitions = 0

        tqdm.write("Running closed loop simulation")
        pbar = tqdm(total = 1)
        for idx_theta in range(n_thetas):
            theta = agent.predict(state, deterministic=True)[0]
            state, reward, terminated, truncated, info = env.step(theta)

            unscaled_state = env.observation_unscaling_func(state)
            unscaled_state, _, _ = env.rev_s(unscaled_state)

            m_total_old = m_total
            m_total = float(unscaled_state[2])

            if m_total >= m_total_possible - tol:
                break

            clc += float(reward)

            pbar.update(float(((m_total - m_total_old) / m_total_possible)))

        pbar.close()
        clc_list.append(clc)
        print(f"Closed loop cost: {clc:.2f}")

        batchtime= env.time
        batch_time_list.append(batchtime)
        print(f"Batch time: {batchtime:.2f}")
        
        n_violations_list.append(env.violations)
        n_transitions = env.x_data.shape[0]
        n_transitions_list.append(n_transitions)
        print(f"Constraints violations: {env.violations} ({env.violations / n_transitions * 100:.2f} %)")

        x_data = env.x_data
        u_data = env.u_data
        sp_data = env.setpoint_data
        if not path.endswith("\\"):
            path += "\\"
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = path + "eval\\"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        figpath = save_path + f"trajectory_{i}.png"
        # plot_trajectory_save(x_data, u_data, env.integration_settings.dt, path)
        plot_trajectory_save(x_data, u_data, sp_data, env.integration_settings.dt, figpath)

        
        with open(save_path + f"x_data_{i}.pkl", "wb") as f:
            pickle.dump(x_data, f)
        with open(save_path + f"u_data_{i}.pkl", "wb") as f:
            pickle.dump(u_data, f)
        with open(save_path + f"clc_{i}.pkl", "wb") as f:
            pickle.dump(clc, f)
        with open(save_path + f"batch_time_{i}.pkl", "wb") as f:
            pickle.dump(batchtime, f)
        with open(save_path + f"violations_{i}.pkl", "wb") as f:
            pickle.dump(env.violations, f)
        with open(save_path + f"n_transitions_{i}.pkl", "wb") as f:
            pickle.dump(n_transitions, f)


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

    n_violations_mean = np.mean(n_violations_list)
    n_violations_std = np.std(n_violations_list)
    n_violations_percentage_mean = np.mean(np.array(n_violations_list) / np.array(n_transitions_list) * 100)
    n_violations_percentage_std = np.std(np.array(n_violations_list) / np.array(n_transitions_list) * 100)
    print(f"Average number of violations: {n_violations_mean:.2f} +- {n_violations_std:.2f} ({n_violations_percentage_mean:.2f} +- {n_violations_percentage_std:.2f} %)")
    with open(save_path + "n_violations_list.pkl", "wb") as f:
        pickle.dump(n_violations_list, f)
    
    return

if __name__ == "__main__":
    main()

