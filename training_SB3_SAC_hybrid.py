import os

from environments_RL import Poly_reactor_SB3_hybrid as Environment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from SaveModelCallback import SaveOnBestTrainingRewardCallback
import torch as th

from stable_baselines3.common.noise import NormalActionNoise

import numpy as np




if __name__ == "__main__":
    # Training parameters
    model_nb = 2

    save_path = f"D:\\recipe_opt\\data\\poly_reactor\\agent\\SB3_SAC_{model_nb}_hybrid\\"
    fig_path = f"D:\\recipe_opt\\figs\\poly_reactor\\agent\\SB3_SAC_{model_nb}_hybrid\\"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # Build the environment
    env = Environment()
    env = Monitor(env, save_path)

    eval_env = Environment(seed = 12)

    log_interval = 50
    check_freq = log_interval
    

    savebest_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=save_path)
    early_stopping_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=25, min_evals=5, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=early_stopping_callback, verbose=1)

    callbacks = [savebest_callback]
    
    action_noise = NormalActionNoise(mean=np.array([0.0]), sigma=np.array([0.1]))

    policy_kwargs = {
        "net_arch": dict(pi=[50, 50], qf=[50, 25]),
        "activation_fn": th.nn.ReLU
    }
    model = SAC(
        "MlpPolicy",
        env,
        gamma = 0.99,
        seed = 12,
        verbose = 1,
        batch_size = 4096,
        action_noise=action_noise,
        learning_rate = 3e-4,
        ent_coef = "auto",
        stats_window_size = log_interval,
        policy_kwargs=policy_kwargs)

    # n_total_time_steps = 200e3
    n_total_time_steps = 500e3
    model.learn(
        total_timesteps=n_total_time_steps,
        progress_bar=True,
        log_interval = log_interval,
        callback = callbacks)
    

    model.save(save_path + "final_model")