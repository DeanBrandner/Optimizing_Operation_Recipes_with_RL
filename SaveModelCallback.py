from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
import pandas as pd

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.best_mean_length = 0.0

        self.n_episodes = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

   
    def _on_step(self) -> bool:
        if self.model.env.envs[0].env.parameter_step_num == 1:
            self.n_episodes += 1

        if self.n_episodes % (self.check_freq + 1) == 0 and self.n_episodes > 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-self.check_freq:])
                mean_length = -(np.mean(x[-self.check_freq:-1] - x[-self.check_freq+1:]))
                self.n_episodes = 1
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    print(f"Best mean length: {self.best_mean_length:.2f} - Last mean length per episode: {mean_length:.2f}")

                # New best model, you could save the agent here
                if (mean_length > self.best_mean_length) or (np.isclose(mean_length, self.best_mean_length, atol=1e-1) and mean_reward > self.best_mean_reward):
                # if mean_reward > self.best_mean_reward:
                    self.best_mean_length = mean_length
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                        self.model.save(self.save_path)

        return True
    
class SaveOnBestTrainingRewardCallback_direct(SaveOnBestTrainingRewardCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.best_termination_ratio: int = int(round(0.0, 2) * 100)
        self.best_truncation_ratio: float = int(round(1.0, 2) * 100)
        self.best_mean_batchtime: float = np.inf
        self.best_std_batchtime: float = np.inf
        self.best_violations: float = np.inf

        self.performance = pd.DataFrame(columns = ["Termination ratio", "Truncation ratio", "Mean batch time", "Std batch time", "Mean violations"])

    def _on_step(self) -> bool:
        if self.model.env.envs[0].env.current_step - 1 == 0:
            self.n_episodes += 1
        
        if self.n_episodes % (self.check_freq + 1) == 0 and self.n_episodes > 0:
            self.n_episodes = 0

            if len(self.model.env.envs[0].env.termination_list) == self.model.env.envs[0].env.termination_list.maxlen and len(self.model.env.envs[0].env.truncation_list) >= self.model.env.envs[0].env.truncation_list.maxlen:
                termination_list = list(self.model.env.envs[0].env.termination_list)[:-1]
                truncation_list = list(self.model.env.envs[0].env.truncation_list)[:-1]
                batchtime_list = list(self.model.env.envs[0].env.batch_time_list)[:-1]
                violations_list = list(self.model.env.envs[0].env.violations_list)[:-1]

                termination_ratio = int(round(np.mean(termination_list), 2) * 100)
                truncation_ratio = int(round(np.mean(truncation_list), 2) * 100)
                mean_batchtime = np.mean(batchtime_list)
                std_batchtime = np.std(batchtime_list)
                mean_violations = np.mean(violations_list)

                self.performance.loc[len(self.performance)] = [termination_ratio, truncation_ratio, mean_batchtime, std_batchtime, mean_violations]
                self.performance.to_csv(os.path.dirname(self.save_path) + "\\performance.csv", index = False)
                
                print(f"Termination ratio: \t{str(termination_ratio).rjust(3)}% \t\t Truncation ratio: \t{str(truncation_ratio).rjust(3)}%")
                print(f"Best termination ratio: {str(self.best_termination_ratio).rjust(3)}% \t\t Best truncation ratio: {str(self.best_truncation_ratio).rjust(3)}%")
                print(f"Mean batch time: \t{mean_batchtime:.2f} +- {std_batchtime:.2f} \t Best mean batch time: \t{self.best_mean_batchtime:.2f} +- {self.best_std_batchtime:.2f}")
                print(f"Mean violations: \t{mean_violations:.2f} \t\t Best mean violations: \t{self.best_violations:.2f}")

                decision1 = termination_ratio > self.best_termination_ratio and truncation_ratio < self.best_truncation_ratio
                decision2 = termination_ratio == self.best_termination_ratio and truncation_ratio == self.best_truncation_ratio and mean_batchtime < self.best_mean_batchtime
                decision3 = termination_ratio == self.best_termination_ratio and truncation_ratio == self.best_truncation_ratio and mean_batchtime <= self.best_mean_batchtime and mean_violations < self.best_violations

                decision = decision1 or decision2 or decision3

                if decision:
                    self.best_termination_ratio = termination_ratio
                    self.best_truncation_ratio = truncation_ratio

                    self.best_mean_batchtime = mean_batchtime
                    self.best_std_batchtime = std_batchtime

                    self.best_violations = mean_violations

                    print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
                # elif termination_ratio == self.best_termination_ratio and truncation_ratio == self.best_truncation_ratio and mean_batchtime < self.best_mean_batchtime:
                #     self.best_mean_batchtime = mean_batchtime
                #     self.best_std_batchtime = std_batchtime

                #     print(f"Saving new best model to {self.save_path}")
                #     self.model.save(self.save_path)


                # # Retrieve training reward
                # x, y = ts2xy(load_results(self.log_dir), "timesteps")
                # if len(x) > 0:
                #     # Mean training reward over the last 100 episodes
                #     mean_reward = np.mean(y[-self.check_freq:])
                #     mean_length = -(np.mean(x[-self.check_freq:-1] - x[-self.check_freq+1:]))
                #     self.n_episodes = 1
                #     if self.verbose >= 1:
                #         print(f"Num timesteps: {self.num_timesteps}")
                #         print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                #         print(f"Best mean length: {self.best_mean_length:.2f} - Last mean length per episode: {mean_length:.2f}")

                #     # New best model, you could save the agent here
                #     if (mean_length > self.best_mean_length) or (np.isclose(mean_length, self.best_mean_length, atol=1e-1) and mean_reward > self.best_mean_reward):
                #     # if mean_reward > self.best_mean_reward:
                #         self.best_mean_length = mean_length
                #         self.best_mean_reward = mean_reward
                #         if self.verbose >= 1:
                #             print(f"Saving new best model to {self.save_path}")
                #             self.model.save(self.save_path)
        return True