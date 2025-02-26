# Standard Libraries
import os
import sys
import itertools
from random import gauss
import threading
gauss_lock = threading.Lock()   

# Plotting and Data Analysis
from matplotlib import pyplot as plt
import numpy as np

# Typing
from typing import Any

# Gymnasium and Environments
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
from logging_tensorboard import TensorboardCallback

# Optuna
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

# PyTorch
import torch
import torch.nn as nn

# Add Parent Directory to Path
sys.path.append('..')

def calculate_gini(array):
    """
    Calculate the Gini coefficient of a numpy array.
    
    Parameters:
    array (numpy.ndarray): Input array to calculate the Gini coefficient for.
    
    Returns:
    float: Gini coefficient.
    """
    
    if np.all(array == 0):
        return 0
    array = np.sort(np.array(array)).astype(np.float64)  # Cast to sorted numpy array
    array = np.clip(array, -1000, 1000)
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]  # Number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

class GiniEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, models_dir, file_dir, log_dir, start, grid_size=5, render_mode=None, method="gini_env"):
        """
        Initialize the environment.

        Parameters:
        models_dir (str): Directory to save models.
        file_dir (str): Directory to save files.
        log_dir (str): Directory to save logs.
        start (list): Starting values.
        grid_size (int): Size of the grid.
        render_mode (str): Mode for rendering.
        method (str): Method to use for reward calculation.
        """
        if not hasattr(self, "index"):
            self.index = 0
        super(GiniEnv, self).__init__()
        self.models_dir = models_dir
        self.file_dir = file_dir
        self.logdir = log_dir
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.start = start
        self.method = method
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(self.grid_size)
        self.observation_space = Dict({
            "required": Box(low=0, high=6, shape=(self.grid_size,), dtype=float),
            "received": Box(low=0, high=500, shape=(self.grid_size,), dtype=float),
            "valuation": Box(low=0, high=101, shape=(self.grid_size,), dtype=float),
            "cur_val": Discrete(5),
            "count_rec": Box(low=0, high=500, shape=(self.grid_size,), dtype=float)
        })
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Parameters:
        seed (int): Seed for random number generation.
        options (dict): Additional options for reset.

        Returns:
        dict: Initial observation.
        """
        self.observation = {
            "required": np.zeros(self.grid_size, dtype=np.float32),
            "received": np.zeros(self.grid_size, dtype=np.float32),
            "valuation": np.zeros(self.grid_size, dtype=np.float32),
            "cur_val": np.array([np.random.randint(1, 5)], dtype=np.float32),
            "count_rec": np.zeros(self.grid_size, dtype=np.float32)
        }
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.steps = 0
        if np.array(self.start).ndim == 3:
            self.frequency = self.start[self.index % len(self.start)]
        else:
            self.frequency = self.start
        self.index += 1
        self.bedarf(self.frequency, count=self.steps)
        return self.observation, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Parameters:
        action (int): Action to take.

        Returns:
        tuple: (observation, reward, terminated, truncated, info)
        """
        self.action = action
        self.observation["received"][action] += min(self.observation["cur_val"], self.observation["required"][action])
        
        for agent in range(self.grid_size):
            self.observation["valuation"][agent] = max(self.observation["valuation"][agent] - 20, 0)
        if self.observation["required"][action] != 0:
            self.observation["valuation"][action] = (min(self.observation["cur_val"], self.observation["received"][action]) / self.observation["required"][action]) * 100
        
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.reward = getattr(self, self.method)(action)
        self.bedarf(self.frequency, count=self.steps)
        
        self.steps += 1
        terminated = self.steps >= 100
        truncated = False
        self.info = {
            "required": self.observation["required"],
            "received": self.observation["received"],
            "valuation": self.observation["valuation"],
            "value": self.observation["cur_val"],
            "gini": self.gini_index,
            "sum_rec": sum(self.observation["received"])
        }
        self.observation["cur_val"] = np.random.randint(1, 5)
        
        return self.observation, self.reward, terminated, truncated, self.info

    def bedarf(self, frequency, count, value=30):
        """
        Calculate the required values for each agent based on frequency and count.

        Parameters:
        frequency (list): Frequency of requirements for each agent.
        count (int): Current count.
        value (int): Base value for calculation.
        """
        bedarf = np.zeros(self.grid_size)
        for agent in range(self.grid_size):
            try:
                if isinstance(frequency[agent], list):
                    count_agent = (count) % len(frequency[agent])
                    with gauss_lock:
                        bedarf[agent] = max(0, gauss(frequency[agent][count_agent], value))
                else:
                    count_agent = (count - 1) % len(frequency)
                    bedarf[agent] = max(0, gauss(frequency[count_agent], value))
            except Exception as e:
                print(f"Error calculating bedarf for agent {agent}: {e}")
                bedarf[agent] = 0
        self.observation["required"] = bedarf

    def render(self, mode='console'):
        """
        Render the environment.

        Parameters:
        mode (str): Mode for rendering.
        """
        if self.render_mode == 'console':
            print(f"reward:{self.reward} Predicted action: {self.action} received: {self.info['received']}, valuation: {self.info['valuation']} value: {self.info['value']}")

    def close(self):
        pass

    def worst_received(self, action):
        return np.clip(-self.observation["received"][action], -10000, 10000)
    
    def round_robin(self, action):
        return np.clip(-np.var(self.observation["count_rec"]), -10000, 10000)
    
    def greedy(self, action):
        return np.clip(self.observation["required"][action], -10000, 10000)
    
    def bedarf_received(self, action):
        if self.observation["received"][action] != 0:
            return np.clip(self.observation["required"][action] / self.observation["received"][action], -10000, 10000)
        else:
            return np.clip(self.observation["required"][action], -10000, 10000)
    
    def gini_env(self, action):
        return np.clip(-100 * self.gini_index + sum(self.observation["received"]) + sum(self.observation["valuation"]), -10000, 10000)

def test(file_dir, log_dir, models_dir, method, iterations=100, start=[[[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]],
                    [[5, 5, 5], [3, 3, 1], [1, 1, 4], [1, 3, 2], [1, 1, 5]],
                    [[5, 1, 3], [3, 5, 1], [2, 4, 1], [2, 3, 4], [2, 4, 1]],
                    [[5, 3, 1], [1, 5, 2], [3, 2, 2], [2, 3, 1], [5, 5, 1]],
                    [[3, 5, 4], [2, 1, 4], [1, 1, 5], [2, 5, 1], [1, 1, 3]]], model_name=PPO):
    """
    Test the trained model.

    Parameters:
    file_dir (str): Directory of the model file.
    log_dir (str): Directory to save logs.
    models_dir (str): Directory to save models.
    method (str): Method to use for testing.
    iterations (int): Number of iterations to run the test.
    start (list): Starting values.
    model_name (class): Model class to use for testing.
    """
    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=start, method=method, models_dir=models_dir, file_dir=file_dir, log_dir=log_dir)])
    model = PPO.load(file_dir, env=env)
    
    obs = env.reset()
    writer = SummaryWriter(log_dir)
    for step in range(iterations):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f' reward:{reward} Predicted action: {action} received: {info[-1]["received"]}, valuation: {info[-1]["valuation"]} value: {info[-1]["value"]}') 
        writer.add_scalar("Test/Reward", reward, step)
        writer.add_scalar("Test/Gini_Index", calculate_gini(info["valuation"]), step)
        writer.add_scalar("Test/Sum_rec", info[-1]["sum_rec"], step)
        writer.add_scalar("Test/Valuation", sum(info[-1]["valuation"]), step)
        env.render()
        if done.any(): 
            print("reward", reward, "last call of episode", info[-1]["terminal_observation"], "Gini Index: ", calculate_gini(info[-1]["received"]))  # always use last element
            print("Episode finished.")
            break

def train(method, logdir, models_dir, timesteps=1000000):
    """
    Train the model.

    Parameters:
    method (str): Method to use for training.
    logdir (str): Directory to save logs.
    models_dir (str): Directory to save models.
    timesteps (int): Number of timesteps to train the model.
    """
    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[[[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]],
                    [[5, 5, 5], [3, 3, 1], [1, 1, 4], [1, 3, 2], [1, 1, 5]],
                    [[5, 1, 3], [3, 5, 1], [2, 4, 1], [2, 3, 4], [2, 4, 1]],
                    [[5, 3, 1], [1, 5, 2], [3, 2, 2], [2, 3, 1], [5, 5, 1]],
                    [[3, 5, 4], [2, 1, 4], [1, 1, 5], [2, 5, 1], [1, 1, 3]]], method=method)], start_method="fork")
    model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir)
    model.learn(total_timesteps=timesteps, tb_log_name="train_test", callback=TensorboardCallback())
    model.save(models_dir)

# https://www.datacamp.com/tutorial/optuna
# https://github.com/araffin/rl-handson-rlvs21/blob/main/optuna/sb3_simple.py
def objective(trial: optuna.Trial, method: str) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    trial (optuna.Trial): Optuna trial object.
    method (str): Method to use for training.

    Returns:
    float: Mean reward of the best trial.
    """
    N_TIMESTEPS = int(2e4)
    N_EVALUATIONS = 2
    EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
    N_EVAL_ENVS = 5
    N_EVAL_EPISODES = 10

    models_dir = f"src/stable_baselines/agents/models/best_model_{method}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    DEFAULT_HYPERPARAMS = {
        "policy": "MultiInputPolicy",
    }

    file_dir = f"src/stable_baselines/agents/models/best_model_{method}.zip"
    logdir = f"src/stable_baselines/logs/{method}"

    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[[[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]],
                    [[5, 5, 5], [3, 3, 1], [1, 1, 4], [1, 3, 2], [1, 1, 5]],
                    [[5, 1, 3], [3, 5, 1], [2, 4, 1], [2, 3, 4], [2, 4, 1]],
                    [[5, 3, 1], [1, 5, 2], [3, 2, 2], [2, 3, 1], [5, 5, 1]],
                    [[3, 5, 4], [2, 1, 4], [1, 1, 5], [2, 5, 1], [1, 1, 3]]], method=method, models_dir=models_dir, file_dir=file_dir, log_dir=logdir)])

    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(ppo_hyper_params(trial))
    kwargs = {key: value for key, value in kwargs.items() if key != "policy"}

    model = PPO("MultiInputPolicy", env, verbose=0, **kwargs)

    eval_envs = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[[[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]],
                    [[5, 5, 5], [3, 3, 1], [1, 1, 4], [1, 3, 2], [1, 1, 5]],
                    [[5, 1, 3], [3, 5, 1], [2, 4, 1], [2, 3, 4], [2, 4, 1]],
                    [[5, 3, 1], [1, 5, 2], [3, 2, 2], [2, 3, 1], [5, 5, 1]],
                    [[3, 5, 4], [2, 1, 4], [1, 1, 5], [2, 5, 1], [1, 1, 3]]], method=method, models_dir=models_dir, file_dir=file_dir, log_dir=logdir)])

    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=models_dir,
        log_path=logdir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    nan_encountered = False

    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_envs.close()

    if nan_encountered:
        return float("nan")

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

def ppo_hyper_params(trial: optuna.Trial) -> dict:
    """
    Sample PPO hyperparameters for Optuna trial.

    Parameters:
    trial (optuna.Trial): Optuna trial object.

    Returns:
    dict: Dictionary of sampled hyperparameters.
    """
    batch_size = 64
    possible_n_steps = [i for i in range(5, 2049) if i % batch_size == 0]
    n_steps = trial.suggest_categorical("n_steps", possible_n_steps)
    
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "n_steps": n_steps,
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 10),
        "batch_size": batch_size,  # Include the batch size in kwargs
    }

if __name__ == "__main__":
    methods = ["gini_env", "round_robin", "greedy", "bedarf_received", "worst_received"]

    for method in methods:
        print(f"Optimizing for method: {method}")

        N_TRIALS = 100
        N_JOBS = 1
        N_STARTUP_TRIALS = 5
        TIMEOUT = int(60 * 15)

        pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        study = optuna.create_study(sampler=sampler, storage="sqlite:///db.sqlite3", pruner=pruner, direction="maximize")

        try:
            study.optimize(lambda trial: objective(trial, method), n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
