import os
import sys
import itertools
from random import gauss
from matplotlib import pyplot as plt
import numpy as np
from typing import Any
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
from logging_tensorboard import TensorboardCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch
import torch.nn as nn


# Add Parent Directory to Path
sys.path.append('..')

#print(optuna.__version__)

def calculate_gini(array):
    if np.all(array==0):
        return 0
    array = np.sort(np.array(array)).astype(np.float16)  # Cast to sorted numpy array
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]  # Number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

class GiniEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    #index=-1

    def __init__(self, models_dir, file_dir, log_dir, grid_size=5, render_mode=None, start=[1,2,3,4,5], method="gini_env"):
        if not hasattr(self, "index"):
            self.index = -1
        super(GiniEnv, self).__init__()
        self.models_dir = models_dir #"src/stable-baselines/agents/models/test_trained" + method
        self.file_dir = file_dir#"src/stable-baselines/agents/models/test_trained" + method +".zip"
        self.logdir = log_dir#"src/stable-baselines/logs"
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.start=start
        self.method = method
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(self.grid_size) 
        self.observation_space = Dict({
            "required": Box(low=0, high=6, shape=(self.grid_size,), dtype=int),
            "received": Box(low=0, high=500, shape=(self.grid_size,), dtype=int),
            "valuation": Box(low=0, high=101, shape=(self.grid_size,), dtype=int),
            "cur_val": Discrete(5),
            "count_rec": Box(low=0, high=500, shape=(self.grid_size,), dtype=int)
        })
        self.render_mode = render_mode 
        self.previous_valuations = np.zeros(self.grid_size)
        self.reset()

    def reset(self, seed=None, options=None ):
        #print(self.num_timesteps)
        self.index = self.index + 1
        if self.start==None:
            self.observation = {
                "required": np.random.randint(0, 5, size=self.grid_size).astype(np.int32),
                "received": np.zeros(self.grid_size),
                "valuation": np.zeros(self.grid_size, dtype=np.int32),
                "cur_val": np.random.randint(1, 5),
                "count_rec": np.zeros(self.grid_size)
            }
        else:
            self.observation = {
                "required": np.array(self.start),
                "received": np.zeros(self.grid_size),
                "valuation": np.zeros(self.grid_size, dtype=np.int32),
                "cur_val": np.random.randint(1, 5),
                "count_rec": np.zeros(self.grid_size)
        }
        self.gini_index = calculate_gini(self.observation["received"])
        self.steps = 0
        steps = [[[2, 2, 2], [3, 3, 1], [1, 1, 1, 4], [1, 3, 2, 3], [1, 1, 1, 5]],
                [[5], [3, 3, 1], [1, 1, 1, 4], [1, 3, 2, 3], [1, 1, 1, 5]],
                [[5, 1, 3], [3, 5, 1], [2, 4, 1, 1], [2, 3, 3, 4], [2, 1, 4, 1]],
                [[5, 3, 1], [1, 5, 2], [3, 2, 2, 1], [2, 3, 2, 1], [5, 4, 5, 1]],
                [[3, 5, 4], [2, 1, 4], [1, 1, 5, 1], [2, 5, 4, 1], [1, 2, 1, 3]]]
        self.frequency = steps[self.index % 5]
        return self.observation, {}

    def step(self, action):
        self.action=action
        self.observation["received"][action] +=  min(self.observation["cur_val"],self.observation["required"][action])
        self.observation["valuation"]=[0] * self.grid_size
        if(self.observation["required"][action]!=0):
            self.observation["valuation"][action] =     (min(self.observation["cur_val"],self.observation["required"][action]) / self.observation["required"][action]) *100
            self.previous_valuations[action] =          (min(self.observation["cur_val"],self.observation["required"][action]) / self.observation["required"][action]) *100
        self.bedarf_(frequency=[[2, 2, 2], [3, 3, 1], [1, 1, 1, 4], [1, 3, 2, 3], [1, 1, 1, 5]],count=self.steps)
        
        #trained: consider both: requirement + received                             check   
        #auch im training unterschiedliche settings                                 
        #one agent always max                                                       check
        #vergleich zwischen trainiert + det. auf allen trainierten settings         


        #next steps: bedarf für tischzuteilung (anzahl) + nichtverfügbarkeiten (an x tagen nicht zuweisbar)
        #modell: verteilung an x agenten (anpassen)
        #gemeinsamer bedarf für agentj & agenti --> min aus bedarfen
        #
        
        self.gini_index = calculate_gini(self.previous_valuations)
        self.reward = getattr(self, self.method)(action)
        #-np.var(self.observation["count_rec"])
        #TO DO: static (Basis)
        #for trained models: one class, reward as python function
        #netzarchitekturen für unterschiedliche anzahl agenten (graph/transformer) stable-baselines? (custom nets) deep sets (ohne index/reihenfolge)
        #
       
        self.steps += 1
        terminated = self.steps >= 50 
        truncated=False
        self.info={}
        self.info["required"] = self.observation["required"]
        self.info["received"] = self.observation["received"]
        self.info["valuation"] = sum(self.previous_valuations)
        self.info["value"] = self.observation["cur_val"]
        self.info["gini"] = self.gini_index
        self.info["sum_rec"] = sum(self.observation["received"])
        self.info["previous_valuations"] = self.previous_valuations
        self.previous_valuations
        #value[0] = np.random.randint(1, 5) #aktuelle verteilung
        self.observation["cur_val"] = np.random.randint(1, 5) #aktuelle verteilung
        
        return self.observation, self.reward, terminated, truncated, self.info
    
    def bedarf(self):
        #bedarf=[]
        bedarf = np.zeros(self.grid_size)
        for agent in range(self.grid_size):
           # bedarf[agent] = np.random.poisson(5, size=1)
           #verschiedene agenten (z.B. niedrig/hoch konstant/periodisch, )
            bedarf[agent] = max(0,(5 - (self.previous_valuations[agent]//20))*np.random.poisson(100, size=1)/100)
            self.previous_valuations[agent] = max(0, self.previous_valuations[agent]-20)
        self.observation["required"]=bedarf

    def bedarf_(self, frequency, count, value=30):
        bedarf = np.zeros(self.grid_size)
        for agent in range(self.grid_size):
            count_agent = (count-1)%len(frequency[agent])
            bedarf[agent] =  max(0,int(gauss(frequency[agent][count_agent] , value)))
           #verschiedene agenten (z.B. niedrig/hoch konstant/periodisch, )
            #bedarf[agent] = (5 - (self.previous_valuations[agent]//20))*np.random.poisson(100, size=1)/100
            self.previous_valuations[agent] = max(0, self.previous_valuations[agent]-20)
        self.observation["required"]=bedarf

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"reward:{self.reward} Predicted action: {self.action} received: {self.info['received']}, valuation: {self.info['valuation']} value: {self.info['value']}")

    def close(self):
        pass



    def worst_received(self, action):
        return (-self.observation["received"][action])
    
    def round_robin(self, action):
        return -np.var(self.observation["count_rec"])
    
    def greedy(self, action):
        return self.observation["required"][action]
    
    def bedarf_received(self, action):
        if(self.observation["received"][action]!=0):
            return self.observation["required"][action]/self.observation["received"][action]
        else:
            return self.observation["required"][action]
    
    def gini_env(self, action):
        return -100*self.gini_index + sum(self.observation["received"]) + sum(self.observation["valuation"]) 
    #print valuation (check)
    #poission verteilung

    #

def test(file_dir, log_dir, models_dir, method, iterations=100, start=[1,2,3,4,5], model_name=PPO):
    env = DummyVecEnv([lambda: GiniEnv(grid_size=len(start), render_mode='console', start=start, method=method, log_dir=log_dir, file_dir=file_dir, models_dir=models_dir)],start_method="fork") 
    #train()

    #test
    #test_env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')])
    model = PPO.load(file_dir, env=env)
    

    obs = env.reset()

    writer = SummaryWriter(log_dir)
    for step in range(iterations):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f' reward:{reward} Predicted action: {action} received: {info[-1]["received"]}, valuation: {info[-1]["valuation"]} value: {info[-1]["value"]}') 
        writer.add_scalar("Test/Reward", reward, step)
        writer.add_scalar("Test/Gini_Index", calculate_gini(info[-1]["previous_valuations"]), step)
        writer.add_scalar("Test/Sum_rec", info[-1]["sum_rec"], step)
        writer.add_scalar("Test/Valuation", info[-1]["valuation"], step)
        env.render()
        if done.any(): 
            print("reward", reward, "last call of episode", info[-1]["terminal_observation"], "Gini Index: ", calculate_gini(info[-1]["received"]))  # always use last element
            print("Episode finished.")
            #experiment_id = "c1KCv3X3QvGwaXfgX1c4tg"
            #experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
            #df = experiment.get_scalars()
            #df  
            break

def train(method, logdir, models_dir, timesteps=1000000):
    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[1, 2, 3, 4, 5], method=method)],start_method="fork") 
    #model = PPO("MlpPolicy", env, verbose=1).learn(1000000)
    model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir)
    model.learn(total_timesteps=timesteps, tb_log_name="train_test", callback=TensorboardCallback())
    model.save(models_dir)


#https://www.datacamp.com/tutorial/optuna
#https://github.com/araffin/rl-handson-rlvs21/blob/main/optuna/sb3_simple.py
def objective(trial: optuna.Trial, method: str) -> float:
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

    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[1, 2, 3, 4, 5], method=method, models_dir=models_dir, file_dir=file_dir, log_dir=logdir)])

    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(ppo_hyper_params(trial))

    kwargs = {key: value for key, value in kwargs.items() if key != "policy"}

    model = PPO("MultiInputPolicy", env, verbose=0, **kwargs)

    eval_envs = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[1, 2, 3, 4, 5], method=method, models_dir=models_dir, file_dir=file_dir, log_dir=logdir)])

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
    """Sample PPO hyperparameters for Optuna trial."""
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
