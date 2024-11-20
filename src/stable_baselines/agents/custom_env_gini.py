import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from logging_tensorboard import TensorboardCallback

models_dir = "src/stable-baselines/agents/models/PPO_non_det_gini"
file_dir = "src/stable-baselines/agents/models/PPO_non_det_gini.zip"
logdir = "src/stable-baselines/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def calculate_gini(array):
    if np.var(array)==0:
        return 0
    array = np.sort(np.array(array)).astype(np.float16)  # Cast to sorted numpy array
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]  # Number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

class GiniEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10, render_mode=None):
        super(GiniEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(self.grid_size) 
        #TO DO:
        #fairness policy return value 
        #maybe all values (no. tables, days, valuation function etc.) (but not recreation of solver)
        #bedarf vs. zufriedenheit
        #reward gini + summe erfüllter need

        self.observation_space = spaces.Box(low=0, high=500, shape=(self.grid_size,), dtype=np.int32)
        self.render_mode = render_mode 
        self.reset()

    def reset(self, seed=None, options=None ):
        self.index = 0
        self.array = [0] * self.grid_size
        self.array = np.array(self.array)
        self.gini_index = calculate_gini(self.array)
        self.steps = 0
        return self.array, {}

    def step(self, action):
        #action = np.clip(np.round(action), 0, self.grid_size-1).astype(np.int8)
        self.array[action] = self.array[action]+1 
        new_gini_index = calculate_gini(self.array)
        reward = - new_gini_index #- np.var(self.array)
        self.gini_index = new_gini_index
        self.steps += 1
        terminated = self.steps >= 50 
        truncated=False
        info={}
        info["terminal_observation"] = self.array

        return self.array, reward, terminated, truncated, info

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"Array: {self.array}, Gini Index: {self.gini_index:.4f}")

    def close(self):
        pass

def train():
    #model = PPO("MlpPolicy", env, verbose=1).learn(1000000)
    model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir)
    model.learn(total_timesteps=100000, tb_log_name="PPO_non_det_gini", callback=TensorboardCallback())
    model.save(models_dir)
env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')]) 
#train()

#test
#test_env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')])
model = PPO.load(file_dir, env=env)

obs = env.reset()
for step in range(50):

    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicted action: {action}") 
    obs, reward, done, info = env.step(action)
    
    
    env.render()
    if done.any(): 
        print("last call of episode", info[-1]["terminal_observation"],"Gini Index: ", calculate_gini(info[-1]["terminal_observation"]))#always use last element
        print("Episode finished.")
        break
