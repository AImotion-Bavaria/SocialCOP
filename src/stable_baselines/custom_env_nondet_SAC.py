import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from logging_tensorboard import TensorboardCallback

models_dir = "src/stable-baselines/models/SAC_non_det"
file_dir = "src/stable-baselines/models/SAC_non_det.zip"
logdir = "src/stable-baselines/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def calculate_gini(array):
    if np.var(array)==0:
        return 0
    array = np.sort(np.array(array))  # Cast to sorted numpy array
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]  # Number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

class GiniEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, grid_size=10, render_mode=None):
        super(GiniEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=20, shape=(self.grid_size,), dtype=np.float32)
        self.render_mode = render_mode 
        self.reset()

    def reset(self, seed=None, options=None):
        self.index = 0
        self.array = np.random.randint(1, 20, size=self.grid_size).astype(np.float32)
        self.gini_index = calculate_gini(self.array)
        self.steps = 0
        return self.array, {}

    def step(self, action):
        
        self.array[self.index] = action
        self.index = (self.index + 1) % self.grid_size
        new_gini_index = calculate_gini(self.array)
        reward = -new_gini_index + 0.1 * sum(self.array)
        self.gini_index = new_gini_index
        self.steps += 1
        terminated = bool(self.gini_index < 0.01)
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
    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')]) 
    #model = PPO("MlpPolicy", env, verbose=1).learn(1000000)
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=100000, tb_log_name="SAC_non_det", callback=TensorboardCallback())
    model.save(models_dir)

train()

#test
test_env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')])
model = SAC.load(file_dir, env=test_env)

obs = test_env.reset()
for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    
    
    test_env.render()
    if done.any(): 
        print("last call of episode", info[-1]["terminal_observation"],"Gini Index: ", calculate_gini(info[-1]["terminal_observation"]))#always use last element
        print("Episode finished.")
        break
action = np.clip(action, 1, 20).astype(np.float32)