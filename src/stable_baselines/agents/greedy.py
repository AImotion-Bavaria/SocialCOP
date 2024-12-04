import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append('..')
from logging_tensorboard import TensorboardCallback
from gymnasium.spaces import Dict, Box, Discrete

models_dir = "src/stable-baselines/agents/models/greedy"
file_dir = "src/stable-baselines/agents/models/greedy.zip"
logdir = "src/stable-baselines/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def calculate_gini(array):
    if np.all(array==0):
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
        self.observation_space = Dict({
            "required": Box(low=0, high=6, shape=(self.grid_size,), dtype=int),
            "received": Box(low=0, high=500, shape=(self.grid_size,), dtype=int),
            "valuation": Box(low=0, high=101, shape=(self.grid_size,), dtype=int),
            "cur_val": Discrete(5)
        })
        self.render_mode = render_mode 
        self.previous_valuations = np.zeros(self.grid_size)
        self.reset()

    def reset(self, seed=None, options=None ):
        self.index = 0
        self.observation = {
            "required": np.random.randint(0, 5, size=self.grid_size).astype(np.int32),
            "received": np.zeros(self.grid_size),
            "valuation": np.zeros(self.grid_size, dtype=np.int32),
            "cur_val": np.random.randint(1, 5)
        }
        self.gini_index = calculate_gini(self.observation["received"])
        self.steps = 0
        return self.observation, {}

    def step(self, action):

        self.observation["received"][action] +=  min(self.observation["cur_val"],self.observation["required"][action])
        self.observation["valuation"]=[0] * self.grid_size
        if(self.observation["required"][action]!=0):
            self.observation["valuation"][action] =     (min(self.observation["cur_val"],self.observation["required"][action]) / self.observation["required"][action]) *100
            self.previous_valuations[action] =          (min(self.observation["cur_val"],self.observation["required"][action]) / self.observation["required"][action]) *100
        
        
        
        self.gini_index = calculate_gini(self.observation["received"])
        reward = self.observation["required"][action] 
       
        self.steps += 1
        terminated = self.steps >= 50 
        truncated=False
        info={}
        info["required"] = self.observation["required"]
        info["received"] = self.observation["received"]
        info["valuation"] = self.observation["valuation"]
        info["value"] = self.observation["cur_val"]
        info["gini"] = self.gini_index
        info["sum_rec"] = sum(self.observation["received"])
        #value[0] = np.random.randint(1, 5) #aktuelle verteilung
        self.observation["cur_val"] = np.random.randint(1, 5) #aktuelle verteilung
        self.bedarf()
        return self.observation, reward, terminated, truncated, info
    
    def bedarf(self):
        bedarf = np.zeros(self.grid_size)
        for agent in range(self.grid_size):
            bedarf[agent] = 5 - (self.previous_valuations[agent]//20)
            self.previous_valuations[agent] = max(0, self.previous_valuations[agent]-20)
        self.observation["required"]=bedarf

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"reward:{reward} Predicted action: {action} received: {info[-1]['received']}, valuation: {info[-1]['valuation']} value: {info[-1]['value']}")

    def close(self):
        pass

def train():
    #model = PPO("MlpPolicy", env, verbose=1).learn(1000000)
    model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir)
    model.learn(total_timesteps=1000000, tb_log_name="greedy", callback=TensorboardCallback())
    model.save(models_dir)
env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')]) 
train()

#test
#test_env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')])
model = PPO.load(file_dir, env=env)

obs = env.reset()
writer = SummaryWriter("src/stable-baselines/logs/greedy")
for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f' reward:{reward} Predicted action: {action} received: {info[-1]["received"]}, valuation: {info[-1]["valuation"]} value: {info[-1]["value"]}') 
    writer.add_scalar("Test/Reward", reward, step)
    writer.add_scalar("Test/Gini_Index", calculate_gini(info[-1]["received"]), step)
    writer.add_scalar("Test/Sum_rec", info[-1]["sum_rec"], step)
    env.render()
    if done.any():  
        print("reward", reward, "last call of episode", info[-1]["terminal_observation"], "Gini Index: ", calculate_gini(info[-1]["received"]))  # always use last element
        print("Episode finished.")
        break


 

