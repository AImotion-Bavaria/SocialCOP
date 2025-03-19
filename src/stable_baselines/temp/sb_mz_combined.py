import gymnasium as gym
from minizinc import Instance, Model, Solver
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append('..')
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True
from gymnasium.spaces import Dict, Box, Discrete

models_dir = "src/stable_baselines/temp/models/greedy"
file_dir = "src/stable_baselines/temp/models/greedy.zip"
logdir = "src/stable_baselines/temp/logs"

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

    def __init__(self, grid_size=5, render_mode=None, start="generic_preferences.dzn"):
        super(GiniEnv, self).__init__()
        self.start=start
        self.grid_size = grid_size

        self.action_space = Box(low=0, high=100, shape=(5,), dtype=np.int32)
        #needs to be n_agents
        self.observation_space = Dict({
            "required": Box(low=0, high=6, shape=(self.grid_size,), dtype=int),
            "received": Box(low=0, high=500, shape=(self.grid_size,), dtype=int),
            "valuation": Box(low=0, high=101, shape=(self.grid_size,), dtype=int)
        })
        self.render_mode = render_mode 
        self.previous_valuations = np.zeros(self.grid_size)
        self.reset()

    def reset(self, seed=None, options=None ):
        self.index = 0
        
        self.observation = {
                "required": np.zeros(self.grid_size, dtype=int),
                "received": np.zeros(self.grid_size),
                "valuation": np.zeros(self.grid_size, dtype=np.int32),
        }

        self.gini_index = calculate_gini(self.observation["valuation"])
        self.steps = 0
        return self.observation, {}

    def step(self, action):
        self.action=action


        simple_agents = Model("src/stable_baselines/temp/table_assignment_generic.mzn")
        filenametest = "src/stable_baselines/temp/"+str(self.index%3)+"_"+self.start
        #simple_agents.add_file("src/stable_baselines/temp/"+str(self.index%3)+self.start,parse_data=True)
        simple_agents.add_file("src/stable_baselines/temp/"+str(self.index%3)+"_"+self.start,parse_data=True)
        # Find the MiniZinc solver configuration for Gecode
        gecode = Solver.lookup("gecode")
        instance = Instance(gecode, simple_agents)
        # Create an Instance of the simple agents model for Gecode

        
        weights = action
        
        with instance.branch() as inst:
            inst["Agents"] = {1,2,3,4,5}
            #inst["m"] = m_python
            inst["weights"] = [int(weight) for weight in weights]
            print(f"... Solving with weights: {weights}")
            result = inst.solve()
            # Output the array selected
            print(result["assigned"])

            # every 0 gets one more in weights
        self.observation["valuation"] += result["utilities"]
        for agent in range(self.grid_size):
            self.observation["received"][agent]=sum(1 for x in result["assigned"][agent] if x != "NoTable")    
        #self.observation["received"]=result["assigned"]
        #for agent in range(self.grid_size):
        ##hier fairness metriken einfügen
            #self.reward=self.observation["valuation"][agent]+self.observation["received"][agent] 
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.reward = - 100*self.gini_index
        
        #self.observation["received"][index] +=  min(self.observation["cur_val"],self.observation["required"][index])
       # self.observation["valuation"]=[0] * self.grid_size
        #if(self.observation["required"][index]!=0):
         #   self.observation["valuation"][index] =     (min(self.observation["cur_val"],self.observation["required"][index]) / self.observation["required"][index]) *100
          #  self.previous_valuations[index] =          (min(self.observation["cur_val"],self.observation["required"][index]) / self.observation["required"][index]) *100
        
        
        
        self.gini_index = calculate_gini(self.observation["valuation"])
        
        #greedy policy (fix)
       
        self.steps += 1
        terminated = self.steps >= 50 
        truncated=False
        self.info={}
        self.info["received"] = self.observation["received"]
        self.info["valuation"] = self.observation["valuation"]
        self.info["gini"] = self.gini_index
        self.info["sum_rec"] = sum(self.observation["received"])
        #value[0] = np.random.randint(1, 5) #aktuelle verteilung
        #self.greedy()
        return self.observation, self.reward, terminated, truncated, self.info
    
    

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"reward:{self.reward} Predicted action: {self.action} received: {self.info['received']}, valuation: {self.info['valuation']}")

    def close(self):
        pass
        
    def greedy(self):
        """
        Get the index of the agent with the highest required value.

        Returns:
        int: Index of the agent with the highest required value.
        """

        self.observation["required"] = np.random.randint(0, 100, size=self.grid_size)
        array = np.zeros(self.grid_size)
        array[np.argmax(self.observation["required"], axis=None, out=None)]=100
        return array
    

    
    def test(self,iterations=10, filedir=file_dir, start="generic_preferences.dzn", model_name=PPO):
        env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=start)]) 
        model = PPO.load(file_dir, env=env)

        obs = env.reset()
        writer = SummaryWriter("src/stable_baselines/temp/logs/greedy_trained")
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f' reward:{reward} Predicted action: {action} received: {info[-1]["received"]}, valuation: {info[-1]["valuation"]} ') 
            writer.add_scalar("Test/Reward", reward, step)
            writer.add_scalar("Test/Gini_Index", calculate_gini(info[-1]["valuation"]), step)
            writer.add_scalar("Test/Sum_rec", info[-1]["sum_rec"], step)
            env.render()
            if done.any():  
                print("reward", reward, "last call of episode", info[-1]["terminal_observation"], "Gini Index: ", calculate_gini(info[-1]["valuation"]))  # always use last element
                print("Episode finished.")
                break

def train():
    #model = PPO("MlpPolicy", env, verbose=1).learn(1000000)
    model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir)
    model.learn(total_timesteps=100, tb_log_name="greedy", callback=TensorboardCallback())
    model.save(models_dir)
env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start="generic_preferences.dzn")]) 
if __name__ == "__main__":
    env = GiniEnv()

    train()
    #env.test()




 

