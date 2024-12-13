from custom_env_gini import GiniEnv
from worst_received import GiniEnv as WorstReceivedEnv
from bedarf_received import GiniEnv as BedarfReceivedEnv
from greedy import GiniEnv as GreedyEnv
from round_robin import GiniEnv as RoundRobinEnv
from json_reader import read_json_file, get_substitution_dictionary

import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from logging_tensorboard import TensorboardCallback


#logdir = "src/stable-baselines/logs"

def calculate_gini(array):
    if np.all(array==0):
        return 0
    array = np.sort(np.array(array)).astype(np.float16)  # Cast to sorted numpy array
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]  # Number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

def gini_env():
    models_dir = "src/stable-baselines/agents/models/gini"
    file_dir = "src/stable-baselines/agents/models/gini.zip"
    return models_dir, file_dir, GiniEnv  
def worst_received_env():
    models_dir = "src/stable-baselines/agents/models/worst_received"
    file_dir = "src/stable-baselines/agents/models/worst_received.zip"
    return models_dir, file_dir, WorstReceivedEnv  
def bedarf_received_env():
    models_dir = "src/stable-baselines/agents/models/bedarf_received"
    file_dir = "src/stable-baselines/agents/models/bedarf_received.zip"
    return  models_dir, file_dir, BedarfReceivedEnv
def greedy_env():
    models_dir = "src/stable-baselines/agents/models/greedy"
    file_dir = "src/stable-baselines/agents/models/greedy.zip"
    return  models_dir, file_dir , GreedyEnv 
def round_robin_env():
    models_dir = "src/stable-baselines/agents/models/round_robin"
    file_dir = "src/stable-baselines/agents/models/round_robin.zip"
    return  models_dir, file_dir,RoundRobinEnv  
    

def train_env(env,model_dir, total_timesteps=1000000, tb_log_name="gini", logdir="src/stable-baselines/logs", model_name="PPO"):
    import importlib
    module_name = "stable_baselines3"
    module = importlib.import_module(module_name)  
    model_class = getattr(module, model_name)  
    model = model_class('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=TensorboardCallback())
    model.save(model_dir)

def run_env(env,file_dir,environment, model_name):
    model = model_name.load(file_dir, env=env)
    
    obs = env.reset()
    writer = SummaryWriter("src/stable-baselines/logs/"+str(environment))
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f' reward:{reward} Predicted action: {action} received: {info[-1]["received"]}, valuation: {info[-1]["valuation"]} value: {info[-1]["value"]}') 
        writer.add_scalar("Test/Reward", reward, step)
        writer.add_scalar("Test/Gini_Index", calculate_gini(info[-1]["received"]), step)
        env.render()
        if done.any():  
            print("count:", info[-1]["count"], "reward", reward, "last call of episode", info[-1]["terminal_observation"], "Gini Index: ", calculate_gini(info[-1]["received"]))  # always use last element
            print("Episode finished.")
            break
    
#env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console')]) 


class Runner:
    def __init__(self) -> None:
        return
        

    def train_experiment(self, environment, total_timesteps, grid_size, model_name):
        
        models_dir, file_dir, env_name = globals()[environment]()
        env = DummyVecEnv([lambda: env_name(grid_size=grid_size, render_mode='console')])
        
        train_env(env=env,model_dir=models_dir,total_timesteps=total_timesteps,tb_log_name=environment, model_name=model_name),

    def run_experiment(self, environment, iterations=100, start_value=None, model_name="PPO"):
        
        models_dir, file_dir, env_name = globals()[environment]()
        #env = DummyVecEnv([lambda: env_name(grid_size=grid_size, render_mode='console')])
        #run_env(env=env,file_dir=file_dir,environment=environment)
        import importlib
        module_name = "stable_baselines3"
        module = importlib.import_module(module_name)  
        my_class = getattr(module, model_name)  
        env_name.test(environment, iterations, file_dir, start=start_value, model_name=my_class)

if __name__ == "__main__":
    runner = Runner()

    sub_dict = get_substitution_dictionary(read_json_file(".\\src\\stable_baselines\\agents\\test.json"))
    #trainloop:
    for agent in sub_dict["train"]:
        runner.train_experiment(agent["train_agent"],agent["iterations"], agent["grid_size"],  model_name=agent["model_name"])
        print("\n\n\n\n")
    #testloop:
    for agent in sub_dict["test"]:
        runner.run_experiment(agent["test_agent"],agent["iterations"],start_value=agent["start_value"], model_name=agent["model_name"])
        print("\n\n\n\n")


