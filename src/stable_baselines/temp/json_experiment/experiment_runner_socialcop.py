import os
import sys
import sqlite3
import logging
import pickle
import datetime
from datetime import timedelta
from os.path import dirname

# Add src folder to Python path
src = dirname(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(src)

# Import custom modules
from stable_baselines.temp.json_experiment.experiment import Experiment, parse_json
from stable_baselines.temp.json_experiment.utilitarian import prepare_utilitarian_runner
from stable_baselines.temp.json_experiment.envy_freeness import (
    prepare_envy_free_runner,
    prepare_envy_min_runner,
    ENVY_PAIRS,
    envy_freeness_mixin,
    enforce_envy_freeness,
    SHARE_FUNCTION,
)
from stable_baselines.temp.json_experiment.simple_runner2 import SimpleRunner
from stable_baselines.temp.json_experiment.leximin_runner import (
    prepare_leximin_runner,
    LeximinRunner,
)
from stable_baselines.temp.json_experiment.pareto_runner import (
    ParetoRunner,
    pareto_only_nondom_mixin,
    ParetoUtilityTracker,
)
from stable_baselines.temp.json_experiment.rawls import prepare_rawls_runner
from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY
from util.mzn_debugger import create_debug_folder

# Import external libraries
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from logging_tensorboard import TensorboardCallback
from minizinc import Model, Solver, Instance, Result, Status
import numpy as np

# Define base and result directories
base_dir = os.path.dirname(__file__)
result_dir = os.path.join('src\\stable_baselines\\temp\\json_experiment\\results')

# Constants
FORCE_OVERRIDE = True  # Use cached versions if False
TIME_LIMIT_EVAL = timedelta(hours=1.0)



def rawls(model: Model, social_mapping, solver: Solver, weights):
    simple_runner = prepare_rawls_runner(social_mapping, use_weights=True)
    #simple_runner.add(set_weights)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model=model, solver=solver, weights=weights)
    print("Result: ", result)
    return result

def utilitarian(model : Model, social_mapping : dict, solver : Solver, weights):
    simple_runner = prepare_utilitarian_runner(social_mapping, use_weights=True)
    if SHARE_FUNCTION in social_mapping: # it is a division problem - I want to record envy counts as well
        simple_runner.add(envy_freeness_mixin)
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

# everything that is associated to envy-freeness has to be a division problem
def utilitarian_envy_free(model : Model, social_mapping : dict, solver : Solver, weights):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
     
    simple_runner : SimpleRunner = prepare_utilitarian_runner(social_mapping, use_weights=True)
    simple_runner.model += [envy_freeness_mixin, enforce_envy_freeness]
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def envy_min(model : Model, social_mapping : dict, solver : Solver, weights):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_min_runner(social_mapping, use_weights=True)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def envy_free(model : Model, social_mapping : dict, solver : Solver, weights):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_free_runner(social_mapping, use_weights=True)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def leximin(model: Model, social_mapping, solver: Solver, weights):
    simple_runner = prepare_leximin_runner(social_mapping, use_weights=True)
    simple_runner.debug = True
    simple_runner.debug_dir = create_debug_folder(os.path.dirname(__file__))
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def leximin_pareto(model: Model, social_mapping, solver: Solver, weights):
    simple_runner : LeximinRunner = prepare_leximin_runner(social_mapping, use_weights=True)
    simple_runner.debug = True
    simple_runner.debug_dir = create_debug_folder(os.path.dirname(__file__))
    simple_runner.timeout = TIME_LIMIT_EVAL

    simple_runner.model += [pareto_only_nondom_mixin] 
    # also need a pareto tracker
    pareto_tracker = ParetoUtilityTracker()
    simple_runner.presolve_step += [pareto_tracker.write_previous_utilities]
    simple_runner.on_result += [pareto_tracker.update_previous_utilities]
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result


def calculate_gini(array):
    if np.all(array==0):
        return 0
    array = np.sort(np.array(array)).astype(np.float16)  # Cast to sorted numpy array
    index = np.arange(1, array.shape[0] + 1)  # Index per array element
    n = array.shape[0]  # Number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

models_dir = "src\\stable_baselines\\temp\\models\\trained_mz"
file_dir = "src\\stable_baselines\\temp\\models\\trained_mz.zip"
logdir = "src\\stable_baselines\\temp\\logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

configurations_map = {
      "rawls" : rawls,  #check
      "leximin": leximin, #check
      "utilitarian" : utilitarian, #check
      "leximin_pareto":  leximin_pareto, #check
      "utilitarian_envy_free":utilitarian_envy_free, #check
      "envy_free": envy_free, #check
      "envy_min": envy_min #check
}

def insert_into_results(database_name, db_result):
    conn = sqlite3.connect(database_name)
    with conn:
        cursor = conn.cursor()
        
        # Insert a new row into the Results table
        cursor.execute('''INSERT INTO Results 
                        (timestamp, model, data_files, utility_vector, max_utility, min_utility, sum_utility, solving_runtime, solver, configuration, envy_pairs)
                        VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (db_result["model"], db_result["data_files"], f'{db_result["utilities"]}', db_result["max_utility"],
                        db_result["min_utility"], db_result["sum_utility"], db_result["solving_runtime"], db_result["solver"], db_result["configuration"], db_result["envy_pairs"]))
        
        conn.commit() 

class GiniEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, experiment, experiment_runner, grid_size=5, render_mode=None, start="generic_preferences.dzn",  ):
        super(GiniEnv, self).__init__()
        self.start=start
        self.grid_size = grid_size
        self.experiment = experiment
        self.experiment_runner = experiment_runner

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
        action=(action*100).astype(int)
        self.action=action


        simple_agents = Model("src\\stable_baselines\\temp\\table_assignment_generic.mzn")
       # simple_agents.add_file("src\\stable_baselines\\temp\\"+str(self.index%3)+"_"+self.start,parse_data=True)
        gecode = Solver.lookup("gecode")
        #chuffed vergleich
        #agenten nach training permutieren --> experiment
        #normierung des environments (boolean flag im init)
        instance = Instance(gecode, simple_agents)
        weights=action

        
        
        with instance.branch() as inst:          
            model = Model()
            base_dir = os.path.join(os.path.dirname(__file__), "")
            model_file = 'src\\stable_baselines\\temp\\table_assignment_generic.mzn'
            model.add_file(model_file)
           # model.add_file('src\\stable_baselines\\temp\\0_generic_preferences.dzn')
            social_mapping_file = os.path.join("src\\stable_baselines\\temp\\json_experiment","social_mapping.json")
            social_mapping = read_social_mapping(social_mapping_file)

            for data_file in self.experiment.model_inst[1]:
                data_file_path = "src\\stable_baselines\\temp\\"+ data_file
                model.add_file(data_file_path, parse_data=True)

            solver = Solver.lookup(self.experiment.solver)
            start_time = datetime.datetime.now()
            weights=action
        result : Result = configurations_map[self.experiment.configuration](model, social_mapping, solver, weights=weights)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        if result and result.status != Status.UNKNOWN:
            utils = result[social_mapping[UTILITY_ARRAY]]
            db_result = {"model": self.experiment.problem, "data_files" : "".join(self.experiment.model_inst[1]), 
                        "utilities" : utils, "max_utility" : max(utils), "min_utility" : min(utils), "sum_utility" : sum(utils),
                        "solving_runtime" : elapsed_time.total_seconds() , 
                        "solver" : self.experiment.solver, "configuration" : self.experiment.configuration,
                        "envy_pairs" : result[ENVY_PAIRS] if hasattr(result.solution, ENVY_PAIRS) else None}
            
            pickle_output = os.path.join(result_dir, self.experiment.get_result_filename())
            # write a pickle file 
            with open(pickle_output, 'wb') as handle:
                pickle.dump(db_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            # write to database
            create_database(self.experiment_runner.database_name)
            insert_into_results(self.experiment_runner.database_name, db_result)


        self.observation["valuation"] += result["utilities"]
        for agent in range(self.grid_size):
            self.observation["received"][agent]=sum(1 for x in result["assigned"][agent] if x != "NoTable")    
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.reward = -100*self.gini_index
        
        self.gini_index = calculate_gini(self.observation["valuation"])
        
       
        self.steps += 1
        terminated = self.steps >= 52 
        truncated=False
        self.info={}
        self.info["received"] = self.observation["received"]
        self.info["valuation"] = self.observation["valuation"]
        self.info["gini"] = self.gini_index
        self.info["sum_rec"] = sum(self.observation["received"])
        self.index += 1
        return self.observation, self.reward, terminated, truncated, self.info
    
    

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"reward:{self.reward} Predicted action: {self.action} received: {self.info['received']}, valuation: {self.info['valuation']}")

    def close(self):
        pass
        
    
    
    def test(self,iterations=10, filedir=file_dir, start="generic_preferences.dzn", model_name=PPO):
        env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=start, experiment=self.experiment, experiment_runner=self.experiment_runner)]) 
        model = PPO.load(file_dir, env=env)

        obs = env.reset()
        writer = SummaryWriter("src\\stable_baselines\\temp\\logs\\greedy_trained")
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

def train(env):
    model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir, n_steps=52, batch_size=52, n_epochs=10)
    model.learn(total_timesteps=1000, tb_log_name="greedy", callback=TensorboardCallback())
    model.save(models_dir)


class ExperimentRunner:

    def __init__(self, database_name) -> None:
        self.database_name = database_name

    def run_all_experiments(self, experiments):
        for experiment in experiments:
            experiment : Experiment = experiment
            logging.info("------------- Running experiment ... "+ experiment.get_identifier())
            try :
                self.run_experiment(experiment)
            except:
                logging.info("An error occurred, continuing")
    

    def run_experiment(self, experiment : Experiment):
        pickle_output = result_dir + "\\"+ experiment.get_result_filename()

        if os.path.exists(pickle_output) and not FORCE_OVERRIDE:
            with open(pickle_output, 'rb') as handle:
                db_result = pickle.load(handle)
            print("Already exists")
            return db_result
        for experiment in experiments:
           env = GiniEnv(grid_size=5, render_mode='console', start="generic_preferences.dzn", experiment=experiment)
           train(env)
           env.test(iterations=10, filedir=file_dir, start="generic_preferences.dzn", model_name=PPO)

def create_database(database_name):
    conn = sqlite3.connect(database_name)
    with conn:
        cursor = conn.cursor()
    
        # Create Results table
        cursor.execute('''CREATE TABLE IF NOT EXISTS Results (
                            timestamp TEXT,
                            model TEXT,
                            data_files TEXT,
                            utility_vector TEXT,
                            max_utility INTEGER,
                            min_utility INTEGER,
                            sum_utility INTEGER,
                            solving_runtime REAL,
                            solver TEXT,
                            configuration TEXT,
                            envy_pairs INTEGER
                        )''')
        
        conn.commit()


if __name__ == "__main__":
    import os 
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    database_name = os.path.join(result_dir, 'test.db')
    create_database(database_name)
    print(f"Database '{database_name}' created successfully.")

    filename =  'src\\stable_baselines\\temp\\json_experiment\\test.json'    
    experiments = parse_json(filename)

    #experiment_runner = ExperimentRunner(database_name)
    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', experiment=experiments[0], experiment_runner=ExperimentRunner(database_name)),]) 
    
    train(env)
    env = GiniEnv(grid_size=5, render_mode='console', experiment=experiments[0], experiment_runner=ExperimentRunner(database_name))
    env.test()
    #experiment_runner.run_all_experiments(experiments)





    
