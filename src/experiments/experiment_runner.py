import os
import sys
import sqlite3
import logging
import pickle
import datetime
from datetime import timedelta
from os.path import dirname

from stable_baselines3.common.callbacks import EvalCallback

import optuna

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src folder to Python path
src = dirname(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../experiments'))
sys.path.append(src)
# Import custom modules
from experiment import Experiment, parse_json
from utilitarian import prepare_utilitarian_runner
from runners.envy_freeness import (
    prepare_envy_free_runner,
    prepare_envy_min_runner,
    ENVY_PAIRS,
    envy_freeness_mixin,
    enforce_envy_freeness,
    SHARE_FUNCTION,
)
from runners.simple_runner import SimpleRunner
from runners.leximin_runner import (
    prepare_leximin_runner,
    LeximinRunner,
)
from pareto_runner import (
    ParetoRunner,
    pareto_only_nondom_mixin,
    ParetoUtilityTracker,
)
from runners.rawls import prepare_rawls_runner
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
result_dir = os.path.join('\\results')

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

models_dir = "src\\experiments\\trained_models\\trained_mz"
file_dir = "src\\experiments\\trained_models\\trained_mz.zip"
logdir = "src\\experiments\\trained_models\\logs"

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

def objective(trial: optuna.Trial, method: str, experiment: Experiment) -> float:
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

    env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start="generic_preferences.dzn", experiment=experiment, experiment_runner=experiment_runner)])

    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(ppo_hyper_params(trial))
    kwargs = {key: value for key, value in kwargs.items() if key != "policy"}

    model = PPO("MultiInputPolicy", env, verbose=0, **kwargs)

    eval_envs = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start="generic_preferences.dzn", experiment=experiment, experiment_runner=experiment_runner)])

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

        simple_agents = Model(self.experiment.path + self.experiment.model_inst[0])
       # simple_agents.add_file("src\\stable_baselines\\temp\\"+str(self.index%3)+"_"+self.start,parse_data=True)
        gecode = Solver.lookup("gecode")
        #chuffed vergleich
        #agenten nach training permutieren --> experiment
        #normierung des environments (boolean flag im init)
        instance = Instance(gecode, simple_agents)
        weights=action

        
        
        with instance.branch() as inst:          
            model = Model()
            #base_dir = os.path.join(os.path.dirname(__file__), "")
            model_file = self.experiment.path + self.experiment.model_inst[0]
            model.add_file(model_file)
           # model.add_file('src\\stable_baselines\\temp\\0_generic_preferences.dzn')
            social_mapping_file = os.path.join(self.experiment.path,"social_mapping.json")
            social_mapping = read_social_mapping(social_mapping_file)

            for data_file in self.experiment.model_inst[1]:
                data_file_path = self.experiment.path + "data\\" + data_file
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
        writer = SummaryWriter(logdir)
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
           
            N_TRIALS = 100
            N_JOBS = 1
            N_STARTUP_TRIALS = 5
            TIMEOUT = int(60 * 15)

            pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)
            sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
            study = optuna.create_study(sampler=sampler, storage="sqlite:///db.sqlite3", pruner=pruner, direction="maximize")
            try:
                study.optimize(lambda trial: objective(trial, "gini_env", experiment), n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
            except KeyboardInterrupt:
                pass

            print("Number of finished trials: ", len(study.trials))

            print("Best trial:")
            trial = study.best_trial

            print(f"  Value: {trial.value}")

            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")


           #env = GiniEnv(grid_size=5, render_mode='console', start="generic_preferences.dzn", experiment=experiment)
           #train(env)
           #env.test(iterations=10, filedir=file_dir, start="generic_preferences.dzn", model_name=PPO)

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

import sqlite3

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


'''class ExperimentRunner:

    def __init__(self, database_name) -> None:
        self.database_name = database_name

    def run_experiment(self, experiment : Experiment):
        pickle_output = os.path.join(result_dir, experiment.get_result_filename())

        if os.path.exists(pickle_output) and not FORCE_OVERRIDE:
            with open(pickle_output, 'rb') as handle:
                db_result = pickle.load(handle)
            print("Already exists")
            return db_result
        
        model = Model()
        base_dir = os.path.join(os.path.dirname(__file__), experiment.path)
        model_file = os.path.join(base_dir, experiment.model_inst[0])
        model.add_file(model_file)

        social_mapping_file = os.path.join(base_dir, "social_mapping.json")
        social_mapping = read_social_mapping(social_mapping_file)

        for data_file in experiment.model_inst[1]:
            data_file_path = os.path.join(base_dir, "data/"+ data_file)
            model.add_file(data_file_path, parse_data=True)

        solver = Solver.lookup(experiment.solver)
        start_time = datetime.datetime.now()
        result : Result = configurations_map[experiment.configuration](model, social_mapping, solver)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        if result and result.status != Status.UNKNOWN:
            utils = result[social_mapping[UTILITY_ARRAY]]
            db_result = {"model": experiment.problem, "data_files" : "".join(experiment.model_inst[1]), 
                        "utilities" : utils, "max_utility" : max(utils), "min_utility" : min(utils), "sum_utility" : sum(utils),
                        "solving_runtime" : elapsed_time.total_seconds() , 
                        "solver" : experiment.solver, "configuration" : experiment.configuration,
                        "envy_pairs" : result[ENVY_PAIRS] if hasattr(result.solution, ENVY_PAIRS) else None}
            

            # write a pickle file 
            with open(pickle_output, 'wb') as handle:
                pickle.dump(db_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            # write to database
            insert_into_results(self.database_name, db_result)
        return result 
    

    def run_all_experiments(self, experiments):
        for experiment in experiments:
            experiment : Experiment = experiment
            logging.info("------------- Running experiment ... "+ experiment.get_identifier())
            try :
                self.run_experiment(experiment)
            except:
                logging.info("An error occurred, continuing")'''


if __name__ == "__main__":
    import os 
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    database_name = os.path.join(result_dir, 'test.db')
    create_database(database_name)
    print(f"Database '{database_name}' created successfully.")

    filename =  os.path.join(os.path.dirname(__file__), 'test.json')    
    experiments = parse_json(filename)

    experiment_runner = ExperimentRunner(database_name)
    #env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', experiment=experiments[0], experiment_runner=ExperimentRunner(database_name)),]) 
    
   # train(env)
    #env = GiniEnv(grid_size=5, render_mode='console', experiment=experiments[0], experiment_runner=experiment_runner)
    #env.test()
    experiment_runner.run_all_experiments(experiments)





    
