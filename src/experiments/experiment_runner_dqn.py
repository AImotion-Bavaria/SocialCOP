import json
import os
import sys
import sqlite3
import logging
import pickle
import datetime
from os.path import dirname
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, A2C, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
#from experiment_runner_det import GiniEnvDQN
from hyperparam import ppo_hyper_params, a2c_hyper_params, sac_hyper_params, dqn_hyper_params
from fairness_models import (
    none,
    rawls,
    leximin,
    utilitarian,
    utilitarian_envy_free,
    envy_free,
    envy_min,
    leximin_pareto
)

# Add src folder to Python path
src = dirname(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../experiments'))
sys.path.append(src)
# Import custom modules
from experiment import Experiment, parse_json
from runners.envy_freeness import ENVY_PAIRS
from util.social_mapping_reader import UTILITY_LOWER_BOUND, UTILITY_UPPER_BOUND, read_social_mapping, UTILITY_ARRAY, ASSIGNED, REQUIRED

# Import external libraries
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Discrete, Box

from torch.utils.tensorboard import SummaryWriter
from logging_tensorboard import TensorboardCallback
from minizinc import Model, Solver, Instance, Result, Status
import numpy as np
from gini import calculate_gini

# Define base and result directories
base_dir = os.path.dirname(__file__)
result_dir = os.path.join('src/experiments/06_07/results')

# Constants
FORCE_OVERRIDE = True  # Use cached versions if False


models_dir = "src/experiments/06_07/trained_models_none/trained_mz"
file_dir = "src/experiments/06_07/trained_models_none/trained_mz.zip"
logdir = "src/experiments/01_07/results"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

configurations_map = {
      "none": none,
      "rawls" : rawls,  #check
      "leximin": leximin, #check
      "utilitarian" : utilitarian, #check
      "leximin_pareto":  leximin_pareto, #check
      "utilitarian_envy_free":utilitarian_envy_free, #check
      "envy_free": envy_free, #check
      "envy_min": envy_min #check
}

def objective(trial: optuna.Trial, experiment: Experiment) -> float:
    #mit weniger zeitschritten --> dann 
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    trial (optuna.Trial): Optuna trial object.
    method (str): Method to use for training.

    Returns:
    float: Mean reward of the best trial.
    """
    N_TIMESTEPS = experiment.iterations# int(2e4)
    N_EVALUATIONS = 10
    EVAL_FREQ = max(1, int(N_TIMESTEPS / N_EVALUATIONS)) 
    N_EVAL_EPISODES = 10

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    DEFAULT_HYPERPARAMS = {
        "policy": "MultiInputPolicy",
    }


    #env = DummyVecEnv([lambda: GiniEnvDQN(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)])
    #env = GiniEnvDQN(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)

    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Dynamically select the correct hyperparameter function based on the model name
    model_hyper_params_func = {
        "ppo": ppo_hyper_params,
        "a2c": a2c_hyper_params,
        "sac": sac_hyper_params,
        "dqn": dqn_hyper_params
    }.get(experiment.model_names.lower())
    if model_hyper_params_func is not None:
        kwargs.update(model_hyper_params_func(trial))
    kwargs = {key: value for key, value in kwargs.items() if key != "policy"}

    model_class = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "dqn": DQN
    }.get(experiment.model_names.lower())
    if model_class is None:
        raise ValueError(f"Unknown model name: {experiment.model_names}")
    model_kwargs = dict(verbose=0, **kwargs)
   # if experiment.model_names.lower() in ["ppo", "a2c"]:
    #    model_kwargs["n_epochs"] = 5
    
    eval_envs = GiniEnvDQN(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)
    model = model_class("MultiInputPolicy", eval_envs, **model_kwargs)
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=models_dir+"/"+experiment.get_identifier_short(),
        log_path=logdir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    try:
        nan_encountered = False
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    except Exception as e:
        print("Exception during learning:", e)
        return float("nan")
    finally:
        model.env.close()
        eval_envs.close()

    if nan_encountered:
        return float("nan")

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return eval_callback.best_mean_reward


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

class GiniEnvDQN(gym.Env):

    #eine klasse für beide (detEnv erbt von GiniEnvDQN)
    metadata = {'render.modes': ['console']}


    def __init__(self, experiment, experiment_runner, render_mode=None,  ):
        super(GiniEnvDQN, self).__init__()
        self.experiment = experiment
        self.experiment_runner = experiment_runner

        self.action_space = gym.spaces.Discrete((2**self.experiment.num_agents)-1)

        #self.action_space = Discrete(low=0, high=100, shape=(self.experiment.num_agents,), dtype=np.int32) # evtl. continouus 
        #needs to be n_agents
        self.observation_space = Dict({
            "required": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64),
            "received": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64),
            "valuation": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64),
            "overall_valuation": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64)
        })
        self.render_mode = render_mode 
        #self.previous_valuations = np.zeros(self.experiment.num_agents)
        self.reset()

    def reset(self, seed=None, options=None ):
        self.index = 0 
        self.observation = {
                "required": np.zeros(self.experiment.num_agents, dtype=np.float64),
                "received": np.zeros(self.experiment.num_agents, dtype=np.float64),
                "valuation": np.zeros(self.experiment.num_agents, dtype=np.float64),
                "overall_valuation": np.zeros(self.experiment.num_agents, dtype=np.float64)
        }
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.steps = 0
        data_file_path = self.experiment.path + "data/" + self.experiment.model_inst[1][self.index%len(self.experiment.model_inst[1])]
        with open(data_file_path, 'r') as f:
            data_dict = json.load(f)
        social_mapping_file = os.path.join(self.experiment.path,"social_mapping.json")
        social_mapping = read_social_mapping(social_mapping_file)
        self.upper_bound = data_dict[social_mapping[UTILITY_UPPER_BOUND]]
        self.lower_bound = data_dict[social_mapping[UTILITY_LOWER_BOUND]] if social_mapping[UTILITY_LOWER_BOUND] in data_dict else 0
        # Convert list of dicts like [{"set": [5]}, ...] to a flat numpy array
        required_list = data_dict[social_mapping[REQUIRED]]
        if isinstance(required_list, list) and isinstance(required_list[0], dict) and "set" in required_list[0]:
            self.observation["required"] = np.array([item["set"][0] if isinstance(item["set"], list) else item["set"] for item in required_list], dtype=np.float64)
        else:
            self.observation["required"] = np.array(required_list, dtype=np.float64)
        return self.observation, {}

    def step(self, action):
        if action is None:
            self.action = np.zeros(self.experiment.num_agents, dtype=np.float64)
        else:
            self.action = bin(action)[2:].zfill(self.experiment.num_agents)  
            self.action = [int(bit) for bit in self.action]
            self.action = [bit * 100 for bit in self.action]



        #print(self.experiment.path + self.experiment.model_inst[0])
        simple_agents = Model(self.experiment.path + self.experiment.model_inst[0])
    
        solver = Solver.lookup(self.experiment.solver)
        #chuffed vergleich
        #agenten nach training permutieren --> experiment
        #normierung des environments (boolean flag im init)
        instance = Instance(solver, simple_agents)
        #weights=self.action
        
        
        with instance.branch() as inst:          
            #model = Model()
            #base_dir = os.path.join(os.path.dirname(__file__), "")
            #model_file = self.experiment.path + self.experiment.model_inst[0]
            #model.add_file(model_file)
           # model.add_file('src/stable_baselines/temp/0_generic_preferences.dzn')
            social_mapping_file = os.path.join(self.experiment.path,"social_mapping.json")
            social_mapping = read_social_mapping(social_mapping_file)

          
            data_file_path = self.experiment.path + "data/" + self.experiment.model_inst[1][self.index%len(self.experiment.model_inst[1])]
            simple_agents.add_file(data_file_path, parse_data=True)

            solver = Solver.lookup(self.experiment.solver)
            start_time = datetime.datetime.now()
            weights=self.action
        result : Result = configurations_map[self.experiment.configuration](simple_agents, social_mapping, solver, weights=weights)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        

       
        if result and result.status != Status.UNKNOWN:
            utils = result[social_mapping[UTILITY_ARRAY]]
            #self.observation["required"] = result[social_mapping[REQUIRED]]
            #vorher notwendig!!! json file als dzn --> zugriff auf beides
            #parse dzn = True
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

        #valuation = result.solution.utilities
            valuation = result[social_mapping[UTILITY_ARRAY]]
            max_val = self.upper_bound # result[social_mapping[UTILITY_UPPER_BOUND]] #allgemeines maximum finden
            min_val = self.lower_bound
            if max_val > min_val:
                # Normalize the valuation to be between 0 and 1
                valuation = (np.array(valuation) - min_val) / (max_val - min_val)
            else:
                valuation = np.array(valuation) / (max_val)
            self.observation["valuation"] = valuation

            #(0.5*self.observation["valuation"]) + result[social_mapping[UTILITY_ARRAY]]
            
            self.observation["received"] = result[social_mapping[ASSIGNED]] 
            self.gini_index = calculate_gini(self.observation["valuation"])
            
            #self.observation["required"] = result[social_mapping[REQUIRED]]
        else:
            utils = np.zeros(self.experiment.num_agents, dtype=np.float64)
            #self.observation["required"] = result[social_mapping[REQUIRED]]
            #vorher notwendig!!! json file als dzn --> zugriff auf beides
            #parse dzn = True
            
            valuation = np.zeros(self.experiment.num_agents, dtype=np.float64)

            #(0.5*self.observation["valuation"]) + result[social_mapping[UTILITY_ARRAY]]
            
            self.observation["received"] = np.zeros(self.experiment.num_agents, dtype=np.float64)
            self.gini_index = 2
            
            #self.observation["required"] = result[social_mapping[REQUIRED]]
       
        #self.observation["valuation"] = valuation
        self.observation["overall_valuation"] =  0.5*(self.observation["overall_valuation"]) + 0.5 * valuation #bounded:  summe aus beiden % = 1
        
            #self.reward = 0
            #for i in range(self.experiment.num_agents):
            #   if self.observation["required"][i] > 0:
            #      self.reward= self.reward + 100*(self.observation["received"][i]/self.observation["required"][i])
            #self.reward = self.reward -100*self.gini_index
        self.reward = (-1000*self.gini_index)+1000
        
        
        #self.gini_index = calculate_gini(self.observation["valuation"])
        
    
        self.steps += 1
        terminated = self.steps >= self.experiment.iterations
        truncated=False
        self.info={}
        self.info["received"] = self.observation["received"]
        self.info["valuation"] = self.observation["valuation"]
        self.info["gini"] = self.gini_index
        self.info["sum_rec"] = sum(self.observation["received"])
        self.info["overall_valuation"] = self.observation["overall_valuation"]
        self.index += 1
        with open(data_file_path, 'r') as f:
            data_dict = json.load(f)
        required_list = data_dict[social_mapping[REQUIRED]]
        if isinstance(required_list, list) and isinstance(required_list[0], dict) and "set" in required_list[0]:
            self.observation["required"] = np.array([item["set"][0] if isinstance(item["set"], list) else item["set"] for item in required_list], dtype=np.float64)
        else:
            self.observation["required"] = np.array(required_list, dtype=np.float64)
        print("action: " + str(action) + "valuation" + str(self.observation["valuation"]) +" reward :"+str(self.reward)+"\n\n")
        return self.observation, self.reward, terminated, truncated, self.info
    
    

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"reward:{self.reward} Predicted action: {self.action} received: {self.info['received']}, valuation: {self.info['valuation']}")

    def close(self):
        pass
        
    
    
    def test(self,iterations=10,  method="worst_received"):
        env = GiniEnvDQN(render_mode='console', experiment=self.experiment, experiment_runner=self.experiment_runner)
        #model = PPO.load(file_dir, env=env)
        unique_logdir = os.path.join(logdir, f"{method}_{self.experiment.solver}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        obs = env.reset()
        writer = SummaryWriter(unique_logdir, filename_suffix=method)
        for step in range(50):
                action = getattr(self, method)()
                observation, reward, terminated, truncated, info = env.step(action)
                print(f' reward:{reward} Predicted action: {action} received: {info["received"]}, valuation: {info["valuation"]} ') 
                writer.add_scalar("Test/Reward", reward, step)
                writer.add_scalar("Test/Gini_Index", calculate_gini(info["valuation"]), step)
                writer.add_scalar("Test/Gini_Index_Overall", calculate_gini(info["overall_valuation"]), step)
                writer.add_scalar("Test/Sum_rec", info["sum_rec"], step)
                env.render()
                if terminated or truncated:  
                    print("reward", reward)
                    print("Gini Index: ", calculate_gini(info["valuation"]))  # always use last element
                    print("Episode finished.")
                    break

    def worst_received(self):
        return_array= np.zeros(self.experiment.num_agents)
        return_array[np.argmin(self.observation["received"], axis=None, out=None)]=1
        return return_array
    
    def none(self):
        return_array= np.zeros(self.experiment.num_agents)
        return return_array
    
    def round_robin(self):
        self.index += 1
        return_array= np.zeros(self.experiment.num_agents)
        return_array[(self.index - 1) % self.experiment.num_agents]=1
        return return_array
    
    def greedy(self):
        return_array = np.zeros(self.experiment.num_agents, dtype=int)
        return_array[np.argmax(self.observation["required"], axis=None, out=None)]=1
        return return_array
    
    def bedarf_received(self):
        received_safe = np.where(self.observation["received"] == 0, 1, self.observation["received"])
        value = max(self.observation["required"] / received_safe)
        return_array= np.zeros(self.experiment.num_agents)
        return_array[np.argmax(self.observation["required"] == value)]=1
        return return_array
        
    
    
def test(render_mode, experiment, experiment_runner):
        env = DummyVecEnv([lambda: GiniEnvDQN(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)]) 
        final_model_path = os.path.join(models_dir, f"{experiment.get_identifier_short()}","best_model.zip")
        #final_model_path = os.path.join("/home/ruttmann/projects/SocialCOP/src/experiments/26_06/trained_models/new/trained_mz/table_assignment_chuffed_rawls_P_P_O_best_model.zip")
        #hier model richtig einfügen
        #!!!!
        model_class = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "dqn": DQN
        }.get(experiment.model_names.lower())
        if model_class is None:
            raise ValueError(f"Unknown model name: {experiment.model_names}")
   # if experiment.model_names.lower() in ["ppo", "a2c"]:
    #    model_kwargs["n_epochs"] = 5
        model = model_class.load(final_model_path, env=env)
        #model = PPO.load(final_model_path, env=env)
        unique_logdir = os.path.join(logdir, f"{experiment.solver}_{experiment.model_names}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        obs = env.reset()
        writer = SummaryWriter(unique_logdir)
        for step in range(50):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                print(f' reward:{reward} Predicted action: {action} received: {info[-1]["received"]}, valuation: {info[-1]["valuation"]} ') 
                writer.add_scalar("Test/Reward", reward, step)
                writer.add_scalar("Test/Gini_Index", calculate_gini(info[-1]["valuation"]), step)
                writer.add_scalar("Test/Gini_Index_Overall", calculate_gini(info[-1]["overall_valuation"]), step)
                writer.add_scalar("Test/Sum_rec", info[-1]["sum_rec"], step)
                env.render()
                if done.any():  
                    print("reward", reward, "last call of episode", info[-1]["terminal_observation"], "Gini Index: ", calculate_gini(info[-1]["valuation"]))  # always use last element
                    print("Episode finished.")
                    break

#def train(env):
 #   model = PPO('MultiInputPolicy', env, verbose=1, ent_coef=0.1, tensorboard_log=logdir, n_steps=52, batch_size=52, n_epochs=10)
  #  model.learn(total_timesteps=1000, tb_log_name="greedy", callback=TensorboardCallback())
   # model.save(models_dir)


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
                logging.info("An error occurred, continuing - "+ experiment.get_identifier())
    

    def run_experiment(self, experiment: Experiment):
        pickle_output = result_dir + "/" + experiment.get_result_filename()

        if os.path.exists(pickle_output) and not FORCE_OVERRIDE:
            with open(pickle_output, 'rb') as handle:
                db_result = pickle.load(handle)
            print("Already exists")
            return db_result

        N_TRIALS = 20
        N_JOBS = 10
        N_STARTUP_TRIALS = 10

        pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        study = optuna.create_study(
            study_name=f"10_07_{experiment.get_identifier_short()}",
            sampler=sampler,
            storage="sqlite:///db.sqlite3",
            pruner=pruner,
            direction="maximize",
            load_if_exists=True
        )

        try:
            study.optimize(lambda trial: objective(trial, experiment), n_trials=N_TRIALS, n_jobs=N_JOBS)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials:", len(study.trials))
        print("Best trial value:", study.best_trial.value)
        print("Best trial params:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        # Rebuild environment
        env = GiniEnvDQN(render_mode='console', experiment=experiment, experiment_runner=self)

        # Prepare hyperparameters
        best_params = study.best_trial.params.copy()
        best_params = {k: v for k, v in best_params.items() if k != "policy"}

        DEFAULT_HYPERPARAMS = {
            "policy": "MultiInputPolicy",
        }
        kwargs = DEFAULT_HYPERPARAMS.copy()

        model_hyper_params_func = {
            "ppo": ppo_hyper_params,
            "a2c": a2c_hyper_params,
            "sac": sac_hyper_params,
            "dqn": dqn_hyper_params
        }.get(experiment.model_names.lower())
        if model_hyper_params_func is not None:
            kwargs.update(model_hyper_params_func(study.best_trial))

        kwargs = {key: value for key, value in kwargs.items() if key != "policy"}

        model_class = {
            "ppo": PPO,
            "a2c": A2C,
            "sac": SAC,
            "dqn": DQN
        }.get(experiment.model_names.lower())
        if model_class is None:
            raise ValueError(f"Unknown model name: {experiment.model_names}")

        model_kwargs = dict(verbose=0, **kwargs)
        model = model_class("MultiInputPolicy", env, **model_kwargs)

        # Use EvalCallback here exactly like in Optuna
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(models_dir, f"{experiment.get_identifier_short()}"),
            log_path=logdir,
            eval_freq=max(1, int(10000 / 10)),  # or use experiment.iterations if preferred
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        )

        # Final training with consistent callback
        model.learn(total_timesteps=100, callback=eval_callback, tb_log_name=experiment.get_identifier_short(), progress_bar=True)

        # Load the true best model saved by EvalCallback
        final_best_model_path = os.path.join(models_dir, f"{experiment.get_identifier_short()}", "best_model.zip")
        if not os.path.exists(final_best_model_path):
            # fallback in case no best model was saved
            print("Warning: No best model found, saving current model instead.")
            final_best_model_path = os.path.join(models_dir, f"{experiment.get_identifier_short()}_best_model.zip")
            model.save(final_best_model_path)
        else:
            print(f"Final best model saved at: {os.path.abspath(final_best_model_path)}")


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



if __name__ == "__main__":
    """ import os 
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    database_name = os.path.join(result_dir, '08.db')
    create_database(database_name)
    print(f"Database '{database_name}' created successfully.")

    filename =  os.path.join(os.path.dirname(__file__), 'test_none_dqn.json')    
    experiments = parse_json(filename)

    experiment_runner = ExperimentRunner(database_name)
    #env = DummyVecEnv([lambda: GiniEnvDQN( render_mode='console', experiment=experiments[0], experiment_runner=ExperimentRunner(database_name)),]) 
    
    #env.train()   #env = GiniEnvDQN(render_mode='console', experiment=experiments[0], experiment_runner=experiment_runner)
    #for experiment in experiments:
     #   print("Running experiment: ", experiment.get_identifier())
    #test(render_mode='console', experiment=experiments[0], experiment_runner=experiment_runner)
    experiment_runner.run_all_experiments(experiments)"""
    import os 
    logging.basicConfig(level=logging.INFO)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    database_name = os.path.join(result_dir, '01_07.db')
    create_database(database_name)
    print(f"Database '{database_name}' created successfully.")
    filename =  os.path.join(os.path.dirname(__file__), 'test_single_configs.json')    
    experiments = parse_json(filename)
    experiment_runner = ExperimentRunner(database_name)
    for experiment in experiments:
        test(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)






