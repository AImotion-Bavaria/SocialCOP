import os
import sys
import sqlite3
import logging
import pickle
import datetime
from os.path import dirname
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from hyperparam import ppo_hyper_params
from fairness_models import (
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
from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY, ASSIGNED, REQUIRED

# Import external libraries
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box

from torch.utils.tensorboard import SummaryWriter
from logging_tensorboard import TensorboardCallback
from minizinc import Model, Solver, Instance, Result, Status
import numpy as np
from gini import calculate_gini

# Define base and result directories
base_dir = os.path.dirname(__file__)
result_dir = os.path.join('src/experiments/results')

# Constants
FORCE_OVERRIDE = True  # Use cached versions if False


models_dir = "src/experiments/trained_models/new/trained_mz"
file_dir = "src/experiments/trained_models/new/trained_mz.zip"
logdir = "src/experiments/trained_models/new/logs"

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


    #env = DummyVecEnv([lambda: GiniEnv(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)])
    env = GiniEnv(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)

    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(ppo_hyper_params(trial))
    kwargs = {key: value for key, value in kwargs.items() if key != "policy"}

    model = PPO("MultiInputPolicy", env, verbose=0, **kwargs, n_epochs=5) # SAC / DQN=discrete

    #eval_envs = DummyVecEnv([lambda: GiniEnv(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)])
    eval_envs = GiniEnv(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)

    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path=models_dir,
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
    finally:
        model.env.close()
        eval_envs.close()

    if nan_encountered:
        return float("nan")

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


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


    def __init__(self, experiment, experiment_runner, render_mode=None,  ):
        super(GiniEnv, self).__init__()
        self.experiment = experiment
        self.experiment_runner = experiment_runner

        self.action_space = Box(low=-1, high=1, shape=(self.experiment.num_agents,), dtype=np.float64) # evtl. continouus 
        #needs to be n_agents
        self.observation_space = Dict({
            "required": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64),
            "received": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64),
            "valuation": Box(low=0, high=1, shape=(self.experiment.num_agents,), dtype=np.float64)
        })
        self.render_mode = render_mode 
        self.previous_valuations = np.zeros(self.experiment.num_agents)
        self.reset()

    def reset(self, seed=None, options=None ):
        self.index = 0 
        self.observation = {
                "required": np.zeros(self.experiment.num_agents, dtype=np.float64),
                "received": np.zeros(self.experiment.num_agents, dtype=np.float64),
                "valuation": np.zeros(self.experiment.num_agents, dtype=np.float64),
        }
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.steps = 0
        return self.observation, {}

    def step(self, action):
        #action=(action*100.0).astype(int)
        self.action=((action+1)*100).astype(int)
        #solver: highs oder cbc

        simple_agents = Model(self.experiment.path + self.experiment.model_inst[0])
    
        gecode = Solver.lookup("gecode")
        #chuffed vergleich
        #agenten nach training permutieren --> experiment
        #normierung des environments (boolean flag im init)
        instance = Instance(gecode, simple_agents)
        #weights=self.action
        
        
        with instance.branch() as inst:          
            model = Model()
            #base_dir = os.path.join(os.path.dirname(__file__), "")
            model_file = self.experiment.path + self.experiment.model_inst[0]
            model.add_file(model_file)
           # model.add_file('src/stable_baselines/temp/0_generic_preferences.dzn')
            social_mapping_file = os.path.join(self.experiment.path,"social_mapping.json")
            social_mapping = read_social_mapping(social_mapping_file)

          
            data_file_path = self.experiment.path + "data/" + self.experiment.model_inst[1][self.index%len(self.experiment.model_inst[1])]
            model.add_file(data_file_path, parse_data=True)

            solver = Solver.lookup(self.experiment.solver)
            start_time = datetime.datetime.now()
            weights=self.action
        result : Result = configurations_map[self.experiment.configuration](model, social_mapping, solver, weights=weights)
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

       
        if result and result.status != Status.UNKNOWN:
            utils = result[social_mapping[UTILITY_ARRAY]]
            self.observation["required"] = result[social_mapping[REQUIRED]]
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

        self.observation["valuation"] = (0.1*self.observation["valuation"]) + result[social_mapping[UTILITY_ARRAY]]
        min_val = np.min(self.observation["valuation"])
        max_val = np.max(self.observation["valuation"])
        if max_val > min_val:
            self.observation["valuation"] = (self.observation["valuation"] - min_val) / (max_val - min_val)
            #zwischen 0 und 1!!
            #random seed fixieren
            #action space normieren (-1 bis 1)

            #behaviour cloning
        

       # for agent in range(self.experiment.num_agents):
        #    self.observation["received"][agent]=sum(1 for x in result[social_mapping[MAIN_VARIABLES]][agent] if x != "NoTable")    
        self.observation["received"] = result[social_mapping[ASSIGNED]]
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.reward = 0
        for i in range(self.experiment.num_agents):
            if self.observation["required"][i] > 0:
                self.reward= self.reward + 100*(self.observation["received"][i]/self.observation["required"][i])
        self.reward = self.reward -100*self.gini_index
        
        
        #self.gini_index = calculate_gini(self.observation["valuation"])
        
       
        self.steps += 1
        terminated = self.steps >=9
        truncated=False
        self.info={}
        self.info["received"] = self.observation["received"]
        self.info["valuation"] = self.observation["valuation"]
        self.info["gini"] = self.gini_index
        self.info["sum_rec"] = sum(self.observation["received"])
        self.index += 1
        print("action: " + str(action) + "received" + str(self.observation["received"]) + "valuation" + str(self.observation["valuation"]) +" reward :"+str(self.reward)+"\n\n")
        return self.observation, self.reward, terminated, truncated, self.info
    
    

    def render(self, mode='console'):
        if self.render_mode == 'console':
            print(f"reward:{self.reward} Predicted action: {self.action} received: {self.info['received']}, valuation: {self.info['valuation']}")

    def close(self):
        pass
        
    
    
def test(render_mode, experiment, experiment_runner):
        env = DummyVecEnv([lambda: GiniEnv(render_mode='console', experiment=experiment, experiment_runner=experiment_runner)]) 
        final_model_path = os.path.join(models_dir, f"{experiment.get_identifier()}_best_model.zip")
        model = PPO.load(final_model_path, env=env)
        unique_logdir = os.path.join(logdir, f"{experiment.solver}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        obs = env.reset()
        writer = SummaryWriter(unique_logdir)
        for step in range(50):
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
                logging.info("An error occurred, continuing")
    

    def run_experiment(self, experiment : Experiment):
        pickle_output = result_dir + "/"+ experiment.get_result_filename()

        if os.path.exists(pickle_output) and not FORCE_OVERRIDE:
            with open(pickle_output, 'rb') as handle:
                db_result = pickle.load(handle)
            print("Already exists")
            return db_result
        for experiment in experiments:
           
            N_TRIALS = experiment.iterations
            N_JOBS = 1
            N_STARTUP_TRIALS = 1
            TIMEOUT = int(60 * 15)

            pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2)
            sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
            study = optuna.create_study(sampler=sampler, storage="sqlite:///db.sqlite3", pruner=pruner, direction="maximize")
            try:
                study.optimize(lambda trial: objective(trial, experiment), n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
               #study.optimize(lambda trial: objective(trial, experiment), n_jobs=N_JOBS, timeout=TIMEOUT)
            except KeyboardInterrupt:
                pass

            print("Number of finished trials: ", len(study.trials))

            print("Best trial:")
            trial = study.best_trial

            print(f"  Value: {trial.value}")

            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            
            # Rebuild environment
            env = GiniEnv(render_mode='console', experiment=experiment, experiment_runner=self)

            # Get best hyperparameters and remove 'policy' if present
            best_params = trial.params.copy()
            policy = "MultiInputPolicy"
            best_params = {k: v for k, v in best_params.items() if k != "policy"}

            # Train final model
            model = PPO(policy, env, verbose=1, **best_params)
            model.learn(total_timesteps=50)

            # Save the final model
            final_model_path = os.path.join(models_dir, f"{experiment.get_identifier()}_best_model.zip")
            model.save(final_model_path)

            print(f"Final trained model saved at: {os.path.abspath(final_model_path)}")



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
    env = GiniEnv(render_mode='console', experiment=experiments[0], experiment_runner=experiment_runner)
    #test(render_mode='console', experiment=experiments[0], experiment_runner=experiment_runner)
    experiment_runner.run_all_experiments(experiments)





    
