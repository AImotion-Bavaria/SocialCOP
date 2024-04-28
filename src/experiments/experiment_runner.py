import sys
from os.path import dirname
import logging
import pickle 
import datetime
from datetime import timedelta

#Add src folder to python path
import os

src = dirname(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))
sys.path.append(src)

from experiment import Experiment, parse_json
from simple_runner import SimpleRunner
from minizinc import Model, Solver, Result, Status
from utilitarian import prepare_utilitarian_runner
from envy_freeness import prepare_envy_free_runner, prepare_envy_min_runner, ENVY_PAIRS, envy_freeness_mixin, enforce_envy_freeness, SHARE_FUNCTION
from leximin_runner import prepare_leximin_runner
from pareto_runner import ParetoRunner
from rawls import prepare_rawls_runner

from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY
from util.mzn_debugger import create_debug_folder

base_dir = os.path.dirname(__file__)
result_dir = os.path.join(base_dir, 'results')

FORCE_OVERRIDE = True  # use cached versions if false
TIME_LIMIT_EVAL = timedelta(hours=1.0)

def rawls(model: Model, social_mapping, solver: Solver):
    simple_runner = prepare_rawls_runner(social_mapping)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result

def utilitarian(model : Model, social_mapping : dict, solver : Solver):
    simple_runner = prepare_utilitarian_runner(social_mapping)
    if SHARE_FUNCTION in social_mapping: # it is a division problem - I want to record envy counts as well
        simple_runner.add(envy_freeness_mixin)
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(model, solver)
    return result

# everything that is associated to envy-freeness has to be a division problem
def utilitarian_envy_free(model : Model, social_mapping : dict, solver : Solver):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
     
    simple_runner : SimpleRunner = prepare_utilitarian_runner(social_mapping)
    simple_runner.add(envy_freeness_mixin)
    simple_runner.add(enforce_envy_freeness)
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(model, solver)
    return result

def envy_min(model : Model, social_mapping : dict, solver : Solver):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_min_runner(social_mapping)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result

def envy_free(model : Model, social_mapping : dict, solver : Solver):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_free_runner(social_mapping)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result

def leximin(model: Model, social_mapping, solver: Solver):
    simple_runner = prepare_leximin_runner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = create_debug_folder(os.path.dirname(__file__))
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result

configurations_map = {
      "rawls" : rawls,
      "leximin": leximin,
      "utilitarian" : utilitarian,
      "leximin_pareto":  rawls,
      "utilitarian_envy_free":utilitarian_envy_free,
      "envy_free": envy_free,
      "envy_min": envy_min
}

import sqlite3

# Function to create a new SQLite database and initialize the table
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


class ExperimentRunner:

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
            self.run_experiment(experiment)

if __name__ == "__main__":
    import os 
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    database_name = os.path.join(result_dir, 'results_database.db')
    create_database(database_name)
    print(f"Database '{database_name}' created successfully.")

    filename =  os.path.join(os.path.dirname(__file__), 'envy_util_experiment.json')    
    experiments = parse_json(filename)

    experiment_runner = ExperimentRunner(database_name)
    experiment_runner.run_all_experiments(experiments)
