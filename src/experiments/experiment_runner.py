import sys
from os.path import dirname
import logging
import pickle 

#Add src folder to python path
import os

src = dirname(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))
sys.path.append(src)

from experiment import Experiment, parse_json
from simple_runner import SimpleRunner
from minizinc import Model, Solver
from utilitarian import add_utilitarian_objective, optimize_utilitarian_objective
from envy_freeness import add_envy_freeness_mixin, optimize_envy, enforce_envy_freeness
from leximin_runner import LeximinRunner
from pareto_runner import ParetoRunner

from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY
from util.mzn_debugger import create_debug_folder

base_dir = os.path.dirname(__file__)
result_dir = os.path.join(base_dir, 'results')

FORCE_OVERRIDE = True  # use cached versions if false

def rawls(model: Model, social_mapping, solver: Solver):
    pass

def utilitarian(model : Model, social_mapping : dict, solver : Solver):
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.add_presolve_handler(add_utilitarian_objective)
    simple_runner.add_presolve_handler(optimize_utilitarian_objective)

    result = simple_runner.run(model, solver)
    return result

configurations_map = {
      "rawls" : rawls,
      "leximin": rawls,
      "utilitarian" : utilitarian,
      "leximin_pareto":  rawls,
      "utilitarian_envy_free":rawls,
      "envy_free": rawls,
      "envy_min":rawls
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
                            configuration TEXT
                        )''')
        
        conn.commit()

import sqlite3

def insert_into_results(database_name, db_result):
    conn = sqlite3.connect(database_name)
    with conn:
        cursor = conn.cursor()
        
        # Insert a new row into the Results table
        cursor.execute('''INSERT INTO Results 
                        (timestamp, model, data_files, utility_vector, max_utility, min_utility, sum_utility, solving_runtime, solver, configuration)
                        VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (db_result["model"], db_result["data_files"], f'{db_result["utilities"]}', db_result["max_utility"],
                        db_result["min_utility"], db_result["sum_utility"], db_result["solving_runtime"], db_result["solver"], db_result["configuration"]))
        
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

        # now for the configurations: 
        result = configurations_map[experiment.configuration](model, social_mapping, solver)
        utils = result[social_mapping[UTILITY_ARRAY]]
        db_result = {"model": experiment.problem, "data_files" : "".join(experiment.model_inst[1]), 
                     "utilities" : utils, "max_utility" : max(utils), "min_utility" : min(utils), "sum_utility" : sum(utils),
                      "solving_runtime" : result.statistics["solveTime"].total_seconds() , 
                      "solver" : experiment.solver, "configuration" : experiment.configuration}
        
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

    filename =  os.path.join(os.path.dirname(__file__), 'test.json')    
    experiments = parse_json(filename)

    experiment_runner = ExperimentRunner(database_name)
    experiment_runner.run_all_experiments(experiments)
