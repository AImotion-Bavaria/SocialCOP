import json
from experiments.experiment import Experiment
from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY, AGENTS_ARRAY, MAIN_VARIABLES, TIME_SPAN, NUM_AGENTS, UTILITY_UPPER_BOUND, UTILITY_LOWER_BOUND
RAWLS_OBJECTIVE = "rawls_objective"

def rawls_objective( instance : Instance, social_mapper, experiment: Experiment=None):
    instance.add_string(f"var int: {RAWLS_OBJECTIVE};")
    instance.add_string(f"constraint {RAWLS_OBJECTIVE} = min({social_mapper[UTILITY_ARRAY]});")

def optimize_rawls_objective(instance : Instance, social_mapper):
    instance.add_string(f"solve maximize ({RAWLS_OBJECTIVE});")

def optimize_rawls_objective_weights_old(instance: Instance, social_mapper):
    # Declare the weights array
    #instance.add_string(f"array[{(social_mapper[AGENTS_ARRAY])}] of int: weights;")
    #array of float ausprobieren
    
    # Assign the weights array values
      # Pass the weights from Python to MiniZinc
    
    # Define the total_weight variable
    #instance.add_string(f"constraint {RAWLS_OBJECTIVE} >= 0;")
    #instance.add_string("constraint total_weight >= 0;")
    #instance.add_string(f"constraint {social_mapper[NUM_AGENTS]} > 0;")
    #instance.add_string(f"constraint {social_mapper[UTILITY_UPPER_BOUND]} > 0;")
    #instance.add_string(f"var int: total_weight = sum(a in {social_mapper[AGENTS_ARRAY]}) ( {social_mapper[UTILITY_ARRAY]}[a] * weights[a]);")

    instance.add_string("var int: normalized_rawls;")
    instance.add_string(f"constraint normalized_rawls * {social_mapper[UTILITY_UPPER_BOUND]} = 100 * {RAWLS_OBJECTIVE};")
 # upperbound / lowerbound from minizinc model
    #maybe calculate beforehand
    #check similar to shared_decision
    
    instance.add_string(f"array[{(social_mapper[AGENTS_ARRAY])}] of int: weights;")
    instance.add_string(f"var int: total_weight = sum(a in {social_mapper[AGENTS_ARRAY]}) ( {social_mapper[UTILITY_ARRAY]}[a] * weights[a]);")
    #instance.add_string(f"solve maximize ({RAWLS_OBJECTIVE} + 1000*total_weight);")
    #instance.add_string("var int: normalized_total_weight;")
    instance.add_string(f"constraint normalized_total_weight * ({social_mapper[NUM_AGENTS]} * 200) = 100 * total_weight;")
    instance.add_string(f"var int: normalized_total_weight = (100 * total_weight) div ({social_mapper[NUM_AGENTS]} * 200);") # read from file
    instance.add_string(f"solve maximize normalized_rawls + normalized_total_weight;")
    # Declare the weights array


def optimize_rawls_objective_weights(instance: Instance, social_mapper):
    instance.add_string(f"array[{(social_mapper[AGENTS_ARRAY])}] of int: weights;")
    instance.add_string(f"var int: {social_mapper[UTILITY_UPPER_BOUND]};")
    instance.add_string(f"var int: max_weight = {social_mapper[NUM_AGENTS]} * {social_mapper[UTILITY_UPPER_BOUND]}*100;")
    instance.add_string(f"var int: total_weight = sum(a in {social_mapper[AGENTS_ARRAY]}) ( {social_mapper[UTILITY_ARRAY]}[a] * weights[a]);")
    instance.add_string(f"var int: normalized_total_weight = (total_weight) * {social_mapper[UTILITY_UPPER_BOUND]};")
    instance.add_string(f"var int: normalized_rawls = {RAWLS_OBJECTIVE} * max_weight;")
    instance.add_string(f"solve maximize (normalized_rawls + normalized_total_weight);")


    """instance.add_string(f"var 0..100: normalized_rawls;")
    instance.add_string(f"var int: {social_mapper[UTILITY_UPPER_BOUND]};")
    instance.add_string(f"constraint normalized_rawls * {social_mapper[UTILITY_UPPER_BOUND]} = 100 * {RAWLS_OBJECTIVE};")
    instance.add_string(f"array[{(social_mapper[AGENTS_ARRAY])}] of int: weights;")
    instance.add_string(f"var 0..1000: total_weight = sum(a in {social_mapper[AGENTS_ARRAY]}) ( {social_mapper[UTILITY_ARRAY]}[a] * weights[a]);")
    instance.add_string(f"constraint normalized_total_weight * ({social_mapper[NUM_AGENTS]} * 200) = 100 * total_weight;")
    instance.add_string(f"var 0..100: normalized_total_weight = (100 * total_weight) div ({social_mapper[NUM_AGENTS]} * 200);") # read from file
    instance.add_string(f"solve maximize normalized_rawls + normalized_total_weight;")"""
    


def prepare_rawls_runner(social_mapping, use_weights=False, experiment: Experiment=None):
    simple_runner = SimpleRunner(social_mapping, experiment)
    simple_runner.add(rawls_objective) 
    if use_weights:
        simple_runner.add(optimize_rawls_objective_weights)
        #simple_runner.add(optimize_rawls_objective_weights)
        
    else:
       
        simple_runner.add(optimize_rawls_objective)
    return simple_runner

if __name__ == "__main__":
    import os
    plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.mzn'))
    plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)


    simple_runner = SimpleRunner(social_mapping)
    simple_runner.add(rawls_objective)
    simple_runner.add(optimize_rawls_objective)
    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)
