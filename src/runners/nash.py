from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance, Result
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from on_result_runner import OnResultRunner

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY
NASH_OBJECTIVE = "nash_objective"

def nash_objective(instance : Instance, social_mapper):
    instance.add_string(f"var int: {NASH_OBJECTIVE};")
    instance.add_string(f"constraint {NASH_OBJECTIVE} = product({social_mapper[UTILITY_ARRAY]});")

def optimize_nash_objective(instance : Instance, social_mapper = None):
    instance.add_string(f"solve maximize {NASH_OBJECTIVE};")

def get_better_nash(instance : Instance, res : Result, social_mapper = None):
    # enforce that the next solution needs to be better than the current one
    instance.add_string(f"constraint {NASH_OBJECTIVE} > {res[NASH_OBJECTIVE]};")

def prepare_nash_runner(social_mapping):
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.add(nash_objective)
    simple_runner.add(optimize_nash_objective)
    return simple_runner

if __name__ == "__main__":
    import os
    social_selection_model = Model(os.path.join(os.path.dirname(__file__), '../models/social_selection/social_selection.mzn'))
    n_agents = 5
    # we can also set some parameters to the Model instance right here in Python (or read from dzh)
    social_selection_model["m"] = 3
    social_selection_model["n"] = n_agents 
    social_selection_model["weights"] = [1+i for i in range(n_agents)]
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/social_selection/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    simple_runner = SimpleRunner(social_mapping)
    # insert all decision variables to calculate the utilitarian objective for this model
    simple_runner.add(nash_objective)
    # actually optimize for the utilitarian objective
    simple_runner.add(optimize_nash_objective)
    result = simple_runner.run(social_selection_model, gecode)
    print(result["selected"])

    # solve the same model using an on result runner 
    on_result_runner = OnResultRunner(social_mapping)
    # insert all decision variables to calculate the utilitarian objective for this model
    on_result_runner.add(nash_objective)
    # optimize for the utilitarian objective using on result constraints
    on_result_runner.add_on_result_handler(optimize_nash_objective)