from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from functools import partial

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY
UTILITARIAN_OBJECTIVE = "utilitarian_objective"

def add_utilitarian_objective(social_mapper, instance : Instance):
    instance.add_string(f"var int: {UTILITARIAN_OBJECTIVE};")
    instance.add_string(f"constraint {UTILITARIAN_OBJECTIVE} = sum({social_mapper[UTILITY_ARRAY]});")

def  optimize_utilitarian_objective(instance : Instance):
    instance.add_string(f"solve maximize {UTILITARIAN_OBJECTIVE};")
    
'''
A utilitarian runner maximizes the sum of utilities; 
it therefore needs to plugin a new objective to the base model
'''
class UtilitarianRunner(SimpleRunner):
    def __init__(self) -> None:
        super().__init__()
        pass

if __name__ == "__main__":
    import os
    social_selection_model = Model(os.path.join(os.path.dirname(__file__), '../models/social_selection/social_selection.mzn'))
    n_agents = 5
    social_selection_model["m"] = 3
    social_selection_model["n"] = n_agents 
    social_selection_model["weights"] = [1+i for i in range(n_agents)]
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/social_selection/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    simple_runner = UtilitarianRunner()
    simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    simple_runner.add_presolve_handler(optimize_utilitarian_objective)
    result = simple_runner.run(social_selection_model, gecode)
    print(result["selected"])