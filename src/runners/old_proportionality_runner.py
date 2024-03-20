import logging
from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os

from utilitarian_runner import add_utilitarian_objective 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from functools import partial

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS



def add_proportionality_objective(social_mapper, instance : Instance):
    #instance.add_string(f"var int: {UTILITARIAN_OBJECTIVE};")
    instance.add_string(f"constraint forall (p in 1..dimension) (max_values[p] = max([utilities[i, p] | i in 1..m]) / dimension);")
    instance.add_string(f"constraint forall (p in 1..dimension) (selected[p] >= max_values[p]);")
    instance.add_string(f"constraint exists(i in 1..m) (forall(j in 1..dimension) (selected[j] = utilities[i, j]));")

def optimize_proportionality_objective(instance : Instance):
    instance.add_string(f"solve satisfy;")
    
'''
A utilitarian runner maximizes the sum of utilities; 
it therefore needs to plugin a new objective to the base model
'''
class ProportionalityRunner(SimpleRunner):
    def __init__(self) -> None:
        super().__init__()
        pass

    def run(self, model, solver=...):
        self.model = model
        return super().run(model, solver)



if __name__ == "__main__":
    import os
    social_selection_model = Model(os.path.join(os.path.dirname(__file__), '../mzn_playground/test_proportionality.mzn'))
    #social_selection_model.add_file("test/rawls2.dzn", parse_data=True)
    social_selection_model["m"] = 6
    social_selection_model["dimension"] = 4  
    social_selection_model["utilities"] = [[98, 30, 98, 98],[98, 30, 98, 99],[42, 37, 80, 12],[20, 90, 60, 30],[99, 30, 30, 99],[1,  1,  1,  1]]
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/social_selection/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    simple_runner = ProportionalityRunner()
    #simple_runner.add_presolve_handler(social_mapping)
    #simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    #simple_runner.add_presolve_handler(optimize_proportional_objective)
    result = simple_runner.run(social_selection_model, gecode)
    print(result["selected"])