from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os

from utilitarian import utilitarian_objective, optimize_utilitarian_objective 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from functools import partial

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS


def add_proportionality_objective(instance : Instance, social_mapper):
    instance.add_string(f"int: m = card(index_set_1of2(possible_solutions));")
    instance.add_string(f"array[1..{social_mapper[NUM_AGENTS]}] of var float: max_values;")
    instance.add_string(f"array[1..{social_mapper[NUM_AGENTS]}] of var int: selected;")
    instance.add_string(f"constraint forall (p in 1..{social_mapper[NUM_AGENTS]}) (max_values[p] = max([possible_solutions[i, p] | i in 1..m]) / {social_mapper[NUM_AGENTS]});")
    instance.add_string(f"constraint forall (p in 1..{social_mapper[NUM_AGENTS]}) (selected[p] >= max_values[p]);")
    instance.add_string(f"constraint exists(i in 1..m) (forall(j in 1..{social_mapper[NUM_AGENTS]}) (selected[j] = possible_solutions[i, j]));")

def optimize_proportionality_objective(instance : Instance, use_weights = True):
    if use_weights:
        instance.add_string(f"var int: total_weight = sum(a in {AGENTS_ARRAY}) (sum(d in 1..m) (selected[a, d] * weights[a]));")
        instance.add_string(f"solve maximize (total_weight);")  
    else:
        instance.add_string(f"solve satisfy;")
    


def prepare_proportionality_runner(social_mapping):
    simple_runner = ProportionalityRunner(social_mapping)
    simple_runner.add(add_proportionality_objective)
    simple_runner.add(optimize_proportionality_objective)
    return simple_runner
    
'''
A utilitarian runner maximizes the sum of utilities; 
it therefore needs to plugin a new objective to the base model
'''
class ProportionalityRunner(SimpleRunner):
    def __init__(self, social_mapping) -> None:
        super().__init__(social_mapping)
        self.social_mapping = social_mapping
        pass

    def run(self, model, solver=...):
        self.model = model
        return super().run(model, solver)
    

if __name__ == "__main__":
    import os
    plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), '../models/project_assignment/project_assignment_core.mzn'))
    plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../models/project_assignment/data/project_assignment_1.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    #     now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)


    simple_runner = ProportionalityRunner(social_mapping)
    simple_runner.add(add_proportionality_objective)
    simple_runner.add(optimize_proportionality_objective)

    result = simple_runner.run(plain_tabular_model, gecode)
    print(result["selected"])


