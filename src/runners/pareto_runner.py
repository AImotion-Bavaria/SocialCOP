from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from functools import partial

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS


def add_pareto_objective(social_mapper, instance : Instance):
    instance.add_string(f"int: m = card(index_set_1of2(possible_solutions));")
    instance.add_string(f"constraint exists(j in 1..m) (forall(i in 1..{social_mapper[NUM_AGENTS]}) (utilities[i] = possible_solutions[j, i]));")
    instance.add_string(f"constraint not exists(j in 1..m) (sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] == possible_solutions[j, i]))+sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] < possible_solutions[j, i]))=={social_mapper[NUM_AGENTS]}/\ sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] < possible_solutions[j, i]))>0);")

def optimize_pareto_objective(instance : Instance):
    instance.add_string(f"solve satisfy;")
    
'''
A utilitarian runner maximizes the sum of utilities; 
it therefore needs to plugin a new objective to the base model
'''
class ProportionalityRunner(SimpleRunner):
    def __init__(self, social_mapping) -> None:
        super().__init__()
        self.social_mapping = social_mapping
        pass

    def run(self, model, solver=...):
        self.model = model
        return super().run(model, solver)
    

if __name__ == "__main__":
    import os
    plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.mzn'))
    plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)


    simple_runner = ProportionalityRunner(social_mapping)
    simple_runner.add_presolve_handler(partial(add_pareto_objective, social_mapping))
    simple_runner.add_presolve_handler(optimize_pareto_objective)
    #simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    #simple_runner.add_presolve_handler(optimize_utilitarian_objective)
    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)




