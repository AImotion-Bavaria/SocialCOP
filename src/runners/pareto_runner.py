from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance, Status
import sys 
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from functools import partial

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS


def add_pareto_objective(social_mapper, instance : Instance):
    #instance.add_string(f"int: m = card(index_set_1of2(possible_solutions));")
    #instance.add_string(f"constraint not exists(j in 1..m) (sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] == possible_solutions[j, i]))+sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] < possible_solutions[j, i]))=={social_mapper[NUM_AGENTS]}/\ sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] < possible_solutions[j, i]))>0);")
    true = 1

def optimize_pareto_objective(social_mapper, instance : Instance):
    res = instance.solve()
    previous_solutions = [res["utilities"]]
    print(res.solution)
    while res.status == Status.SATISFIED:
        with instance.branch() as child:
            child.add_string(f"int: m = {len(previous_solutions)};")
            child.add_string(f"array[int, int] of int: previous_solutions = {convert_to_minizinc_syntax(previous_solutions)};")
            child.add_string(f"constraint not exists(j in 1..m) (forall(i in 1..{social_mapper[NUM_AGENTS]}) (utilities[i] = previous_solutions[j, i]));")
            child.add_string(f"constraint not exists(j in 1..m) (sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] == previous_solutions[j, i]))+sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] < previous_solutions[j, i]))=={social_mapper[NUM_AGENTS]}/\ sum(i in 1..{social_mapper[NUM_AGENTS]}) (bool2int(utilities[i] < previous_solutions[j, i]))>0);")
            res = child.solve()
            if res.solution is not None:
                print(previous_solutions)
                new_solution = res["utilities"]
                for result in previous_solutions:
                    counter = 0
                    for agent in range (len(result)):
                        if result[agent]<=new_solution[agent]:
                            counter+=1
                    if counter == len(result):
                        previous_solutions.remove(result)
                previous_solutions.append(res["utilities"])
    counter=0            
    print("previous_solutions="+str(previous_solutions))
    return previous_solutions

def convert_to_minizinc_syntax(array):
    return_value="["
    for row in array:
        return_value+="|"+", ".join(str(entry) for entry in row)
    return_value+="|]"
    return return_value
    
    
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
    simple_runner.add_presolve_handler(partial(optimize_pareto_objective, social_mapping))
    #simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    #simple_runner.add_presolve_handler(optimize_utilitarian_objective)
    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)



