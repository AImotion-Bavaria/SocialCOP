from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY
RAWLS_OBJECTIVE = "rawls_objective"

def rawls_objective( instance : Instance, social_mapper):
    instance.add_string(f"var int: {RAWLS_OBJECTIVE};")
    instance.add_string(f"constraint {RAWLS_OBJECTIVE} = min({social_mapper[UTILITY_ARRAY]});")

def  optimize_rawls_objective(instance : Instance, social_mapper = None):
    instance.add_string(f"solve maximize {RAWLS_OBJECTIVE};")
    
def prepare_rawls_runner(social_mapping):
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.add(rawls_objective)
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