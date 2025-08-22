from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, UTILITY_ARRAY, AGENTS_ARRAY, MAIN_VARIABLES, TIME_SPAN
none_OBJECTIVE = "none_objective"

def none_objective( instance : Instance, social_mapper):
    instance.add_string(f"var int: {none_OBJECTIVE};")
    instance.add_string(f"constraint {none_OBJECTIVE} = min({social_mapper[UTILITY_ARRAY]});")

def optimize_none_objective(instance : Instance, social_mapper):
    instance.add_string(f"solve maximize ({none_OBJECTIVE});")

def optimize_none_objective_weights(instance: Instance, social_mapper):
    # Declare the weights array
    instance.add_string(f"array[{(social_mapper[AGENTS_ARRAY])}] of int: weights;")
    #array of float ausprobieren
    
    # Assign the weights array values
      # Pass the weights from Python to MiniZinc
    
    # Define the total_weight variable
    instance.add_string(f"var int: total_weight = sum(a in {social_mapper[AGENTS_ARRAY]}) ( {social_mapper[UTILITY_ARRAY]}[a] * weights[a]);")
    #var float: total_weight

    # Solve the objective
    instance.add_string(f"solve maximize total_weight;")
    

    
        
def prepare_none_runner(social_mapping, use_weights=False):
    simple_runner = SimpleRunner(social_mapping)
    #simple_runner.add(none_objective)
    if use_weights:
        simple_runner.add(optimize_none_objective_weights)
        #simple_runner.add(optimize_none_objective_weights)
        
    else:
        simple_runner.add(optimize_none_objective)
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
    simple_runner.add(none_objective)
    simple_runner.add(optimize_none_objective)
    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)
