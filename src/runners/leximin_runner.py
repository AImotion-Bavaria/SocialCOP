from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os 
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from functools import partial

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS
LEXIMIN_AGENTS_PLACEHOLDER = "Agents_Rawls_Mixin"
LEXIMIN_UTILITIES_PLACEHOLDER = "utilities_Rawls_Mixin"
MAXMIN_VALUES = "maxmin_values"
MAXMIN_AGENTS = "maxmin_agents"
NEXT_WORST_UTIL = "next_worst_util"
NEXT_WORST_AGENT = "next_worst_agent"

def add_leximin_mixin(social_mapper, instance : Instance):
    instance.add_file(os.path.join(os.path.dirname(__file__), '../models/leximin_mixin.mzn'))
    instance.add_string(f"{LEXIMIN_AGENTS_PLACEHOLDER} = {social_mapper[AGENTS_ARRAY]};")
    instance.add_string(f"{LEXIMIN_UTILITIES_PLACEHOLDER} = {social_mapper[UTILITY_ARRAY]};")

class LeximinRunner(SimpleRunner):
    def __init__(self, social_mapping) -> None:
        super().__init__()
        self.social_mapping = social_mapping

    def run(self, model, solver=...):
        self.model = model
        return super().run(model, solver)
    
    def solve(self, child: Instance):
        # have to ask the model, otherwise the parameter might not get passed to the child instance
        num_agents = self.model[social_mapping[NUM_AGENTS]]

        # gets initialized to be empty, updated with minimal values as we go
        maxmin_values = []
        updated_result = 0
        for i in range(num_agents):
            with child.branch() as inst:
                inst[MAXMIN_VALUES] = maxmin_values
                result = inst.solve()

                #calculate and store preconditions for next iteration 
                worst_util = result[NEXT_WORST_UTIL]
                maxmin_values.append(worst_util)

                # some logging
                logging.info(f"Currently worst-off agent {result[NEXT_WORST_AGENT]} with utility {result[NEXT_WORST_UTIL]}")
                logging.info(f"Maxmin agents: {result[MAXMIN_AGENTS]}")
                updated_result = result[self.social_mapping[UTILITY_ARRAY]]
        return updated_result

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO)
    plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.mzn'))
    plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    simple_runner = LeximinRunner(social_mapping)
    simple_runner.add_presolve_handler(partial(add_leximin_mixin, social_mapping))
    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)