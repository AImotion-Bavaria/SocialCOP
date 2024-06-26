from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance, Status 
import sys 
import os 
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS
from util.mzn_debugger import create_debug_folder, log_and_debug_generated_files

LEXIMIN_AGENTS_PLACEHOLDER = "Agents_Rawls_Mixin"
LEXIMIN_UTILITIES_PLACEHOLDER = "utilities_Rawls_Mixin"
MAXMIN_VALUES = "maxmin_values"
MAXMIN_AGENTS = "maxmin_agents"
NEXT_WORST_UTIL = "next_worst_util"
NEXT_WORST_AGENT = "next_worst_agent"

def add_leximin_mixin(instance : Instance, social_mapper):
    instance.add_file(os.path.join(os.path.dirname(__file__), '../models/leximin_mixin.mzn'))
    instance.add_string(f"\n{LEXIMIN_AGENTS_PLACEHOLDER} = {social_mapper[AGENTS_ARRAY]};\n")
    instance.add_string(f"\n{LEXIMIN_UTILITIES_PLACEHOLDER} = {social_mapper[UTILITY_ARRAY]};\n")

class LeximinRunner(SimpleRunner):
    def __init__(self, social_mapping) -> None:
        super().__init__(social_mapping)
        self.add(add_leximin_mixin)
        self.presolve_step = []
    
    def presolve_step_hook(self, instance):
        for handler in self.presolve_step:
            handler(instance, self.social_mapping)

    def solve(self, child: Instance):
        # have to ask the model, otherwise the parameter might not get passed to the child instance
        num_agents = self.mzn_model[self.social_mapping[NUM_AGENTS]]

        # gets initialized to be empty, updated with minimal values as we go
        maxmin_values = []
        for i in range(num_agents):
            with child.branch() as inst:
                inst[MAXMIN_VALUES] = maxmin_values
                self.presolve_step_hook(inst)
                
                if self.debug:
                    log_and_debug_generated_files(inst, "leximin_runner_inst", i, self.debug_dir)
                result = inst.solve(timeout=self.timeout)
                # TODO what happens here in a timeout
                if not result.solution or result.status == Status.UNKNOWN:
                    return last_result
                
                # here, we could have some on result constraints 
                self.on_result_hook(child, result)

                #calculate and store preconditions for next iteration 
                worst_util = result[NEXT_WORST_UTIL]
                maxmin_values.append(worst_util)

                # some logging
                logging.info(f"Currently worst-off agent {result[NEXT_WORST_AGENT]} with utility {result[NEXT_WORST_UTIL]}")
                logging.info(f"Maxmin agents: {result[MAXMIN_AGENTS]}")
                last_result = result 

        return result

def prepare_leximin_runner(social_mapping):
    return LeximinRunner(social_mapping)

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO)
    debug_dir = create_debug_folder(os.path.dirname(__file__))
    plain_tabular_model = Model()
    plain_tabular_model_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.mzn')
    plain_tabular_model.add_file(plain_tabular_model_file, parse_data=True)
    plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    simple_runner = LeximinRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir
    
    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)