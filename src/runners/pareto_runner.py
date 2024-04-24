from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance, Status
import sys 
import os 
import logging
from functools import partial
from string import Template
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS, get_substitution_dictionary
from util.mzn_debugger import create_debug_folder, log_and_debug_generated_files

PREVIOUS_UTILITIES = "previous_utilities"

def prepare_pareto_runner(social_mapping):
    return ParetoRunner(social_mapping)

def add_pareto_mixin(instance : Instance, social_mapper):
    pareto_mixin_template_file = os.path.join(os.path.dirname(__file__), '../models/pareto_mixin_template.mzn')
    pareto_mixin_template = Template(Path(pareto_mixin_template_file).read_text())
    sub_dict = get_substitution_dictionary(social_mapper)
    pareto_mixin = pareto_mixin_template.substitute(sub_dict)
    logging.info(pareto_mixin)
    instance.add_string(pareto_mixin)

class ParetoRunner(SimpleRunner):
    def __init__(self, social_mapping) -> None:
        super().__init__(social_mapping)
        self.add_presolve_handler(add_pareto_mixin)

    def solve(self, instance: Instance):
        # we need an empty list of previous utilities to begin with
        with instance.branch() as child:
            child[PREVIOUS_UTILITIES] = []
            res = child.solve()
        previous_utilities = [res[self.social_mapping[UTILITY_ARRAY]]]
        previous_solutions = [res] # the actual solutions might be important, too
        logging.info(previous_utilities)
        i = 0

        while res.status == Status.SATISFIED:
            with instance.branch() as child:
               child[PREVIOUS_UTILITIES] = previous_utilities
               if self.debug:  
                    log_and_debug_generated_files(child, "pareto_runner", i, debug_dir_=self.debug_dir)
               res = child.solve()
               if res.solution is not None:
                    logging.info(previous_utilities)
                    new_utilities = res[self.social_mapping[UTILITY_ARRAY]]
                    dominated_indices = []
                    for solution_index, prev_sol_utils in enumerate(previous_utilities):
                        counter = 0
                        # compare two utility vectors, e.g. [2,5,7] is pareto-dominated by [2, 6, 9]
                        for prev_sol_utility, new_utility in zip (prev_sol_utils, new_utilities):
                            if prev_sol_utility <= new_utility:
                                counter += 1
                        if counter == len(prev_sol_utils): # less than or equal for all agents -> previous solution is dominated
                            dominated_indices.append(solution_index)

                    for solution_index in dominated_indices[::-1]:
                        del previous_utilities[solution_index]
                        del previous_solutions[solution_index]
                    
                    previous_utilities.append(res[self.social_mapping[UTILITY_ARRAY]])
                    previous_solutions.append(res)
            i += 1
        logging.info(f"Previous utilities: {previous_utilities}")
        #logging.info(f"Previous solutions: {previous_solutions}")
        return previous_utilities, previous_solutions


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

    pareto_runner = ParetoRunner(social_mapping)
    pareto_runner.debug = True
    pareto_runner.debug_dir = debug_dir
    result = pareto_runner.run(plain_tabular_model, gecode)
