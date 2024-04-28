from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance, Status 
import sys 
import os 
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, AGENTS_ARRAY, UTILITY_ARRAY, NUM_AGENTS
from util.mzn_debugger import create_debug_folder, log_and_debug_generated_files

"""
An on_result_runner implements tree search (Branch-and-Bound)
where on having a result from a previous search, new constraints get posted
"""

class OnResultRunner(SimpleRunner):
    def __init__(self, social_mapping) -> None:
        super().__init__(social_mapping)
        self.on_result_handlers = [] # a list of functions applied after seeing a result

    def add_on_result_handler(self, handler):
        self.on_result_handlers.append(handler)

    def solve(self, inst: Instance):
        res = inst.solve()
        previous_solutions = [res]
                 
        solutions = []
        while res.status == Status.SATISFIED:
            solutions.append(res)
            logging.info(f"Found solution: {res}") 

            with inst.branch() as child:
                for on_result_handler in self.on_result_handlers:
                    on_result_handler(child, res)
                if self.debug:
                    log_and_debug_generated_files(child)
                res = child.solve()
        return previous_solutions

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO)
    debug_dir = create_debug_folder(os.path.dirname(__file__))
    social_selection_model = Model(os.path.join(os.path.dirname(__file__), '../models/social_selection/social_selection.mzn'))
    n_agents = 5
    # we can also set some parameters to the Model instance right here in Python (or read from dzh)
    social_selection_model["m"] = 3
    social_selection_model["n"] = n_agents 
    social_selection_model["weights"] = [1+i for i in range(n_agents)]
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/social_selection/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    # solve the same model using an on result runner 
    on_result_runner = OnResultRunner(social_mapping)
    from utilitarian import add_utilitarian_objective, get_better_utilitarian
    # insert all decision variables to calculate the utilitarian objective for this model
    on_result_runner.add(add_utilitarian_objective)
    # optimize for the utilitarian objective using on result constraints
    on_result_runner.add_on_result_handler(get_better_utilitarian)
    on_result_runner.run(social_selection_model)