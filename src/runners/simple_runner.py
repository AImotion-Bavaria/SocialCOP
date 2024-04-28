# Executes the introductory MiniZinc model
# with data specified in Python and prints the solution
# Follow the installation instructions at
# https://minizinc-python.readthedocs.io/en/latest/getting_started.html#installation
from minizinc import Instance, Model, Solver
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from util.social_mapping_reader import read_social_mapping
from util.mzn_debugger import create_debug_folder, log_and_debug_generated_files

class SimpleRunner:
    def __init__(self, social_mapping) -> None:
        self.model = []   # a list of functions applied before solving
        self.on_result = [] # a list of functions applied after seeing a result
        self.debug = False
        self.debug_dir = None
        self.social_mapping = social_mapping
        self.timeout = None

    def run(self, model, solver = Solver.lookup("gecode")):
        self.instance = Instance(solver, model)
        self.mzn_model = model 
        with self.instance.branch() as child:
            self.presolve_hook(child)
            # immediately before solving, log this model
            if self.debug:
                    log_and_debug_generated_files(child, "simple_runner_child", 0, self.debug_dir)
            result = self.solve(child)
            self.on_result_hook(child, result)
        return result 
    
    def solve(self, child : Instance):
        """
        intended to be overwritten by specialized runners 
        """ 
        return child.solve(timeout = self.timeout)
    
    def add(self, presolve_handler):
        """_summary_

        Args:
            presolve_handler (function): function takes a child instance as argument
        """
        self.model.append(presolve_handler)

    def presolve_hook(self, instance):
        for handler in self.model:
            handler(instance, self.social_mapping)

    def on_result_hook(self, instance, result):
        for handler in self.on_result:
            handler(instance, self.social_mapping, result)

if __name__ == "__main__":
    import os
    hello_social_model = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.mzn')
    hello_social_data = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/plain_tabular.dzn')
    simple_agents = Model(hello_social_model)
    simple_agents.add_file(hello_social_data)
    # Find the MiniZinc solver configuration for Gecode
    gecode = Solver.lookup("gecode")
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/plain_tabular/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)
    simple_runner = SimpleRunner(social_mapping)
    result = simple_runner.run(simple_agents, gecode)
    print(result)