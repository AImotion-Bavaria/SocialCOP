# Executes the introductory MiniZinc model
# with data specified in Python and prints the solution
# Follow the installation instructions at
# https://minizinc-python.readthedocs.io/en/latest/getting_started.html#installation
from minizinc import Instance, Model, Solver

import os
hello_social_model = os.path.join(os.path.dirname(__file__), '../mzn_models/hello_social_cop.mzn')
hello_social_data = os.path.join(os.path.dirname(__file__), '../mzn_models/test1.dzn')

simple_agents = Model(hello_social_model)
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
simple_agents.add_file(hello_social_data)
# Create an Instance of the simple agents model for Gecode
instance = Instance(gecode, simple_agents)

# Assign 4 to n
#instance["n"] = 4
#instance["m"] = 2
result = instance.solve()
# Output the array q
print(result["selected"]) 