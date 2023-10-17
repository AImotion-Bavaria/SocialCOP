# Executes the introductory MiniZinc model
# with data specified in Python and prints the solution
# Follow the installation instructions at
# https://minizinc-python.readthedocs.io/en/latest/getting_started.html#installation
from minizinc import Instance, Model, Solver

# weighted_social_cop.mzn
simple_agents = Model("mzn_models/weighted_social_cop.mzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")

m_python = 2
n_python = 5
# Create an Instance of the simple agents model for Gecode
instance = Instance(gecode, simple_agents)

# initially, all get the same chance
weights = [1 for i in range(n_python)]

for i in range(10):
    with instance.branch() as inst:
        inst["n"] = n_python
        inst["m"] = m_python
        inst["weights"] = weights
        print(f"... Solving with weights: {weights}")
        result = inst.solve()
        # Output the array selected
        print(result["selected"]) 
        selected = result["selected"]

        # every 0 gets one more in weights
        for a in range(n_python):
            weights[a] += 1 - selected[a]
