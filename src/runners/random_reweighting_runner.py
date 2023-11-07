# Executes the introductory MiniZinc model
# with data specified in Python and prints the solution
# Follow the installation instructions at
# https://minizinc-python.readthedocs.io/en/latest/getting_started.html#installation
import random
from minizinc import Instance, Model, Solver

# weighted_social_cop.mzn
simple_agents = Model("./src/mzn_models/weighted_social_cop.mzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")

# Create an Instance of the simple agents model for Gecode
instance = Instance(gecode, simple_agents)

def run_solving_random(m_python, n_python, repetitions):
# initially, all get the same chance
    weights = [1 for i in range(n_python)]
    distribution = [0 for i in range(n_python)]

    for i in range(repetitions):
        with instance.branch() as inst:
            inst["n"] = n_python
            inst["m"] = m_python
            inst["weights"] = weights
            #print(f"... Solving with weights: {weights}")
            result = inst.solve()
            # Output the array selected
            #print(result["selected"]) 
            selected = result["selected"]

            # every 0 gets one more in weights
            for a in range(n_python):
                weights[a] = random.randrange(0, n_python)
                distribution[a] += selected[a]
    return distribution