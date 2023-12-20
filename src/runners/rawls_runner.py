from minizinc import Instance, Model, Solver
import re

model = Model("src/mzn_models/rawls_test_table_python.mzn")
model.add_file("src/mzn_models/rawls2.dzn", parse_data=True)
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the simple agents model for Gecode
instance = Instance(gecode, model)

#Init counter, dimension and minimal_value array
counter = 1
dimension = model["dimension"]
min_val = [0] * dimension

#iterate for number of agents
while(counter <= dimension):
    with instance.branch() as inst:
        #solve with given preconditions
        inst["index"] = counter
        inst["minimal_values"] = min_val
        result = inst.solve()

        print(result)

        #calculate and store preconditions for next iteration 
        worst_util = result["worst_util"]# int(re.search(r'worst_util=(\d+)', str(result)).group(1))
        min_val[counter-1]=worst_util
        counter = counter + 1

   