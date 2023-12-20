from minizinc import Instance, Model, Solver
import re

model = Model("src/mzn_models/rawls_test_table_python_assignment.mzn")
model.add_file("src/mzn_models/rawls2.dzn", parse_data=True)
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the simple agents model for Gecode
instance = Instance(gecode, model)

#Init counter, dimension and minimal_value array
counter = 1
dimension = model["dimension"]
maxmin_values = []

#iterate for number of agents
while(counter <= dimension):
    with instance.branch() as inst:
        #solve with given preconditions
        inst["maxmin_values"] = maxmin_values
        result = inst.solve()

        print(result)

        #calculate and store preconditions for next iteration 
        worst_util = result["worst_util"]
        maxmin_values.append(worst_util)
        counter = counter + 1

   