from minizinc import Instance, Model, Solver
import re
import pymzn

model = Model("src/mzn_models/rawls_test_table_python.mzn")
model.add_file("src/mzn_models/rawls.dzn")
# Find the MiniZinc solver configuration for Gecode
gecode = Solver.lookup("gecode")
# Create an Instance of the simple agents model for Gecode
instance = Instance(gecode, model)
#get initial values from dzn file
data = pymzn.dzn2dict('src/mzn_models/rawls.dzn')

#Init counter, dimension and minimal_value array
counter = 1
dimension = data["dimension"]
min_val = [0] * dimension

#iterate for number of agents
while(counter<=dimension):
    with instance.branch() as inst:
        #solve with given preconditions
        inst["index"]=counter
        inst["minimal_values"] = min_val
        result = inst.solve()

        print(result)

        #calculate and store preconditions for next iteration 
        worst_util = int(re.search(r'worst_util=(\d+)', str(result)).group(1))
        min_val[counter-1]=worst_util
        counter=counter+1

   