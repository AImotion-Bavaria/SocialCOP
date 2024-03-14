# this just runs a model that would timeout and see if we actually get the best solution so far
# when running this benchmark, make sure that the "worst case" heuristic is activated
# to provoke long times for optimization

import minizinc
from datetime import timedelta
from minizinc import Instance, Model, Result, Solver, Status

tsp_model = Model("vehicle_routing_standalone.mzn")
tsp_model.add_file("data/3.dzn")
solver = minizinc.Solver.lookup("gecode")

inst = minizinc.Instance(solver, tsp_model)
timeout = timedelta(seconds=3)

result = inst.solve(timeout=timeout)

if result.status == Status.SATISFIED or result.status == Status.OPTIMAL_SOLUTION:
    print("sounds good")
    tour = result["next"]
    print(tour)
    print(result["social_welfare"])

    print(result.status)

