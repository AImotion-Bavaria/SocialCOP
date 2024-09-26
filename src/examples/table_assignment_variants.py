""" table_assignment_variants.py
This file shows different use cases with the table assgiment core model
1. Simply maximize the utilitarian welfare
2. Look for all pareto optimal solutions
3. Get the leximin solution
4. Post envy-freeness and compare utilitarian / leximin
"""
import sys 
import os 
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))

from simple_runner import SimpleRunner
from minizinc import Model, Solver
from utilitarian import utilitarian_objective, optimize_utilitarian_objective
from envy_freeness import envy_freeness_mixin, optimize_envy, enforce_envy_freeness
from leximin_runner import LeximinRunner
from pareto_runner import ParetoRunner, pareto_mixin, pareto_only_nondom_mixin, ParetoUtilityTracker

from util.social_mapping_reader import read_social_mapping
from util.mzn_debugger import create_debug_folder

def maximize_utilitarian_welfare(model : Model, solver : Solver, social_mapping):
    logging.info("Maximizing utilitarian welfare ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add(utilitarian_objective)
    simple_runner.add(optimize_utilitarian_objective)

    result = simple_runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

def maximize_utilitarian_welfare_envyfree(model : Model, solver : Solver, social_mapping):
    logging.info("Maximizing utilitarian welfare with envy-freeness ...")
    runner = SimpleRunner(social_mapping)
    runner.debug = True
    runner.debug_dir = debug_dir

    runner.model += [envy_freeness_mixin]
    runner.model += [enforce_envy_freeness]
    runner.add(utilitarian_objective)
    runner.add(optimize_utilitarian_objective)

    result = runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

def minimize_envy(model : Model, solver : Solver, social_mapping):
    logging.info("Minimizing envy ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add(envy_freeness_mixin)
    simple_runner.add(optimize_envy)

    result = simple_runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

def leximin(model : Model, solver : Solver, social_mapping):
    logging.info("Leximin ...")
    leximin_runner = LeximinRunner(social_mapping)
    leximin_runner.debug = False
    leximin_runner.debug_dir = debug_dir

    result = leximin_runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

def leximin_envyfree(model : Model, solver : Solver, social_mapping):
    logging.info("Leximin Envy Free ...")
    leximin_runner = LeximinRunner(social_mapping)
    leximin_runner.debug = False
    leximin_runner.debug_dir = debug_dir

    leximin_runner.model += [envy_freeness_mixin]
    leximin_runner.model += [enforce_envy_freeness]
    result = leximin_runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

def leximin_pareto(model : Model, solver : Solver, social_mapping):
    logging.info("Leximin ...")
    leximin_runner = LeximinRunner(social_mapping)
    leximin_runner.model += [pareto_only_nondom_mixin] 
    # also need a pareto tracker
    pareto_tracker = ParetoUtilityTracker()
    leximin_runner.presolve_step += [pareto_tracker.write_previous_utilities]
    leximin_runner.on_result += [pareto_tracker.update_previous_utilities]

    # Warning: This doesn't work well for know because we have 
    # 
    #constraint not exists(j in prev_util_index_set) 
    #   (forall(i in Agents) (utilities[i] = previous_utilities[j, i]));
    # and that could be a problem if we want to find solutions for Leximin for every agent

    leximin_runner.debug = True
    leximin_runner.debug_dir = debug_dir

    result = leximin_runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

def pareto(model : Model, solver : Solver, social_mapping):
    logging.info("Pareto ...")
    pareto_runner = ParetoRunner(social_mapping)
    pareto_runner.debug = False
    pareto_runner.debug_dir = debug_dir

    result = pareto_runner.run(table_assignment_model, gecode)
    print(result) 
    print("--"*50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    debug_dir = create_debug_folder(os.path.dirname(__file__))
    table_assignment_model = Model()
    model_file = os.path.join(os.path.dirname(__file__), '../models/table_assignment/table_assignment_generic.mzn')
    table_assignment_model.add_file(model_file, parse_data = True)
    table_assignment_model.add_file(os.path.join(os.path.dirname(__file__), '../models/table_assignment/data/generic_2.dzn'), parse_data=True)
    table_assignment_model.add_file(os.path.join(os.path.dirname(__file__), '../models/table_assignment/data/generic_2_preferences.dzn'), parse_data=True)
    gecode = Solver.lookup("chuffed")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/table_assignment/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)
    
    # 6. A Leximin run with a redundant Pareto constraints posted on top 
    #leximin_pareto(table_assignment_model, gecode, social_mapping)

    # 1. Simply maximize the utilitarian welfare
    #maximize_utilitarian_welfare(table_assignment_model, gecode, social_mapping)

    # 2. Minimize envy
    #minimize_envy(table_assignment_model, gecode, social_mapping)

    # 3. Enforce envy-freeness, maximize utilitarian
    #maximize_utilitarian_welfare_envyfree(table_assignment_model, gecode, social_mapping)

    # 4. A Leximin run
    #leximin(table_assignment_model, gecode, social_mapping)

    # 5. An envy-free Leximin run
    leximin_envyfree(table_assignment_model, gecode, social_mapping)

    # 6. A Pareto run 
    #pareto(table_assignment_model, gecode, social_mapping)




   