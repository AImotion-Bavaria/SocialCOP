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

from util.social_mapping_reader import read_social_mapping
from util.mzn_debugger import create_debug_folder

def maximize_utilitarian_welfare_envyfree(model : Model, solver : Solver, social_mapping):
    logging.info("Maximizing utilitarian welfare with envy-freeness ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add(envy_freeness_mixin)
    simple_runner.add(enforce_envy_freeness)
    simple_runner.add(utilitarian_objective)
    simple_runner.add(optimize_utilitarian_objective)

    result = simple_runner.run(table_assignment_model, solver)
    print(result) 
    print("--"*50)

def maximize_utilitarian_welfare(model : Model, solver : Solver, social_mapping):
    logging.info("Maximizing utilitarian welfare without envy-freeness ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add(envy_freeness_mixin)
    #simple_runner.add_presolve_handler(enforce_envy_freeness)
    simple_runner.add(utilitarian_objective)
    simple_runner.add(optimize_utilitarian_objective)

    result = simple_runner.run(table_assignment_model, solver)
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
    solver = Solver.lookup("chuffed")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/table_assignment/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)
    
    # Enforce envy-freeness, maximize utilitarian
    maximize_utilitarian_welfare_envyfree(table_assignment_model, solver, social_mapping)
    maximize_utilitarian_welfare(table_assignment_model, solver, social_mapping)
    
   