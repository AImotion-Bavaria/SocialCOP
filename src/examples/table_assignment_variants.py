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
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))

from simple_runner import SimpleRunner
from minizinc import Model, Solver
from utilitarian import add_utilitarian_objective, optimize_utilitarian_objective
from envy_freeness import add_envy_freeness_mixin, optimize_envy, enforce_envy_freeness
from leximin_runner import LeximinRunner

from util.social_mapping_reader import read_social_mapping
from util.mzn_debugger import create_debug_folder

def maximize_utilitarian_welfare(model : Model, solver : Solver, social_mapping):
    logging.info("Maximizing utilitarian welfare ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    simple_runner.add_presolve_handler(optimize_utilitarian_objective)

    result = simple_runner.run(table_assignment_model, gecode)
    print(result) 

def maximize_utilitarian_welfare_envyfree(model : Model, solver : Solver, social_mapping):
    logging.info("Maximizing utilitarian welfare with envy-freeness ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add_presolve_handler(partial(add_envy_freeness_mixin, social_mapping))
    simple_runner.add_presolve_handler(enforce_envy_freeness)
    simple_runner.add_presolve_handler(partial(add_utilitarian_objective, social_mapping))
    simple_runner.add_presolve_handler(optimize_utilitarian_objective)

    result = simple_runner.run(table_assignment_model, gecode)
    print(result) 

def minimize_envy(model : Model, solver : Solver, social_mapping):
    logging.info("Minimizing envy ...")
    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir

    simple_runner.add_presolve_handler(partial(add_envy_freeness_mixin, social_mapping))
    simple_runner.add_presolve_handler(optimize_envy)

    result = simple_runner.run(table_assignment_model, gecode)
    print(result) 

def leximin(model : Model, solver : Solver, social_mapping):
    logging.info("Leximin ...")
    leximin_runner = LeximinRunner(social_mapping)
    leximin_runner.debug = True
    leximin_runner.debug_dir = debug_dir

    result = leximin_runner.run(table_assignment_model, gecode)
    print(result) 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    debug_dir = create_debug_folder(os.path.dirname(__file__))
    table_assignment_model = Model()
    model_file = os.path.join(os.path.dirname(__file__), '../models/table_assignment/table_assignment_generic.mzn')
    table_assignment_model.add_file(model_file, parse_data = True)
    table_assignment_model.add_file(os.path.join(os.path.dirname(__file__), '../models/table_assignment/data/generic_1.dzn'), parse_data=True)
    table_assignment_model.add_file(os.path.join(os.path.dirname(__file__), '../models/table_assignment/data/generic_1_preferences.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/table_assignment/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    # 1. Simply maximize the utilitarian welfare
    maximize_utilitarian_welfare(table_assignment_model, gecode, social_mapping)

    # 2. Minimize envy
    minimize_envy(table_assignment_model, gecode, social_mapping)

    # 3. Enforce envy-freeness, maximize utilitarian
    maximize_utilitarian_welfare_envyfree(table_assignment_model, gecode, social_mapping)

    # 4. A Leximin run
    leximin(table_assignment_model, gecode, social_mapping)

    # 5. A Pareto run 