from datetime import timedelta
import os
import sys
from minizinc import Model, Solver
from os.path import dirname



# Add src folder to Python path
src = dirname(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../runners'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../util'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../experiments'))

from experiment import Experiment
from envy_freeness import enforce_envy_freeness, envy_freeness_mixin, prepare_envy_free_runner, prepare_envy_min_runner
from leximin_runner import LeximinRunner, prepare_leximin_runner
from mzn_debugger import create_debug_folder
from pareto_runner import ParetoUtilityTracker, pareto_only_nondom_mixin
from rawls_normalized import prepare_rawls_runner
from none_runner import prepare_none_runner
from simple_runner import SimpleRunner
from social_mapping_reader import SHARE_FUNCTION
from utilitarian import prepare_utilitarian_runner


TIME_LIMIT_EVAL = timedelta(hours=1.0)

def none(model: Model, social_mapping, solver: Solver, weights):
    #simple_runner = prepare_rawls_runner(social_mapping, use_weights=True, experiment=experiment)
    simple_runner = prepare_none_runner(social_mapping, use_weights=True)
    #simple_runner.add(set_weights)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model=model, solver=solver, weights=weights)
    print("Result: ", result)
    return result

def rawls(model: Model, social_mapping, solver: Solver, weights):
    #simple_runner = prepare_rawls_runner(social_mapping, use_weights=True, experiment=experiment)
    simple_runner = prepare_rawls_runner(social_mapping, use_weights=True)
    #simple_runner.add(set_weights)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model=model, solver=solver, weights=weights)
    print("Result: ", result)
    return result

def utilitarian(model : Model, social_mapping : dict, solver : Solver, weights):
    simple_runner = prepare_utilitarian_runner(social_mapping, use_weights=True)
    #if SHARE_FUNCTION in social_mapping: # it is a division problem - I want to record envy counts as well
     #   simple_runner.add(envy_freeness_mixin)
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

# everything that is associated to envy-freeness has to be a division problem
def utilitarian_envy_free(model : Model, social_mapping : dict, solver : Solver, weights):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
     
    simple_runner : SimpleRunner = prepare_utilitarian_runner(social_mapping, use_weights=True)
    simple_runner.model += [envy_freeness_mixin, enforce_envy_freeness]
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def envy_min(model : Model, social_mapping : dict, solver : Solver, weights):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_min_runner(social_mapping, use_weights=True)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def envy_free(model : Model, social_mapping : dict, solver : Solver, weights):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_free_runner(social_mapping, use_weights=True)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def leximin(model: Model, social_mapping, solver: Solver, weights):
    simple_runner = prepare_leximin_runner(social_mapping, use_weights=True)
    simple_runner.debug = True
    simple_runner.debug_dir = create_debug_folder(os.path.dirname(__file__))
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result

def leximin_pareto(model: Model, social_mapping, solver: Solver, weights):
    simple_runner : LeximinRunner = prepare_leximin_runner(social_mapping, use_weights=True)
    simple_runner.debug = True
    simple_runner.debug_dir = create_debug_folder(os.path.dirname(__file__))
    simple_runner.timeout = TIME_LIMIT_EVAL

    simple_runner.model += [pareto_only_nondom_mixin] 
    # also need a pareto tracker
    pareto_tracker = ParetoUtilityTracker()
    simple_runner.presolve_step += [pareto_tracker.write_previous_utilities]
    simple_runner.on_result += [pareto_tracker.update_previous_utilities]
    result = simple_runner.run(weights=weights, model=model, solver=solver)
    print("Result: ", result)
    return result