from simple_runner import SimpleRunner
from minizinc import Model, Solver, Instance
import sys 
import os 
import logging
from pathlib import Path
from functools import partial
from string import Template

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from util.social_mapping_reader import read_social_mapping, get_substitution_dictionary, AGENTS_ARRAY, SHARE_UTIL_AGENT, SHARE_FUNCTION
from util.mzn_debugger import create_debug_folder, log_and_debug_generated_files

def add_envy_freeness_mixin(social_mapper, instance : Instance):
    # That's roughly what the mixin template looks like:

    # predicate envy_free() =
    # envy_pairs() == 0;

    # function var int: envy_pairs() = 
    # sum(i in ${AGENTS_ARRAY}, j in ${AGENTS_ARRAY} where i != j) (
    #     bool2int( ${SHARE_UTIL_AGENT}(i, ${SHARE_FUNCTION}(i)) >= ${SHARE_UTIL_AGENT}(i, ${SHARE_FUNCTION}(j)) )

    # constraint envy_free();

    # TODO test if model is amenable to fair division (needs share function and share util specified)
    envy_free_mixin_template_file = os.path.join(os.path.dirname(__file__), '../models/envy_freeness_mixin_template.mzn')
    envy_free_mixin_template = Template(Path(envy_free_mixin_template_file).read_text())
    sub_dict = get_substitution_dictionary(social_mapper)
    envy_free_mixin = envy_free_mixin_template.substitute(sub_dict)
    logging.info(envy_free_mixin)
    instance.add_string(envy_free_mixin)

def optimize_envy(instance : Instance):
    instance.add_string(f"\nsolve minimize envy_pairs;\n")

def enforce_envy_freeness(instance : Instance):
    instance.add_string(f"\nconstraint envy_free();\n")

if __name__ == "__main__":    
    logging.basicConfig(level=logging.INFO)
    debug_dir = create_debug_folder(os.path.dirname(__file__))
    plain_tabular_model = Model()
    plain_tabular_model_file = os.path.join(os.path.dirname(__file__), '../models/pure_division/pure_division.mzn')
    plain_tabular_model.add_file(plain_tabular_model_file, parse_data=True)
    plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../models/pure_division/1.dzn'), parse_data=True)
    gecode = Solver.lookup("gecode")
    
    # now let's read the social mapping file 
    social_mapping_file = os.path.join(os.path.dirname(__file__), '../models/pure_division/social_mapping.json')
    social_mapping = read_social_mapping(social_mapping_file)

    simple_runner = SimpleRunner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = debug_dir
    simple_runner.add_presolve_handler(partial(add_envy_freeness_mixin, social_mapping))
    simple_runner.add_presolve_handler(optimize_envy)

    result = simple_runner.run(plain_tabular_model, gecode)
    print(result)