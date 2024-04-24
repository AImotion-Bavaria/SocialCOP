import unittest
import sys
import os
import warnings
import logging
from minizinc import Model

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/util"))
from social_mapping_reader import read_social_mapping
from mzn_debugger import create_debug_folder
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/runners"))

from simple_runner import SimpleRunner
from pareto_runner import prepare_pareto_runner

# make sure that the basic minizinc models are properly executed
class ParetoTest(unittest.TestCase):
    def setUp(self) -> None:
        
        self.debug_dir = create_debug_folder(os.path.dirname(__file__))
        logging.basicConfig(level=logging.INFO)
        return super().setUp()
    
    def test_plain_tabular(self):

        plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), "../src/models/plain_tabular/plain_tabular_deterministic.mzn")) 
        plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../src/models/plain_tabular/pareto_test.dzn'), parse_data=True)
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/plain_tabular/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        runner = prepare_pareto_runner(social_mapping)
        runner.debug = True 
        runner.debug_dir = self.debug_dir
        result = runner.run(plain_tabular_model)
        print(result)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

