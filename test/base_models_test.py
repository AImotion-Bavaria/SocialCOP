import unittest
import sys
import warnings
from minizinc import Model
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('src/runners')
import os

from simple_runner import SimpleRunner

# make sure that the basic minizinc models are properly executed
class BaseModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.simple_runner = SimpleRunner()
        return super().setUp()
    
    def test_plain_tabular(self):
        plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), "../src/models/plain_tabular/plain_tabular.mzn")) 
        result = self.simple_runner.run(plain_tabular_model)
        self.assertIsNotNone(result)

    def test_table_assignment(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/table_assignment/table_assignment.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/table_assignment/data/1.dzn")
        model.add_file(data)        
        result = self.simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_social_selection(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/social_selection/social_selection.mzn")) 
        n_agents = 5
        model["m"] = 3
        model["n"] = n_agents
        model["weights"] = [1 for i in range(n_agents)]
        result = self.simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_photo_placement(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/photo_placement/photo_placement.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/photo_placement/data/4.dzn")
        model.add_file(data)        
        model.add_string("old_solutions = [];")
        result = self.simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_project_assignment(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/project_assignment/project_assignment.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/project_assignment/project_assignment_1_with_old_solutions.dzn")
        model.add_file(data)
        result = self.simple_runner.run(model)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

