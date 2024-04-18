import unittest
import sys
import os
import warnings
from minizinc import Model

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/util"))
from social_mapping_reader import read_social_mapping
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/runners"))


from simple_runner import SimpleRunner

# make sure that the basic minizinc models are properly executed
class BaseModelsTest(unittest.TestCase):
    def setUp(self) -> None:

        return super().setUp()
    
    def test_plain_tabular(self):
        plain_tabular_model = Model(os.path.join(os.path.dirname(__file__), "../src/models/plain_tabular/plain_tabular.mzn")) 
        plain_tabular_model.add_file(os.path.join(os.path.dirname(__file__), '../src/models/plain_tabular/plain_tabular.dzn'), parse_data=True)
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/plain_tabular/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        simple_runner = SimpleRunner(social_mapping)
        result = simple_runner.run(plain_tabular_model)
        result = simple_runner.run(plain_tabular_model)
        self.assertIsNotNone(result)

    def test_table_assignment(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/table_assignment/table_assignment.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/table_assignment/data/1.dzn")
        model.add_file(data)        
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/table_assignment/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        simple_runner = SimpleRunner(social_mapping)
        result = simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_social_selection(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/social_selection/social_selection.mzn")) 
        n_agents = 5
        model["m"] = 3
        model["n"] = n_agents
        model["weights"] = [1 for i in range(n_agents)]
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/social_selection/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        simple_runner = SimpleRunner(social_mapping)
        result = simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_photo_placement(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/photo_placement/photo_placement.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/photo_placement/data/4.dzn")
        model.add_file(data)   
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/photo_placement/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        simple_runner = SimpleRunner(social_mapping)     
        result = simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_project_assignment(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/project_assignment/project_assignment.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/project_assignment/project_assignment_1_with_old_solutions.dzn")
        model.add_file(data)
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/project_assignment/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        simple_runner = SimpleRunner(social_mapping)
        result = simple_runner.run(model)
        self.assertIsNotNone(result)

    def test_bus_tour(self):
        model = Model(os.path.join(os.path.dirname(__file__), "../src/models/bus_tour/bus_tour.mzn")) 
        data = os.path.join(os.path.dirname(__file__), "../src/models/bus_tour/data/bus_tour_1.dzn")
        model.add_file(data)
        social_mapping_file = os.path.join(os.path.dirname(__file__), '../src/models/bus_tour/social_mapping.json')
        social_mapping = read_social_mapping(social_mapping_file)
        simple_runner = SimpleRunner(social_mapping)
        result = simple_runner.run(model)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()

