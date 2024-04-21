import json
from dataclasses import dataclass

@dataclass
class Experiment:
    solver: str
    configuration: str
    problem: str
    path: str
    model_inst : tuple

    def get_identifier(self):
        ident = f"{self.problem}_{self.solver}_{self.configuration}_{ '_'.join(self.model_inst[1])}" 
        ident = ident.replace(".", "_")
        ident = ident.replace("[", "_")
        ident = ident.replace("]", "_")
        return ident

    def get_result_filename(self):
        return f"{self.get_identifier()}.pkl"
    
def parse_json(filename):
    experiments = []
    
    with open(filename, 'r') as file:
        data = json.load(file)
        
        for problem_instance in data['problems']:
            model = problem_instance['model']
            problem = problem_instance["problem"]
            path = problem_instance["path"]
            data_files = problem_instance['data']
            
            for solver in data['solvers']:
                for configuration in data['configurations']:
                    experiment = Experiment(solver, configuration, problem, path, (model, data_files))
                    experiments.append(experiment)
                        
    
    return experiments

if __name__ == "__main__":
    # Example usage:
    import os 
    filename =  os.path.join(os.path.dirname(__file__), 'ki2024_experiment_plan.json')    
    experiments = parse_json(filename)

    # Now you have a list of Experiment instances, you can use them as needed.
    for exp  in experiments:
        exp : Experiment = exp
        print(f"Solver: {exp.solver}, Configuration: {exp.configuration}, Problem: {exp.problem} Model-Inst: {exp.model_inst}")
        print(exp.get_result_filename())
