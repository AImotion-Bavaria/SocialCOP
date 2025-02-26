import numpy as np
from json_reader import get_substitution_dictionary, read_json_file
from trained import train, test, GiniEnv
from det import DetEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def run_tests(steps, sub_dict):
    """
    Run tests for both trained and deterministic methods.
    
    Parameters:
    steps (list): List of steps for the environment.
    sub_dict (dict): Dictionary containing test methods and parameters.
    """
    for j in range(5):
        # Test loop for trained methods
        for method in sub_dict["test"]:
            for i in range(5):
                models_dir = "./src/stable_baselines/models/"
                file_dir = f"src/stable_baselines/models/best_model_{method['test_agent']}.zip"
                logdir = f"src/stable_baselines/logs/trained_{j}_{method['test_agent']}_{i}"
                test(file_dir=file_dir, log_dir=logdir, method=method["test_agent"], iterations=method["iterations"], start=steps[j], models_dir=models_dir)
                print("\n\n\n\n")

        # Test loop for deterministic methods
        for method in sub_dict["deterministic"]:
            env = DetEnv(start=steps[j])
            env.test(method["test_agent"], method["iterations"], experimentNo=j)
            print("\n\n\n\n")

def training():
    """
    Train models using specified methods and parameters.
    """
    steps = [
        [[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]],
        [[5, 5, 5], [3, 3, 1], [1, 1, 4], [1, 3, 2], [1, 1, 5]],
        [[5, 1, 3], [3, 5, 1], [2, 4, 1], [2, 3, 4], [2, 4, 1]],
        [[5, 3, 1], [1, 5, 2], [3, 2, 2], [2, 3, 1], [5, 5, 1]],
        [[3, 5, 4], [2, 1, 4], [1, 1, 5], [2, 5, 1], [1, 1, 3]]
    ]

    sub_dict = get_substitution_dictionary(read_json_file(".\\src\\stable_baselines\\agents\\test.json"))
    for j in range(5):
        # Training loop
        for method in sub_dict["train"]:
            env = DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', start=[1, 2, 3, 4, 5], method=method["train_agent"])])
            models_dir = f"src/stable_baselines/agents/models/test_trained_{method['train_agent']}"
            logdir = "src/stable_baselines/logs/training_"
            train(method["train_agent"], log_dir=logdir, models_dir=models_dir, timesteps=method["iterations"])

if __name__ == "__main__":
    steps = [
        [[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]],
        [[5, 5, 5], [3, 3, 1], [1, 1, 4], [1, 3, 2], [1, 1, 5]],
        [[5, 1, 3], [3, 5, 1], [2, 4, 1], [2, 3, 4], [2, 4, 1]],
        [[5, 3, 1], [1, 5, 2], [3, 2, 2], [2, 3, 1], [5, 5, 1]],
        [[3, 5, 4], [2, 1, 4], [1, 1, 5], [2, 5, 1], [1, 1, 3]]
    ]

    sub_dict = get_substitution_dictionary(read_json_file(".\\src\\stable_baselines\\agents\\test.json"))
    run_tests(steps, sub_dict)



