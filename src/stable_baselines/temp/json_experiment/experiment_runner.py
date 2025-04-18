import numpy as np
from json_reader import get_substitution_dictionary, read_json_file
from sb_mz_combined import GiniEnv, train
#from sb_mz_combined_chuffed import GiniEnv, train as chuffed_train
from stable_baselines3.common.vec_env import DummyVecEnv
from deterministic import DetEnv

def run_tests(sub_dict):
    """
    Run tests for both trained and deterministic methods.
    
    Parameters:
    steps (list): List of steps for the environment.
    sub_dict (dict): Dictionary containing test methods and parameters.
    """
    for j in range(len(sub_dict["data"][-1])):
        # Test loop for trained methods
        for method in sub_dict["test"]:
            for i in range(1):
                models_dir = "./src/stable_baselines/temp/json_experiment/models/"
                file_dir = f"src/stable_baselines/temp/json_experiment/models/best_model_{method['test_agent']}.zip"
                logdir = f"src/stable_baselines/temp/json_experiment/logs/trained_{j}_{method['test_agent']}_{i}"
                trial = sub_dict["data"][-1][j]
                env = GiniEnv(solver=method["solver"])
                env.test(file_dir=file_dir, log_dir=logdir, iterations=method["iterations"])
                print("\n\n\n\n")

        # Test loop for deterministic methods
        for method in sub_dict["deterministic"]:
            env = DetEnv(start=sub_dict["data"][-1][j])
            env.test(method["test_agent"], method["iterations"], experimentNo=j)
            print("\n\n\n\n")

def training():
    """
    Train models using specified methods and parameters.
    """
    sub_dict = get_substitution_dictionary(read_json_file(".\\src\\stable_baselines\\temp\\json_experiment\\test.json"))
    for j in range(1):
        # Training loop
        for method in sub_dict["train"]:
                    env=DummyVecEnv([lambda: GiniEnv(grid_size=5, render_mode='console', solver=method["solver"])])
                    models_dir = f"src/stable_baselines/temp/json_experiment/models/best_model_{method['train_agent']}"
                    logdir = f"src/stable_baselines/logs/training_gecode_{method['train_agent']}"
                    train(env, models_dir)
            

if __name__ == "__main__":
    sub_dict = get_substitution_dictionary(read_json_file(".\\src\\stable_baselines\\temp\\json_experiment\\test.json"))
    training()
    run_tests(sub_dict)



