from random import gauss
import threading

gauss_lock = threading.Lock()
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete
from torch.utils.tensorboard import SummaryWriter

from minizinc import Instance, Model, Solver


def calculate_gini(array):
    """
    Calculate the Gini coefficient of a numpy array.
    
    Parameters:
    array (numpy.ndarray): Input array to calculate the Gini coefficient for.
    
    Returns:
    float: Gini coefficient.
    """
    if np.all(array == 0):
        return 0
    
    # Sort the array and cast to float16
    array = np.sort(np.array(array)).astype(np.float16)
    
    # Create an index array
    index = np.arange(1, array.shape[0] + 1)
    
    # Number of elements in the array
    n = array.shape[0]
    
    # Calculate the Gini coefficient
    gini_coefficient = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))
    
    return gini_coefficient


class DetEnv:
    """
    Custom environment for deterministic simulations.
    """

    def __init__(self, grid_size=5, render_mode=None):
        """
        Initialize the environment.

        Parameters:
        start (int): Starting point.
        grid_size (int): Size of the grid.
        render_mode (str): Mode for rendering.
        """
        #self.index = -1
        self.action_space = Box(low=0, high=100, shape=(grid_size,), dtype=np.int32)
        self.grid_size = grid_size
        
        # Define the observation space
        self.observation_space = Dict({
            "required": Box(low=0, high=6, shape=(self.grid_size,), dtype=np.int32),
            "received": Box(low=0, high=500, shape=(self.grid_size,), dtype=np.int32),
            "valuation": Box(low=0, high=101, shape=(self.grid_size,), dtype=np.int32),
            "cur_val": Discrete(5),
            "count_rec": Box(low=0, high=500, shape=(self.grid_size,), dtype=np.int32)
        })
        
        self.previous_valuations = np.zeros(self.grid_size)
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Parameters:
        seed (int): Seed for random number generation.
        options (dict): Additional options for reset.

        Returns:
        dict: Initial observation.
        """
        self.index =0
        
        self.observation = {
            "required": np.zeros(self.grid_size, dtype=np.float32),
            "received": np.zeros(self.grid_size),
            "valuation": np.zeros(self.grid_size, dtype=np.int32),
            "cur_val": np.random.randint(1, 5),
            "count_rec": np.zeros(self.grid_size)
        }
        
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.steps = 0
        
        return self.observation, {}

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Parameters:
        action (int): Action to take.

        Returns:
        tuple: (observation, terminated, truncated, info)
        """
        self.action = action
        index = np.argmax(action)
        simple_agents = Model("src/stable_baselines/temp/table_assignment_generic.mzn")
    # Find the MiniZinc solver configuration for Gecode
        gecode = Solver.lookup("gecode")
        
        simple_agents.add_file("src/stable_baselines/temp/"+str(self.index%3)+"_generic_preferences.dzn",parse_data=True)
        # Create an Instance of the simple agents model for Gecode

        instance = Instance(gecode, simple_agents)
        weights = action
        
        with instance.branch() as inst:
            #instance.add_file("src/stable_baselines/temp/generic_1_preferences.dzn", parse_data=True)
            #inst["m"] = m_python
            inst["weights"] = [int(weight) for weight in weights]
            print(f"... Solving with weights: {weights}")
            result = inst.solve()
            # Output the array selected
            print(result["assigned"]) 

            # every 0 gets one more in weights
            

        self.observation["received"]=result["assigned"]
        
        
        self.observation["valuation"] += result["utilities"]
        
       # if self.observation["required"][index] != 0:
        #    self.observation["valuation"][index] = (min(self.observation["cur_val"], sum(1 for x in self.observation["received"][index] if x != "NoTable")) / self.observation["required"][index]) * 100
        
        self.gini_index = calculate_gini(self.observation["valuation"])
        
        
        self.steps += 1
        self.index+=1
        
        terminated = self.steps >= 52
        truncated = False
        self.info = {
            "required": self.observation["required"],
            "received": self.observation["received"],
            "valuation": self.observation["valuation"],
            "value": self.observation["cur_val"],
            "gini": self.gini_index,
            "sum_rec": sum(1 for x in self.observation["received"][index] if x != "NoTable")
        }
        
        self.observation["cur_val"] = np.random.randint(1, 5)
        
        return self.observation, terminated, truncated, self.info

    
    
    def greedy(self):
        """
        Get the index of the agent with the highest required value.

        Returns:
        int: Index of the agent with the highest required value.
        """

        self.observation["required"] = np.random.randint(0, 100, size=self.grid_size)
        array = np.zeros(self.grid_size)
        array[np.argmax(self.observation["required"], axis=None, out=None)]=100
        return array
    
    

    def test(self, method, iterations, experimentNo=0):
        """
        Test the environment using a specified method over a number of iterations.

        Parameters:
        method (str): Method to use for testing.
        iterations (int): Number of iterations to run the test.
        experimentNo (int): Experiment number for logging.
        """
        log_dir = f"src/stable_baselines/temp/logs/deterministic_{experimentNo}_{method}"
        writer = SummaryWriter(log_dir)
        
        for s in range(iterations):
            action = getattr(self, method)()
            observation, terminated, truncated, info = self.step(action)
            
            writer.add_scalar("Test/Gini_Index", calculate_gini(info["valuation"]), s)
            writer.add_scalar("Test/Sum_rec", info["sum_rec"], s)
            writer.add_scalar("Test/Valuation", sum(info["valuation"]), s)
            
            if terminated: 
                print("reward", "Gini Index: ", calculate_gini(info["valuation"]))  # always use last element
                print("Episode finished.")
                self.reset()
                break


if __name__ == "__main__":
    #for _ in range(50):
        env = DetEnv()
        env.test("greedy", 50)
        
        action = env.greedy()
        observation, terminated, truncated, info = env.step(action)
        env.reset()
        print("Episode finished.", action)
        if terminated:
            print("Action taken:", action)
            action = env.greedy()



    
    

    