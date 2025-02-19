from random import gauss
import threading

gauss_lock = threading.Lock()
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete
from torch.utils.tensorboard import SummaryWriter


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

    def __init__(self, start, grid_size=5, render_mode=None):
        """
        Initialize the environment.

        Parameters:
        start (int): Starting point.
        grid_size (int): Size of the grid.
        render_mode (str): Mode for rendering.
        """
        self.index = -1
        self.start = start
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
        self.index += 1
        
        self.observation = {
            "required": np.zeros(self.grid_size, dtype=np.float32),
            "received": np.zeros(self.grid_size),
            "valuation": np.zeros(self.grid_size, dtype=np.int32),
            "cur_val": np.random.randint(1, 5),
            "count_rec": np.zeros(self.grid_size)
        }
        
        self.gini_index = calculate_gini(self.observation["valuation"])
        self.steps = 0
        self.frequency = self.start
        self.bedarf(self.start, count=self.steps)
        
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

        self.observation["received"][self.action] += min(self.observation["cur_val"], self.observation["required"][self.action])
        
        for agent in range(self.grid_size):
            self.observation["valuation"][agent] = max(self.observation["valuation"][agent] - 20, 0)
        
        if self.observation["required"][self.action] != 0:
            self.observation["valuation"][self.action] = (min(self.observation["cur_val"], self.observation["received"][self.action]) / self.observation["required"][self.action]) * 100
        
        self.gini_index = calculate_gini(self.previous_valuations)
        self.bedarf(self.start, count=self.steps)
        self.steps += 1
        
        terminated = self.steps >= 30
        truncated = False
        self.info = {
            "required": self.observation["required"],
            "received": self.observation["received"],
            "valuation": self.observation["valuation"],
            "value": self.observation["cur_val"],
            "gini": self.gini_index,
            "sum_rec": sum(self.observation["received"])
        }
        
        self.observation["cur_val"] = np.random.randint(1, 5)
        
        return self.observation, terminated, truncated, self.info

    def worst_received(self):
        """
        Get the index of the agent with the least received value.

        Returns:
        int: Index of the agent with the least received value.
        """
        return np.argmin(self.observation["received"], axis=None, out=None)
    
    def round_robin(self):
        """
        Get the next agent index in a round-robin fashion.

        Returns:
        int: Index of the next agent.
        """
        self.index += 1
        return (self.index - 1) % self.grid_size
    
    def greedy(self):
        """
        Get the index of the agent with the highest required value.

        Returns:
        int: Index of the agent with the highest required value.
        """
        return np.argmax(self.observation["required"], axis=None, out=None)
    
    def bedarf_received(self):
        """
        Get the index of the agent with the highest need based on received values.

        Returns:
        int: Index of the agent with the highest need.
        """
        received_safe = np.where(self.observation["received"] == 0, 1, self.observation["received"])
        value = max(self.observation["required"] / received_safe)
        return np.argmax(self.observation["required"] == value)
    
    def bedarf(self, frequency, count, value=30):
        """
        Calculate the required values for each agent based on frequency and count.

        Parameters:
        frequency (list): Frequency of requirements for each agent.
        count (int): Current count.
        value (int): Base value for calculation.
        """
        bedarf = np.zeros(self.grid_size)
        for agent in range(self.grid_size):
            try:
                count_agent = (count - 1) % len(frequency[agent])
                with gauss_lock:
                    bedarf[agent] = max(0, gauss(frequency[agent][count_agent], value))
            except IndexError:
                count_agent = (count - 1) % len(frequency)
                bedarf[agent] = max(0, gauss(frequency[count_agent], value))
            except Exception as e:
                print(f"Error calculating bedarf for agent {agent}: {e}")
                bedarf[agent] = 0
        self.observation["required"] = bedarf

    def test(self, method, iterations, experimentNo=0):
        """
        Test the environment using a specified method over a number of iterations.

        Parameters:
        method (str): Method to use for testing.
        iterations (int): Number of iterations to run the test.
        experimentNo (int): Experiment number for logging.
        """
        log_dir = f"src/stable_baselines/logs/deterministic_{experimentNo}_{method}"
        writer = SummaryWriter(log_dir)
        
        for s in range(iterations):
            action = getattr(self, method)()
            observation, terminated, truncated, info = self.step(action)
            
            writer.add_scalar("Test/Gini_Index", calculate_gini(info["valuation"]), s)
            writer.add_scalar("Test/Sum_rec", info["sum_rec"], s)
            writer.add_scalar("Test/Valuation", sum(info["valuation"]), s)
            
            if terminated: 
                print("reward", "Gini Index: ", calculate_gini(info["received"]))  # always use last element
                print("Episode finished.")
                self.reset()
                break


if __name__ == "__main__":
    env = DetEnv(start=[[2, 2, 2], [3, 3, 1], [1, 1, 4], [1, 2, 3], [1, 1, 5]])
    env.test("greedy", 5)
    env.test("round_robin", 5)
    env.test("worst_received", 5)
    env.test("bedarf_received", 5)