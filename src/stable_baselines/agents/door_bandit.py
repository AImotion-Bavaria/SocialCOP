import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DoorBanditEnv(gym.Env):
    def __init__(self, num_doors=5, ignore_door_up_to = 0):
        super(DoorBanditEnv, self).__init__()
        self.num_doors = num_doors
        self.ignore_door_up_to = ignore_door_up_to
        self.action_space = spaces.Discrete(num_doors)
        # Update observation space to be a binary array of length num_doors
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_doors,), dtype=np.int8)
        self.prize_door = np.random.randint(num_doors)
        
    def reset(self, seed=41):
        self.prize_door = np.random.randint(self.ignore_door_up_to, self.num_doors)
        # Create observation array with 1 at prize door location
        observation = np.zeros(self.num_doors, dtype=np.int8)
        observation[self.prize_door] = 1
        return observation, {}  # In Gymnasium, reset returns (observation, info)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        if action == self.prize_door:
            reward = 100
        else:
            reward = -1
        done = True
        # Return current observation state
        observation = np.zeros(self.num_doors, dtype=np.int8)
        observation[self.prize_door] = 1
        truncated = False
        return observation, reward, done, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Example usage
if __name__ == "__main__":
    env = DoorBanditEnv(num_doors=5)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    action = 2  # Example action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")