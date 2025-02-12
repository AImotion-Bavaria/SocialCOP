from door_bandit import DoorBanditEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

if __name__ == "__main__":

    env = DoorBanditEnv(num_doors=5)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    action = 2  # Example action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

    # Create the vectorized environment
    # In training, we ignore the first door, but in test it is possible again
    env = make_vec_env(lambda: DoorBanditEnv(num_doors=5, ignore_door_up_to=1), n_envs=1, seed=41)

    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)

    model_path = "ppo_door_bandit"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print("Model loaded successfully.")
    else:
        # Train the agent
        model.learn(total_timesteps=10000)        
        # Save the agent
        model.save(model_path)

    # Test the trained agent, but now let's not ignore the first door
    env = make_vec_env(lambda: DoorBanditEnv(num_doors=5, ignore_door_up_to=0), n_envs=1, seed=41)

    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        print(f"Obs {obs}, Selected action: {action}")
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            obs = env.reset()