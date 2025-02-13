import os
from typing import Callable, Tuple

import torch as th
from torch import nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from door_bandit import DoorBanditEnv

class CustomDoorPolicyNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    https://medium.com/@vishal93ranjan/understanding-transformers-implementing-self-attention-in-pytorch-4256f680f0b3

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # see policies._build() for the usage, this is expected
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), 
            nn.ReLU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.ReLU(),  
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), 
            nn.ReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomDoorPolicyNetwork(self.features_dim)


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
    model = PPO(CustomActorCriticPolicy, env, verbose=1)
 
    model_path = "ppo_door_bandit_custom"
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