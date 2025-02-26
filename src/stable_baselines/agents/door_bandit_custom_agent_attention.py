import os
from typing import Callable, Tuple

import torch as th
from torch import nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from door_bandit import DoorBanditEnv

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding layer to convert binary inputs to embeddings
        self.embedding = nn.Linear(1, embedding_dim)
        
        # Query, Key, and Value transformations
        self.W_query = nn.Linear(embedding_dim, embedding_dim)
        self.W_key = nn.Linear(embedding_dim, embedding_dim)
        self.W_value = nn.Linear(embedding_dim, embedding_dim)
        
        # Scaling factor for dot product attention
        self.scale = th.sqrt(th.FloatTensor([embedding_dim]))

    def forward(self, x):
        # x shape: (batch_size, seq_len=5)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Reshape input to (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Create embeddings for each position
        # Shape: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(x)
        
        # Calculate Query, Key, Value
        Q = self.W_query(embeddings)  # (batch_size, seq_len, embedding_dim)
        K = self.W_key(embeddings)    # (batch_size, seq_len, embedding_dim)
        V = self.W_value(embeddings)  # (batch_size, seq_len, embedding_dim)
        
        # Calculate attention scores
        # (batch_size, seq_len, seq_len)
        attention = th.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        # (batch_size, seq_len, embedding_dim)
        out = th.matmul(attention_weights, V)
        
        return out

class CustomDoorPolicyNetwork(nn.Module):
    """
    Custom network for policy and value function using self-attention.
    Each binary input in the sequence represents a door state.
    """
    def __init__(
        self,
        feature_dim: int,  # This will be 5 for 5 doors
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super().__init__()

        # Save output dimensions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        embedding_dim = 16  # Dimension for token embeddings

        # Shared attention layer
        self.attention = SingleHeadAttention(embedding_dim)
        
        # Shared transformation after attention
        self.shared_transform = nn.Linear(embedding_dim, embedding_dim)
        
        # Final projections to policy and value dimensions
        self.policy_proj = nn.Linear(embedding_dim, last_layer_dim_pi)
        self.value_proj = nn.Linear(embedding_dim, last_layer_dim_vf)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
        """
        # Get attention outputs for all positions
        # shape: (batch_size, seq_len, embedding_dim)
        attention_outputs = self.attention(features)
        
        # Apply shared transformation and ReLU
        # shape: (batch_size, seq_len, embedding_dim)
        transformed = F.relu(self.shared_transform(attention_outputs))
        
        # Average pool across sequence dimension
        # shape: (batch_size, embedding_dim)
        pooled = transformed.mean(dim=1)
        
        # Project to policy and value dimensions
        policy_output = self.policy_proj(pooled)
        value_output = self.value_proj(pooled)
        
        return policy_output, value_output

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        attention_outputs = self.attention(features)
        transformed = F.relu(self.shared_transform(attention_outputs))
        pooled = transformed.mean(dim=1)
        return self.policy_proj(pooled)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        attention_outputs = self.attention(features)
        transformed = F.relu(self.shared_transform(attention_outputs))
        pooled = transformed.mean(dim=1)
        return self.value_proj(pooled)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomDoorPolicyNetwork(self.features_dim)


if __name__ == "__main__":
    # The rest of the code remains the same as in the original file
    env = DoorBanditEnv(num_doors=5)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    action = 2
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

    env = make_vec_env(lambda: DoorBanditEnv(num_doors=5, ignore_door_up_to=1), n_envs=1, seed=41)

    model = PPO(CustomActorCriticPolicy, env, verbose=1)
 
    model_path = "ppo_door_bandit_custom_attention"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print("Model loaded successfully.")
    else:
        model.learn(total_timesteps=10000)        
        model.save(model_path)

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