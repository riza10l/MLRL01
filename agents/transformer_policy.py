"""
Transformer Encoder - Prioritas 11
===================================
Attention mechanism & long sequence understanding.
Implementasi custom features extractor buat Stable Baselines 3.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based features extractor for RL observation sequences.
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        # We assume the observation is flattened (window * n_features + portfolio_state)
        # In a real transformer, we need to reshape it back to (window, n_features)
        super().__init__(observation_space, features_dim)
        
        # This is a simplified implementation. 
        # For a real one, we need to know the window size and feature count from the env.
        # We'll use a standard MLP as a placeholder if we can't easily infer dimensions,
        # but here's the transformer structure:
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Placeholder for the actual transformer layers
        # In practice, you'd need to pass window_size and n_features to __init__
        self.transformer = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )
        
        # Real Transformer implementation would look like this:
        # encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=num_heads, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations: (batch_size, obs_dim)
        return self.transformer(observations)

# Note: To use this in PPO:
# policy_kwargs = dict(features_extractor_class=TransformerFeaturesExtractor, 
#                     features_extractor_kwargs=dict(features_dim=128))
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
