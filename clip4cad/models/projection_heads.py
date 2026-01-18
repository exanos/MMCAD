"""
Projection heads for contrastive learning.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension (defaults to input_dim)
            num_layers: Number of layers (2 or 3)
        """
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        if num_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        elif num_layers == 3:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]

        Returns:
            [..., output_dim]
        """
        return self.net(x)


class LearnableTemperature(nn.Module):
    """Learnable temperature parameter for contrastive learning."""

    def __init__(self, init_value: float = 0.07, min_value: float = 0.01, max_value: float = 1.0):
        """
        Args:
            init_value: Initial temperature value
            min_value: Minimum temperature
            max_value: Maximum temperature
        """
        super().__init__()

        import math

        self.log_temp = nn.Parameter(torch.tensor(math.log(init_value)))
        self.min_value = min_value
        self.max_value = max_value

    @property
    def temperature(self) -> torch.Tensor:
        """Get clamped temperature value."""
        return self.log_temp.exp().clamp(min=self.min_value, max=self.max_value)

    def forward(self) -> torch.Tensor:
        """Return temperature value."""
        return self.temperature
