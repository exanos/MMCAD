"""
InfoNCE Contrastive Loss

Implements symmetric InfoNCE for multimodal alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for contrastive learning.

    Aligns embeddings from two modalities where corresponding
    pairs (diagonal) should be similar and non-corresponding
    pairs should be dissimilar.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling factor
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embed1: torch.Tensor,
        embed2: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss.

        Args:
            embed1: [B, D] embeddings from modality 1
            embed2: [B, D] embeddings from modality 2
            temperature: Optional learnable temperature

        Returns:
            Scalar loss value
        """
        temp = temperature if temperature is not None else self.temperature

        # L2 normalize
        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        # Similarity matrix
        sim = embed1 @ embed2.T / temp  # [B, B]

        # Labels: diagonal is positive
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        # Cross-entropy in both directions
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.T, labels)

        return (loss_12 + loss_21) / 2

    @torch.no_grad()
    def compute_accuracy(
        self,
        embed1: torch.Tensor,
        embed2: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute retrieval accuracy (for logging)."""
        temp = temperature if temperature is not None else self.temperature

        embed1 = F.normalize(embed1, dim=-1)
        embed2 = F.normalize(embed2, dim=-1)

        sim = embed1 @ embed2.T / temp
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        acc_12 = (sim.argmax(dim=1) == labels).float().mean()
        acc_21 = (sim.argmax(dim=0) == labels).float().mean()

        return ((acc_12 + acc_21) / 2).item()


class MultiModalInfoNCE(nn.Module):
    """
    InfoNCE loss for multiple modality pairs.

    Computes pairwise loss for all available modality combinations.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.infonce = InfoNCELoss(temperature)

    def forward(
        self,
        embeddings: dict,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE across all modality pairs.

        Args:
            embeddings: Dict mapping modality name to [B, D] embeddings
            temperature: Optional learnable temperature

        Returns:
            Average loss across all pairs
        """
        modalities = list(embeddings.keys())

        if len(modalities) < 2:
            return torch.tensor(0.0, device=list(embeddings.values())[0].device)

        total_loss = 0.0
        n_pairs = 0

        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                e1 = embeddings[modalities[i]]
                e2 = embeddings[modalities[j]]

                loss = self.infonce(e1, e2, temperature)
                total_loss = total_loss + loss
                n_pairs += 1

        return total_loss / n_pairs
