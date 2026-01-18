"""
Rotation Robustness Evaluation

Evaluates the rotation invariance of learned representations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import math


def get_rotation_matrix_z(angle: float, device: torch.device) -> torch.Tensor:
    """Get 3x3 rotation matrix around Z axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], device=device, dtype=torch.float32)


def get_rotation_matrix_y(angle: float, device: torch.device) -> torch.Tensor:
    """Get 3x3 rotation matrix around Y axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], device=device, dtype=torch.float32)


def get_rotation_matrix_x(angle: float, device: torch.device) -> torch.Tensor:
    """Get 3x3 rotation matrix around X axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], device=device, dtype=torch.float32)


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Generate random 3D rotation matrix using Gram-Schmidt."""
    # Random angles
    angles = torch.rand(3, device=device) * 2 * math.pi

    Rx = get_rotation_matrix_x(angles[0].item(), device)
    Ry = get_rotation_matrix_y(angles[1].item(), device)
    Rz = get_rotation_matrix_z(angles[2].item(), device)

    return Rz @ Ry @ Rx


def rotate_batch(
    batch: Dict[str, torch.Tensor],
    rotation_matrix: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Apply rotation to geometry in batch.

    Args:
        batch: Input batch with brep_faces, brep_edges, points
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        Rotated batch
    """
    rotated = {}

    for key, value in batch.items():
        if key == "brep_faces" and value is not None:
            # [B, F, H, W, 3] -> rotate last dim
            shape = value.shape
            flat = value.reshape(-1, 3)
            rotated_flat = flat @ rotation_matrix.T
            rotated[key] = rotated_flat.reshape(shape)

        elif key == "brep_edges" and value is not None:
            # [B, E, L, 3] -> rotate last dim
            shape = value.shape
            flat = value.reshape(-1, 3)
            rotated_flat = flat @ rotation_matrix.T
            rotated[key] = rotated_flat.reshape(shape)

        elif key == "points" and value is not None:
            # [B, N, 3] -> rotate last dim
            shape = value.shape
            flat = value.reshape(-1, 3)
            rotated_flat = flat @ rotation_matrix.T
            rotated[key] = rotated_flat.reshape(shape)

        else:
            rotated[key] = value

    return rotated


@torch.no_grad()
def evaluate_rotation_robustness(
    model,
    dataloader,
    device: torch.device,
    n_rotations: int = 12,
    angles: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Evaluate rotation robustness of model.

    Args:
        model: CLIP4CAD model
        dataloader: Data loader
        device: Device
        n_rotations: Number of random rotations to test
        angles: Specific angles to test (overrides n_rotations)

    Returns:
        Rotation robustness metrics
    """
    model.eval()

    if angles is None:
        # Random rotations
        rotation_matrices = [
            random_rotation_matrix(device) for _ in range(n_rotations)
        ]
    else:
        # Specific angles around Z axis
        rotation_matrices = [
            get_rotation_matrix_z(a, device) for a in angles
        ]

    # Collect embeddings for original and rotated
    original_embeddings = {"brep": [], "point": []}
    rotated_embeddings = {i: {"brep": [], "point": []} for i in range(len(rotation_matrices))}

    for batch in tqdm(dataloader, desc="Evaluating rotation robustness"):
        # Original embeddings
        outputs = model(batch)

        if "brep_global" in outputs:
            original_embeddings["brep"].append(outputs["brep_global"].cpu())
        if "point_global" in outputs:
            original_embeddings["point"].append(outputs["point_global"].cpu())

        # Rotated embeddings
        for i, R in enumerate(rotation_matrices):
            rotated_batch = rotate_batch(batch, R)
            rotated_outputs = model(rotated_batch)

            if "brep_global" in rotated_outputs:
                rotated_embeddings[i]["brep"].append(rotated_outputs["brep_global"].cpu())
            if "point_global" in rotated_outputs:
                rotated_embeddings[i]["point"].append(rotated_outputs["point_global"].cpu())

    # Concatenate
    for mod in ["brep", "point"]:
        if original_embeddings[mod]:
            original_embeddings[mod] = torch.cat(original_embeddings[mod], dim=0)
        for i in range(len(rotation_matrices)):
            if rotated_embeddings[i][mod]:
                rotated_embeddings[i][mod] = torch.cat(rotated_embeddings[i][mod], dim=0)

    # Compute metrics
    metrics = {}

    for mod in ["brep", "point"]:
        if mod not in original_embeddings or not isinstance(original_embeddings[mod], torch.Tensor):
            continue

        orig = F.normalize(original_embeddings[mod], dim=-1)
        similarities = []

        for i in range(len(rotation_matrices)):
            if mod not in rotated_embeddings[i] or not isinstance(rotated_embeddings[i][mod], torch.Tensor):
                continue

            rot = F.normalize(rotated_embeddings[i][mod], dim=-1)

            # Cosine similarity between original and rotated
            sim = (orig * rot).sum(dim=-1)  # [N]
            similarities.append(sim)

        if similarities:
            all_sims = torch.stack(similarities, dim=1)  # [N, n_rotations]

            # Average similarity (higher = more invariant)
            avg_sim = all_sims.mean().item()
            metrics[f"{mod}_rotation_sim_mean"] = avg_sim

            # Minimum similarity (worst case)
            min_sim = all_sims.min(dim=1)[0].mean().item()
            metrics[f"{mod}_rotation_sim_min"] = min_sim

            # Standard deviation (lower = more consistent)
            std_sim = all_sims.std(dim=1).mean().item()
            metrics[f"{mod}_rotation_sim_std"] = std_sim

    return metrics


def compute_rotation_invariance_score(
    original_embedding: torch.Tensor,
    rotated_embeddings: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute rotation invariance score for a single sample.

    Args:
        original_embedding: [D] original embedding
        rotated_embeddings: List of [D] rotated embeddings

    Returns:
        Rotation invariance metrics
    """
    orig = F.normalize(original_embedding.unsqueeze(0), dim=-1)

    similarities = []
    for rot_emb in rotated_embeddings:
        rot = F.normalize(rot_emb.unsqueeze(0), dim=-1)
        sim = (orig * rot).sum().item()
        similarities.append(sim)

    similarities = np.array(similarities)

    return {
        "mean_sim": float(similarities.mean()),
        "min_sim": float(similarities.min()),
        "max_sim": float(similarities.max()),
        "std_sim": float(similarities.std()),
    }


@torch.no_grad()
def evaluate_discrete_rotations(
    model,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate on discrete 90-degree rotations (axis-aligned).

    These are the rotations used during training augmentation.

    Args:
        model: CLIP4CAD model
        dataloader: Data loader
        device: Device

    Returns:
        Discrete rotation metrics
    """
    # 24 discrete rotations (cube symmetry group)
    discrete_angles = [
        (0, 0, 0),
        (0, 0, math.pi/2),
        (0, 0, math.pi),
        (0, 0, 3*math.pi/2),
        (0, math.pi/2, 0),
        (0, math.pi/2, math.pi/2),
        (0, math.pi/2, math.pi),
        (0, math.pi/2, 3*math.pi/2),
        (0, math.pi, 0),
        (0, math.pi, math.pi/2),
        (0, math.pi, math.pi),
        (0, math.pi, 3*math.pi/2),
        (0, 3*math.pi/2, 0),
        (0, 3*math.pi/2, math.pi/2),
        (0, 3*math.pi/2, math.pi),
        (0, 3*math.pi/2, 3*math.pi/2),
        (math.pi/2, 0, 0),
        (math.pi/2, 0, math.pi/2),
        (math.pi/2, 0, math.pi),
        (math.pi/2, 0, 3*math.pi/2),
        (3*math.pi/2, 0, 0),
        (3*math.pi/2, 0, math.pi/2),
        (3*math.pi/2, 0, math.pi),
        (3*math.pi/2, 0, 3*math.pi/2),
    ]

    rotation_matrices = []
    for ax, ay, az in discrete_angles:
        Rx = get_rotation_matrix_x(ax, device)
        Ry = get_rotation_matrix_y(ay, device)
        Rz = get_rotation_matrix_z(az, device)
        R = Rz @ Ry @ Rx
        rotation_matrices.append(R)

    # Use general evaluation with these specific rotations
    return evaluate_rotation_robustness(
        model, dataloader, device,
        n_rotations=len(rotation_matrices),
        angles=None,  # We override internally
    )
