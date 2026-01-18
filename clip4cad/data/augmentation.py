"""
Data augmentation for 3D CAD data.

Provides consistent rotation augmentation across modalities.
"""

import numpy as np
import torch
from typing import Optional, Tuple


def sample_discrete_rotation_matrix(device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Sample a random rotation matrix from discrete 90-degree rotations.
    24 possible rotations (all axis-aligned orientations).

    Args:
        device: Target device for tensor

    Returns:
        [3, 3] rotation matrix
    """
    # Random 90-degree rotations around each axis
    angles = np.random.choice([0, 90, 180, 270], size=3) * np.pi / 180

    cos_x, sin_x = np.cos(angles[0]), np.sin(angles[0])
    cos_y, sin_y = np.cos(angles[1]), np.sin(angles[1])
    cos_z, sin_z = np.cos(angles[2]), np.sin(angles[2])

    Rx = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    Ry = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    Rz = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    R = (Rz @ Ry @ Rx).astype(np.float32)
    R = torch.from_numpy(R)

    if device is not None:
        R = R.to(device)

    return R


def sample_continuous_rotation_matrix(
    max_angle: float = np.pi,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Sample a random continuous rotation matrix.

    Args:
        max_angle: Maximum rotation angle in radians
        device: Target device

    Returns:
        [3, 3] rotation matrix
    """
    # Random rotation axis
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # Random angle
    angle = np.random.uniform(-max_angle, max_angle)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    R = R.astype(np.float32)
    R = torch.from_numpy(R)

    if device is not None:
        R = R.to(device)

    return R


def apply_rotation_augmentation(
    points: torch.Tensor,
    rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotation to points.

    Args:
        points: [..., 3] points to rotate
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        Rotated points with same shape
    """
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    rotated = points_flat @ rotation_matrix.T
    return rotated.reshape(original_shape)


def rotate_face_grids(
    face_grids: torch.Tensor,
    rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Rotate face point grids.

    Args:
        face_grids: [B, F, H, W, 3] or [F, H, W, 3] face point grids
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        Rotated face grids with same shape
    """
    return apply_rotation_augmentation(face_grids, rotation_matrix)


def rotate_edge_curves(
    edge_curves: torch.Tensor,
    rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Rotate edge curves.

    Args:
        edge_curves: [B, E, L, 3] or [E, L, 3] edge curves
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        Rotated edge curves with same shape
    """
    return apply_rotation_augmentation(edge_curves, rotation_matrix)


def rotate_pointcloud(
    points: torch.Tensor,
    rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Rotate point cloud.

    Args:
        points: [B, N, 3] or [N, 3] point cloud
        rotation_matrix: [3, 3] rotation matrix

    Returns:
        Rotated point cloud with same shape
    """
    return apply_rotation_augmentation(points, rotation_matrix)


def random_point_jitter(
    points: torch.Tensor,
    std: float = 0.01,
    clip: float = 0.05
) -> torch.Tensor:
    """
    Add random jitter to points.

    Args:
        points: [..., 3] points
        std: Standard deviation of noise
        clip: Maximum noise magnitude

    Returns:
        Jittered points
    """
    noise = torch.randn_like(points) * std
    noise = torch.clamp(noise, -clip, clip)
    return points + noise


def random_scale(
    points: torch.Tensor,
    scale_range: Tuple[float, float] = (0.8, 1.2)
) -> torch.Tensor:
    """
    Apply random uniform scaling.

    Args:
        points: [..., 3] points
        scale_range: (min_scale, max_scale)

    Returns:
        Scaled points
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return points * scale


def random_point_dropout(
    points: torch.Tensor,
    max_dropout_ratio: float = 0.5
) -> torch.Tensor:
    """
    Randomly drop points from point cloud.

    Args:
        points: [N, 3] point cloud
        max_dropout_ratio: Maximum ratio of points to drop

    Returns:
        Point cloud with some points dropped (may have different size)
    """
    n_points = points.shape[0]
    dropout_ratio = np.random.uniform(0, max_dropout_ratio)
    n_keep = max(1, int(n_points * (1 - dropout_ratio)))

    indices = np.random.choice(n_points, n_keep, replace=False)
    return points[indices]


def normalize_pointcloud(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize point cloud to unit sphere.

    Args:
        points: [N, 3] or [B, N, 3] point cloud

    Returns:
        Normalized point cloud
    """
    if points.dim() == 2:
        centroid = points.mean(dim=0)
        points = points - centroid
        scale = points.abs().max()
        if scale > 1e-8:
            points = points / scale
    else:
        # Batched
        centroid = points.mean(dim=1, keepdim=True)
        points = points - centroid
        scale = points.abs().amax(dim=(1, 2), keepdim=True)
        scale = torch.clamp(scale, min=1e-8)
        points = points / scale

    return points


def normalize_to_bbox(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize points to [-1, 1] bounding box.

    Args:
        points: [..., 3] points

    Returns:
        Normalized points in [-1, 1] range
    """
    flat_points = points.reshape(-1, 3)
    min_vals = flat_points.min(dim=0)[0]
    max_vals = flat_points.max(dim=0)[0]
    center = (min_vals + max_vals) / 2
    scale = (max_vals - min_vals).max() / 2
    scale = max(scale.item(), 1e-8)

    normalized = (points - center) / scale
    return normalized
