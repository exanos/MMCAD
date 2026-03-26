"""
AutoBrep BFS Ordering and Spatial Extraction Utilities

Provides:
- BFS ordering of faces with parent/edge tracking
- Spatial property extraction (centroids, normals, areas)
- Face/edge type classification
- Edge reordering based on face BFS order
"""

from typing import Dict, Tuple, List
import numpy as np

# Face type constants
FACE_TYPE_PLANE = 0
FACE_TYPE_CYLINDER = 1
FACE_TYPE_CONE = 2
FACE_TYPE_SPHERE = 3
FACE_TYPE_TORUS = 4
FACE_TYPE_BSPLINE = 5

FACE_TYPE_NAMES = {
    FACE_TYPE_PLANE: 'plane',
    FACE_TYPE_CYLINDER: 'cylinder',
    FACE_TYPE_CONE: 'cone',
    FACE_TYPE_SPHERE: 'sphere',
    FACE_TYPE_TORUS: 'torus',
    FACE_TYPE_BSPLINE: 'bspline',
}

# Edge type constants
EDGE_TYPE_LINE = 0
EDGE_TYPE_CIRCLE = 1
EDGE_TYPE_ELLIPSE = 2
EDGE_TYPE_BSPLINE = 3

EDGE_TYPE_NAMES = {
    EDGE_TYPE_LINE: 'line',
    EDGE_TYPE_CIRCLE: 'circle',
    EDGE_TYPE_ELLIPSE: 'ellipse',
    EDGE_TYPE_BSPLINE: 'bspline',
}


def build_face_graph_from_incidence(face_edge_incidence: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], int]]:
    """
    Build face adjacency graph from face-edge incidence matrix.

    Args:
        face_edge_incidence: [F, E] binary matrix where 1 indicates face touches edge

    Returns:
        face_neighbors: dict mapping face_idx -> list of neighbor face indices
        edge_between_faces: dict mapping (min_face, max_face) -> edge_idx
    """
    F, E = face_edge_incidence.shape

    face_neighbors = {i: [] for i in range(F)}
    edge_between_faces = {}

    for edge_idx in range(E):
        faces = np.where(face_edge_incidence[:, edge_idx])[0]
        if len(faces) == 2:
            f1, f2 = int(faces[0]), int(faces[1])
            face_neighbors[f1].append(f2)
            face_neighbors[f2].append(f1)
            edge_key = (min(f1, f2), max(f1, f2))
            edge_between_faces[edge_key] = edge_idx

    return face_neighbors, edge_between_faces


def bfs_order_faces_with_parents(
    face_edge_incidence: np.ndarray,
    face_bboxes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    BFS traverse faces with full parent/edge tracking for hierarchy.

    Following AutoBrep's deterministic BFS ordering:
    1. Start from face with smallest XYZ bounding box (lexicographic sort)
    2. Visit neighbors in XYZ-sorted order
    3. Track parent face and connecting edge for tree structure

    Args:
        face_edge_incidence: [F, E] binary incidence matrix
        face_bboxes: [F, 6] bounding boxes [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        dict with:
            bfs_to_original_face: [F] original face indices in BFS order
            original_to_bfs_face: [F] BFS position for each original face
            bfs_level: [F] level in BFS tree (0 = root)
            bfs_parent_face: [F] parent face BFS index (-1 for root)
            bfs_parent_edge: [F] edge connecting to parent (-1 for root)
    """
    F, E = face_edge_incidence.shape

    if F == 0:
        return {
            'bfs_to_original_face': np.array([], dtype=np.int32),
            'original_to_bfs_face': np.array([], dtype=np.int32),
            'bfs_level': np.array([], dtype=np.int32),
            'bfs_parent_face': np.array([], dtype=np.int32),
            'bfs_parent_edge': np.array([], dtype=np.int32),
        }

    # Build face graph
    face_neighbors, edge_between_faces = build_face_graph_from_incidence(face_edge_incidence)

    # XYZ lexicographic sort for deterministic ordering
    # Sort by xmin, ymin, zmin, xmax, ymax, zmax
    xyz_order = np.lexsort((
        face_bboxes[:, 5],  # zmax
        face_bboxes[:, 4],  # ymax
        face_bboxes[:, 3],  # xmax
        face_bboxes[:, 2],  # zmin
        face_bboxes[:, 1],  # ymin
        face_bboxes[:, 0],  # xmin
    ))
    xyz_rank = np.argsort(xyz_order)  # Map face_idx -> rank in xyz order

    # BFS with parent tracking
    start_face = int(xyz_order[0])

    bfs_to_original = []
    bfs_levels = []
    bfs_parent_face = []
    bfs_parent_edge = []

    visited = set()
    # Queue: (face_idx, level, parent_bfs_idx, parent_edge_idx)
    queue = [(start_face, 0, -1, -1)]

    while queue:
        face, level, parent_bfs, parent_edge = queue.pop(0)

        if face in visited:
            continue
        visited.add(face)

        current_bfs_idx = len(bfs_to_original)
        bfs_to_original.append(face)
        bfs_levels.append(level)
        bfs_parent_face.append(parent_bfs)
        bfs_parent_edge.append(parent_edge)

        # Get unvisited neighbors, sorted by XYZ rank
        neighbors = [n for n in face_neighbors[face] if n not in visited]
        if neighbors:
            # Sort neighbors by their XYZ rank
            neighbor_ranks = [xyz_rank[n] for n in neighbors]
            sorted_indices = np.argsort(neighbor_ranks)

            for idx in sorted_indices:
                n = neighbors[idx]
                edge_key = (min(face, n), max(face, n))
                edge_idx = edge_between_faces.get(edge_key, -1)
                queue.append((n, level + 1, current_bfs_idx, edge_idx))

    # Handle disconnected components - add remaining faces
    for face in xyz_order:
        face = int(face)
        if face not in visited:
            visited.add(face)
            bfs_to_original.append(face)
            bfs_levels.append(0)  # New component root
            bfs_parent_face.append(-1)
            bfs_parent_edge.append(-1)

    # Build reverse mapping
    original_to_bfs = np.zeros(F, dtype=np.int32)
    for bfs_idx, orig_idx in enumerate(bfs_to_original):
        original_to_bfs[orig_idx] = bfs_idx

    return {
        'bfs_to_original_face': np.array(bfs_to_original, dtype=np.int32),
        'original_to_bfs_face': original_to_bfs,
        'bfs_level': np.array(bfs_levels, dtype=np.int32),
        'bfs_parent_face': np.array(bfs_parent_face, dtype=np.int32),
        'bfs_parent_edge': np.array(bfs_parent_edge, dtype=np.int32),
    }


def bfs_order_edges(
    face_edge_incidence: np.ndarray,
    bfs_to_original_face: np.ndarray,
    edge_bboxes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Reorder edges based on face BFS order.

    Edges are ordered by the minimum BFS index of their adjacent faces,
    with XYZ bbox as tiebreaker.

    Args:
        face_edge_incidence: [F, E] binary incidence matrix
        bfs_to_original_face: [F] original face indices in BFS order
        edge_bboxes: [E, 6] edge bounding boxes

    Returns:
        dict with:
            bfs_to_original_edge: [E] original edge indices in BFS order
            original_to_bfs_edge: [E] BFS position for each original edge
            edge_to_faces: [E, 2] adjacent face BFS indices for each edge
    """
    F, E = face_edge_incidence.shape

    if E == 0:
        return {
            'bfs_to_original_edge': np.array([], dtype=np.int32),
            'original_to_bfs_edge': np.array([], dtype=np.int32),
            'edge_to_faces': np.zeros((0, 2), dtype=np.int32),
        }

    # Build original_to_bfs_face mapping
    original_to_bfs_face = np.zeros(F, dtype=np.int32)
    for bfs_idx, orig_idx in enumerate(bfs_to_original_face):
        original_to_bfs_face[orig_idx] = bfs_idx

    # For each edge, find its adjacent faces (in BFS order)
    edge_to_faces_orig = []
    edge_min_face_bfs = []

    for edge_idx in range(E):
        faces = np.where(face_edge_incidence[:, edge_idx])[0]
        if len(faces) >= 2:
            f1_bfs = original_to_bfs_face[faces[0]]
            f2_bfs = original_to_bfs_face[faces[1]]
            edge_to_faces_orig.append([f1_bfs, f2_bfs])
            edge_min_face_bfs.append(min(f1_bfs, f2_bfs))
        elif len(faces) == 1:
            f1_bfs = original_to_bfs_face[faces[0]]
            edge_to_faces_orig.append([f1_bfs, -1])
            edge_min_face_bfs.append(f1_bfs)
        else:
            edge_to_faces_orig.append([-1, -1])
            edge_min_face_bfs.append(F + 1)  # Put at end

    edge_min_face_bfs = np.array(edge_min_face_bfs)

    # Sort edges by min face BFS index, then by XYZ bbox
    # Create sort key: (min_face_bfs, xmin, ymin, zmin, xmax, ymax, zmax)
    sort_keys = np.column_stack([
        edge_min_face_bfs,
        edge_bboxes[:, 0],  # xmin
        edge_bboxes[:, 1],  # ymin
        edge_bboxes[:, 2],  # zmin
        edge_bboxes[:, 3],  # xmax
        edge_bboxes[:, 4],  # ymax
        edge_bboxes[:, 5],  # zmax
    ])

    # Lexicographic sort
    bfs_to_original_edge = np.lexsort(sort_keys.T[::-1])

    # Build reverse mapping
    original_to_bfs_edge = np.zeros(E, dtype=np.int32)
    for bfs_idx, orig_idx in enumerate(bfs_to_original_edge):
        original_to_bfs_edge[orig_idx] = bfs_idx

    # Reorder edge_to_faces to BFS edge order
    edge_to_faces_orig = np.array(edge_to_faces_orig, dtype=np.int32)
    edge_to_faces_bfs = edge_to_faces_orig[bfs_to_original_edge]

    return {
        'bfs_to_original_edge': bfs_to_original_edge.astype(np.int32),
        'original_to_bfs_edge': original_to_bfs_edge,
        'edge_to_faces': edge_to_faces_bfs,
    }


def extract_spatial_properties(
    face_point_grids: np.ndarray,
    edge_point_grids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extract spatial properties (centroids, normals, areas) from point grids.

    Args:
        face_point_grids: [F, 32, 32, 3] surface point samples
        edge_point_grids: [E, 32, 3] curve point samples

    Returns:
        dict with:
            face_centroids: [F, 3] center of each face
            face_normals: [F, 3] estimated outward normal
            face_areas: [F] approximate surface area
            face_bboxes: [F, 6] bounding boxes
            edge_midpoints: [E, 3] center of each edge
            edge_directions: [E, 3] tangent direction at midpoint
            edge_lengths: [E] approximate arc length
            edge_bboxes: [E, 6] bounding boxes
    """
    F = face_point_grids.shape[0] if len(face_point_grids.shape) > 0 else 0
    E = edge_point_grids.shape[0] if len(edge_point_grids.shape) > 0 else 0

    # Initialize arrays
    face_centroids = np.zeros((F, 3), dtype=np.float32)
    face_normals = np.zeros((F, 3), dtype=np.float32)
    face_areas = np.zeros(F, dtype=np.float32)
    face_bboxes = np.zeros((F, 6), dtype=np.float32)

    edge_midpoints = np.zeros((E, 3), dtype=np.float32)
    edge_directions = np.zeros((E, 3), dtype=np.float32)
    edge_lengths = np.zeros(E, dtype=np.float32)
    edge_bboxes = np.zeros((E, 6), dtype=np.float32)

    # Process faces
    for i in range(F):
        grid = face_point_grids[i]  # [32, 32, 3]

        # Centroid: mean of all points
        face_centroids[i] = grid.mean(axis=(0, 1))

        # Bounding box
        face_bboxes[i, :3] = grid.min(axis=(0, 1))
        face_bboxes[i, 3:] = grid.max(axis=(0, 1))

        # Estimate normal from cross product of grid vectors at center
        center_i, center_j = grid.shape[0] // 2, grid.shape[1] // 2

        # Vectors along u and v directions at center
        u_vec = grid[min(center_i + 1, grid.shape[0] - 1), center_j] - grid[max(center_i - 1, 0), center_j]
        v_vec = grid[center_i, min(center_j + 1, grid.shape[1] - 1)] - grid[center_i, max(center_j - 1, 0)]

        normal = np.cross(u_vec, v_vec)
        norm = np.linalg.norm(normal)
        face_normals[i] = normal / norm if norm > 1e-8 else np.array([0, 0, 1])

        # Approximate area: sum of small quad areas
        # Simplified: use bbox diagonal as proxy
        dims = face_bboxes[i, 3:] - face_bboxes[i, :3]
        # Approximate surface area as product of two larger dimensions
        sorted_dims = np.sort(dims)[::-1]
        face_areas[i] = sorted_dims[0] * sorted_dims[1]

    # Process edges
    for i in range(E):
        curve = edge_point_grids[i]  # [32, 3]

        # Midpoint: center point on curve
        mid_idx = curve.shape[0] // 2
        edge_midpoints[i] = curve[mid_idx]

        # Direction: tangent at midpoint (difference of neighbors)
        direction = curve[min(mid_idx + 1, curve.shape[0] - 1)] - curve[max(mid_idx - 1, 0)]
        norm = np.linalg.norm(direction)
        edge_directions[i] = direction / norm if norm > 1e-8 else np.array([1, 0, 0])

        # Bounding box
        edge_bboxes[i, :3] = curve.min(axis=0)
        edge_bboxes[i, 3:] = curve.max(axis=0)

        # Arc length: sum of segment lengths
        segments = np.diff(curve, axis=0)  # [31, 3]
        edge_lengths[i] = np.linalg.norm(segments, axis=1).sum()

    return {
        'face_centroids': face_centroids,
        'face_normals': face_normals,
        'face_areas': face_areas,
        'face_bboxes': face_bboxes,
        'edge_midpoints': edge_midpoints,
        'edge_directions': edge_directions,
        'edge_lengths': edge_lengths,
        'edge_bboxes': edge_bboxes,
    }


def extract_geometry_types(
    face_point_grids: np.ndarray,
    edge_point_grids: np.ndarray,
    planarity_threshold: float = 0.01,
    linearity_threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify face/edge types from point grids using geometric analysis.

    Face types: 0=plane, 1=cylinder, 2=cone, 3=sphere, 4=torus, 5=bspline
    Edge types: 0=line, 1=circle, 2=ellipse, 3=bspline

    Note: This is a simplified classification. For more accurate results,
    the original CAD geometry types should be extracted from STEP file.

    Args:
        face_point_grids: [F, 32, 32, 3] surface point samples
        edge_point_grids: [E, 32, 3] curve point samples
        planarity_threshold: threshold for plane classification
        linearity_threshold: threshold for line classification

    Returns:
        face_types: [F] face type indices
        edge_types: [E] edge type indices
    """
    F = face_point_grids.shape[0] if len(face_point_grids.shape) > 0 else 0
    E = edge_point_grids.shape[0] if len(edge_point_grids.shape) > 0 else 0

    face_types = np.full(F, FACE_TYPE_BSPLINE, dtype=np.int8)
    edge_types = np.full(E, EDGE_TYPE_BSPLINE, dtype=np.int8)

    # Classify faces
    for i in range(F):
        grid = face_point_grids[i]  # [32, 32, 3]

        # Flatten to point cloud
        points = grid.reshape(-1, 3)

        # Compute centroid and center points
        centroid = points.mean(axis=0)
        centered = points - centroid

        # SVD to get principal components
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)

            # Planarity: ratio of smallest to largest singular value
            planarity = s[2] / (s[0] + 1e-8)

            if planarity < planarity_threshold:
                face_types[i] = FACE_TYPE_PLANE
            else:
                # Could add more sophisticated classification here
                # For now, default to bspline for non-planar surfaces
                face_types[i] = FACE_TYPE_BSPLINE
        except Exception:
            face_types[i] = FACE_TYPE_BSPLINE

    # Classify edges
    for i in range(E):
        curve = edge_point_grids[i]  # [32, 3]

        # Check if straight line
        start = curve[0]
        end = curve[-1]
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-8:
            edge_types[i] = EDGE_TYPE_LINE
            continue

        direction = direction / length

        # Compute perpendicular distances from line
        vectors = curve - start
        # Project onto direction
        projections = np.outer(np.dot(vectors, direction), direction)
        perpendiculars = vectors - projections
        max_deviation = np.linalg.norm(perpendiculars, axis=1).max()

        # Normalize by length
        linearity = max_deviation / (length + 1e-8)

        if linearity < linearity_threshold:
            edge_types[i] = EDGE_TYPE_LINE
        else:
            # Could check for circle/ellipse here
            # For now, default to bspline
            edge_types[i] = EDGE_TYPE_BSPLINE

    return face_types, edge_types


def build_face_adjacency_matrix(
    face_edge_incidence: np.ndarray,
    bfs_to_original_face: np.ndarray = None,
) -> np.ndarray:
    """
    Build sparse face-face adjacency matrix.

    Args:
        face_edge_incidence: [F, E] binary incidence matrix
        bfs_to_original_face: optional [F] BFS ordering to reorder result

    Returns:
        adjacency: [F, F] sparse binary matrix (1 if faces share edge)
    """
    F, E = face_edge_incidence.shape
    adjacency = np.zeros((F, F), dtype=np.int8)

    for edge_idx in range(E):
        faces = np.where(face_edge_incidence[:, edge_idx])[0]
        if len(faces) == 2:
            f1, f2 = int(faces[0]), int(faces[1])
            adjacency[f1, f2] = 1
            adjacency[f2, f1] = 1

    # Reorder to BFS order if provided
    if bfs_to_original_face is not None:
        # Create inverse mapping
        original_to_bfs = np.zeros(F, dtype=np.int32)
        for bfs_idx, orig_idx in enumerate(bfs_to_original_face):
            original_to_bfs[orig_idx] = bfs_idx

        # Reorder both dimensions
        adjacency_bfs = np.zeros((F, F), dtype=np.int8)
        for i in range(F):
            for j in range(F):
                if adjacency[i, j]:
                    bfs_i = original_to_bfs[i]
                    bfs_j = original_to_bfs[j]
                    adjacency_bfs[bfs_i, bfs_j] = 1
        return adjacency_bfs

    return adjacency


def reorder_to_bfs(
    arrays_dict: Dict[str, np.ndarray],
    bfs_order: np.ndarray,
    axis: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Reorder arrays to BFS order along specified axis.

    Args:
        arrays_dict: dict of name -> array pairs
        bfs_order: [N] indices giving BFS order (bfs_to_original mapping)
        axis: axis along which to reorder

    Returns:
        dict with reordered arrays
    """
    result = {}
    for name, arr in arrays_dict.items():
        if arr is None or len(arr) == 0:
            result[name] = arr
        else:
            result[name] = np.take(arr, bfs_order, axis=axis)
    return result
