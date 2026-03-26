"""
Attention Visualization for CLIP4CAD

Provides tools to trace attention back to specific faces/edges with full context,
enabling interpretable analysis of what the model is attending to.

Features:
- Get top-K attended faces with full visualization context
- 3D scatter plots of attention heatmaps
- Point cloud rendering of attended faces
- BFS hierarchy visualization

Usage:
    from clip4cad.visualization import AttentionVisualizer
    from clip4cad.data import AutoBrepDataset

    ds = AutoBrepDataset("brep_autobrep.h5", raw_hdf5="brep_autobrep_raw.h5")
    viz = AttentionVisualizer(ds)

    # Get top attended faces
    top_faces = viz.get_top_attended_faces(0, grounding_matrix, slot_idx=0)

    # Plot 3D attention heatmap
    viz.plot_attention_heatmap_3d(0, grounding_matrix, slot_idx=0)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


# Type constants
FACE_TYPE_NAMES = {
    0: 'plane',
    1: 'cylinder',
    2: 'cone',
    3: 'sphere',
    4: 'torus',
    5: 'bspline',
}

EDGE_TYPE_NAMES = {
    0: 'line',
    1: 'circle',
    2: 'ellipse',
    3: 'bspline',
}


class AttentionVisualizer:
    """
    Trace attention back to specific faces/edges with full context.

    Requires AutoBrepDataset with raw geometry for visualization.

    Args:
        dataset: AutoBrepDataset instance with raw_hdf5 loaded
    """

    def __init__(self, dataset):
        """
        Initialize visualizer.

        Args:
            dataset: AutoBrepDataset instance
        """
        self.dataset = dataset

        # Check that raw geometry is available
        if dataset.raw_hdf5_path is None:
            print("Warning: Raw HDF5 not provided. Some visualization features unavailable.")

    def get_top_attended_faces(
        self,
        sample_idx: int,
        grounding_matrix: torch.Tensor,
        slot_idx: int,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Get top-K attended faces with full visualization context.

        Args:
            sample_idx: Index of sample in dataset
            grounding_matrix: [K, N] attention weights from GFA model (K=slots, N=faces)
            slot_idx: Which slot (text token) to analyze
            top_k: Number of top faces to return

        Returns:
            List of dicts, each containing:
                bfs_idx: Position in BFS order
                original_face_id: Original CAD face index
                bfs_level: Level in BFS tree (0=root)
                parent_face_bfs: Parent face in BFS tree
                centroid: [3] face center
                normal: [3] face normal
                area: Face area
                bbox: [6] bounding box
                face_type: Surface type name (plane, cylinder, etc.)
                point_grid: [32, 32, 3] raw geometry (if raw file available)
                attention_weight: Attention score
        """
        # Get sample data
        sample = self.dataset[sample_idx]
        num_faces = sample['num_faces']

        # Get attention for this slot, only for valid faces
        if isinstance(grounding_matrix, torch.Tensor):
            grounding_matrix = grounding_matrix.detach().cpu()

        face_attention = grounding_matrix[slot_idx, :num_faces]

        # Get top-K indices
        top_k_actual = min(top_k, num_faces)
        top_values, top_indices = torch.topk(face_attention, top_k_actual)

        # Get raw geometry if available
        raw_geom = None
        if self.dataset.raw_hdf5_path is not None:
            try:
                raw_geom = self.dataset.get_raw_geometry(sample_idx)
            except Exception:
                pass

        results = []
        for i, bfs_idx in enumerate(top_indices.numpy()):
            bfs_idx = int(bfs_idx)

            result = {
                'bfs_idx': bfs_idx,
                'original_face_id': int(sample['bfs_to_original_face'][bfs_idx].item()),
                'bfs_level': int(sample['face_bfs_level'][bfs_idx].item()),
                'parent_face_bfs': int(sample['bfs_parent_face'][bfs_idx].item()),
                'centroid': sample['face_centroids'][bfs_idx].numpy(),
                'attention_weight': top_values[i].item(),
            }

            # Add raw geometry info if available
            if raw_geom is not None:
                result['normal'] = raw_geom['face_normals'][bfs_idx]
                result['area'] = float(raw_geom['face_areas'][bfs_idx])
                result['bbox'] = raw_geom['face_bboxes'][bfs_idx]
                result['face_type'] = FACE_TYPE_NAMES.get(
                    int(raw_geom['face_types'][bfs_idx]), 'unknown'
                )
                result['point_grid'] = raw_geom['face_point_grids'][bfs_idx]

            results.append(result)

        return results

    def get_top_attended_edges(
        self,
        sample_idx: int,
        grounding_matrix: torch.Tensor,
        slot_idx: int,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Get top-K attended edges with full visualization context.

        Args:
            sample_idx: Index of sample in dataset
            grounding_matrix: [K, M] attention weights (K=slots, M=edges)
            slot_idx: Which slot to analyze
            top_k: Number of top edges to return

        Returns:
            List of dicts with edge info
        """
        sample = self.dataset[sample_idx]
        num_edges = sample['num_edges']

        if isinstance(grounding_matrix, torch.Tensor):
            grounding_matrix = grounding_matrix.detach().cpu()

        edge_attention = grounding_matrix[slot_idx, :num_edges]

        top_k_actual = min(top_k, num_edges)
        top_values, top_indices = torch.topk(edge_attention, top_k_actual)

        raw_geom = None
        if self.dataset.raw_hdf5_path is not None:
            try:
                raw_geom = self.dataset.get_raw_geometry(sample_idx)
            except Exception:
                pass

        results = []
        for i, bfs_idx in enumerate(top_indices.numpy()):
            bfs_idx = int(bfs_idx)

            result = {
                'bfs_idx': bfs_idx,
                'original_edge_id': int(sample['bfs_to_original_edge'][bfs_idx].item()),
                'adjacent_faces': sample['edge_to_faces'][bfs_idx].numpy(),
                'attention_weight': top_values[i].item(),
            }

            if raw_geom is not None:
                result['midpoint'] = raw_geom['edge_midpoints'][bfs_idx]
                result['direction'] = raw_geom['edge_directions'][bfs_idx]
                result['length'] = float(raw_geom['edge_lengths'][bfs_idx])
                result['edge_type'] = EDGE_TYPE_NAMES.get(
                    int(raw_geom['edge_types'][bfs_idx]), 'unknown'
                )
                result['point_curve'] = raw_geom['edge_point_grids'][bfs_idx]

            results.append(result)

        return results

    def plot_attention_heatmap_3d(
        self,
        sample_idx: int,
        grounding_matrix: torch.Tensor,
        slot_idx: int,
        ax=None,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        size_by_area: bool = True,
    ):
        """
        Plot 3D scatter of face centroids colored by attention.

        Args:
            sample_idx: Index of sample
            grounding_matrix: [K, N] attention weights
            slot_idx: Which slot to visualize
            ax: Matplotlib 3D axis (created if None)
            show_colorbar: Whether to show colorbar
            title: Plot title (auto-generated if None)
            size_by_area: Scale point sizes by face area

        Returns:
            matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

        sample = self.dataset[sample_idx]
        num_faces = sample['num_faces']

        if isinstance(grounding_matrix, torch.Tensor):
            grounding_matrix = grounding_matrix.detach().cpu().numpy()

        attention = grounding_matrix[slot_idx, :num_faces]
        centroids = sample['face_centroids'][:num_faces].numpy()

        # Get face areas for sizing
        sizes = np.ones(num_faces) * 50
        if size_by_area and self.dataset.raw_hdf5_path is not None:
            try:
                raw_geom = self.dataset.get_raw_geometry(sample_idx)
                areas = raw_geom['face_areas'][:num_faces]
                # Normalize areas to reasonable point sizes
                areas = np.clip(areas, 0, np.percentile(areas, 95))
                sizes = 50 + 200 * (areas / (areas.max() + 1e-8))
            except Exception:
                pass

        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c=attention,
            cmap='hot',
            s=sizes,
            alpha=0.7,
        )

        if show_colorbar:
            plt.colorbar(scatter, ax=ax, label='Attention Weight', shrink=0.6)

        if title is None:
            title = f'Face Attention for Slot {slot_idx}'
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return ax

    def plot_bfs_hierarchy(
        self,
        sample_idx: int,
        ax=None,
        color_by_level: bool = True,
        show_edges: bool = True,
    ):
        """
        Plot BFS hierarchy of faces.

        Args:
            sample_idx: Index of sample
            ax: Matplotlib 3D axis
            color_by_level: Color faces by BFS level
            show_edges: Draw lines connecting adjacent faces

        Returns:
            matplotlib axis
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        sample = self.dataset[sample_idx]
        num_faces = sample['num_faces']

        centroids = sample['face_centroids'][:num_faces].numpy()
        bfs_levels = sample['face_bfs_level'][:num_faces].numpy()
        parent_faces = sample['bfs_parent_face'][:num_faces].numpy()

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Color by level
        if color_by_level:
            colors = bfs_levels
            cmap = 'viridis'
        else:
            colors = 'blue'
            cmap = None

        scatter = ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c=colors,
            cmap=cmap,
            s=50,
            alpha=0.7,
        )

        # Draw edges to parents
        if show_edges:
            for i in range(num_faces):
                parent = parent_faces[i]
                if parent >= 0:
                    ax.plot(
                        [centroids[i, 0], centroids[parent, 0]],
                        [centroids[i, 1], centroids[parent, 1]],
                        [centroids[i, 2], centroids[parent, 2]],
                        'k-',
                        alpha=0.3,
                        linewidth=0.5,
                    )

        if color_by_level:
            plt.colorbar(scatter, ax=ax, label='BFS Level', shrink=0.6)

        ax.set_title(f'BFS Hierarchy (Sample {sample_idx})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return ax

    def render_attended_faces_pointcloud(
        self,
        sample_idx: int,
        grounding_matrix: torch.Tensor,
        slot_idx: int,
        threshold: float = 0.1,
    ):
        """
        Render point grids of highly-attended faces as a colored point cloud.

        Args:
            sample_idx: Index of sample
            grounding_matrix: [K, N] attention weights
            slot_idx: Which slot to visualize
            threshold: Minimum attention weight to include

        Returns:
            trimesh.PointCloud or (points, colors) tuple if trimesh unavailable
        """
        if self.dataset.raw_hdf5_path is None:
            raise ValueError("Raw HDF5 file required for point cloud rendering")

        sample = self.dataset[sample_idx]
        raw_geom = self.dataset.get_raw_geometry(sample_idx)
        num_faces = sample['num_faces']

        if isinstance(grounding_matrix, torch.Tensor):
            grounding_matrix = grounding_matrix.detach().cpu().numpy()

        attention = grounding_matrix[slot_idx, :num_faces]

        # Get faces above threshold
        attended_mask = attention > threshold
        attended_indices = np.where(attended_mask)[0]

        if len(attended_indices) == 0:
            print(f"No faces above threshold {threshold}")
            return None

        # Combine point grids
        all_points = []
        all_colors = []

        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.hot
        except ImportError:
            # Fallback: simple gradient
            def cmap(x):
                return (x, 0, 1 - x, 1)

        for bfs_idx in attended_indices:
            grid = raw_geom['face_point_grids'][bfs_idx]  # [32, 32, 3]
            points = grid.reshape(-1, 3)

            # Color by attention
            color = cmap(attention[bfs_idx])[:3]
            colors = np.tile(color, (len(points), 1))

            all_points.append(points)
            all_colors.append(colors)

        points = np.vstack(all_points)
        colors = np.vstack(all_colors)

        # Try to return trimesh PointCloud
        try:
            import trimesh
            # Colors need to be 0-255 for trimesh
            colors_uint8 = (colors * 255).astype(np.uint8)
            return trimesh.PointCloud(points, colors=colors_uint8)
        except ImportError:
            return points, colors

    def render_model_pointcloud(
        self,
        sample_idx: int,
        include_edges: bool = False,
    ):
        """
        Render complete model as point cloud.

        Args:
            sample_idx: Index of sample
            include_edges: Include edge curves in point cloud

        Returns:
            trimesh.PointCloud or (points, colors) tuple
        """
        if self.dataset.raw_hdf5_path is None:
            raise ValueError("Raw HDF5 file required for point cloud rendering")

        raw_geom = self.dataset.get_raw_geometry(sample_idx)
        num_faces = raw_geom['num_faces']
        num_edges = raw_geom['num_edges']

        # Collect face points
        all_points = []
        all_colors = []

        # Face points (gray)
        for i in range(num_faces):
            grid = raw_geom['face_point_grids'][i]
            points = grid.reshape(-1, 3)
            colors = np.tile([0.7, 0.7, 0.7], (len(points), 1))
            all_points.append(points)
            all_colors.append(colors)

        # Edge points (red)
        if include_edges:
            for i in range(num_edges):
                curve = raw_geom['edge_point_grids'][i]
                colors = np.tile([1.0, 0.2, 0.2], (len(curve), 1))
                all_points.append(curve)
                all_colors.append(colors)

        points = np.vstack(all_points)
        colors = np.vstack(all_colors)

        try:
            import trimesh
            colors_uint8 = (colors * 255).astype(np.uint8)
            return trimesh.PointCloud(points, colors=colors_uint8)
        except ImportError:
            return points, colors

    def compare_slots_attention(
        self,
        sample_idx: int,
        grounding_matrix: torch.Tensor,
        slot_indices: List[int],
        figsize: Tuple[int, int] = (15, 5),
    ):
        """
        Compare attention patterns across multiple slots.

        Args:
            sample_idx: Index of sample
            grounding_matrix: [K, N] attention weights
            slot_indices: List of slot indices to compare
            figsize: Figure size

        Returns:
            matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        n_slots = len(slot_indices)
        fig = plt.figure(figsize=figsize)

        for i, slot_idx in enumerate(slot_indices):
            ax = fig.add_subplot(1, n_slots, i + 1, projection='3d')
            self.plot_attention_heatmap_3d(
                sample_idx,
                grounding_matrix,
                slot_idx,
                ax=ax,
                show_colorbar=(i == n_slots - 1),
                title=f'Slot {slot_idx}',
            )

        plt.tight_layout()
        return fig
