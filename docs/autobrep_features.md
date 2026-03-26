# AutoBrep BFS-Ordered Feature Storage

This document describes the storage format and usage of AutoBrep BFS-ordered B-Rep features for CLIP4CAD.

## Overview

The AutoBrep feature storage preserves the full BFS-ordered sequence representation from the AutoBrep paper, enabling:
1. **Attention Visualization** - Trace attention back to specific faces/edges in the original CAD model
2. **Information Preservation** - Store everything AutoBrep uses for generation
3. **Topology Awareness** - Face-face adjacency, BFS hierarchy, edge connectivity

## Two-File Storage Design

The features are stored in two HDF5 files, optimized for different use cases:

### Main File (`brep_autobrep.h5`) - ~2GB
**Used for training.** Small enough to load into RAM for fast batch iteration.

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `face_features` | [N, F, 48] | float32 | FSQ latents (BFS ordered) |
| `edge_features` | [N, E, 12] | float32 | FSQ latents (BFS ordered) |
| `face_masks` | [N, F] | float32 | Valid face mask |
| `edge_masks` | [N, E] | float32 | Valid edge mask |
| `bfs_to_original_face` | [N, F] | int32 | BFS idx → original CAD face ID |
| `bfs_to_original_edge` | [N, E] | int32 | BFS idx → original CAD edge ID |
| `bfs_level` | [N, F] | int32 | BFS tree level (0=root) |
| `bfs_parent_face` | [N, F] | int32 | Parent face in BFS tree (-1=root) |
| `bfs_parent_edge` | [N, F] | int32 | Edge connecting to parent |
| `edge_to_faces` | [N, E, 2] | int32 | Adjacent face BFS indices |
| `face_centroids` | [N, F, 3] | float32 | Face centers (for quick 3D viz) |
| `num_faces` | [N] | int32 | Actual face count |
| `num_edges` | [N] | int32 | Actual edge count |
| `uids` | [N] | string | Sample identifiers |

### Raw File (`brep_autobrep_raw.h5`) - ~50GB
**Used for visualization.** Loaded lazily on demand.

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `face_point_grids` | [N, F, 32, 32, 3] | float32 | Raw surface samples |
| `edge_point_grids` | [N, E, 32, 3] | float32 | Raw curve samples |
| `face_normals` | [N, F, 3] | float32 | Outward normal vectors |
| `face_areas` | [N, F] | float32 | Surface areas |
| `face_bboxes` | [N, F, 6] | float32 | [xmin,ymin,zmin,xmax,ymax,zmax] |
| `edge_midpoints` | [N, E, 3] | float32 | Curve centers |
| `edge_directions` | [N, E, 3] | float32 | Tangent at midpoint |
| `edge_lengths` | [N, E] | float32 | Arc lengths |
| `edge_bboxes` | [N, E, 6] | float32 | Edge bounding boxes |
| `face_types` | [N, F] | int8 | Surface type (see below) |
| `edge_types` | [N, E] | int8 | Curve type (see below) |
| `face_adjacency` | [N, F, F] | int8 | Sparse adjacency matrix |
| `uids` | [N] | string | For joining with main file |

### Geometry Type Constants

**Face Types:**
- 0: plane
- 1: cylinder
- 2: cone
- 3: sphere
- 4: torus
- 5: bspline

**Edge Types:**
- 0: line
- 1: circle
- 2: ellipse
- 3: bspline

## BFS Ordering

Following the AutoBrep paper, faces are ordered by BFS traversal:

1. **Start Face**: Face with smallest XYZ bounding box (lexicographic sort)
2. **Traversal**: Visit neighbors in XYZ-sorted order
3. **Deterministic**: Same input always produces same BFS order

```python
# BFS gives you:
bfs_to_original_face[i]  # BFS position i → original CAD face index
original_to_bfs_face[j]  # Original face j → BFS position
bfs_level[i]             # How many hops from start face
bfs_parent_face[i]       # Parent in BFS tree (-1 for root)
bfs_parent_edge[i]       # Edge connecting to parent
```

## Usage

### Training (Fast)

```python
from clip4cad.data import AutoBrepDataset, create_autobrep_dataloader

# Load main file to RAM for fast training
ds = AutoBrepDataset(
    "embeddings/brep_autobrep.h5",
    load_main_to_memory=True  # ~2GB RAM
)

# Or use dataloader
train_loader = create_autobrep_dataloader(
    "embeddings/brep_autobrep.h5",
    batch_size=512,
    shuffle=True,
    num_workers=4,
)

for batch in train_loader:
    face_features = batch['brep_face_features']  # [B, F, 48]
    edge_features = batch['brep_edge_features']  # [B, E, 12]
    face_mask = batch['brep_face_mask']          # [B, F] bool
    # ... train model
```

### Visualization

```python
from clip4cad.data import AutoBrepDataset
from clip4cad.visualization import AttentionVisualizer

# Load both files
ds = AutoBrepDataset(
    "embeddings/brep_autobrep.h5",
    raw_hdf5="embeddings/brep_autobrep_raw.h5",
    load_main_to_memory=True
)

# Create visualizer
viz = AttentionVisualizer(ds)

# After running model, get attention matrix
grounding_matrix = model.get_grounding_weights(...)  # [K, N]

# Get top attended faces for slot 0
top_faces = viz.get_top_attended_faces(
    sample_idx=0,
    grounding_matrix=grounding_matrix,
    slot_idx=0,
    top_k=5
)

for face in top_faces:
    print(f"BFS idx: {face['bfs_idx']}")
    print(f"Original CAD face: {face['original_face_id']}")
    print(f"BFS level: {face['bfs_level']}")
    print(f"Attention: {face['attention_weight']:.3f}")
    print(f"Face type: {face['face_type']}")
    print()

# Plot 3D attention heatmap
viz.plot_attention_heatmap_3d(0, grounding_matrix, slot_idx=0)

# Render attended faces as point cloud
pc = viz.render_attended_faces_pointcloud(0, grounding_matrix, slot_idx=0, threshold=0.1)
pc.show()  # If trimesh installed
```

### Raw Geometry Access

```python
# Get raw geometry for visualization
raw = ds.get_raw_geometry(sample_idx=0)

# Access specific face point grid
face_grid = raw['face_point_grids'][face_bfs_idx]  # [32, 32, 3]

# Or use convenience method
face_grid = ds.get_face_point_cloud(sample_idx=0, face_bfs_idx=5)
```

## Preprocessing

Generate features from STEP files:

```bash
python scripts/precompute_brep_autobrep.py \
    --step-dir ../data/extracted_step_files \
    --csv ../data/169k.csv \
    --output-dir ../data/embeddings \
    --surface-checkpoint pretrained/autobrep/surf-fsq.ckpt \
    --edge-checkpoint pretrained/autobrep/edge-fsq.ckpt \
    --num-workers 8 \
    --batch-size 32 \
    --max-faces 192 \
    --max-edges 512 \
    --checkpoint-every 100

# Resume from crash
python scripts/precompute_brep_autobrep.py ... --resume

# Skip raw file for faster preprocessing (no visualization)
python scripts/precompute_brep_autobrep.py ... --skip-raw
```

## Comparison with Previous Format

| Feature | Old (`brep_features.h5`) | New (`brep_autobrep.h5`) |
|---------|-------------------------|--------------------------|
| Face ordering | Arbitrary | BFS ordered |
| Face-face adjacency | Not stored | `face_adjacency` matrix |
| BFS hierarchy | Not stored | `bfs_level`, `bfs_parent_*` |
| Index mapping | Not stored | `bfs_to_original_*` |
| Raw geometry | Not stored | Separate raw file |
| Attention viz | Not possible | Full traceability |

## File Size Estimates

For ~170K samples with max_faces=192, max_edges=512:

| File | Size | Load Time |
|------|------|-----------|
| `brep_autobrep.h5` (main) | ~2 GB | <10s to RAM |
| `brep_autobrep_raw.h5` (raw) | ~50 GB | Lazy load |

## Integration with GFA Model

The BFS-ordered features are a drop-in replacement for training:

```python
# Model doesn't need changes - features are already BFS ordered
from clip4cad.models import CLIP4CAD_GFA

model = CLIP4CAD_GFA(config)

# Train with AutoBrep dataset
for batch in train_loader:
    outputs = model(
        brep_face_features=batch['brep_face_features'],
        brep_edge_features=batch['brep_edge_features'],
        brep_face_mask=batch['brep_face_mask'],
        brep_edge_mask=batch['brep_edge_mask'],
        # ... other inputs
    )
```

Optionally, add BFS positional encoding:

```python
class BFSPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_levels: int = 32):
        super().__init__()
        self.level_embed = nn.Embedding(max_levels, d_model)

    def forward(self, tokens, bfs_levels):
        return tokens + self.level_embed(bfs_levels)
```
