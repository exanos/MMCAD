# Topology Fix Reference - CRITICAL INFORMATION

## Problem Summary

**Training was stuck because `edge_to_faces` is 100% -1 and `bfs_level` is 100% 0.**

This breaks the v4.8.1 architecture which relies on:
- `EdgeMessageLayer` using `edge_to_faces` for message passing
- BFS ordering using `bfs_level` for hierarchy
- Face-edge connectivity for topology-aware encoding

## Root Cause

In `scripts/precompute_brep_autobrep_fast.py` lines 169-188:

```python
# BROKEN CODE (C++ iterator pattern doesn't work in PythonOCC):
flist = emap.FindFromIndex(idx)
it = flist.begin()
while it != flist.end():  # ← ALWAYS FALSE IN PYTHONOCC!
    face = it.Value()
    it.Next()

# This silently fails, leaving incidence matrix as all zeros
except:
    pass  # ← SILENTLY SWALLOWS ERROR
```

## The Fix

Use correct PythonOCC iterator pattern:

```python
from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

flist = emap.FindFromIndex(idx)
it = TopTools_ListIteratorOfListOfShape(flist)  # CORRECT!
while it.More():  # CORRECT!
    face = topods.Face(it.Value())
    it.Next()
```

## Fix Script

**Location:** `scripts/fix_topology_fast.py`

**Usage:**
```bash
python scripts/fix_topology_fast.py \
    --step-dir D:/Defect_Det/MMCAD/data/extracted_step_files \
    --h5-dir D:/Defect_Det/MMCAD/data/embeddings \
    --temp-dir C:/Users/User/Desktop/temp_h5 \
    --num-workers 28
```

**Optimizations:**
1. Copies HDF5 to SSD (C:) for faster I/O
2. Chunks files for HDD locality
3. Incremental saves with resume support
4. Timeout handling for stuck OCC calls
5. Uses 28 workers on Xeon 8352Y

**Expected time:** 4-8 hours for 169k samples

## Hardware

- **CPU:** Intel Xeon 8352Y (32 cores @ 2.2GHz)
- **RAM:** 256GB
- **GPU:** RTX 4090 24GB
- **Storage:** D: = HDD (data), C: = SSD (temp)

## Data Flow

```
STEP files (D: HDD)
    ↓
[1] OCC reads shape
    ↓
[2] TopExp_Explorer extracts faces/edges
    ↓
[3] MapShapesAndAncestors builds edge→face mapping
    ↓
[4] TopTools_ListIteratorOfListOfShape iterates faces per edge  ← THE FIX
    ↓
[5] Build face_edge_incidence matrix
    ↓
[6] bfs_order_faces_with_parents() computes BFS level
    ↓
[7] bfs_order_edges() computes edge_to_faces
    ↓
[8] Save to HDF5 (C: SSD, then copy to D:)
```

## Key Functions (clip4cad/data/autobrep_utils.py)

### bfs_order_faces_with_parents()
- Input: `face_edge_incidence` [F, E], `face_bboxes` [F, 6]
- Output: `bfs_to_original_face`, `bfs_level`, `bfs_parent_face`, `bfs_parent_edge`
- Algorithm: BFS from face with smallest XYZ bbox, neighbors sorted by XYZ

### bfs_order_edges()
- Input: `face_edge_incidence`, `bfs_to_original_face`, `edge_bboxes`
- Output: `bfs_to_original_edge`, `edge_to_faces` [E, 2]
- Algorithm: Edges ordered by min BFS face index, then XYZ bbox

### build_face_graph_from_incidence()
- Builds face adjacency graph from incidence matrix
- Returns `face_neighbors` dict and `edge_between_faces` dict

## HDF5 Structure (after fix)

```
train_brep_autobrep.h5:
├── face_features      [N, 192, 48]   float32  - FSQ face latents
├── edge_features      [N, 512, 12]   float32  - FSQ edge latents
├── face_masks         [N, 192]       float32  - Valid face mask
├── edge_masks         [N, 512]       float32  - Valid edge mask
├── edge_to_faces      [N, 512, 2]    int32    - Edge connectivity ← FIXED
├── bfs_level          [N, 192]       int32    - BFS hierarchy ← FIXED
├── face_centroids     [N, 192, 3]    float32  - Face centers
├── face_normals       [N, 192, 3]    float32  - Face normals (optional)
├── face_areas         [N, 192]       float32  - Face areas (optional)
├── edge_midpoints     [N, 512, 3]    float32  - Edge centers (optional)
├── edge_lengths       [N, 512]       float32  - Edge lengths (optional)
├── num_faces          [N]            int32    - Valid face count
├── num_edges          [N]            int32    - Valid edge count
└── uids               [N]            string   - Sample IDs
```

## Verification

After running the fix, verify with:

```python
import h5py
import numpy as np

with h5py.File("data/embeddings/train_brep_autobrep.h5", "r") as f:
    e2f = f["edge_to_faces"][:]  # [N, 512, 2]
    bfs = f["bfs_level"][:]      # [N, 192]

    # Check if ANY edge has valid face connections per sample
    valid_e2f = ((e2f != -1).any(axis=(1, 2))).sum()  # Correct for 3D array
    valid_bfs = (bfs != 0).any(axis=1).sum()

    print(f"Valid edge_to_faces: {valid_e2f}/{len(e2f)}")
    print(f"Valid bfs_level: {valid_bfs}/{len(bfs)}")

# Should show >95% valid (some samples may genuinely have issues)
```

## Model Architecture (v4.8.1)

After topology fix, the model flow is:

```
BRep Input:
├── face_features [B, 192, 48]
├── edge_features [B, 512, 12]
├── edge_to_faces [B, 512, 2]  ← NOW VALID
└── bfs_level [B, 192]         ← NOW VALID

    ↓ face_proj, edge_proj

├── F [B, 192, 256]
└── E [B, 512, 256]

    ↓ EdgeMessageLayer (uses edge_to_faces!)

├── F' [B, 192, 256]  (faces updated with edge info)
└── E' [B, 512, 256]  (edges updated with face info)

    ↓ BRep Transformer (uses bfs_level for positional encoding!)

└── z_brep [B, 256]  (global BRep embedding)
```

## Alternative: AutoBrep-style Encoder

If topology fix doesn't work, use `clip4cad/models/brep_encoder_autobrep.py`:
- Doesn't rely on edge_to_faces or bfs_level
- Uses transformer attention to learn topology implicitly
- Sorts faces by centroid XYZ order
- Simpler but may be less effective

## Files Created/Modified

1. **scripts/fix_topology_fast.py** - The fix script
2. **clip4cad/models/brep_encoder_autobrep.py** - Fallback encoder
3. **notebooks/test_autobrep_encoder.py** - Test script for fallback
4. **docs/TOPOLOGY_FIX_REFERENCE.md** - This document

## Next Steps

1. Run `fix_topology_fast.py` (4-8 hours)
2. Verify with the verification code above
3. Train v4.8.1 model with fixed data
4. If still not learning, try AutoBrep-style encoder as fallback
