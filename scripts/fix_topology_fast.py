#!/usr/bin/env python3
"""
FAST TOPOLOGY FIX - Extracts face_edge_incidence and fixes edge_to_faces/bfs_level

ROOT CAUSE OF BUG:
The original precompute_brep_autobrep_fast.py used C++ iterator pattern that doesn't
work with PythonOCC:
    it = flist.begin()
    while it != flist.end():  # BROKEN!

CORRECT PATTERN:
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
    it = TopTools_ListIteratorOfListOfShape(flist)
    while it.More():  # CORRECT!

OPTIMIZATIONS (from augment_spatial_fields.py):
1. Sort files by path for HDD locality
2. Chunk-based processing (many files per worker)
3. Incremental HDF5 writes with checkpoints
4. Resume support
5. Timeout handling for stuck OCC calls
6. Write HDF5 to SSD (C:) for speed, then move to final location

HARDWARE TARGET:
- Intel Xeon 8352Y (32 cores)
- 256GB RAM
- RTX 4090 (not used here - CPU only)
- D: = HDD (STEP files)
- C: = SSD (temp HDF5 for speed)

USAGE:
    python scripts/fix_topology_fast.py \
        --step-dir D:/Defect_Det/MMCAD/data/extracted_step_files \
        --h5-dir D:/Defect_Det/MMCAD/data/embeddings \
        --temp-dir C:/Users/User/Desktop/temp_h5 \
        --num-workers 28

Expected time: 4-8 hours for 169k samples
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import gc
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================
MAX_FACES = 192
MAX_EDGES = 512


# =============================================================================
# Worker Functions
# =============================================================================

def process_chunk(chunk: List[Tuple[str, str]]) -> List[Optional[Dict]]:
    """
    Worker: Process a CHUNK of STEP files to extract topology.
    Uses CORRECT PythonOCC iterator pattern.
    """
    import sys
    import os
    import platform

    # Suppress OCC stderr
    if platform.system() == "Windows":
        sys.stderr = open('NUL', 'w')
        os.dup2(sys.stderr.fileno(), 2)
    else:
        sys.stderr = open('/dev/null', 'w')
        os.dup2(sys.stderr.fileno(), 2)

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopTools import (
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_ListIteratorOfListOfShape,  # CRITICAL: Correct iterator!
        )
    except Exception as e:
        return [{"uid": uid, "error": f"Import failed: {e}"} for uid, _ in chunk]

    results = []
    for uid, filepath in chunk:
        result = extract_topology_single(
            uid, filepath,
            STEPControl_Reader, IFSelect_RetDone,
            TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE,
            topods, topexp,
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_ListIteratorOfListOfShape,
        )
        results.append(result)

    return results


def extract_topology_single(
    uid, filepath,
    STEPControl_Reader, IFSelect_RetDone,
    TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE,
    topods, topexp,
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape,
) -> Optional[Dict]:
    """
    Extract face_edge_incidence matrix from a single STEP file.

    Returns:
        dict with uid, num_faces, num_edges, incidence, face_bboxes, edge_bboxes
        or dict with uid, error if failed
    """
    try:
        # Read STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(filepath))
        if status != IFSelect_RetDone:
            return {"uid": uid, "error": "ReadFile failed"}

        reader.TransferRoots()
        shape = reader.OneShape()

        if shape.IsNull():
            return {"uid": uid, "error": "Null shape"}

        # Collect faces with their shapes for later comparison
        face_shapes = []
        face_bboxes = []

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More() and len(face_shapes) < MAX_FACES:
            try:
                face = topods.Face(exp.Current())
                face_shapes.append(face)

                # Get bounding box via surface sampling (fast approximation)
                from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
                surface = BRepAdaptor_Surface(face)
                u_min, u_max = surface.FirstUParameter(), surface.LastUParameter()
                v_min, v_max = surface.FirstVParameter(), surface.LastVParameter()

                # Clamp infinite parameters
                u_min, u_max = max(u_min, -1e6), min(u_max, 1e6)
                v_min, v_max = max(v_min, -1e6), min(v_max, 1e6)

                # Sample corners for bbox
                points = []
                for u in [u_min, u_max]:
                    for v in [v_min, v_max]:
                        try:
                            pnt = surface.Value(u, v)
                            points.append([pnt.X(), pnt.Y(), pnt.Z()])
                        except:
                            pass

                if points:
                    points = np.array(points)
                    bbox = np.concatenate([points.min(axis=0), points.max(axis=0)])
                else:
                    bbox = np.zeros(6)
                face_bboxes.append(bbox)
            except:
                pass
            exp.Next()

        # Collect edges
        edge_shapes = []
        edge_bboxes = []

        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More() and len(edge_shapes) < MAX_EDGES:
            try:
                edge = topods.Edge(exp.Current())
                edge_shapes.append(edge)

                # Get edge bbox via curve sampling
                from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
                curve = BRepAdaptor_Curve(edge)
                t_min, t_max = curve.FirstParameter(), curve.LastParameter()
                t_min, t_max = max(t_min, -1e6), min(t_max, 1e6)

                points = []
                for t in np.linspace(t_min, t_max, 5):
                    try:
                        pnt = curve.Value(t)
                        points.append([pnt.X(), pnt.Y(), pnt.Z()])
                    except:
                        pass

                if points:
                    points = np.array(points)
                    bbox = np.concatenate([points.min(axis=0), points.max(axis=0)])
                else:
                    bbox = np.zeros(6)
                edge_bboxes.append(bbox)
            except:
                pass
            exp.Next()

        num_f = len(face_shapes)
        num_e = len(edge_shapes)

        if num_f == 0:
            return {"uid": uid, "error": "No faces"}

        # =====================================================================
        # BUILD INCIDENCE MATRIX - THE CRITICAL FIX
        # =====================================================================
        incidence = np.zeros((num_f, num_e), dtype=np.int8)
        incidence_populated = 0

        # Method 1: Use MapShapesAndAncestors (edge -> faces)
        try:
            emap = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, emap)

            for ei, edge in enumerate(edge_shapes):
                idx = emap.FindIndex(edge)
                if idx > 0:
                    face_list = emap.FindFromIndex(idx)

                    # CORRECT ITERATOR PATTERN FOR PYTHONOCC!
                    it = TopTools_ListIteratorOfListOfShape(face_list)
                    while it.More():
                        try:
                            ancestor_face = topods.Face(it.Value())
                            # Find matching face in our list
                            for fi, f in enumerate(face_shapes):
                                if f.IsSame(ancestor_face):
                                    incidence[fi, ei] = 1
                                    incidence_populated += 1
                                    break
                        except:
                            pass
                        it.Next()
        except Exception as e:
            # Method 1 failed, try Method 2
            pass

        # Method 2: If Method 1 failed, try direct edge exploration per face
        if incidence_populated == 0:
            try:
                for fi, face in enumerate(face_shapes):
                    edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
                    while edge_exp.More():
                        try:
                            face_edge = topods.Edge(edge_exp.Current())
                            for ei, e in enumerate(edge_shapes):
                                if e.IsSame(face_edge):
                                    incidence[fi, ei] = 1
                                    incidence_populated += 1
                                    break
                        except:
                            pass
                        edge_exp.Next()
            except:
                pass

        # Pad to max sizes
        incidence_padded = np.zeros((MAX_FACES, MAX_EDGES), dtype=np.int8)
        incidence_padded[:num_f, :num_e] = incidence

        face_bboxes_padded = np.zeros((MAX_FACES, 6), dtype=np.float32)
        edge_bboxes_padded = np.zeros((MAX_EDGES, 6), dtype=np.float32)
        if face_bboxes:
            face_bboxes_padded[:num_f] = np.array(face_bboxes)
        if edge_bboxes:
            edge_bboxes_padded[:num_e] = np.array(edge_bboxes)

        return {
            "uid": uid,
            "num_faces": num_f,
            "num_edges": num_e,
            "incidence": incidence_padded,
            "face_bboxes": face_bboxes_padded,
            "edge_bboxes": edge_bboxes_padded,
            "incidence_populated": incidence_populated,
        }

    except Exception as e:
        return {"uid": uid, "error": str(e)[:100]}


def compute_topology_fields(result: Dict) -> Dict:
    """
    Compute edge_to_faces and bfs_level from incidence matrix.
    Uses the same logic as autobrep_utils.py
    """
    if "error" in result:
        return result

    uid = result["uid"]
    num_f = result["num_faces"]
    num_e = result["num_edges"]
    incidence = result["incidence"][:num_f, :num_e]
    face_bboxes = result["face_bboxes"][:num_f]
    edge_bboxes = result["edge_bboxes"][:num_e]

    # Import utilities
    from clip4cad.data.autobrep_utils import (
        bfs_order_faces_with_parents,
        bfs_order_edges,
    )

    try:
        # BFS order faces
        face_bfs = bfs_order_faces_with_parents(incidence, face_bboxes)

        # BFS order edges
        edge_bfs = bfs_order_edges(incidence, face_bfs["bfs_to_original_face"], edge_bboxes)

        # Pad to max sizes
        bfs_level = np.zeros(MAX_FACES, dtype=np.int32)
        bfs_level[:num_f] = face_bfs["bfs_level"]

        edge_to_faces = np.full((MAX_EDGES, 2), -1, dtype=np.int32)
        edge_to_faces[:num_e] = edge_bfs["edge_to_faces"]

        return {
            "uid": uid,
            "bfs_level": bfs_level,
            "edge_to_faces": edge_to_faces,
            "bfs_to_original_face": face_bfs["bfs_to_original_face"],
            "bfs_to_original_edge": edge_bfs["bfs_to_original_edge"],
            "incidence_sum": incidence.sum(),  # For verification
        }
    except Exception as e:
        return {"uid": uid, "error": f"BFS computation failed: {e}"}


def save_results_to_hdf5(h5_path: Path, results: Dict, uid_to_idx: Dict) -> int:
    """
    Incrementally save topology results to HDF5.
    Returns number of samples saved.
    """
    if not results:
        return 0

    saved = 0
    with h5py.File(h5_path, "a") as f:
        edge_to_faces = f["edge_to_faces"]
        bfs_level = f["bfs_level"]
        done_flags = f["_topology_done"]

        for uid, result in results.items():
            if uid not in uid_to_idx:
                continue
            if "error" in result:
                continue

            idx = uid_to_idx[uid]

            try:
                edge_to_faces[idx] = result["edge_to_faces"]
                bfs_level[idx] = result["bfs_level"]
                done_flags[idx] = 1
                saved += 1
            except Exception as e:
                pass

        f.flush()

    return saved


def main():
    parser = argparse.ArgumentParser(description="Fix topology fields in HDF5 files")
    parser.add_argument("--step-dir", required=True, help="Directory with STEP files (D: HDD)")
    parser.add_argument("--h5-dir", required=True, help="Directory with HDF5 files to fix")
    parser.add_argument("--temp-dir", default="C:/Users/User/Desktop/temp_h5",
                       help="Temp dir on SSD for faster writes")
    parser.add_argument("--num-workers", type=int, default=28, help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=200, help="Files per worker chunk")
    parser.add_argument("--save-every", type=int, default=1000, help="Checkpoint every N files")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per chunk (seconds)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't fix")
    parser.add_argument("--split", type=str, default=None, help="Only process this split (train/val/test)")
    args = parser.parse_args()

    num_workers = min(args.num_workers, cpu_count() - 2, 30)  # Leave 2 cores for system
    step_dir = Path(args.step_dir)
    h5_dir = Path(args.h5_dir)
    temp_dir = Path(args.temp_dir)

    print("=" * 70)
    print("FAST TOPOLOGY FIX")
    print("=" * 70)
    print(f"Workers: {num_workers} | Chunk size: {args.chunk_size}")
    print(f"Step dir: {step_dir}")
    print(f"H5 dir: {h5_dir}")
    print(f"Temp dir (SSD): {temp_dir}")
    print()

    # Create temp directory on SSD
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Build STEP file map
    print("Scanning STEP directory...")
    step_map = {}
    for ext in ["*.step", "*.stp", "*.STEP", "*.STP"]:
        for p in step_dir.glob(ext):
            step_map[p.stem] = p
    print(f"Found {len(step_map):,} STEP files")

    # Find H5 files to fix
    h5_files = list(h5_dir.glob("*_brep_autobrep.h5"))

    # Filter to specific split if requested
    if args.split:
        h5_files = [f for f in h5_files if f.name.startswith(args.split)]

    print(f"Found {len(h5_files)} HDF5 files to fix")

    for h5_path in h5_files:
        print(f"\n{'='*70}")
        print(f"Processing: {h5_path.name}")
        print("=" * 70)

        # Copy to SSD for faster read/write
        temp_h5_path = temp_dir / h5_path.name
        if not temp_h5_path.exists():
            print(f"Copying to SSD: {temp_h5_path}")
            shutil.copy2(h5_path, temp_h5_path)

        # Load UIDs and check current state
        with h5py.File(temp_h5_path, "r") as f:
            uids = [u.decode() if isinstance(u, bytes) else str(u) for u in f["uids"][:]]
            n_samples = len(uids)

            # Check current edge_to_faces state
            e2f = f["edge_to_faces"][:]
            valid_e2f = (e2f != -1).any(axis=(1, 2)).sum() if len(e2f.shape) == 3 else (e2f != -1).any(axis=1).sum()
            print(f"Current state: {valid_e2f:,}/{n_samples:,} samples have valid edge_to_faces")

            if args.verify_only:
                # Just show statistics
                bfs = f["bfs_level"][:]
                valid_bfs = (bfs != 0).any(axis=1).sum()
                print(f"Valid bfs_level: {valid_bfs:,}/{n_samples:,}")
                continue

        uid_to_idx = {uid: i for i, uid in enumerate(uids)}

        # Create tracking dataset if needed
        with h5py.File(temp_h5_path, "a") as f:
            if "_topology_done" not in f:
                f.create_dataset("_topology_done", shape=(n_samples,), dtype=np.uint8)
                already_done = set()
            else:
                done_flags = f["_topology_done"][:]
                already_done = set(uids[i] for i in range(n_samples) if done_flags[i] == 1)
            print(f"Already processed: {len(already_done):,}")

        # Build work list
        work_items = [(uid, str(step_map[uid])) for uid in uids
                     if uid in step_map and uid not in already_done]
        print(f"Remaining work: {len(work_items):,} files")

        if not work_items:
            print("All done! Copying back to original location...")
            shutil.copy2(temp_h5_path, h5_path)
            continue

        # Sort by path for HDD locality
        work_items.sort(key=lambda x: x[1])

        # Split into chunks
        chunks = [work_items[i:i + args.chunk_size] for i in range(0, len(work_items), args.chunk_size)]
        print(f"Created {len(chunks)} chunks")

        # Process
        print(f"\nProcessing with {num_workers} workers...")
        start_time = time.time()

        pending_results = {}
        processed = 0
        failed = 0
        saved = 0
        incidence_total = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}

            pbar = tqdm(total=len(work_items), desc="Extracting topology")
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    chunk_results = future.result(timeout=args.timeout)

                    for result in chunk_results:
                        if result is None:
                            failed += 1
                        elif "error" in result:
                            failed += 1
                        else:
                            # Compute BFS fields
                            topo_result = compute_topology_fields(result)
                            if "error" not in topo_result:
                                pending_results[topo_result["uid"]] = topo_result
                                incidence_total += topo_result.get("incidence_sum", 0)
                                processed += 1
                            else:
                                failed += 1
                        pbar.update(1)

                except FuturesTimeoutError:
                    failed += len(chunk)
                    pbar.update(len(chunk))
                    tqdm.write(f"  [TIMEOUT] Chunk skipped")
                except Exception as e:
                    failed += len(chunk)
                    pbar.update(len(chunk))
                    tqdm.write(f"  [ERROR] {str(e)[:50]}")

                # Checkpoint
                if len(pending_results) >= args.save_every:
                    save_count = save_results_to_hdf5(temp_h5_path, pending_results, uid_to_idx)
                    saved += save_count
                    avg_incidence = incidence_total / max(processed, 1)
                    tqdm.write(f"  [CHECKPOINT] Saved {save_count} (avg incidence: {avg_incidence:.1f})")
                    pending_results.clear()
                    incidence_total = 0

            pbar.close()

        # Final save
        if pending_results:
            save_count = save_results_to_hdf5(temp_h5_path, pending_results, uid_to_idx)
            saved += save_count

        elapsed = time.time() - start_time
        print(f"\nProcessed: {processed:,} | Saved: {saved:,} | Failed: {failed:,}")
        print(f"Time: {elapsed/3600:.1f} hours ({processed/elapsed:.1f} files/sec)")

        # Verify
        with h5py.File(temp_h5_path, "r") as f:
            e2f = f["edge_to_faces"][:]
            valid_e2f = (e2f != -1).any(axis=1).sum()
            bfs = f["bfs_level"][:]
            valid_bfs = (bfs != 0).any(axis=1).sum()
            print(f"Final state: {valid_e2f:,}/{n_samples:,} valid edge_to_faces")
            print(f"             {valid_bfs:,}/{n_samples:,} valid bfs_level")

        # Copy back to original location
        print(f"\nCopying back to {h5_path}...")
        shutil.copy2(temp_h5_path, h5_path)

        # Cleanup tracking dataset
        with h5py.File(h5_path, "a") as f:
            if "_topology_done" in f:
                del f["_topology_done"]
            f.flush()

        gc.collect()

    print(f"\n{'='*70}")
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
