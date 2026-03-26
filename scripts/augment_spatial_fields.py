#!/usr/bin/env python3
"""
OPTIMIZED v3: Fast spatial field augmentation with RESUME support.

Strategy:
1. Sort files by path for better HDD locality
2. Split work into chunks (workers process many files sequentially)
3. **Incremental HDF5 writes** - checkpoint every batch so progress isn't lost
4. **Resume support** - skip already-processed UIDs on restart
5. **Timeout handling** - skip files that hang OCC

Usage:
    python scripts/augment_spatial_fields.py \
        --step-dir ../data/extracted_step_files \
        --h5-dir ../data/embeddings \
        --num-workers 32

Resume after interruption - just run the same command again!
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import time
import gc
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
# Global config
# =============================================================================
GRID_SIZE = 32
CURVE_SIZE = 32
MAX_FACES = 192
MAX_EDGES = 512


def process_chunk(chunk: List[Tuple[str, str]]) -> List[Optional[Dict]]:
    """
    Worker function: Process a CHUNK of files sequentially.
    Each worker handles many files to reduce HDD seek overhead.
    """
    # Suppress OCC stderr once per worker
    import sys
    import platform
    if platform.system() == "Windows":
        sys.stderr = open('NUL', 'w')
        os.dup2(sys.stderr.fileno(), 2)
    else:
        sys.stderr = open('/dev/null', 'w')
        os.dup2(sys.stderr.fileno(), 2)

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    except Exception:
        return [None] * len(chunk)

    results = []
    for uid, filepath in chunk:
        result = extract_single(
            uid, filepath,
            STEPControl_Reader, IFSelect_RetDone,
            TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE,
            topods, BRepAdaptor_Surface, BRepAdaptor_Curve
        )
        results.append(result)

    return results


def extract_single(
    uid, filepath,
    STEPControl_Reader, IFSelect_RetDone,
    TopExp_Explorer, TopAbs_FACE, TopAbs_EDGE,
    topods, BRepAdaptor_Surface, BRepAdaptor_Curve
) -> Optional[Dict]:
    """Extract spatial properties from a single STEP file."""
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(filepath)) != IFSelect_RetDone:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()

        if shape.IsNull():
            return None

        # Sample face grids
        face_grids = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More() and len(face_grids) < MAX_FACES:
            try:
                face = topods.Face(exp.Current())
                surface = BRepAdaptor_Surface(face)
                u_min, u_max = surface.FirstUParameter(), surface.LastUParameter()
                v_min, v_max = surface.FirstVParameter(), surface.LastVParameter()
                u_min, u_max = max(u_min, -1e6), min(u_max, 1e6)
                v_min, v_max = max(v_min, -1e6), min(v_max, 1e6)

                grid = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
                for i in range(GRID_SIZE):
                    u = u_min + (u_max - u_min) * i / max(GRID_SIZE - 1, 1)
                    for j in range(GRID_SIZE):
                        v = v_min + (v_max - v_min) * j / max(GRID_SIZE - 1, 1)
                        try:
                            pnt = surface.Value(u, v)
                            grid[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]
                        except:
                            pass

                if not np.isnan(grid).any():
                    face_grids.append(grid)
            except:
                pass
            exp.Next()

        # Sample edge curves
        edge_curves = []
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More() and len(edge_curves) < MAX_EDGES:
            try:
                edge = topods.Edge(exp.Current())
                curve = BRepAdaptor_Curve(edge)
                t_min, t_max = curve.FirstParameter(), curve.LastParameter()
                t_min, t_max = max(t_min, -1e6), min(t_max, 1e6)

                points = np.zeros((CURVE_SIZE, 3), dtype=np.float32)
                for i in range(CURVE_SIZE):
                    t = t_min + (t_max - t_min) * i / max(CURVE_SIZE - 1, 1)
                    try:
                        pnt = curve.Value(t)
                        points[i] = [pnt.X(), pnt.Y(), pnt.Z()]
                    except:
                        pass

                if not np.isnan(points).any():
                    edge_curves.append(points)
            except:
                pass
            exp.Next()

        if not face_grids:
            return None

        num_f = len(face_grids)
        num_e = len(edge_curves)

        # Compute spatial properties directly (inline for speed)
        face_normals = np.zeros((num_f, 3), dtype=np.float32)
        face_areas = np.zeros(num_f, dtype=np.float32)

        for i, grid in enumerate(face_grids):
            ci, cj = GRID_SIZE // 2, GRID_SIZE // 2
            u_vec = grid[min(ci+1, GRID_SIZE-1), cj] - grid[max(ci-1, 0), cj]
            v_vec = grid[ci, min(cj+1, GRID_SIZE-1)] - grid[ci, max(cj-1, 0)]
            normal = np.cross(u_vec, v_vec)
            norm = np.linalg.norm(normal)
            face_normals[i] = normal / norm if norm > 1e-8 else np.array([0, 0, 1])

            bbox_min = grid.min(axis=(0, 1))
            bbox_max = grid.max(axis=(0, 1))
            dims = np.sort(bbox_max - bbox_min)[::-1]
            face_areas[i] = dims[0] * dims[1]

        edge_midpoints = np.zeros((num_e, 3), dtype=np.float32)
        edge_lengths = np.zeros(num_e, dtype=np.float32)

        for i, curve in enumerate(edge_curves):
            mid_idx = CURVE_SIZE // 2
            edge_midpoints[i] = curve[mid_idx]
            segments = np.diff(curve, axis=0)
            edge_lengths[i] = np.linalg.norm(segments, axis=1).sum()

        return {
            "uid": uid,
            "num_faces": num_f,
            "num_edges": num_e,
            "face_normals": face_normals,
            "face_areas": face_areas,
            "edge_midpoints": edge_midpoints,
            "edge_lengths": edge_lengths,
        }

    except Exception:
        return None


def save_results_to_hdf5(h5_path, results, uid_to_idx, existing_num_faces, existing_num_edges):
    """
    Incrementally save results to HDF5.
    Returns number of samples successfully saved.
    """
    if not results:
        return 0

    saved = 0
    with h5py.File(h5_path, "a") as f:
        face_normals = f["face_normals"]
        face_areas = f["face_areas"]
        edge_midpoints = f["edge_midpoints"]
        edge_lengths = f["edge_lengths"]
        done_flags = f["_augment_done"]

        for uid, result in results.items():
            if uid not in uid_to_idx:
                continue

            idx = uid_to_idx[uid]
            nf = min(result["num_faces"], int(existing_num_faces[idx]), MAX_FACES)
            ne = min(result["num_edges"], int(existing_num_edges[idx]), MAX_EDGES)

            try:
                if nf > 0 and result["face_normals"].shape[0] >= nf:
                    face_normals[idx, :nf] = result["face_normals"][:nf]
                    face_areas[idx, :nf] = result["face_areas"][:nf]
                if ne > 0 and result["edge_midpoints"].shape[0] >= ne:
                    edge_midpoints[idx, :ne] = result["edge_midpoints"][:ne]
                    edge_lengths[idx, :ne] = result["edge_lengths"][:ne]

                # Mark as done
                done_flags[idx] = 1
                saved += 1
            except Exception:
                pass

        f.flush()

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", required=True, help="Directory with STEP files")
    parser.add_argument("--h5-dir", required=True, help="Directory with existing HDF5 files")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=500, help="Files per worker chunk")
    parser.add_argument("--save-every", type=int, default=5000, help="Save checkpoint every N files")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per chunk in seconds")
    args = parser.parse_args()

    # For HDD, use fewer workers to reduce seek contention
    # Xeon 8352Y has 32 cores - use 24-32 workers
    num_workers = min(args.num_workers, cpu_count(), 32)
    chunk_size = args.chunk_size
    step_dir = Path(args.step_dir)
    h5_dir = Path(args.h5_dir)

    print("=" * 70)
    print("OPTIMIZED v3: Spatial Field Augmentation with RESUME Support")
    print("=" * 70)
    print(f"Strategy: Sort by path → Chunk files → Incremental saves")
    print(f"Workers: {num_workers} | Chunk size: {chunk_size} | Save every: {args.save_every}")
    print(f"Timeout per chunk: {args.timeout}s")
    print(f"Step dir: {step_dir}")
    print(f"H5 dir: {h5_dir}")
    print()

    # Build global STEP file map once
    print("Scanning STEP directory...")
    step_map = {}
    for ext in ["*.step", "*.stp", "*.STEP", "*.STP"]:
        for p in step_dir.glob(ext):
            step_map[p.stem] = p
    print(f"Found {len(step_map):,} STEP files")

    # Find H5 files
    h5_files = list(h5_dir.glob("*_brep_autobrep.h5"))
    print(f"Found {len(h5_files)} HDF5 files to augment")

    for h5_path in h5_files:
        print(f"\n{'='*70}")
        print(f"Processing: {h5_path.name}")
        print("=" * 70)

        # =====================================================================
        # RESUME SUPPORT: Check existing state and create/resume datasets
        # =====================================================================
        with h5py.File(h5_path, "r") as f:
            uids = [u.decode() if isinstance(u, bytes) else str(u) for u in f["uids"][:]]
            n_samples = len(uids)
            existing_num_faces = f["num_faces"][:]
            existing_num_edges = f["num_edges"][:]

        uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        print(f"Samples in HDF5: {n_samples:,}")

        # Create or open datasets for incremental writing
        with h5py.File(h5_path, "a") as f:
            if "face_normals" not in f:
                print("Creating new spatial field datasets...")
                f.create_dataset("face_normals", shape=(n_samples, MAX_FACES, 3),
                                dtype=np.float32, compression="lzf")
                f.create_dataset("face_areas", shape=(n_samples, MAX_FACES),
                                dtype=np.float32, compression="lzf")
                f.create_dataset("edge_midpoints", shape=(n_samples, MAX_EDGES, 3),
                                dtype=np.float32, compression="lzf")
                f.create_dataset("edge_lengths", shape=(n_samples, MAX_EDGES),
                                dtype=np.float32, compression="lzf")
                # Track which samples are processed (1 = done, 0 = pending)
                f.create_dataset("_augment_done", shape=(n_samples,), dtype=np.uint8)
                f.flush()
                already_done = set()
            else:
                # Resume: find already-processed samples
                if "_augment_done" in f:
                    done_flags = f["_augment_done"][:]
                    already_done = set(uids[i] for i in range(n_samples) if done_flags[i] == 1)
                else:
                    # Legacy: check face_areas for non-zero values
                    face_areas_data = f["face_areas"][:]
                    already_done = set(uids[i] for i in range(n_samples)
                                      if face_areas_data[i].sum() > 0)
                print(f"RESUMING: {len(already_done):,} samples already processed")

        # Filter to UIDs with STEP files that haven't been processed yet
        work_items = [(uid, str(step_map[uid])) for uid in uids
                     if uid in step_map and uid not in already_done]
        print(f"Remaining work: {len(work_items):,} files")

        if not work_items:
            print("All files already processed!")
            # Clean up temp flag dataset
            with h5py.File(h5_path, "a") as f:
                if "_augment_done" in f:
                    del f["_augment_done"]
                f.flush()
            continue

        # Sort by path for better HDD locality
        print("Sorting by path for HDD locality...")
        work_items.sort(key=lambda x: x[1])

        # Split into chunks
        chunks = []
        for i in range(0, len(work_items), chunk_size):
            chunks.append(work_items[i:i + chunk_size])
        print(f"Created {len(chunks)} chunks of ~{chunk_size} files each")

        # =====================================================================
        # Process chunks with INCREMENTAL SAVING
        # =====================================================================
        print(f"\nProcessing with {num_workers} workers (incremental save every {args.save_every} files)...")
        process_start = time.time()

        pending_results = {}  # Buffer results before saving
        failed = 0
        processed = 0
        saved = 0
        timedout = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): (i, chunk) for i, chunk in enumerate(chunks)}

            pbar = tqdm(total=len(work_items), desc="Processing")
            for future in as_completed(futures):
                chunk_idx, chunk = futures[future]
                try:
                    # Timeout handling to avoid getting stuck
                    chunk_results = future.result(timeout=args.timeout)
                    for (uid, _), result in zip(chunk, chunk_results):
                        if result is not None:
                            pending_results[result["uid"]] = result
                            processed += 1
                        else:
                            failed += 1
                        pbar.update(1)
                except FuturesTimeoutError:
                    # Chunk timed out - mark all as failed and continue
                    timedout += len(chunk)
                    pbar.update(len(chunk))
                    tqdm.write(f"  [TIMEOUT] Chunk {chunk_idx} ({len(chunk)} files) - skipping")
                except Exception as e:
                    failed += len(chunk)
                    pbar.update(len(chunk))
                    tqdm.write(f"  [ERROR] Chunk {chunk_idx}: {str(e)[:50]}")

                # Incremental save checkpoint
                if len(pending_results) >= args.save_every:
                    save_count = save_results_to_hdf5(
                        h5_path, pending_results, uid_to_idx,
                        existing_num_faces, existing_num_edges
                    )
                    saved += save_count
                    tqdm.write(f"  [CHECKPOINT] Saved {save_count} samples (total: {saved:,})")
                    pending_results.clear()

            pbar.close()

        # Final save of remaining results
        if pending_results:
            save_count = save_results_to_hdf5(
                h5_path, pending_results, uid_to_idx,
                existing_num_faces, existing_num_edges
            )
            saved += save_count
            print(f"Final save: {save_count} samples")
            pending_results.clear()

        process_time = time.time() - process_start
        throughput = processed / process_time if process_time > 0 else 0

        # Summary
        print(f"\n{'='*50}")
        print(f"Processed: {processed:,} | Saved: {saved:,} | Failed: {failed:,} | Timeout: {timedout:,}")
        print(f"Time: {process_time:.1f}s ({throughput:.1f} files/s)")

        # Verify final state
        with h5py.File(h5_path, "r") as f:
            if "_augment_done" in f:
                done_count = f["_augment_done"][:].sum()
                print(f"Total augmented: {done_count:,} / {n_samples:,}")

        # Clean up tracking dataset if complete
        with h5py.File(h5_path, "a") as f:
            if "_augment_done" in f:
                done_count = f["_augment_done"][:].sum()
                if done_count == n_samples:
                    del f["_augment_done"]
                    print("Augmentation complete - removed tracking dataset")
            f.flush()

        gc.collect()

    print(f"\n{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
