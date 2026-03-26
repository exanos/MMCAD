#!/usr/bin/env python3
"""
Pre-compute AutoBrep BFS-ordered B-Rep features from STEP files.

Hybrid approach:
- Multiprocessing for parallel OCC geometry extraction
- Threading for async GPU encoding
- Proper stderr suppression to prevent terminal freezing

Usage:
    python scripts/precompute_brep_autobrep_fast.py \
        --step-dir ../data/extracted_step_files \
        --csv ../data/169k.csv \
        --mapping-csv ../data/combined_sorted_models.csv \
        --splits-csv ../data/aligned/all_splits.csv \
        --output-dir ../data/embeddings \
        --num-workers 8
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import gc
import platform
import queue
import threading
from multiprocessing import Pool, cpu_count
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# =============================================================================
# Worker Functions (run in separate processes)
# =============================================================================

def init_worker():
    """Initialize worker: suppress stderr at OS level to capture C-level OCC output."""
    import sys
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Suppress stderr at OS level (captures C-level output from OCC)
    if platform.system() == "Windows":
        # Redirect file descriptor 2 (stderr) to NUL
        sys.stderr = open('NUL', 'w')
        os.dup2(sys.stderr.fileno(), 2)
    else:
        sys.stderr = open('/dev/null', 'w')
        os.dup2(sys.stderr.fileno(), 2)


def process_single_file(args):
    """Worker: Extract geometry from STEP file."""
    sample, max_faces, max_edges, grid_size, curve_size = args

    # Import OCC in worker (after stderr suppression from init_worker)
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    except:
        return None

    filepath = sample["path"]
    uid = sample["uid"]
    split = sample.get("split", "train")

    def sample_face_grid(face):
        try:
            surface = BRepAdaptor_Surface(face)
            u_min, u_max = surface.FirstUParameter(), surface.LastUParameter()
            v_min, v_max = surface.FirstVParameter(), surface.LastVParameter()
            u_min, u_max = max(u_min, -1e6), min(u_max, 1e6)
            v_min, v_max = max(v_min, -1e6), min(v_max, 1e6)

            grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
            for i in range(grid_size):
                u = u_min + (u_max - u_min) * i / max(grid_size - 1, 1)
                for j in range(grid_size):
                    v = v_min + (v_max - v_min) * j / max(grid_size - 1, 1)
                    try:
                        pnt = surface.Value(u, v)
                        grid[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]
                    except:
                        pass
            return grid if not np.isnan(grid).any() else None
        except:
            return None

    def sample_edge_curve(edge):
        try:
            curve = BRepAdaptor_Curve(edge)
            t_min, t_max = curve.FirstParameter(), curve.LastParameter()
            t_min, t_max = max(t_min, -1e6), min(t_max, 1e6)

            points = np.zeros((curve_size, 3), dtype=np.float32)
            for i in range(curve_size):
                t = t_min + (t_max - t_min) * i / max(curve_size - 1, 1)
                try:
                    pnt = curve.Value(t)
                    points[i] = [pnt.X(), pnt.Y(), pnt.Z()]
                except:
                    pass
            return points if not np.isnan(points).any() else None
        except:
            return None

    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(filepath)) != IFSelect_RetDone:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return None

        # Extract faces
        faces, face_bboxes, face_shapes = [], [], []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More() and len(faces) < max_faces:
            try:
                face = topods.Face(exp.Current())
                grid = sample_face_grid(face)
                if grid is not None:
                    faces.append(grid)
                    face_bboxes.append(np.concatenate([grid.min(axis=(0,1)), grid.max(axis=(0,1))]))
                    face_shapes.append(face)
            except:
                pass
            exp.Next()

        # Extract edges
        edges, edge_bboxes, edge_shapes = [], [], []
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More() and len(edges) < max_edges:
            try:
                edge = topods.Edge(exp.Current())
                curve = sample_edge_curve(edge)
                if curve is not None:
                    edges.append(curve)
                    edge_bboxes.append(np.concatenate([curve.min(axis=0), curve.max(axis=0)]))
                    edge_shapes.append(edge)
            except:
                pass
            exp.Next()

        if not faces and not edges:
            return None

        num_f, num_e = len(faces), len(edges)

        # Build incidence
        incidence = np.zeros((num_f, num_e), dtype=np.int8)
        try:
            emap = TopTools_IndexedDataMapOfShapeListOfShape()
            topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, emap)
            for ei, e in enumerate(edge_shapes):
                idx = emap.FindIndex(e)
                if idx > 0:
                    flist = emap.FindFromIndex(idx)
                    it = flist.begin()
                    while it != flist.end():
                        try:
                            af = topods.Face(it.Value())
                            for fi, f in enumerate(face_shapes):
                                if f.IsSame(af):
                                    incidence[fi, ei] = 1
                                    break
                        except:
                            pass
                        it.Next()
        except:
            pass

        # Pad arrays
        face_arr = np.zeros((max_faces, grid_size, grid_size, 3), dtype=np.float32)
        edge_arr = np.zeros((max_edges, curve_size, 3), dtype=np.float32)
        fbox = np.zeros((max_faces, 6), dtype=np.float32)
        ebox = np.zeros((max_edges, 6), dtype=np.float32)
        inc = np.zeros((max_faces, max_edges), dtype=np.int8)

        for i, f in enumerate(faces): face_arr[i] = f
        for i, e in enumerate(edges): edge_arr[i] = e
        for i, b in enumerate(face_bboxes): fbox[i] = b
        for i, b in enumerate(edge_bboxes): ebox[i] = b
        inc[:num_f, :num_e] = incidence

        return {
            "uid": uid,
            "split": split,
            "faces": face_arr,
            "edges": edge_arr,
            "face_bboxes": fbox,
            "edge_bboxes": ebox,
            "incidence": inc,
            "num_faces": num_f,
            "num_edges": num_e,
        }
    except:
        return None


# =============================================================================
# Normalization
# =============================================================================

def normalize_geometry(faces, edges, num_f, num_e, max_f, max_e):
    fm = np.zeros(max_f, dtype=np.float32)
    em = np.zeros(max_e, dtype=np.float32)
    fm[:num_f] = 1.0
    em[:num_e] = 1.0

    for i in range(num_f):
        f = faces[i]
        mn, mx = f.min(axis=(0,1), keepdims=True), f.max(axis=(0,1), keepdims=True)
        r = mx - mn
        if r.max() < 1e-8:
            fm[i] = 0
        else:
            faces[i] = 2*(f-mn)/np.maximum(r,1e-8) - 1

    for i in range(num_e):
        e = edges[i]
        mn, mx = e.min(axis=0, keepdims=True), e.max(axis=0, keepdims=True)
        r = mx - mn
        if r.max() < 1e-8:
            em[i] = 0
        else:
            edges[i] = 2*(e-mn)/np.maximum(r,1e-8) - 1

    return faces, edges, fm, em


# =============================================================================
# HDF5 Utilities
# =============================================================================

def create_hdf5(path, mf, me, fd, ed, total):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("face_features", (0, mf, fd), maxshape=(total, mf, fd),
                        dtype=np.float32, chunks=(10, mf, fd), compression="lzf")
        f.create_dataset("edge_features", (0, me, ed), maxshape=(total, me, ed),
                        dtype=np.float32, chunks=(10, me, ed), compression="lzf")
        f.create_dataset("face_masks", (0, mf), maxshape=(total, mf),
                        dtype=np.float32, chunks=(100, mf), compression="lzf")
        f.create_dataset("edge_masks", (0, me), maxshape=(total, me),
                        dtype=np.float32, chunks=(100, me), compression="lzf")
        f.create_dataset("edge_to_faces", (0, me, 2), maxshape=(total, me, 2),
                        dtype=np.int32, chunks=(100, me, 2), compression="lzf")
        f.create_dataset("bfs_level", (0, mf), maxshape=(total, mf),
                        dtype=np.int32, chunks=(100, mf), compression="lzf")
        f.create_dataset("face_centroids", (0, mf, 3), maxshape=(total, mf, 3),
                        dtype=np.float32, chunks=(100, mf, 3), compression="lzf")
        f.create_dataset("num_faces", (0,), maxshape=(total,), dtype=np.int32)
        f.create_dataset("num_edges", (0,), maxshape=(total,), dtype=np.int32)
        f.create_dataset("uids", (0,), maxshape=(total,), dtype=h5py.special_dtype(vlen=str))
        f.attrs["n_samples"] = 0
        f.flush()


def append_hdf5(path, data, uids):
    """Append batch to HDF5 with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with h5py.File(path, "a") as f:
                if "face_features" not in f:
                    # File was corrupted, skip this batch
                    print(f"Warning: {path} missing datasets, skipping batch")
                    return False
                cur = f["face_features"].shape[0]
                n = len(uids)
                new = cur + n
                for k in data:
                    if k in f:
                        f[k].resize(new, axis=0)
                        f[k][cur:new] = data[k]
                f["uids"].resize(new, axis=0)
                for i, u in enumerate(uids):
                    f["uids"][cur + i] = str(u)
                f.attrs["n_samples"] = new
                f.flush()
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
            else:
                print(f"Warning: Failed to append to {path}: {e}")
                return False
    return False


def load_processed_uids(h5_path):
    if not h5_path.exists():
        return set()
    try:
        with h5py.File(h5_path, "r") as f:
            if "uids" in f:
                return set(str(u.decode() if isinstance(u, bytes) else u) for u in f["uids"][:])
    except:
        pass
    return set()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--splits-csv")
    parser.add_argument("--mapping-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-faces", type=int, default=192)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--curve-size", type=int, default=32)
    parser.add_argument("--face-dim", type=int, default=48)
    parser.add_argument("--edge-dim", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    num_workers = min(args.num_workers, 61 if platform.system() == "Windows" else cpu_count())
    step_dir = Path(args.step_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    split_files = {
        "train": output_dir / "train_brep_autobrep.h5",
        "val": output_dir / "val_brep_autobrep.h5",
        "test": output_dir / "test_brep_autobrep.h5",
    }

    print("=" * 60)
    print("AutoBrep Feature Extraction (Parallel + GPU)")
    print("=" * 60)
    print(f"Workers: {num_workers}, Device: {device}")

    # Load CSV
    print("\nLoading CSV...")
    df = pd.read_csv(args.csv, low_memory=False).dropna(subset=["uid"])
    df["uid"] = df["uid"].astype(str)

    # Load mapping
    abc_to_canonical = {}
    if args.mapping_csv:
        mdf = pd.read_csv(args.mapping_csv, low_memory=False)
        mdf["abc_uid"] = mdf["modelname"].str.strip('"').apply(lambda x: str(int(x)))
        mdf["canonical_uid"] = mdf["uid"].astype(str)
        abc_to_canonical = dict(zip(mdf["abc_uid"], mdf["canonical_uid"]))

    # Build UID mapping
    uid_to_canonical = {}
    if "quality_score" in df.columns:
        for _, row in df.iterrows():
            uid = str(row["uid"])
            if row["quality_score"] == 0.6 and uid in abc_to_canonical:
                uid_to_canonical[uid] = abc_to_canonical[uid]
            else:
                uid_to_canonical[uid] = uid
    else:
        uid_to_canonical = {str(row["uid"]): str(row["uid"]) for _, row in df.iterrows()}

    # Load splits
    if args.splits_csv:
        sdf = pd.read_csv(args.splits_csv, low_memory=False)
        sdf["uid"] = sdf["uid"].astype(str)
        uid_to_split = dict(zip(sdf["uid"], sdf["split"]))
    else:
        uid_to_split = {uid: "train" for uid in df["uid"]}

    # Resume
    processed = set()
    if args.resume:
        for split, path in split_files.items():
            p = load_processed_uids(path)
            processed.update(p)
            if p:
                print(f"  {split}: {len(p):,} done")
    else:
        for path in split_files.values():
            if path.exists():
                path.unlink()

    # Scan STEP files
    print("Scanning STEP directory...")
    step_map = {}
    for ext in ["*.step", "*.stp", "*.STEP", "*.STP"]:
        for p in step_dir.glob(ext):
            step_map[p.stem] = p

    # Build sample list (skip test split for now)
    samples = []
    for uid in uid_to_canonical:
        cuid = uid_to_canonical[uid]
        split = uid_to_split.get(uid, "train")
        if split == "test":
            continue  # Skip test for now
        if cuid in step_map and cuid not in processed:
            samples.append({
                "uid": cuid,
                "path": str(step_map[cuid]),
                "split": split,
            })

    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in samples:
        split_counts[s["split"]] += 1

    print(f"To process: {len(samples):,}")
    print(f"  train: {split_counts['train']:,}, val: {split_counts['val']:,}, test: {split_counts['test']:,}")

    if not samples:
        return

    # Load encoder
    print("\nLoading encoder...")
    from clip4cad.models.encoders.brep_encoder import BRepEncoder
    model = BRepEncoder(
        face_dim=args.face_dim, edge_dim=args.edge_dim,
        face_grid_size=args.grid_size, edge_curve_size=args.curve_size,
        auto_download=True, freeze=True,
    ).to(device).eval()

    from clip4cad.data.autobrep_utils import (
        bfs_order_faces_with_parents, bfs_order_edges, extract_spatial_properties,
    )

    # Create HDF5 files
    for split, path in split_files.items():
        if not path.exists():
            create_hdf5(path, args.max_faces, args.max_edges, args.face_dim, args.edge_dim,
                       max(split_counts.get(split, 1000), 1000))

    # Prepare worker args
    worker_args = [(s, args.max_faces, args.max_edges, args.grid_size, args.curve_size) for s in samples]

    # Shared state
    buffers = {split: {"data": {k: [] for k in [
        "face_features", "edge_features", "face_masks", "edge_masks",
        "edge_to_faces", "bfs_level", "face_centroids", "num_faces", "num_edges"
    ]}, "uids": []} for split in split_files}

    geom_queue = queue.Queue(maxsize=args.batch_size * 4)  # Buffer for GPU
    done_event = threading.Event()
    gpu_done = 0
    gpu_lock = threading.Lock()

    hdf5_lock = threading.Lock()

    def flush_split(split):
        buf = buffers[split]
        if not buf["uids"]:
            return
        data = {k: np.stack(v) for k, v in buf["data"].items()}
        with hdf5_lock:
            success = append_hdf5(split_files[split], data, buf["uids"])
        if success:
            for k in buf["data"]:
                buf["data"][k] = []
            buf["uids"] = []

    def gpu_worker():
        """Thread: GPU encoding."""
        nonlocal gpu_done
        geom_batch = []
        meta_batch = []

        while True:
            try:
                item = geom_queue.get(timeout=1.0)
            except queue.Empty:
                if done_event.is_set():
                    break
                continue

            if item is None:  # Poison pill
                break

            geom, meta = item
            geom_batch.append(geom)
            meta_batch.append(meta)

            if len(geom_batch) >= args.batch_size:
                process_batch(geom_batch, meta_batch)
                with gpu_lock:
                    gpu_done += len(geom_batch)
                geom_batch = []
                meta_batch = []

        # Process remaining
        if geom_batch:
            process_batch(geom_batch, meta_batch)
            with gpu_lock:
                gpu_done += len(geom_batch)

    def process_batch(geom_batch, meta_batch):
        """GPU encode a batch."""
        batch_f, batch_e, batch_fm, batch_em = [], [], [], []
        for g in geom_batch:
            f, e, fm, em = normalize_geometry(
                g["faces"].copy(), g["edges"].copy(),
                g["num_faces"], g["num_edges"],
                args.max_faces, args.max_edges
            )
            batch_f.append(f)
            batch_e.append(e)
            batch_fm.append(fm)
            batch_em.append(em)

        with torch.no_grad():
            ft = torch.from_numpy(np.stack(batch_f)).to(device)
            et = torch.from_numpy(np.stack(batch_e)).to(device)
            fmt = torch.from_numpy(np.stack(batch_fm)).to(device)
            emt = torch.from_numpy(np.stack(batch_em)).to(device)
            ff, ef = model(ft, et, fmt, emt)
            ff, ef = ff.cpu().numpy(), ef.cpu().numpy()

        for i, (g, m) in enumerate(zip(geom_batch, meta_batch)):
            nf, ne = g["num_faces"], g["num_edges"]
            uid, split = m["uid"], m["split"]

            fb = bfs_order_faces_with_parents(g["incidence"][:nf, :ne], g["face_bboxes"][:nf])
            eb = bfs_order_edges(g["incidence"][:nf, :ne], fb["bfs_to_original_face"], g["edge_bboxes"][:ne])
            sp = extract_spatial_properties(g["faces"][:nf], g["edges"][:ne])
            fo, eo = fb["bfs_to_original_face"], eb["bfs_to_original_edge"]

            def pad(arr, order, mx, fill=0):
                out = np.full((mx,) + arr.shape[1:], fill, dtype=arr.dtype) if len(arr.shape) > 1 else np.full(mx, fill, dtype=arr.dtype)
                if len(order) > 0:
                    out[:len(order)] = arr[order]
                return out

            buf = buffers[split]
            buf["data"]["face_features"].append(pad(ff[i, :nf], fo, args.max_faces))
            buf["data"]["edge_features"].append(pad(ef[i, :ne], eo, args.max_edges))
            buf["data"]["face_masks"].append(pad(batch_fm[i][:nf], fo, args.max_faces))
            buf["data"]["edge_masks"].append(pad(batch_em[i][:ne], eo, args.max_edges))

            bfs_lv = np.zeros(args.max_faces, dtype=np.int32)
            bfs_lv[:nf] = fb["bfs_level"]
            buf["data"]["bfs_level"].append(bfs_lv)

            e2f = np.full((args.max_edges, 2), -1, dtype=np.int32)
            e2f[:ne] = eb["edge_to_faces"]
            buf["data"]["edge_to_faces"].append(e2f)

            fc = np.zeros((args.max_faces, 3), dtype=np.float32)
            fc[:nf] = sp["face_centroids"][fo]
            buf["data"]["face_centroids"].append(fc)

            buf["data"]["num_faces"].append(np.array(nf, dtype=np.int32))
            buf["data"]["num_edges"].append(np.array(ne, dtype=np.int32))
            buf["uids"].append(uid)

    # Start GPU thread
    gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
    gpu_thread.start()

    # Process with multiprocessing
    print(f"\nProcessing with {num_workers} workers...")
    done = 0
    fail = 0
    last_save = 0

    with Pool(num_workers, initializer=init_worker) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_single_file, worker_args, chunksize=20),
            total=len(worker_args),
            desc="Extracting",
            mininterval=0.5
        )
        for result in pbar:
            if result is None:
                fail += 1
            else:
                done += 1
                meta = {"uid": result["uid"], "split": result["split"]}
                geom_queue.put((result, meta))

            with gpu_lock:
                gd = gpu_done

            pbar.set_postfix({"ext": done, "gpu": gd, "fail": fail})

            # Periodic save
            if gd - last_save >= 2000:
                for split in split_files:
                    flush_split(split)
                last_save = gd
                gc.collect()

    # Signal GPU thread to finish
    done_event.set()
    geom_queue.put(None)
    gpu_thread.join(timeout=60)

    # Final flush
    for split in split_files:
        flush_split(split)

    print(f"\n{'='*60}")
    print(f"Done! Extracted: {done:,}, GPU encoded: {gpu_done:,}, Failed: {fail:,}")
    for split, path in split_files.items():
        if path.exists():
            with h5py.File(path, "r") as f:
                n = f.attrs.get("n_samples", 0)
            if n > 0:
                print(f"  {split}: {n:,} samples, {path.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
