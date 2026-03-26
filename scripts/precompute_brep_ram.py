#!/usr/bin/env python3
"""
RAM-Based Pre-compute B-Rep features - Optimized for 256GB RAM + HDD.

Strategy:
1. Load ALL STEP files into RAM (~60GB) - sequential I/O
2. Process in batches to avoid IPC bottleneck
3. Maximize CPU parallelism and GPU throughput

Usage:
    python scripts/precompute_brep_ram.py \
        --step-dir ../data/extracted_step_files \
        --csv ../data/169k.csv \
        --output-dir ../data/embeddings \
        --num-workers 16
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import gc
import json
import sys
import tempfile
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')


def load_all_files_to_ram(samples: List[Dict], num_threads: int = 16) -> Dict[str, bytes]:
    """Load all STEP files into RAM. With 256GB RAM, 60GB is easy."""
    print(f"\n{'='*60}")
    print("PHASE 1: Loading ALL STEP files to RAM")
    print(f"{'='*60}")

    file_cache = {}
    total_bytes = 0
    failed = 0

    def load_file(sample):
        try:
            with open(sample["path"], "rb") as f:
                return sample["uid"], f.read()
        except:
            return sample["uid"], None

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(
            executor.map(load_file, samples),
            total=len(samples),
            desc="Loading to RAM",
            unit="files"
        ))

    for uid, data in results:
        if data is not None:
            file_cache[uid] = data
            total_bytes += len(data)
        else:
            failed += 1

    elapsed = time.time() - start
    speed = total_bytes / 1e6 / elapsed if elapsed > 0 else 0

    print(f"\nLoaded {len(file_cache):,} files ({total_bytes / 1e9:.1f} GB)")
    print(f"Time: {elapsed:.1f}s, Speed: {speed:.1f} MB/s")
    if failed:
        print(f"Failed: {failed}")
    return file_cache


def import_occ():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCC.Core.TopExp import topexp
        return {
            'STEPControl_Reader': STEPControl_Reader,
            'IFSelect_RetDone': IFSelect_RetDone,
            'TopExp_Explorer': TopExp_Explorer,
            'TopAbs_FACE': TopAbs_FACE,
            'TopAbs_EDGE': TopAbs_EDGE,
            'topods': topods,
            'BRepAdaptor_Surface': BRepAdaptor_Surface,
            'BRepAdaptor_Curve': BRepAdaptor_Curve,
            'TopTools_IndexedDataMapOfShapeListOfShape': TopTools_IndexedDataMapOfShapeListOfShape,
            'topexp': topexp,
        }
    except:
        return None


def process_step_bytes(args):
    """Process STEP from bytes. Writes to temp file, extracts geometry."""
    uid, file_bytes, max_faces, max_edges, grid_size, curve_size = args

    if file_bytes is None:
        return None

    occ = import_occ()
    if occ is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            reader = occ['STEPControl_Reader']()
            if reader.ReadFile(tmp_path) != occ['IFSelect_RetDone']:
                return None

            reader.TransferRoots()
            shape = reader.OneShape()
            if shape.IsNull():
                return None

            # Extract faces
            faces, face_bboxes, face_shapes = [], [], []
            exp = occ['TopExp_Explorer'](shape, occ['TopAbs_FACE'])
            while exp.More() and len(faces) < max_faces:
                try:
                    face = occ['topods'].Face(exp.Current())
                    grid = _sample_face(face, grid_size, occ)
                    if grid is not None:
                        faces.append(grid)
                        face_bboxes.append(np.concatenate([grid.min(axis=(0,1)), grid.max(axis=(0,1))]))
                        face_shapes.append(face)
                except:
                    pass
                exp.Next()

            # Extract edges
            edges, edge_bboxes, edge_shapes = [], [], []
            exp = occ['TopExp_Explorer'](shape, occ['TopAbs_EDGE'])
            while exp.More() and len(edges) < max_edges:
                try:
                    edge = occ['topods'].Edge(exp.Current())
                    curve = _sample_edge(edge, curve_size, occ)
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

            # Build face-edge incidence
            incidence = np.zeros((num_f, num_e), dtype=np.int8)
            try:
                emap = occ['TopTools_IndexedDataMapOfShapeListOfShape']()
                occ['topexp'].MapShapesAndAncestors(shape, occ['TopAbs_EDGE'], occ['TopAbs_FACE'], emap)
                for ei, e in enumerate(edge_shapes):
                    idx = emap.FindIndex(e)
                    if idx > 0:
                        flist = emap.FindFromIndex(idx)
                        it = flist.begin()
                        while it != flist.end():
                            af = occ['topods'].Face(it.Value())
                            for fi, f in enumerate(face_shapes):
                                if f.IsSame(af):
                                    incidence[fi, ei] = 1
                                    break
                            it.Next()
            except:
                pass

            # Pad
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

            return {"uid": uid, "faces": face_arr, "edges": edge_arr,
                    "face_bboxes": fbox, "edge_bboxes": ebox,
                    "incidence": inc, "num_faces": num_f, "num_edges": num_e}
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    except:
        return None


def _sample_face(face, size, occ):
    try:
        surf = occ['BRepAdaptor_Surface'](face)
        u0, u1 = surf.FirstUParameter(), surf.LastUParameter()
        v0, v1 = surf.FirstVParameter(), surf.LastVParameter()
        if abs(u0) > 1e10: u0 = -100
        if abs(u1) > 1e10: u1 = 100
        if abs(v0) > 1e10: v0 = -100
        if abs(v1) > 1e10: v1 = 100
        grid = np.zeros((size, size, 3), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                u = u0 + (u1-u0)*i/max(size-1,1)
                v = v0 + (v1-v0)*j/max(size-1,1)
                try:
                    p = surf.Value(u, v)
                    grid[i,j] = [p.X(), p.Y(), p.Z()]
                except:
                    pass
        return grid
    except:
        return None


def _sample_edge(edge, size, occ):
    try:
        crv = occ['BRepAdaptor_Curve'](edge)
        t0, t1 = crv.FirstParameter(), crv.LastParameter()
        if abs(t0) > 1e10: t0 = -100
        if abs(t1) > 1e10: t1 = 100
        pts = np.zeros((size, 3), dtype=np.float32)
        for i in range(size):
            t = t0 + (t1-t0)*i/max(size-1,1)
            try:
                p = crv.Value(t)
                pts[i] = [p.X(), p.Y(), p.Z()]
            except:
                pass
        return pts
    except:
        return None


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


def create_hdf5(path, mf, me, fd, ed, total):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("face_features", (0,mf,fd), maxshape=(total,mf,fd), dtype=np.float32, chunks=(10,mf,fd), compression="lzf")
        f.create_dataset("edge_features", (0,me,ed), maxshape=(total,me,ed), dtype=np.float32, chunks=(10,me,ed), compression="lzf")
        f.create_dataset("face_masks", (0,mf), maxshape=(total,mf), dtype=np.float32, chunks=(100,mf), compression="lzf")
        f.create_dataset("edge_masks", (0,me), maxshape=(total,me), dtype=np.float32, chunks=(100,me), compression="lzf")
        f.create_dataset("bfs_to_original_face", (0,mf), maxshape=(total,mf), dtype=np.int32, chunks=(100,mf), compression="lzf")
        f.create_dataset("bfs_to_original_edge", (0,me), maxshape=(total,me), dtype=np.int32, chunks=(100,me), compression="lzf")
        f.create_dataset("bfs_level", (0,mf), maxshape=(total,mf), dtype=np.int32, chunks=(100,mf), compression="lzf")
        f.create_dataset("bfs_parent_face", (0,mf), maxshape=(total,mf), dtype=np.int32, chunks=(100,mf), compression="lzf")
        f.create_dataset("bfs_parent_edge", (0,mf), maxshape=(total,mf), dtype=np.int32, chunks=(100,mf), compression="lzf")
        f.create_dataset("edge_to_faces", (0,me,2), maxshape=(total,me,2), dtype=np.int32, chunks=(100,me,2), compression="lzf")
        f.create_dataset("face_centroids", (0,mf,3), maxshape=(total,mf,3), dtype=np.float32, chunks=(100,mf,3), compression="lzf")
        f.create_dataset("num_faces", (0,), maxshape=(total,), dtype=np.int32)
        f.create_dataset("num_edges", (0,), maxshape=(total,), dtype=np.int32)
        f.create_dataset("uids", (0,), maxshape=(total,), dtype=h5py.special_dtype(vlen=str))
        f.attrs["n_samples"] = 0


def append_hdf5(path, data, uids):
    with h5py.File(path, "a") as f:
        cur = f["face_features"].shape[0]
        n = len(uids)
        new = cur + n
        for k in data:
            f[k].resize(new, axis=0)
            f[k][cur:new] = data[k]
        f["uids"].resize(new, axis=0)
        for i, u in enumerate(uids):
            f["uids"][cur+i] = u
        f.attrs["n_samples"] = new


def load_processed(h5):
    if not h5.exists():
        return set()
    try:
        with h5py.File(h5, "r") as f:
            return set(u.decode() if isinstance(u, bytes) else u for u in f["uids"][:])
    except:
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128, help="GPU batch size")
    parser.add_argument("--num-workers", type=int, default=16, help="CPU workers for geometry extraction")
    parser.add_argument("--process-batch", type=int, default=1000, help="Files to process per pool batch (reduces IPC)")
    parser.add_argument("--max-faces", type=int, default=192)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--curve-size", type=int, default=32)
    parser.add_argument("--face-dim", type=int, default=48)
    parser.add_argument("--edge-dim", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    step_dir = Path(args.step_dir)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    suffix = f"_{args.split}" if args.split else ""
    out_path = output_dir / f"brep_autobrep{suffix}.h5"
    ckpt_path = output_dir / f"brep_autobrep{suffix}.ckpt.json"

    device = torch.device(args.device)

    print("="*60)
    print("RAM-Based B-Rep Feature Extraction (256GB RAM Optimized)")
    print("="*60)
    print(f"Split: {args.split or 'all'}")
    print(f"CPU Workers: {args.num_workers}")
    print(f"GPU Batch: {args.batch_size}")
    print(f"Process Batch: {args.process_batch} (files per pool submission)")
    print(f"Output: {out_path}")

    # Resume
    processed = set()
    if args.resume:
        processed = load_processed(out_path)
        print(f"\nResuming: {len(processed):,} already done")
    else:
        for p in [out_path, ckpt_path]:
            if p.exists():
                p.unlink()

    # Load CSV
    print("\nLoading CSV...")
    df = pd.read_csv(csv_path, low_memory=False).dropna(subset=["uid"])
    if args.split and "split" in df.columns:
        df = df[df["split"] == args.split]
    uids = [str(u) for u in df["uid"].tolist()]
    print(f"UIDs in CSV: {len(uids):,}")

    # Scan STEP dir
    print("\nScanning STEP directory...")
    step_files = {}
    for ext in ["*.step", "*.stp", "*.STEP", "*.STP"]:
        for p in step_dir.glob(ext):
            step_files[p.stem] = p
    print(f"STEP files found: {len(step_files):,}")

    # Match
    samples = [{"uid": u, "path": str(step_files[u])} for u in uids if u in step_files and u not in processed]
    print(f"To process: {len(samples):,}")

    if not samples:
        print("Nothing to do!")
        return

    # ========== PHASE 1: Load ALL to RAM ==========
    file_cache = load_all_files_to_ram(samples, num_threads=16)

    # ========== PHASE 2: Process from RAM ==========
    print(f"\n{'='*60}")
    print("PHASE 2: Processing from RAM (no more disk I/O!)")
    print(f"{'='*60}")

    print("\nCreating encoder...")
    from clip4cad.models.encoders.brep_encoder import BRepEncoder
    model = BRepEncoder(
        face_dim=args.face_dim, edge_dim=args.edge_dim,
        face_grid_size=args.grid_size, edge_curve_size=args.curve_size,
        auto_download=True, freeze=True,
    ).to(device).eval()

    from clip4cad.data.autobrep_utils import (
        bfs_order_faces_with_parents, bfs_order_edges, extract_spatial_properties,
    )

    total = len(samples) + len(processed)
    if not out_path.exists():
        create_hdf5(out_path, args.max_faces, args.max_edges, args.face_dim, args.edge_dim, total)

    # Buffers
    buf = {k: [] for k in ["face_features", "edge_features", "face_masks", "edge_masks",
                           "bfs_to_original_face", "bfs_to_original_edge", "bfs_level",
                           "bfs_parent_face", "bfs_parent_edge", "edge_to_faces",
                           "face_centroids", "num_faces", "num_edges"]}
    buf_uids = []
    all_done = list(processed)
    done_cnt, fail_cnt, batch_cnt = 0, 0, 0

    def save_buf():
        nonlocal batch_cnt
        if not buf_uids:
            return
        data = {k: np.stack(v) for k, v in buf.items()}
        append_hdf5(out_path, data, buf_uids)
        for k in buf: buf[k] = []
        buf_uids.clear()
        batch_cnt += 1

    def process_geom_batch(geom_list):
        """GPU encode + BFS order a batch of geometry results."""
        if not geom_list:
            return

        # Normalize
        batch_f, batch_e, batch_fm, batch_em = [], [], [], []
        for g in geom_list:
            f, e, fm, em = normalize_geometry(
                g["faces"].copy(), g["edges"].copy(),
                g["num_faces"], g["num_edges"],
                args.max_faces, args.max_edges
            )
            batch_f.append(f)
            batch_e.append(e)
            batch_fm.append(fm)
            batch_em.append(em)

        # GPU encode
        ft = torch.from_numpy(np.stack(batch_f)).to(device)
        et = torch.from_numpy(np.stack(batch_e)).to(device)
        fmt = torch.from_numpy(np.stack(batch_fm)).to(device)
        emt = torch.from_numpy(np.stack(batch_em)).to(device)

        with torch.no_grad():
            ff, ef = model(ft, et, fmt, emt)
        ff, ef = ff.cpu().numpy(), ef.cpu().numpy()

        # BFS order each
        for i, g in enumerate(geom_list):
            nf, ne = g["num_faces"], g["num_edges"]

            fb = bfs_order_faces_with_parents(g["incidence"][:nf,:ne], g["face_bboxes"][:nf])
            eb = bfs_order_edges(g["incidence"][:nf,:ne], fb["bfs_to_original_face"], g["edge_bboxes"][:ne])
            sp = extract_spatial_properties(g["faces"][:nf], g["edges"][:ne])

            fo, eo = fb["bfs_to_original_face"], eb["bfs_to_original_edge"]

            def pad(arr, order, mx):
                out = np.zeros((mx,)+arr.shape[1:], dtype=arr.dtype)
                out[:len(order)] = arr[order]
                return out

            buf["face_features"].append(pad(ff[i,:nf], fo, args.max_faces))
            buf["edge_features"].append(pad(ef[i,:ne], eo, args.max_edges))
            buf["face_masks"].append(pad(batch_fm[i][:nf], fo, args.max_faces))
            buf["edge_masks"].append(pad(batch_em[i][:ne], eo, args.max_edges))

            bff = np.zeros(args.max_faces, dtype=np.int32); bff[:nf] = fo
            bfe = np.zeros(args.max_edges, dtype=np.int32); bfe[:ne] = eo
            bfl = np.zeros(args.max_faces, dtype=np.int32); bfl[:nf] = fb["bfs_level"]
            bpf = np.full(args.max_faces, -1, dtype=np.int32); bpf[:nf] = fb["bfs_parent_face"]
            bpe = np.full(args.max_faces, -1, dtype=np.int32); bpe[:nf] = fb["bfs_parent_edge"]
            e2f = np.full((args.max_edges,2), -1, dtype=np.int32); e2f[:ne] = eb["edge_to_faces"]
            fc = np.zeros((args.max_faces,3), dtype=np.float32); fc[:nf] = sp["face_centroids"][fo]

            buf["bfs_to_original_face"].append(bff)
            buf["bfs_to_original_edge"].append(bfe)
            buf["bfs_level"].append(bfl)
            buf["bfs_parent_face"].append(bpf)
            buf["bfs_parent_edge"].append(bpe)
            buf["edge_to_faces"].append(e2f)
            buf["face_centroids"].append(fc)
            buf["num_faces"].append(nf)
            buf["num_edges"].append(ne)
            buf_uids.append(g["uid"])

    # Process in batches to reduce IPC overhead
    print(f"\nProcessing {len(samples):,} files with {args.num_workers} workers...")
    print(f"Batch size: {args.process_batch} files per pool submission\n")

    pbar = tqdm(total=len(samples), desc="Processing", unit="files")
    geom_buf = []

    # Process in chunks to reduce IPC
    for batch_start in range(0, len(samples), args.process_batch):
        batch_end = min(batch_start + args.process_batch, len(samples))
        batch_samples = samples[batch_start:batch_end]

        # Prepare work for this batch
        work = [(s["uid"], file_cache.get(s["uid"]), args.max_faces, args.max_edges, args.grid_size, args.curve_size)
                for s in batch_samples]

        # Process with pool
        with Pool(args.num_workers, maxtasksperchild=100) as pool:
            results = pool.map(process_step_bytes, work, chunksize=max(1, len(work)//args.num_workers))

        for result in results:
            if result is None:
                fail_cnt += 1
            else:
                geom_buf.append(result)
                all_done.append(result["uid"])

            done_cnt += 1
            pbar.update(1)

            # GPU batch
            if len(geom_buf) >= args.batch_size:
                process_geom_batch(geom_buf)
                geom_buf = []
                pbar.set_postfix({"buf": len(buf_uids), "fail": fail_cnt})

        # Checkpoint
        if len(buf_uids) >= args.batch_size * args.checkpoint_every:
            process_geom_batch(geom_buf)
            geom_buf = []
            save_buf()
            with open(ckpt_path, "w") as f:
                json.dump({"count": done_cnt, "uids": all_done[-10000:]}, f)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    pbar.close()

    # Final
    process_geom_batch(geom_buf)
    save_buf()

    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"Processed: {done_cnt:,}, Failed: {fail_cnt:,}")

    with h5py.File(out_path, "r") as f:
        print(f"Output: {out_path}")
        print(f"  Samples: {f.attrs['n_samples']:,}")


if __name__ == "__main__":
    main()
