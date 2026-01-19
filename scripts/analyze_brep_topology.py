#!/usr/bin/env python3
"""
Analyze B-Rep topology (face/edge counts) in the dataset.

Samples STEP files and computes topology statistics to validate
the max_faces and max_edges parameters.

Requirements:
    conda install -c conda-forge pythonocc-core

Usage:
    python scripts/analyze_brep_topology.py --step-dir ../data/extracted_step_files --csv ../data/169k.csv
    python scripts/analyze_brep_topology.py --step-dir ../data/extracted_step_files --uids ../data/sample_uids.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import OCC
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    print("Warning: pythonOCC not installed")
    print("Install with: conda install -c conda-forge pythonocc-core")


def count_faces_edges(filepath: str) -> tuple:
    """Count faces and edges in a STEP file."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)

    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read: {filepath}")

    reader.TransferRoots()
    shape = reader.OneShape()

    # Count faces
    face_count = 0
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face_count += 1
        face_explorer.Next()

    # Count edges
    edge_count = 0
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge_count += 1
        edge_explorer.Next()

    return face_count, edge_count


def main():
    parser = argparse.ArgumentParser(description="Analyze B-Rep topology")
    parser.add_argument("--step-dir", type=str, required=True, help="Directory with STEP files")
    parser.add_argument("--csv", type=str, default=None, help="CSV file to sample UIDs from")
    parser.add_argument("--uids", type=str, default=None, help="CSV file with pre-sampled UIDs")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not HAS_OCC:
        print("Error: pythonOCC required")
        return

    print("=" * 60)
    print("B-REP TOPOLOGY ANALYSIS")
    print("=" * 60)
    print(f"STEP dir: {args.step_dir}")
    print(f"Samples: {args.num_samples}")
    print()

    step_dir = Path(args.step_dir)

    # Get UIDs
    if args.uids:
        print(f"Loading UIDs from: {args.uids}")
        uid_df = pd.read_csv(args.uids)
        uids = [str(u) for u in uid_df.iloc[:, 0].tolist()]
    elif args.csv:
        print(f"Sampling UIDs from: {args.csv}")
        df = pd.read_csv(args.csv)
        df = df.dropna(subset=["uid"])
        sample_df = df.sample(n=args.num_samples, random_state=args.seed)
        uids = [str(u) for u in sample_df["uid"].tolist()]
    else:
        print("Error: Provide --csv or --uids")
        return

    print(f"UIDs to process: {len(uids)}")

    # Find existing STEP files
    samples = []
    for uid in uids:
        step_path = step_dir / f"{uid}.step"
        if not step_path.exists():
            step_path = step_dir / f"{uid}.stp"
        if step_path.exists():
            samples.append({"uid": uid, "path": str(step_path)})

    print(f"STEP files found: {len(samples)}")
    print()

    # Analyze
    print("Analyzing B-Rep topology...")
    face_counts = []
    edge_counts = []
    failed = 0

    for sample in tqdm(samples):
        try:
            faces, edges = count_faces_edges(sample["path"])
            face_counts.append(faces)
            edge_counts.append(edges)
        except Exception as e:
            failed += 1

    face_counts = np.array(face_counts)
    edge_counts = np.array(edge_counts)

    print(f"\nProcessed: {len(face_counts)}, Failed: {failed}")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFace counts:")
    print(f"  Min: {face_counts.min()}, Max: {face_counts.max()}, Mean: {face_counts.mean():.1f}")
    print(f"  Percentiles: 50%={np.percentile(face_counts, 50):.0f}, "
          f"90%={np.percentile(face_counts, 90):.0f}, "
          f"95%={np.percentile(face_counts, 95):.0f}, "
          f"99%={np.percentile(face_counts, 99):.0f}")

    print(f"\nEdge counts:")
    print(f"  Min: {edge_counts.min()}, Max: {edge_counts.max()}, Mean: {edge_counts.mean():.1f}")
    print(f"  Percentiles: 50%={np.percentile(edge_counts, 50):.0f}, "
          f"90%={np.percentile(edge_counts, 90):.0f}, "
          f"95%={np.percentile(edge_counts, 95):.0f}, "
          f"99%={np.percentile(edge_counts, 99):.0f}")

    print(f"\nCoverage at different max_faces:")
    for max_f in [32, 48, 64, 96, 128, 192, 256]:
        coverage = (face_counts <= max_f).mean() * 100
        print(f"  max_faces={max_f}: {coverage:.1f}%")

    print(f"\nCoverage at different max_edges:")
    for max_e in [64, 96, 128, 192, 256, 384, 512]:
        coverage = (edge_counts <= max_e).mean() * 100
        print(f"  max_edges={max_e}: {coverage:.1f}%")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    face_p95 = np.percentile(face_counts, 95)
    face_p99 = np.percentile(face_counts, 99)
    edge_p95 = np.percentile(edge_counts, 95)
    edge_p99 = np.percentile(edge_counts, 99)

    # Face recommendation
    if face_p95 <= 64:
        print(f"✓ max_faces=64 covers 95th percentile ({face_p95:.0f} faces)")
        if face_p99 <= 64:
            print(f"✓ max_faces=64 also covers 99th percentile ({face_p99:.0f} faces)")
        else:
            print(f"⚠ 99th percentile is {face_p99:.0f} faces - some truncation")
    else:
        recommended = 96 if face_p95 <= 96 else 128 if face_p95 <= 128 else 192
        print(f"⚠ 95th percentile is {face_p95:.0f} faces - consider max_faces={recommended}")

    # Edge recommendation
    if edge_p95 <= 128:
        print(f"✓ max_edges=128 covers 95th percentile ({edge_p95:.0f} edges)")
        if edge_p99 <= 128:
            print(f"✓ max_edges=128 also covers 99th percentile ({edge_p99:.0f} edges)")
        else:
            print(f"⚠ 99th percentile is {edge_p99:.0f} edges - some truncation")
    else:
        recommended = 192 if edge_p95 <= 192 else 256 if edge_p95 <= 256 else 384
        print(f"⚠ 95th percentile is {edge_p95:.0f} edges - consider max_edges={recommended}")


if __name__ == "__main__":
    main()
