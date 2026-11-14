#!/usr/bin/env python3
"""
STEP -> raw B-Rep inputs for the released BRepFormer (baseline_trimodal_v4.pth).

This is the extractor that produced the training caches, reproduced verbatim:
  * brep_features.h5  -> face_features (192,16), face_centroids (192,3),
                         edge_to_faces (512,2), face_mask (192,)
  * brep_topology.h5  -> face_normals (192,3)

Output is RAW (unnormalized). Feed it through inference.prepare_brep() with the
released stats (brep_norm_stats_v4.npz) before calling the model.

Requires pythonocc-core (tested with 7.7.x):
    conda install -c conda-forge pythonocc-core=7.7.2

Usage:
    python scripts/step_to_brep.py part.step -o part_brep.npz

Face features (16 columns), all in raw STEP units, no shape normalization:
    0-2   mean xyz of a 32x32 grid sampled over the face's UV parameter box
    3-5   std  xyz of that grid
    6-8   min  xyz
    9-11  max  xyz
    12-14 unit normal = cross(dP/du, dP/dv) by central differences at grid
          cell (16,16)  (0 if degenerate)
    15    product of the two largest bbox extents (cols 9-11 minus 6-8)
The UV box is the UNTRIMMED parameter range (BRepAdaptor_Surface First/Last
U/V); infinite bounds are clamped to +-100. Trimming is not applied, so a face's
descriptor reflects its whole underlying surface patch.

face_centroids = cols 0-2 (mean of the same 1024 grid points).

face_normals: BRepLProp_SLProps normal (no face-orientation flip) at cell (5,5)
of a 10x10 grid over the same UV box. This is what the attention bias uses.

edge_to_faces: edges in TopExp_Explorer(EDGE) order (NOT deduplicated -- shared
edges appear once per traversal, as in training), truncated to 512. Each row
holds up to two face indices (explorer face order, truncated to 192) from
MapShapesAndAncestors; missing entries are -1.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

MAX_FACES = 192
MAX_EDGES = 512
GRID_SIZE = 32
NORMAL_GRID = 10


def _uv_box(s):
    u0, u1 = s.FirstUParameter(), s.LastUParameter()
    v0, v1 = s.FirstVParameter(), s.LastVParameter()
    if abs(u0) > 1e10: u0 = -100.0
    if abs(u1) > 1e10: u1 = 100.0
    if abs(v0) > 1e10: v0 = -100.0
    if abs(v1) > 1e10: v1 = 100.0
    return u0, u1, v0, v1


def extract(filepath: str) -> dict | None:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    reader = STEPControl_Reader()
    if reader.ReadFile(str(filepath)) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        return None

    gs = GRID_SIZE
    face_grids, face_shapes = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        try:
            face = topods.Face(exp.Current())
            s = BRepAdaptor_Surface(face)
            u0, u1, v0, v1 = _uv_box(s)
            grid = np.zeros((gs, gs, 3), dtype=np.float32)
            for i in range(gs):
                for j in range(gs):
                    u = u0 + (u1 - u0) * i / max(gs - 1, 1)
                    v = v0 + (v1 - v0) * j / max(gs - 1, 1)
                    try:
                        p = s.Value(u, v)
                        grid[i, j] = [p.X(), p.Y(), p.Z()]
                    except Exception:
                        pass
            face_grids.append(grid)
            face_shapes.append(face)
        except Exception:
            pass
        exp.Next()

    edge_shapes = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        try:
            edge_shapes.append(topods.Edge(exp.Current()))
        except Exception:
            pass
        exp.Next()

    if not face_grids:
        return None
    nf = min(len(face_grids), MAX_FACES)
    ne = min(len(edge_shapes), MAX_EDGES)

    # Topology
    e2f = np.full((MAX_EDGES, 2), -1, dtype=np.int32)
    efm = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, efm)
    for ei, edge in enumerate(edge_shapes[:ne]):
        try:
            idx = efm.FindIndex(edge)
            if idx > 0:
                fl = efm.FindFromIndex(idx)
                it, side = fl.begin(), 0
                while it != fl.end() and side < 2:
                    af = topods.Face(it.Value())
                    for fi in range(nf):
                        if face_shapes[fi].IsSame(af):
                            e2f[ei, side] = fi
                            side += 1
                            break
                    it.Next()
        except Exception:
            pass

    fm = np.zeros(MAX_FACES, dtype=np.float32)
    fm[:nf] = 1.0

    # 16-d face features + centroids
    fc = np.zeros((MAX_FACES, 3), dtype=np.float32)
    ff = np.zeros((MAX_FACES, 16), dtype=np.float32)
    mid = gs // 2
    for f in range(nf):
        g = face_grids[f]
        p = g.reshape(-1, 3)
        fc[f] = p.mean(0)
        ff[f, 0:3] = p.mean(0)
        ff[f, 3:6] = p.std(0)
        ff[f, 6:9] = p.min(0)
        ff[f, 9:12] = p.max(0)
        du = g[min(mid + 1, gs - 1), mid] - g[max(mid - 1, 0), mid]
        dv = g[mid, min(mid + 1, gs - 1)] - g[mid, max(mid - 1, 0)]
        n = np.cross(du, dv)
        nm = np.linalg.norm(n)
        if nm > 1e-8:
            n /= nm
        ff[f, 12:15] = n
        bb = ff[f, 9:12] - ff[f, 6:9]
        ff[f, 15] = np.prod(np.sort(np.abs(bb))[-2:])

    # Face normals for the angular attention-bias channel
    fn = np.zeros((MAX_FACES, 3), dtype=np.float32)
    k = NORMAL_GRID // 2
    for f in range(nf):
        try:
            s = BRepAdaptor_Surface(face_shapes[f])
            u0, u1, v0, v1 = _uv_box(s)
            u = u0 + (u1 - u0) * k / max(NORMAL_GRID - 1, 1)
            v = v0 + (v1 - v0) * k / max(NORMAL_GRID - 1, 1)
            props = BRepLProp_SLProps(s, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                d = props.Normal()
                fn[f] = [d.X(), d.Y(), d.Z()]
        except Exception:
            pass

    return {
        "face_features": ff,
        "face_centroids": fc,
        "edge_to_faces": e2f,
        "face_mask": fm,
        "face_normals": fn,
        "num_faces": np.int32(nf),
        "num_edges": np.int32(ne),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    out = extract(args.step)
    if out is None:
        sys.exit(f"failed to read {args.step}")
    path = args.out or args.step.rsplit(".", 1)[0] + "_brep.npz"
    np.savez(path, **out)
    print(f"{path}: {int(out['num_faces'])} faces, {int(out['num_edges'])} edges"
          + (" (truncated to 192 faces)" if out["num_faces"] >= MAX_FACES else ""))


if __name__ == "__main__":
    main()
