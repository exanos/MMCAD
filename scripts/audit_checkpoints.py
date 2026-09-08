#!/usr/bin/env python3
"""
Inventory every .pth under a directory and print what each one actually is --
which encoder towers it contains, which epoch, and every retrieval metric
stored in it -- WITHOUT loading any tensors.

A .pth is a zip archive whose `data.pkl` member holds the object graph; the
tensor storages live in separate members. Reading only `data.pkl` means a 4 GB
checkpoint is inspected in milliseconds, so this is safe to run over a whole
Google Drive folder.

Usage
-----
    python audit_checkpoints.py /content/drive/MyDrive/MMCAD
    python audit_checkpoints.py /content/drive/MyDrive/MMCAD --sort r@1
"""

from __future__ import annotations

import argparse
import io
import pickletools
import re
import zipfile
from pathlib import Path

METRIC_RE = re.compile(r"R@|recall|mAP|loss|epoch|acc", re.I)

TOWER_HINTS = {
    "text_encoder": "text (EmbeddingGemma)",
    "brep_encoder": "B-Rep (BRepFormer)",
    "pc_encoder": "point cloud (DGCNN)",
    "vit": "sketch (ViT-Base)",
    "vision": "photoreal image (SigLIP)",
    "dgcnn": "point cloud (DGCNN)",
}


def read_header(path: Path):
    """Return (tokens, module_roots) parsed from the checkpoint's pickle."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        blob = z.read(name)

    out = io.StringIO()
    try:
        pickletools.dis(blob, out)
    except Exception:
        pass
    text = out.getvalue()

    tokens = []
    for line in text.split("\n"):
        m = re.search(r"(?:SHORT_BINUNICODE|BINUNICODE)\s+'([^']*)'", line)
        if m:
            tokens.append(("S", m.group(1)))
            continue
        m = re.search(r"BINFLOAT\s+([-\d.eE+]+)", line)
        if m:
            tokens.append(("F", float(m.group(1))))
            continue
        m = re.search(r"BININT\d?\s+(-?\d+)", line)
        if m:
            tokens.append(("I", int(m.group(1))))

    roots = sorted({t.split(".")[0] for k, t in tokens if k == "S" and "." in t})
    return tokens, roots


def extract_metrics(tokens):
    metrics = {}
    for i in range(len(tokens) - 1):
        kind, key = tokens[i]
        kind2, val = tokens[i + 1]
        if kind == "S" and kind2 in ("F", "I") and METRIC_RE.search(str(key)):
            metrics.setdefault(key, val)
    return metrics


def describe_towers(roots):
    found = []
    for r in roots:
        for hint, label in TOWER_HINTS.items():
            if r == hint and label not in found:
                found.append(label)
    return found or roots[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory to scan recursively")
    ap.add_argument("--pattern", default="*.pth")
    ap.add_argument("--sort", default="date", choices=["date", "size", "name", "r@1"])
    ap.add_argument("--max-metrics", type=int, default=14)
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob(args.pattern))
    if not files:
        print(f"no {args.pattern} under {args.root}")
        return

    rows = []
    for f in files:
        try:
            tokens, roots = read_header(f)
        except Exception as exc:
            rows.append((f, None, None, f"UNREADABLE: {exc}"))
            continue
        rows.append((f, extract_metrics(tokens), describe_towers(roots), None))

    def best_r1(metrics):
        if not metrics:
            return -1.0
        vals = [v for k, v in metrics.items() if re.search(r"(R|recall)@1(_|$)", k, re.I)]
        return max(vals) if vals else -1.0

    if args.sort == "date":
        rows.sort(key=lambda r: r[0].stat().st_mtime, reverse=True)
    elif args.sort == "size":
        rows.sort(key=lambda r: r[0].stat().st_size, reverse=True)
    elif args.sort == "r@1":
        rows.sort(key=lambda r: best_r1(r[1]), reverse=True)
    else:
        rows.sort(key=lambda r: str(r[0]))

    for path, metrics, towers, err in rows:
        size_gb = path.stat().st_size / 1e9
        import datetime
        when = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print("=" * 78)
        print(f"{path}")
        print(f"  {size_gb:.2f} GB   modified {when}")
        if err:
            print(f"  {err}")
            continue
        print(f"  towers: {', '.join(towers)}")
        epoch = metrics.get("epoch")
        if epoch is not None:
            print(f"  epoch: {epoch}")
        ranked = sorted(
            ((k, v) for k, v in metrics.items()
             if k != "epoch" and re.search(r"R@|recall|mAP", k, re.I)),
            key=lambda kv: -kv[1],
        )
        if ranked:
            print("  metrics (best first):")
            for k, v in ranked[:args.max_metrics]:
                print(f"      {k:<34} {v:.2f}")
            if len(ranked) > args.max_metrics:
                print(f"      … {len(ranked) - args.max_metrics} more")
        else:
            losses = {k: v for k, v in metrics.items() if "loss" in k.lower()}
            if losses:
                print("  losses only (no retrieval metrics stored):")
                for k, v in list(losses.items())[:6]:
                    print(f"      {k:<34} {v:.4f}")
    print("=" * 78)
    print(f"{len(rows)} checkpoint(s) scanned.")


if __name__ == "__main__":
    main()
