"""
Build an enriched tree JSON that maps each node to the UIDs from abc_dataset_clean.csv.

Output structure per node:
{
  "name": "...",
  "depth": N,
  "type": "internal" | "leaf",
  "n_keywords": N,
  "n_models": N,              # number of unique UIDs under this node
  "keywords": [...],           # leaf only: keyword list with freq + uid list
  "model_uids": [...],         # leaf only: flat list of all UIDs
  "children": [...]            # internal only
}

Internal nodes get n_models (aggregated from children) but NOT the full
uid list (would be huge and redundant). The leaf nodes carry the actual
uid lists, both per-keyword and as a flat deduplicated set.

Also produces a lightweight index file:
  keyword_uid_index.json  ->  { keyword: [uid1, uid2, ...], ... }
"""

import json
import csv
import re
import sys
import io
from collections import defaultdict
from pathlib import Path
import time

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATASET = Path(__file__).parent.parent / 'abc_dataset_clean.csv'
TREE_JSON = Path(__file__).parent / 'application_tree_dpgmm.json'
OUT_TREE = Path(__file__).parent / 'application_tree_dpgmm_uids.json'
OUT_INDEX = Path(__file__).parent / 'keyword_uid_index.json'


def parse_applications(apps_str):
    """Parse the applications column into a list of keyword strings."""
    if not apps_str or apps_str.strip() == '':
        return []
    # Format: 'Keyword1", "Keyword2", "Keyword3'  (leading quote often missing)
    # Split on '", "' pattern
    apps_str = apps_str.strip().strip('"')
    keywords = re.split(r'"\s*,\s*"', apps_str)
    return [k.strip().strip('"') for k in keywords if k.strip()]


def load_tree(path):
    """Load tree JSON, handling potential trailing data."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    depth = 0
    end = 0
    for i, ch in enumerate(content):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(content[:end])


def build_keyword_uid_index(dataset_path):
    """Build keyword -> set of UIDs mapping from the dataset."""
    print(f"Reading dataset: {dataset_path}")
    t0 = time.time()

    kw_to_uids = defaultdict(set)
    uid_count = 0

    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row['uid']
            keywords = parse_applications(row.get('applications', ''))
            for kw in keywords:
                kw_to_uids[kw].add(uid)
            uid_count += 1

            if uid_count % 50000 == 0:
                print(f"  {uid_count} rows, {len(kw_to_uids)} unique keywords...")

    dt = time.time() - t0
    print(f"  Done: {uid_count} models, {len(kw_to_uids)} keywords in {dt:.1f}s")
    return kw_to_uids


def enrich_tree(node, kw_to_uids, depth=0):
    """Recursively enrich tree nodes with UID information."""
    if node.get('type') == 'leaf':
        # Leaf node: attach UIDs per keyword and as flat set
        all_uids = set()
        enriched_keywords = []

        for kw_info in node.get('keywords', []):
            kw = kw_info['keyword']
            uids = sorted(kw_to_uids.get(kw, set()))
            enriched_keywords.append({
                'keyword': kw,
                'frequency': kw_info.get('frequency', 0),
                'n_models': len(uids),
                'uids': uids
            })
            all_uids.update(uids)

        return {
            'name': node['name'],
            'depth': node.get('depth', depth),
            'type': 'leaf',
            'n_keywords': node.get('n_keywords', len(enriched_keywords)),
            'n_models': len(all_uids),
            'model_uids': sorted(all_uids),
            'keywords': enriched_keywords,
        }

    else:
        # Internal node: recurse into children, aggregate model count
        enriched_children = []
        all_child_uids = set()

        for child in node.get('children', []):
            enriched_child = enrich_tree(child, kw_to_uids, depth + 1)
            enriched_children.append(enriched_child)
            # Collect UIDs from child
            if enriched_child['type'] == 'leaf':
                all_child_uids.update(enriched_child.get('model_uids', []))
            else:
                # For internal children, we need to walk down to get UIDs
                # But we stored n_models, so just collect from the recursive result
                all_child_uids.update(collect_uids(enriched_child))

        return {
            'name': node['name'],
            'depth': node.get('depth', depth),
            'type': 'internal',
            'n_keywords': node.get('n_keywords', 0),
            'n_children': len(enriched_children),
            'n_models': len(all_child_uids),
            'children': enriched_children,
        }


def collect_uids(node):
    """Recursively collect all UIDs under a node."""
    if node['type'] == 'leaf':
        return set(node.get('model_uids', []))
    uids = set()
    for child in node.get('children', []):
        uids.update(collect_uids(child))
    return uids


def tree_stats(node, depth=0):
    """Print summary stats."""
    if node['type'] == 'leaf':
        return 1, 0, node['n_models']
    leaves = 0
    internals = 1
    models = node['n_models']
    for child in node.get('children', []):
        l, i, _ = tree_stats(child, depth + 1)
        leaves += l
        internals += i
    return leaves, internals, models


def main():
    print("=" * 60)
    print("Building UID-enriched tree")
    print("=" * 60)

    # Step 1: Build keyword -> UID index
    kw_to_uids = build_keyword_uid_index(DATASET)

    # Step 2: Save the flat keyword-uid index
    print(f"\nSaving keyword-UID index to {OUT_INDEX}...")
    index_data = {kw: sorted(uids) for kw, uids in kw_to_uids.items()}
    with open(OUT_INDEX, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=None, separators=(',', ':'))
    size_mb = OUT_INDEX.stat().st_size / 1e6
    print(f"  {len(index_data)} keywords, {size_mb:.1f} MB")

    # Step 3: Load and enrich tree
    print(f"\nLoading tree from {TREE_JSON}...")
    tree = load_tree(TREE_JSON)

    print("Enriching tree with UIDs...")
    t0 = time.time()
    enriched = enrich_tree(tree, kw_to_uids)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    # Step 4: Stats
    leaves, internals, total_models = tree_stats(enriched)
    print(f"\nTree stats:")
    print(f"  Leaves: {leaves}")
    print(f"  Internal: {internals}")
    print(f"  Unique models at root: {enriched['n_models']}")

    # Step 5: Save enriched tree
    print(f"\nSaving enriched tree to {OUT_TREE}...")
    with open(OUT_TREE, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2)
    size_mb = OUT_TREE.stat().st_size / 1e6
    print(f"  {size_mb:.1f} MB")

    print("\nDone.")


if __name__ == '__main__':
    main()
