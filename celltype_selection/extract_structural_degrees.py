"""
Extract Per-Cell-Type Structural Degree from the Flyvis Connectome

Computes in-degree and out-degree per cell type directly from the connectome
JSON (fib25-fib19_v2.2.json), with NO activity or simulation involved --
this is wiring structure alone, matching the same "number of distinct
connection partners" definition already used elsewhere in this project (see
the degree-preserving swap scheme, which preserves exactly this quantity).

Since each edge in this connectome file already represents one unique
(src, tar) cell-type pair (605 edges = 605 unique pairs across 65 cell
types), degree is a direct count:
    out_degree[cell_type] = number of edges where src == cell_type
    in_degree[cell_type]  = number of edges where tar == cell_type

Also computes a synapse-weighted variant (total synapse count per cell
type, summed across all its edges' offset slots), a richer structural
measure than raw partner count, in case that ranking differs meaningfully
from plain degree.

Output format matches what structural_vs_activity_ranking.py expects:
a .npz with 'cell_types', 'in_degree', 'out_degree' arrays (plain-degree
version), plus a second file with synapse-weighted degree for comparison.

USAGE:
    python extract_structural_degrees.py fib25-fib19_v2_2.json --outdir ../results
"""

import json
import argparse
import os
from collections import Counter
import numpy as np


def load_spec(path):
    with open(path) as f:
        return json.load(f)


def compute_plain_degree(edges):
    """Number of distinct connection partners per cell type -- the same
    definition already used by the degree-preserving swap scheme.
    """
    out_degree = Counter(e["src"] for e in edges)
    in_degree = Counter(e["tar"] for e in edges)
    return in_degree, out_degree


def compute_synapse_weighted_degree(edges):
    """Total synapse count per cell type, summed across all its edges'
    offset slots (in as target, out as source).
    """
    out_weighted = Counter()
    in_weighted = Counter()
    for e in edges:
        n_syn = sum(off[1] for off in e["offsets"])
        out_weighted[e["src"]] += n_syn
        in_weighted[e["tar"]] += n_syn
    return in_weighted, out_weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connectome_json", help="Path to fib25-fib19_v2_2.json")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.connectome_json} ...")
    spec = load_spec(args.connectome_json)
    edges = spec["edges"]
    print(f"  {len(edges)} edges")

    all_cell_types = sorted(
        set(e["src"] for e in edges) | set(e["tar"] for e in edges)
    )
    print(f"  {len(all_cell_types)} unique cell types")

    print("\nComputing plain degree (number of distinct partners) ...")
    in_deg, out_deg = compute_plain_degree(edges)
    in_degree_arr = np.array([in_deg.get(ct, 0) for ct in all_cell_types])
    out_degree_arr = np.array([out_deg.get(ct, 0) for ct in all_cell_types])

    plain_path = os.path.join(args.outdir, "structural_degrees_plain.npz")
    np.savez(
        plain_path,
        cell_types=np.array(all_cell_types, dtype=object),
        in_degree=in_degree_arr,
        out_degree=out_degree_arr,
    )
    print(f"  Saved: {plain_path}")

    print("\nComputing synapse-weighted degree ...")
    in_w, out_w = compute_synapse_weighted_degree(edges)
    in_weighted_arr = np.array([in_w.get(ct, 0) for ct in all_cell_types])
    out_weighted_arr = np.array([out_w.get(ct, 0) for ct in all_cell_types])

    weighted_path = os.path.join(args.outdir, "structural_degrees_synapse_weighted.npz")
    np.savez(
        weighted_path,
        cell_types=np.array(all_cell_types, dtype=object),
        in_degree=in_weighted_arr,
        out_degree=out_weighted_arr,
    )
    print(f"  Saved: {weighted_path}")

    print("\n" + "=" * 60)
    print("TOP 10 BY TOTAL DEGREE (in + out), PLAIN")
    print("=" * 60)
    total_plain = in_degree_arr + out_degree_arr
    top10_idx = np.argsort(total_plain)[::-1][:10]
    for rank, i in enumerate(top10_idx, 1):
        print(f"  {rank:2d}. {all_cell_types[i]:20s} "
              f"in={in_degree_arr[i]:3d}  out={out_degree_arr[i]:3d}  "
              f"total={total_plain[i]:3d}")

    print("\n" + "=" * 60)
    print("TOP 10 BY TOTAL SYNAPSE-WEIGHTED DEGREE")
    print("=" * 60)
    total_weighted = in_weighted_arr + out_weighted_arr
    top10_idx_w = np.argsort(total_weighted)[::-1][:10]
    for rank, i in enumerate(top10_idx_w, 1):
        print(f"  {rank:2d}. {all_cell_types[i]:20s} "
              f"in={in_weighted_arr[i]:6.0f}  out={out_weighted_arr[i]:6.0f}  "
              f"total={total_weighted[i]:6.0f}")

    print("\nDone. Feed either output into structural_vs_activity_ranking.py "
          "via --structural_degrees.")


if __name__ == "__main__":
    main()
