"""
Structural vs. Activity-Based Cell-Type Informativeness

Randal Koene's question, precisely: can the connectome ITSELF (wiring alone,
no activity) tell us which neurons are sufficient? The greedy selection in
optimal_celltype_selection.py answers a related but different question: it
ranks cell types by how informative their TRAINED ACTIVITY is for
reconstructing the full population's representational geometry.

This script closes that gap: it compares the activity-based ranking against
a purely structural ranking (in-degree, out-degree, or total degree per cell
type, computed directly from the connectome adjacency, no simulation or
training involved) via Spearman rank correlation.

Strong agreement (high Spearman rho) would mean wiring structure alone
predicts informativeness -- a real, direct answer to Randal's question.
Weak or no agreement would mean trained activity carries information beyond
what raw connectivity reveals -- an equally real, and arguably more
interesting, answer.

Note: this script does NOT compute the structural degree values itself.
You need to supply a per-cell-type degree array from your own connectome
data (likely already computed somewhere in randomize_connectome_schemes.py
or wherever the degree-preserving swap logic lives). See the
`--structural_degrees` argument below.

Expected input format for --structural_degrees: a .npz file containing
  cell_types: array of cell type names (matching the names used in the
               population-vector file)
  in_degree: array of in-degree per cell type
  out_degree: array of out-degree per cell type
If your existing pipeline stores this differently, adjust the loading
section below accordingly -- flag me the actual format and I'll rewrite
this section precisely rather than guess at it.
"""

import argparse
import numpy as np
from scipy.stats import spearmanr


def load_activity_ranking(path):
    """Load the greedy activity-based selection order and scores.

    best_order in the saved file is a list of integer indices into the
    cell_types array from the ORIGINAL population-vector file, not cell
    type name strings. Convert to names here so downstream comparisons
    are name-to-name, not index-to-name.
    """
    d = np.load(path, allow_pickle=True)
    cell_types = list(d["cell_types"])
    best_order_idx = list(d["best_order"])
    best_order_names = [cell_types[i] for i in best_order_idx]
    return best_order_names, cell_types


def load_structural_degrees(path):
    """Load per-cell-type structural degree values.

    Adjust this function if your connectome data is stored in a different
    format -- this assumes a .npz with 'cell_types', 'in_degree', and
    'out_degree' arrays, aligned by index.
    """
    d = np.load(path, allow_pickle=True)
    cell_types = list(d["cell_types"])
    in_degree = np.asarray(d["in_degree"], dtype=float)
    out_degree = np.asarray(d["out_degree"], dtype=float)
    total_degree = in_degree + out_degree
    return cell_types, in_degree, out_degree, total_degree


def rank_from_values(cell_types, values, descending=True):
    """Convert a value array into a ranked list of cell types, highest
    (most informative / highest degree) first.
    """
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    return [cell_types[i] for i in order]


def compare_rankings(activity_order, structural_order, label):
    """Spearman correlation between two full rankings of the same items,
    plus overlap in the top-10 of each.
    """
    # Spearman correlation needs numeric rank vectors over the same items,
    # in a consistent order. Use activity_order's item order as the
    # reference and look up each item's rank in structural_order.
    structural_rank_lookup = {ct: i for i, ct in enumerate(structural_order)}
    activity_ranks = list(range(len(activity_order)))
    structural_ranks = [structural_rank_lookup[ct] for ct in activity_order]

    rho, p = spearmanr(activity_ranks, structural_ranks)

    top10_activity = set(activity_order[:10])
    top10_structural = set(structural_order[:10])
    overlap = top10_activity & top10_structural

    print(f"\n--- {label} ---")
    print(f"Spearman rho (activity rank vs. structural rank): {rho:.3f} "
          f"(p = {p:.4f})")
    print(f"Top-10 overlap: {len(overlap)}/10 cell types in common")
    if overlap:
        print(f"  Shared: {sorted(overlap)}")
    print(f"Activity top 5:    {activity_order[:5]}")
    print(f"Structural top 5:  {structural_order[:5]}")

    return rho, p, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activity_results",
        default="../results/optimal_celltype_selection_results.npz",
        help="Path to saved output from optimal_celltype_selection.py",
    )
    parser.add_argument(
        "--structural_degrees",
        required=True,
        help="Path to .npz with per-cell-type structural degree data "
             "(see module docstring for expected format)",
    )
    args = parser.parse_args()

    print(f"Loading activity-based ranking from {args.activity_results} ...")
    activity_order, activity_cell_types = load_activity_ranking(
        args.activity_results
    )

    print(f"Loading structural degree data from {args.structural_degrees} ...")
    struct_cell_types, in_deg, out_deg, total_deg = load_structural_degrees(
        args.structural_degrees
    )

    # sanity check: cell type sets should match
    missing = set(activity_cell_types) - set(struct_cell_types)
    if missing:
        print(f"\nWARNING: {len(missing)} cell types in activity data have "
              f"no structural match: {sorted(missing)[:5]}...")

    print("\n" + "=" * 60)
    print("STRUCTURAL vs. ACTIVITY-BASED INFORMATIVENESS RANKING")
    print("=" * 60)

    for label, values in [
        ("Total degree (in + out)", total_deg),
        ("In-degree only", in_deg),
        ("Out-degree only", out_deg),
    ]:
        structural_order = rank_from_values(struct_cell_types, values)
        compare_rankings(activity_order, structural_order, label)

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("High Spearman rho (roughly > 0.5) and strong top-10 overlap would")
    print("suggest wiring structure alone substantially predicts which cell")
    print("types are informative -- a direct answer to Randal's question.")
    print("Low or near-zero rho would suggest trained activity carries")
    print("information beyond what raw connectivity reveals, which is a")
    print("real, honest finding in its own right, not a failed test.")


if __name__ == "__main__":
    main()
