"""
Optimal Cell-Type Selection: Adapting Beiran & Litwin-Kumar (2025) to
Representational Geometry

Beiran & Litwin-Kumar (Nature Neuroscience, 2025) develop a method for
ranking which neurons are most informative to record from, in a
teacher-student RNN setting where the goal is to constrain unknown
single-neuron parameters given a known connectome. Their exact procedure
(greedy selection based on the singular value decomposition of a
parameter-to-activity mapping matrix) assumes an explicit gain/bias
fitting problem that does not directly apply here: our models are already
fully trained, not being fit to unknown per-neuron parameters.

This script adapts the SAME UNDERLYING QUESTION, which subset of units is
most informative, to our setting: given the full 65-cell-type population
response, which subset of cell types is sufficient to reconstruct the full
population's representational geometry (RDM)? This is the natural analogue
of "which neurons to record from" in a representational-geometry framework
rather than a parameter-fitting one.

Method: greedy forward selection. At each step, add whichever remaining
cell type, when combined with the already-selected set, best reconstructs
the full-population RDM (Spearman correlation), averaged across all real
(connectome-constrained) models. Compare against random and worst-case
selection orderings, following the same comparison structure Beiran &
Litwin-Kumar use in their own Figure 8.

Run this from the directory containing results_exp1_50models_full_shiu.npz,
or pass --data pointing to it directly.
"""

import argparse
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform


def compute_rdm(pop_vectors):
    """Cosine-distance RDM from a set of population vectors.

    Args:
        pop_vectors: (n_stimuli, n_units) array

    Returns:
        (n_stimuli, n_stimuli) RDM
    """
    eps = 1e-10
    return squareform(pdist(pop_vectors + eps, metric="cosine"))


def rdm_upper_triangle(rdm):
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def reconstruction_score(pop_matrices, unit_subset, full_rdms):
    """Average Spearman correlation between the full-population RDM and
    the RDM built from only `unit_subset`, across all models.

    Args:
        pop_matrices: (n_models, n_stimuli, n_units)
        unit_subset: list/array of unit indices to include
        full_rdms: precomputed list of full-population RDMs, one per model

    Returns:
        mean Spearman r across models
    """
    scores = []
    for m in range(pop_matrices.shape[0]):
        sub_vectors = pop_matrices[m][:, unit_subset]
        if len(unit_subset) == 1:
            # cosine distance is degenerate for 1D vectors of a single
            # feature; fall back to absolute difference in that case
            sub_rdm = squareform(pdist(sub_vectors, metric="euclidean"))
        else:
            sub_rdm = compute_rdm(sub_vectors)
        r, _ = spearmanr(
            rdm_upper_triangle(full_rdms[m]), rdm_upper_triangle(sub_rdm)
        )
        scores.append(0.0 if np.isnan(r) else r)
    return float(np.mean(scores))


def greedy_selection(pop_matrices, full_rdms, n_units, mode="best"):
    """Greedy forward selection of cell types.

    Args:
        mode: "best" selects the unit maximizing reconstruction score at
              each step; "worst" selects the unit minimizing it, for
              comparison, matching Beiran & Litwin-Kumar's own
              best/random/worst comparison in their Figure 8.

    Returns:
        order: list of unit indices in selection order
        scores: reconstruction score after each addition
    """
    remaining = list(range(n_units))
    selected = []
    scores = []
    while remaining:
        candidate_scores = {}
        for cand in remaining:
            trial_subset = selected + [cand]
            candidate_scores[cand] = reconstruction_score(
                pop_matrices, trial_subset, full_rdms
            )
        if mode == "best":
            chosen = max(candidate_scores, key=candidate_scores.get)
        elif mode == "worst":
            chosen = min(candidate_scores, key=candidate_scores.get)
        else:
            raise ValueError("mode must be 'best' or 'worst'")
        selected.append(chosen)
        scores.append(candidate_scores[chosen])
        remaining.remove(chosen)
    return selected, scores


def random_baseline(pop_matrices, full_rdms, n_units, n_repeats=10, seed=42):
    """Average reconstruction score curve under random selection order,
    for comparison against the greedy best/worst orderings.
    """
    rng = np.random.default_rng(seed)
    all_scores = np.zeros((n_repeats, n_units))
    for rep in range(n_repeats):
        order = rng.permutation(n_units)
        selected = []
        for i, unit in enumerate(order):
            selected.append(unit)
            all_scores[rep, i] = reconstruction_score(
                pop_matrices, selected, full_rdms
            )
    return all_scores.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="../results/results_exp1_50models_full_shiu.npz",
        help="Path to the saved population-vector .npz file",
    )
    parser.add_argument(
        "--n_random_repeats",
        type=int,
        default=10,
        help="Number of random orderings to average for the baseline",
    )
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    d = np.load(args.data, allow_pickle=True)
    pop_matrices = d["cc_pop_matrices"]  # (n_models, n_stimuli, n_units)
    cell_types = (
        list(d["cell_types"]) if "cell_types" in d else
        [f"unit_{i}" for i in range(pop_matrices.shape[2])]
    )
    n_models, n_stimuli, n_units = pop_matrices.shape
    print(f"  {n_models} models, {n_stimuli} stimuli, {n_units} cell types")

    print("Computing full-population RDMs (all cell types) per model ...")
    full_rdms = [compute_rdm(pop_matrices[m]) for m in range(n_models)]

    print("\nRunning greedy BEST selection ...")
    best_order, best_scores = greedy_selection(
        pop_matrices, full_rdms, n_units, mode="best"
    )

    print("Running greedy WORST selection (for comparison) ...")
    worst_order, worst_scores = greedy_selection(
        pop_matrices, full_rdms, n_units, mode="worst"
    )

    print(f"Running {args.n_random_repeats} random-order baselines ...")
    random_scores = random_baseline(
        pop_matrices, full_rdms, n_units, n_repeats=args.n_random_repeats
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nTop 10 most informative cell types (greedy BEST order):")
    for i, unit in enumerate(best_order[:10]):
        print(f"  {i + 1:2d}. {cell_types[unit]:20s} "
              f"(cumulative r = {best_scores[i]:.3f})")

    # number of cell types needed to reach various reconstruction thresholds
    for threshold in (0.8, 0.9, 0.95):
        n_needed = next(
            (i + 1 for i, s in enumerate(best_scores) if s >= threshold),
            None,
        )
        if n_needed is not None:
            print(f"\nCell types needed for r >= {threshold}: {n_needed} "
                  f"of {n_units} ({100 * n_needed / n_units:.1f}%)")
        else:
            print(f"\nThreshold r >= {threshold} not reached with any subset size")

    print("\nComparison at n=10 recorded cell types:")
    print(f"  Best-order reconstruction:   r = {best_scores[9]:.3f}")
    print(f"  Random-order reconstruction: r = {random_scores[9]:.3f}")
    print(f"  Worst-order reconstruction:  r = {worst_scores[9]:.3f}")

    print("\nSaving full results to ../results/optimal_celltype_selection_results.npz")
    np.savez(
        "../results/optimal_celltype_selection_results.npz",
        best_order=best_order,
        best_scores=best_scores,
        worst_order=worst_order,
        worst_scores=worst_scores,
        random_scores=random_scores,
        cell_types=cell_types,
    )
    print("Done.")


if __name__ == "__main__":
    main()
