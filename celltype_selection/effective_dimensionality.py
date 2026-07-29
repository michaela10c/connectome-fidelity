"""
Effective Dimensionality of Trained Computation: Real vs. Random Wiring

Candidate experiment 3: does real wiring reach a lower-dimensional, more
compressed solution than trained random wiring? Rather than asking whether
the two populations' geometries are distinguishable (the paper's existing
RSA-based question), this asks how many effective degrees of freedom each
population actually uses to represent the stimulus set, a more direct
notion of computational efficiency.

METHOD: for each model, treat its population response as a (n_stimuli,
n_cell_types) matrix. Compute the participation ratio of its PCA
eigenvalue spectrum:

    PR = (sum of eigenvalues)^2 / sum of (eigenvalues^2)

PR ranges from 1 (all variance in one dimension) to n_stimuli (variance
spread evenly across all dimensions). A lower PR means a more compressed,
lower-dimensional representation.

Compare the PR distribution across all real (connectome-constrained)
models against all random (weight-shuffled) models via Mann-Whitney U,
matching the statistical convention used throughout the rest of this
project.

USAGE:
    python effective_dimensionality.py --data results_exp1_50models_full_shiu.npz
"""

import argparse
import numpy as np
from scipy.stats import mannwhitneyu


def participation_ratio(pop_matrix):
    """pop_matrix: (n_stimuli, n_units). Returns the participation ratio
    of its PCA eigenvalue spectrum (equivalently, of the eigenvalues of
    its covariance matrix across stimuli).
    """
    centered = pop_matrix - pop_matrix.mean(axis=0, keepdims=True)
    cov = centered @ centered.T  # (n_stimuli, n_stimuli), avoids needing
                                   # n_units >= n_stimuli
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)  # numerical safety
    num = eigvals.sum() ** 2
    denom = (eigvals ** 2).sum()
    if denom == 0:
        return 0.0
    return num / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="results_exp1_50models_full_shiu.npz",
        help="Path to the saved population-vector .npz file",
    )
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    d = np.load(args.data, allow_pickle=True)
    cc_matrices = d["cc_pop_matrices"]      # (n_models, n_stimuli, n_units)
    rand_matrices = d["rand_pop_matrices"]  # (n_models, n_stimuli, n_units)

    n_cc, n_stim, n_units = cc_matrices.shape
    n_rand = rand_matrices.shape[0]
    print(f"  {n_cc} real (connectome-constrained) models")
    print(f"  {n_rand} random (weight-shuffled) models")
    print(f"  {n_stim} stimuli, {n_units} cell types per model")

    print("\nComputing participation ratio per model ...")
    cc_pr = np.array([participation_ratio(cc_matrices[m]) for m in range(n_cc)])
    rand_pr = np.array([participation_ratio(rand_matrices[m]) for m in range(n_rand)])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nReal (CC) participation ratio:    "
          f"mean = {cc_pr.mean():.3f}, std = {cc_pr.std():.3f}")
    print(f"Random (shuffled) participation ratio: "
          f"mean = {rand_pr.mean():.3f}, std = {rand_pr.std():.3f}")
    print(f"\nMax possible PR (n_stimuli): {n_stim}")

    stat, p = mannwhitneyu(cc_pr, rand_pr, alternative="two-sided")
    print(f"\nMann-Whitney U test (real vs. random): p = {p:.4g}")

    direction = "LOWER (more compressed)" if cc_pr.mean() < rand_pr.mean() else "HIGHER (less compressed)"
    print(f"\nReal wiring's dimensionality is {direction} than random wiring's.")

    # effect size: rank-biserial correlation, consistent with reporting
    # practice elsewhere in this project favoring effect sizes alongside p
    n1, n2 = len(cc_pr), len(rand_pr)
    effect_size = 1 - (2 * stat) / (n1 * n2)
    print(f"Rank-biserial effect size: {effect_size:.3f}")

    print("\nSaving results to effective_dimensionality_results.npz")
    np.savez(
        "effective_dimensionality_results.npz",
        cc_pr=cc_pr,
        rand_pr=rand_pr,
        p_value=p,
        effect_size=effect_size,
    )
    print("Done.")


if __name__ == "__main__":
    main()
