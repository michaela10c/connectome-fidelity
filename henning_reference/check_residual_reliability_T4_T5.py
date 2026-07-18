"""
check_residual_reliability_T4_T5.py -- direct test of whether the 17.7%
non-circular residual in the Henning-derived DIRECTION RDM (not the
cell-group profiles checked previously) is real, reproducible signal or
noise.

DESIGN: T4 (ON pathway) and T5 (OFF pathway) are independently measured,
biologically distinct populations. If the non-circular residual in the
direction RDM reflects real structure, it should be reproducible: a
direction RDM built from ONLY T4 columns and one built from ONLY T5
columns should show CORRELATED residuals after each is separately
rank-residualized against circular distance. If the residual is mostly
noise specific to how the 8 columns happened to combine, the two
pathways' residuals should be uncorrelated with each other -- each
pathway's "leftover after circularity" would look like independent
noise, not shared signal.

This is a split-half reliability check, using a real biological split
(T4 vs T5) rather than an arbitrary one, applied to the actual quantity
in question (the direction RDM's residual), not an adjacent one.

USAGE:
    python check_residual_reliability_T4_T5.py
    (run after build_henning_reference.py, needs henning_population_matrix.npy)
"""

import numpy as np
from scipy.stats import rankdata, pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)
LAYER_CELLTYPES = ['T4A', 'T4B', 'T4C', 'T4D', 'T5A', 'T5B', 'T5C', 'T5D']
IS_T4 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)


def rank_residualize_matrix(rdm, against_rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    return r_vals - predicted, iu


def main():
    print("=== Loading population matrix ===")
    population_matrix = np.load("henning_population_matrix.npy")
    print(f"  shape: {population_matrix.shape} (8 directions x 8 layer/celltype groups)")

    circ_ref = squareform(pdist(
        STIMULUS_DIRECTIONS_DEG.reshape(-1, 1),
        metric=lambda a, b: min(abs(a[0] - b[0]), 360 - abs(a[0] - b[0]))
    ))

    print("\n=== Building SEPARATE direction RDMs from T4-only and T5-only "
          "columns ===")
    t4_matrix = population_matrix[:, IS_T4]   # 8 directions x 4 T4 groups
    t5_matrix = population_matrix[:, ~IS_T4]  # 8 directions x 4 T5 groups

    rdm_t4 = squareform(pdist(t4_matrix, metric='cosine'))
    rdm_t5 = squareform(pdist(t5_matrix, metric='cosine'))
    print(f"  T4-only RDM off-diagonal range: {rdm_t4[rdm_t4>0].min():.4f} to {rdm_t4.max():.4f}")
    print(f"  T5-only RDM off-diagonal range: {rdm_t5[rdm_t5>0].min():.4f} to {rdm_t5.max():.4f}")

    print("\n=== Circularity check on each pathway SEPARATELY ===")
    iu = np.triu_indices(N_DIRECTIONS, k=1)
    r_t4_circ, _ = pearsonr(rdm_t4[iu], circ_ref[iu])
    r_t5_circ, _ = pearsonr(rdm_t5[iu], circ_ref[iu])
    print(f"  T4-only RDM vs circular reference: r={r_t4_circ:.4f}")
    print(f"  T5-only RDM vs circular reference: r={r_t5_circ:.4f}")
    print(f"  (combined 8-column RDM was r=0.9074 -- compare against these)")

    print("\n=== Rank-residualizing EACH pathway's RDM against circular "
          "distance, separately ===")
    resid_t4, _ = rank_residualize_matrix(rdm_t4, circ_ref)
    resid_t5, _ = rank_residualize_matrix(rdm_t5, circ_ref)

    print("\n=== THE ACTUAL TEST: do the two independent pathways' residuals "
          "agree with each other? ===")
    r_reliability, p_reliability = pearsonr(resid_t4, resid_t5)
    rho_reliability, p_rho = spearmanr(resid_t4, resid_t5)
    print(f"  Pearson r between T4-residual and T5-residual: {r_reliability:.4f}  (p={p_reliability:.3f})")
    print(f"  Spearman rho: {rho_reliability:.4f}  (p={p_rho:.3f})")
    print(f"  (n={len(resid_t4)} direction-pairs; with only 28 points this "
          f"test has limited power, so treat p-values as suggestive, not "
          f"definitive, at this sample size)")

    print("\n=== Interpretation ===")
    if r_reliability > 0.5 and p_reliability < 0.05:
        print(f"  RELIABLE: T4 and T5's independent non-circular residuals "
              f"agree substantially (r={r_reliability:.3f}). This is real "
              f"evidence the 17.7% residual reflects genuine, reproducible "
              f"structure present in two independently measured pathways, "
              f"not noise specific to how the columns were combined.")
    elif r_reliability > 0.2:
        print(f"  WEAKLY RELIABLE: some agreement between the two pathways' "
              f"residuals (r={r_reliability:.3f}) but not strong or clearly "
              f"significant at this sample size. Consistent with a real but "
              f"modest signal, or with noise that happens to partially "
              f"align -- underpowered to fully distinguish these with only "
              f"28 direction-pairs.")
    else:
        print(f"  NOT RELIABLE: T4 and T5's independent residuals do not "
              f"agree (r={r_reliability:.3f}). This is a real, concerning "
              f"finding -- it suggests the 17.7% non-circular residual in "
              f"the combined RDM may not reflect structure present "
              f"independently in both pathways, and could be more "
              f"consistent with noise or an artifact of how the 8 columns "
              f"were pooled together, rather than genuine biological signal.")


if __name__ == "__main__":
    main()
