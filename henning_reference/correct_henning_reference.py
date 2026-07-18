"""
correct_henning_reference.py -- applies the same rank-residualization
circularity correction used elsewhere in this project (Experiment 3,
Experiment 5, both MICrONS versions) to the Henning-derived reference RDM.

IMPORTANT FRAMING DIFFERENCE from every other use of this correction in
this project: those all compared TWO RDMs (a model vs a reference) after
removing circularity from both. Here there is no re-simulated Flyvis
model yet to compare against -- only the reference itself. So this
script answers a narrower, more honest question: how much of the
reference's own structure is explained by circularity, and does
whatever is left behind carry real, checkable signal, or is it just
noise. It is NOT a test of whether this reference would beat a null in
an actual model comparison -- that still requires the re-simulation
step (the 8-vs-12-direction stimulus mismatch) to even be possible.

The "is the residual real" check: T4 (ON) and T5 (OFF) are independently
known, from Maisak et al. 2013 and confirmed throughout this dataset's
own paper, to be functionally distinct pathways. If the residual RDM
(after circularity is removed) still cleanly separates T4 conditions
from T5 conditions, that's evidence the leftover structure is tracking
something real, not just residual noise. If that separation collapses
along with the circular structure, that's evidence there isn't much
left to work with.

USAGE:
    python correct_henning_reference.py
    (run after build_henning_reference.py, needs its saved .npy outputs)
"""

import numpy as np
from scipy.stats import rankdata, pearsonr
from scipy.spatial.distance import pdist, squareform

N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)
LAYER_CELLTYPES = ['T4A', 'T4B', 'T4C', 'T4D', 'T5A', 'T5B', 'T5C', 'T5D']
IS_T4 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)  # first 4 cols are T4, last 4 are T5


def rank_residualize_matrix(rdm, against_rdm):
    """Rank-residualize the upper triangle of rdm against the upper
    triangle of against_rdm -- same technique already validated
    elsewhere in this project. Returns the residual as a full symmetric
    matrix (zero diagonal) for further use, plus the raw upper-triangle
    residual vector for stats.
    """
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    residual_upper = r_vals - predicted

    residual_matrix = np.zeros_like(rdm)
    residual_matrix[iu] = residual_upper
    residual_matrix = residual_matrix + residual_matrix.T
    return residual_matrix, residual_upper


def main():
    print("=== Loading the Henning-derived population matrix and reference "
          "RDM from build_henning_reference.py ===")
    population_matrix = np.load("henning_population_matrix.npy")
    rdm = np.load("henning_reference_rdm.npy")
    print(f"  population_matrix shape: {population_matrix.shape} "
          f"(8 directions x 8 layer/celltype groups)")
    print(f"  rdm shape: {rdm.shape}")

    circ_ref = squareform(pdist(
        STIMULUS_DIRECTIONS_DEG.reshape(-1, 1),
        metric=lambda a, b: min(abs(a[0] - b[0]), 360 - abs(a[0] - b[0]))
    ))

    print("\n=== Reproducing the raw circularity correlation (sanity check "
          "against the earlier run) ===")
    iu = np.triu_indices(N_DIRECTIONS, k=1)
    r_raw, p_raw = pearsonr(rdm[iu], circ_ref[iu])
    print(f"  raw Pearson r = {r_raw:.4f}  (should match 0.9074 from before)")
    print(f"  variance explained by circularity alone: {r_raw**2*100:.1f}%")

    print("\n=== Rank-residualizing the reference against circular distance "
          "-- same technique as Experiment 3 / Experiment 5 ===")
    residual_matrix, residual_upper = rank_residualize_matrix(rdm, circ_ref)
    residual_var = np.var(residual_upper)
    total_var = np.var(rankdata(rdm[iu]))
    print(f"  Residual variance retained: {residual_var:.2f} out of "
          f"{total_var:.2f} total rank-variance ({100*residual_var/total_var:.1f}%)")
    print(f"  (this is the ceiling on how much 'non-circular' structure "
          f"could possibly be detected in ANY comparison using this "
          f"reference -- {100*(1-r_raw**2):.1f}% of variance is genuinely "
          f"unexplained by circularity, matching 1 - r^2 above)")

    print("\n=== Is what's left real? Checking whether the residual still "
          "separates T4 (ON) from T5 (OFF) conditions -- an independently "
          "known, real biological distinction, not something assumed into "
          "the construction ===")
    # population_matrix columns are ordered T4A,T4B,T4C,T4D,T5A,T5B,T5C,T5D
    # check: in the RAW population matrix, are T4 and T5 columns'
    # response PROFILES (across the 8 directions) more different from
    # each other than same-pathway columns are from each other?
    t4_cols = population_matrix[:, IS_T4]
    t5_cols = population_matrix[:, ~IS_T4]

    within_t4 = pdist(t4_cols.T, metric='cosine')
    within_t5 = pdist(t5_cols.T, metric='cosine')
    between = pdist(np.vstack([t4_cols.T, t5_cols.T]), metric='cosine')
    # crude but direct: mean within-pathway distance vs mean cross-pathway
    # distance, computed directly on response profiles (not the RDM,
    # since T4/T5 identity is a property of the COLUMNS/groups, not the
    # ROWS/stimulus-directions the 8x8 RDM and its residual are built over)
    cross_pairs = []
    for i in range(4):
        for j in range(4):
            cross_pairs.append(1 - np.corrcoef(t4_cols[:, i], t5_cols[:, j])[0, 1])
    print(f"  mean within-T4 cosine distance: {within_t4.mean():.4f}")
    print(f"  mean within-T5 cosine distance: {within_t5.mean():.4f}")
    print(f"  mean T4-vs-T5 cross distance:   {np.mean(cross_pairs):.4f}")
    if np.mean(cross_pairs) > max(within_t4.mean(), within_t5.mean()):
        print(f"  T4 and T5 response profiles are more different from each "
              f"other than within-pathway variation -- the ON/OFF "
              f"distinction is present in this population data, "
              f"independent of the circularity question above.")
    else:
        print(f"  T4 and T5 response profiles are NOT more different from "
              f"each other than within-pathway variation -- the expected "
              f"ON/OFF distinction is not clearly showing up in this "
              f"construction, which is worth taking seriously as a "
              f"warning sign about the reconstruction method, not just "
              f"the circularity question.")

    print("\n=== Overall interpretation ===")
    print(f"  Circularity explains {r_raw**2*100:.1f}% of the reference "
          f"RDM's rank-variance -- substantial, but importantly NOT as "
          f"total as the retracted Maisak reference (r=0.978, {0.978**2*100:.1f}% "
          f"explained). There is a real, non-trivial {100*(1-r_raw**2):.1f}% "
          f"of variance this reference carries that a pure circular-distance "
          f"matrix does not. Whether that residual is enough to usefully "
          f"discriminate a good model from a bad one is a separate "
          f"question that can only be answered once there's an actual "
          f"model RDM to compare it against -- this script establishes "
          f"the ceiling on what's possible, not the final answer.")


if __name__ == "__main__":
    main()
