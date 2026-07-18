"""
analyze_raw_reference.py -- runs the full, already-validated analysis
pipeline (circularity check, split-half T4/T5 reliability, comparison
against Flyvis CC/random) on the population matrix built directly from
real R_teta data (build_reference_from_raw.py's output), using the
identical methodology already applied to the von Mises version, for a
direct comparison between the two.

USAGE:
    python analyze_raw_reference.py
    (needs raw_population_matrix.npy, results_exp1_8dir_50models_full_shiu.npz)
"""

import numpy as np
from scipy.stats import rankdata, pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)
IS_T4 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)


def rank_residualize(rdm, against_rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    return r_vals - predicted


def main():
    print("=== Loading the raw-data-derived population matrix ===")
    population_matrix = np.load("raw_population_matrix.npy")
    print(f"  shape: {population_matrix.shape}")

    circ_ref = np.zeros((N_DIRECTIONS, N_DIRECTIONS))
    for i in range(N_DIRECTIONS):
        for j in range(N_DIRECTIONS):
            d = abs(STIMULUS_DIRECTIONS_DEG[i] - STIMULUS_DIRECTIONS_DEG[j])
            circ_ref[i, j] = min(d, 360 - d)
    iu = np.triu_indices(N_DIRECTIONS, k=1)

    print("\n=== STEP 1: Circularity check (von Mises version: r=0.9074, "
          "82.3% variance explained) ===")
    rdm = squareform(pdist(population_matrix, metric='cosine'))
    r_pearson, p_pearson = pearsonr(rdm[iu], circ_ref[iu])
    r_spearman, p_spearman = spearmanr(rdm[iu], circ_ref[iu])
    print(f"  Pearson r vs circular-distance:  {r_pearson:.4f}  (p={p_pearson:.2e})")
    print(f"  Spearman rho vs circular-distance: {r_spearman:.4f}  (p={p_spearman:.2e})")
    print(f"  Variance explained by circularity: {r_pearson**2*100:.1f}%")

    print("\n=== STEP 2: Split-half reliability, T4 vs T5 (von Mises "
          "version: r=0.901, p<0.001) ===")
    t4_matrix = population_matrix[:, IS_T4]
    t5_matrix = population_matrix[:, ~IS_T4]
    rdm_t4 = squareform(pdist(t4_matrix, metric='cosine'))
    rdm_t5 = squareform(pdist(t5_matrix, metric='cosine'))
    resid_t4 = rank_residualize(rdm_t4, circ_ref)
    resid_t5 = rank_residualize(rdm_t5, circ_ref)
    r_reliability, p_reliability = pearsonr(resid_t4, resid_t5)
    print(f"  Pearson r between T4-residual and T5-residual: {r_reliability:.4f}  (p={p_reliability:.3f})")

    print("\n=== STEP 3: Comparison against Flyvis CC and random (von Mises "
          "version, corrected: CC r=-0.4094, Random r=-0.5753) ===")
    flyvis = np.load("results_exp1_8dir_50models_full_shiu.npz", allow_pickle=True)
    cc_rdm = flyvis["cc_rdm_cosine"]
    rand_rdm = flyvis["rand_rdm_cosine"]

    r_cc_raw, _ = spearmanr(cc_rdm[iu], rdm[iu])
    r_rand_raw, _ = spearmanr(rand_rdm[iu], rdm[iu])
    print(f"  RAW: CC vs reference r = {r_cc_raw:.4f}  |  Random vs reference r = {r_rand_raw:.4f}")

    cc_resid = rank_residualize(cc_rdm, circ_ref)
    rand_resid = rank_residualize(rand_rdm, circ_ref)
    ref_resid = rank_residualize(rdm, circ_ref)
    r_cc_corrected, p_cc = spearmanr(cc_resid, ref_resid)
    r_rand_corrected, p_rand = spearmanr(rand_resid, ref_resid)
    print(f"  CORRECTED: CC vs reference r = {r_cc_corrected:.4f}  (p={p_cc:.4f})  |  Random vs reference r = {r_rand_corrected:.4f}  (p={p_rand:.4f})")

    print("\n=== Interpretation ===")
    same_sign_cc = np.sign(r_cc_corrected) == np.sign(-0.4094)
    same_sign_rand = np.sign(r_rand_corrected) == np.sign(-0.5753)
    cc_less_negative_than_rand = r_cc_corrected > r_rand_corrected

    if same_sign_cc and same_sign_rand and cc_less_negative_than_rand:
        print(f"  CONFIRMED: same qualitative pattern as the von Mises version "
              f"(both negative, CC less negative than random) using REAL "
              f"response data with no parametric reconstruction at all.")
    else:
        print(f"  CHANGED: the raw-data version does NOT reproduce the same "
              f"pattern as the von Mises version. Important -- the earlier "
              f"finding may have depended on the parametric reconstruction "
              f"in a way that wasn't obvious.")


if __name__ == "__main__":
    main()
