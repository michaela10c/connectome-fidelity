"""
check_per_model_consistency_raw.py -- per-model consistency check against
the RAW-DATA reference specifically. Reuses the exact reconstruction logic
already validated in check_per_model_consistency.py (confirmed byte-identical
against saved ground truth before trusting it).
"""

import numpy as np
from scipy.stats import rankdata, spearmanr, mannwhitneyu, ttest_ind
from scipy.spatial.distance import cosine, pdist, squareform

N_DIRECTIONS = 8
STIMULUS_DIRECTIONS_DEG = np.arange(0, 360, 360 // N_DIRECTIONS)


def build_rdm_cosine(pop_matrix):
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    pop_matrix = pop_matrix + 1e-10
    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j])
    return rdm


def rank_residualize(rdm, against_rdm):
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    r_vals = rankdata(rdm[iu])
    r_against = rankdata(against_rdm[iu])
    slope, intercept = np.polyfit(r_against, r_vals, 1)
    predicted = slope * r_against + intercept
    return r_vals - predicted


def main():
    flyvis = np.load("results_exp1_8dir_50models_full_shiu.npz", allow_pickle=True)
    cc_rdms = flyvis["cc_rdms_cosine"]
    rand_pop_matrices = flyvis["rand_pop_matrices"]

    raw_population_matrix = np.load("raw_population_matrix.npy")
    raw_rdm = squareform(pdist(raw_population_matrix, metric='cosine'))

    circ_ref = np.zeros((N_DIRECTIONS, N_DIRECTIONS))
    for i in range(N_DIRECTIONS):
        for j in range(N_DIRECTIONS):
            d = abs(STIMULUS_DIRECTIONS_DEG[i] - STIMULUS_DIRECTIONS_DEG[j])
            circ_ref[i, j] = min(d, 360 - d)

    raw_ref_resid = rank_residualize(raw_rdm, circ_ref)

    rand_rdms = np.array([build_rdm_cosine(rand_pop_matrices[i])
                          for i in range(rand_pop_matrices.shape[0])])

    cc_correlations = []
    for i in range(cc_rdms.shape[0]):
        model_resid = rank_residualize(cc_rdms[i], circ_ref)
        r, _ = spearmanr(model_resid, raw_ref_resid)
        cc_correlations.append(r)
    cc_correlations = np.array(cc_correlations)

    rand_correlations = []
    for i in range(rand_rdms.shape[0]):
        model_resid = rank_residualize(rand_rdms[i], circ_ref)
        r, _ = spearmanr(model_resid, raw_ref_resid)
        rand_correlations.append(r)
    rand_correlations = np.array(rand_correlations)

    u_stat, p_mwu = mannwhitneyu(cc_correlations, rand_correlations, alternative='two-sided')
    print(f"CC mean={cc_correlations.mean():.4f}, {100*np.mean(cc_correlations<0):.0f}% negative")
    print(f"Random mean={rand_correlations.mean():.4f}, {100*np.mean(rand_correlations<0):.0f}% negative")
    print(f"Mann-Whitney p={p_mwu:.4f}")

if __name__ == "__main__":
    main()
