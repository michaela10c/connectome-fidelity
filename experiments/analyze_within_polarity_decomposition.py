#!/usr/bin/env python
"""
Proper, individual-pairwise version of the within-polarity decomposition
check: does removing the ON-vs-OFF cross-block from the ON+OFF RDM collapse
the apparent CC-vs-null convergence back down to what ON-only and Henning
already show?

The earlier version of this check used the weaker mean-vs-mean statistic
(two averaged RDMs correlated against each other) -- this uses the same
individual-pairwise + within-CC-baseline Mann-Whitney test already validated
for the main comparison, now applied to within-polarity sub-blocks instead
of the full 24x24 RDM.

Requires the *_raw_for_replotting.npz produced by the version of
test_item1_all_null_schemes.py that saves individual RDMs (cc_rdms_individual,
{scheme}_null_rdms_individual) -- older .npz files without those keys will
fail with a clear error rather than silently produce wrong numbers.

Usage:
    python analyze_within_polarity_decomposition.py \
        --npz item1_all_null_schemes_results_moving_edge_12dir_on_off_first_raw_for_replotting.npz
"""
import argparse
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu


def pairwise_r(rdms_a, rdms_b, indices):
    """All pairwise Spearman correlations between two lists of RDMs,
    restricted to the given stimulus sub-indices (e.g. ON-only or OFF-only)."""
    idx = np.triu_indices(len(indices), k=1)
    out = []
    for ra in rdms_a:
        sub_a = ra[np.ix_(indices, indices)]
        for rb in rdms_b:
            sub_b = rb[np.ix_(indices, indices)]
            r, _ = spearmanr(sub_a[idx], sub_b[idx])
            out.append(r)
    return np.array(out)


def within_group_r(rdms, indices):
    idx = np.triu_indices(len(indices), k=1)
    out = []
    for i in range(len(rdms)):
        for j in range(i + 1, len(rdms)):
            sub_i = rdms[i][np.ix_(indices, indices)]
            sub_j = rdms[j][np.ix_(indices, indices)]
            r, _ = spearmanr(sub_i[idx], sub_j[idx])
            out.append(r)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "cc_rdms_individual" not in d.files:
        raise KeyError(
            f"{args.npz} does not contain individual RDMs (cc_rdms_individual) -- "
            f"this is an older .npz from before that was saved. Rerun with the "
            f"updated test_item1_all_null_schemes.py to produce a compatible file."
        )

    cc_rdms = list(d["cc_rdms_individual"])
    n_stim = cc_rdms[0].shape[0]
    if n_stim != 24:
        raise ValueError(f"Expected 24 stimuli (ON+OFF) for within-polarity "
                          f"decomposition, got {n_stim} -- this only makes sense "
                          f"for the on_off polarity condition.")

    # Same interleaving as plotting_utils.py's stim_labels: even=OFF, odd=ON
    on_idx = np.arange(1, 24, 2)
    off_idx = np.arange(0, 24, 2)
    full_idx_pairs = np.arange(24)

    schemes = sorted(set(k[:-len("_null_rdms_individual")] for k in d.files
                          if k.endswith("_null_rdms_individual")))
    print(f"Found schemes: {schemes}\n")

    for scheme in schemes:
        null_rdms = list(d[f"{scheme}_null_rdms_individual"])
        print(f"{'='*70}\n{scheme}\n{'='*70}")

        for label, indices in [("Full ON+OFF (24 stim)", full_idx_pairs),
                                ("ON-ON only (12 stim, within-polarity)", on_idx),
                                ("OFF-OFF only (12 stim, within-polarity)", off_idx)]:
            within_cc = within_group_r(cc_rdms, indices)
            cc_vs_null = pairwise_r(cc_rdms, null_rdms, indices)
            u, p = mannwhitneyu(within_cc, cc_vs_null, alternative="greater")
            print(f"  {label}:")
            print(f"    within-CC mean r = {within_cc.mean():.3f} +/- {within_cc.std():.3f} "
                  f"(n={len(within_cc)})")
            print(f"    CC-vs-{scheme} mean r = {cc_vs_null.mean():.3f} +/- "
                  f"{cc_vs_null.std():.3f} (n={len(cc_vs_null)})")
            print(f"    Mann-Whitney p = {p:.2e}")
            print()


if __name__ == "__main__":
    main()
