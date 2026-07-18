#!/usr/bin/env python
"""
pool_trained_vs_untrained.py -- pools the two Experiment 5 null schemes
(degree_preserving_swap + erdos_renyi, n=10 each) into a single n=20
trained-random group and tests it against the untrained weight-shuffled-random
population (n=50), for real statistical power on the trained-vs-untrained
question that compare_all_populations_henning.py's per-scheme breakdown could
only suggest at n=10.

Also reports the maximum-power version: all task-trained populations (CC +
both Exp5 schemes, n=70) vs. the untrained population (n=50).

Requires: compare_all_populations_results.json (output of
compare_all_populations_henning.py) in the same directory.

USAGE:
    python pool_trained_vs_untrained.py
"""

import json

import numpy as np
from scipy.stats import mannwhitneyu, binomtest, ttest_ind


def main():
    with open("compare_all_populations_results.json") as f:
        d = json.load(f)

    pop_degree = d["Exp5 degree_preserving_swap (trained-random)"]
    pop_er = d["Exp5 erdos_renyi (trained-random)"]
    pop_shuffled = d["weight-shuffled random (stability-constrained)"]
    pop_cc = d["CC (pretrained, connectome-constrained)"]

    print("=" * 78)
    print("POOLED TRAINED-RANDOM (degree_preserving_swap + erdos_renyi, n=20)")
    print("vs. WEIGHT-SHUFFLED RANDOM (untrained, n=50)")
    print("=" * 78)

    results = {}
    for ref in ["von_mises", "raw"]:
        degree_r = np.array(pop_degree[ref]["per_network_r"])
        er_r = np.array(pop_er[ref]["per_network_r"])
        pooled_trained_random = np.concatenate([degree_r, er_r])
        shuffled_r = np.array(pop_shuffled[ref]["per_network_r"])
        cc_r = np.array(pop_cc[ref]["per_network_r"])

        print(f"\n--- Reference: {ref} ---")
        print(f"Pooled trained-random (n={len(pooled_trained_random)}): "
              f"mean={pooled_trained_random.mean():+.4f}, "
              f"{100 * np.mean(pooled_trained_random < 0):.0f}% negative")
        print(f"Weight-shuffled random, untrained (n={len(shuffled_r)}): "
              f"mean={shuffled_r.mean():+.4f}, "
              f"{100 * np.mean(shuffled_r < 0):.0f}% negative")

        n_neg = int(np.sum(pooled_trained_random < 0))
        n_tot = len(pooled_trained_random)
        binom_p = binomtest(n_neg, n_tot, p=0.5, alternative="two-sided").pvalue
        print(f"Pooled trained-random sign test: {n_neg}/{n_tot} negative, "
              f"binomial p={binom_p:.4f}")

        u, p_mw = mannwhitneyu(pooled_trained_random, shuffled_r, alternative="two-sided")
        print(f"Mann-Whitney (pooled trained-random vs untrained): "
              f"U={u:.1f}, p={p_mw:.4f}")

        t, p_t = ttest_ind(pooled_trained_random, shuffled_r)
        print(f"t-test (pooled trained-random vs untrained): t={t:.3f}, p={p_t:.4f}")

        all_trained = np.concatenate([cc_r, degree_r, er_r])
        n_neg_all = int(np.sum(all_trained < 0))
        n_tot_all = len(all_trained)
        binom_p_all = binomtest(n_neg_all, n_tot_all, p=0.5, alternative="two-sided").pvalue
        u2, p_mw2 = mannwhitneyu(all_trained, shuffled_r, alternative="two-sided")
        print(f"ALL trained (CC+degree+ER, n={n_tot_all}): {n_neg_all}/{n_tot_all} "
              f"negative, binomial p={binom_p_all:.6f}")
        print(f"Mann-Whitney (ALL trained vs untrained): U={u2:.1f}, p={p_mw2:.6f}")

        results[ref] = dict(
            pooled_n=n_tot, pooled_mean=float(pooled_trained_random.mean()),
            pooled_pct_negative=float(100 * n_neg / n_tot),
            pooled_binom_p=float(binom_p),
            pooled_vs_untrained_mw_p=float(p_mw),
            pooled_vs_untrained_ttest_p=float(p_t),
            all_trained_n=n_tot_all, all_trained_pct_negative=float(100 * n_neg_all / n_tot_all),
            all_trained_binom_p=float(binom_p_all),
            all_trained_vs_untrained_mw_p=float(p_mw2),
        )

    with open("pool_trained_vs_untrained_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote pool_trained_vs_untrained_results.json")


if __name__ == "__main__":
    main()
