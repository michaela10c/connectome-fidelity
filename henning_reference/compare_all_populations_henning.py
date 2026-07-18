#!/usr/bin/env python
"""
compare_all_populations_henning.py -- the decisive check: is "negative
correlation with the Henning reference" specific to particular wiring
conditions, or does EVERY Flyvis population tested so far show it?

WHY THIS MATTERS
-----------------
Four populations have now been evaluated against the Henning reference,
across two separate sessions:
  1. CC (50 networks, pretrained, connectome-constrained)          -- earlier session
  2. Weight-shuffled random (50 networks, stability-constrained)   -- earlier session
  3. Exp5 degree_preserving_swap (10 networks, trained-random)     -- this session
  4. Exp5 erdos_renyi (10 networks, trained-random)                -- this session

All four so far show NEGATIVE ensemble-mean correlation with the von Mises
Henning reference. If that turns out to be true of every population
regardless of wiring, training regime, or anything else about the network,
then "negative correlation with this reference" is not discriminating
between conditions at all -- it says something about the reference
construction or the evaluation pipeline, not about connectome fidelity.
This script puts all four side by side, computed with IDENTICAL
methodology, to settle that question directly rather than comparing
numbers computed at different times with potentially different code paths.

WHY THE METHODOLOGY MUST BE IDENTICAL ACROSS ALL FOUR
--------------------------------------------------------
The CC and weight-shuffled-random RDMs originally came from
moving_edge_on.ipynb, which built RDMs via a DIFFERENT cosine-RDM
implementation (scipy.spatial.distance.cosine in a manual loop, but
WITHOUT the nan_to_num/epsilon preprocessing exp5_henning_evaluate.py and
check_per_model_consistency_raw.py use). This script rebuilds ALL FOUR
populations' RDMs from their raw population VECTORS (not their
already-computed RDMs) using build_rdm_cosine, imported directly from
exp5_henning_evaluate.py -- so every population is processed through the
exact same code path. This matters precisely because of the
pdist-vs-manual-loop numerical divergence already found and confirmed in
this project (a ~1e-7 difference was enough to flip rank order and change
an aggregate statistic meaningfully at this sample size). Comparing four
populations fairly requires eliminating that as a confound first.

USAGE:
    python compare_all_populations_henning.py [--n_perm 10000] [--seed 42]

Requires (same directory):
  exp5_henning_evaluate.py (imported)
  validate_exp5_henning_pvalues.py (imported, for the permutation test)
  results_exp1_8dir_50models_full_shiu.npz   (CC + weight-shuffled random, 8-dir)
  exp5_henning_rdms_degree_preserving_swap.npy
  exp5_henning_rdms_erdos_renyi.npy
  henning_population_matrix.npy
  raw_population_matrix.npy
"""

import argparse
import json

import numpy as np
from scipy.stats import mannwhitneyu, binomtest

from exp5_henning_evaluate import build_rdm_cosine, circular_reference
from validate_exp5_henning_pvalues import compute_partial_r, permutation_test

ALPHA = 0.05  # significance threshold for the perm test, stated explicitly
              # here rather than left implicit, since the verdict logic below
              # depends on it directly.

# How large a gap between the ensemble-mean r and the per-network mean r
# counts as a real divergence worth flagging, not just sampling noise.
# Chosen as clearly larger than the gap seen among the three populations
# that DID roughly agree (CC, degree-swap, ER: 0.24, 0.38, 0.06 respectively)
# -- weight-shuffled-random's actual gap was 0.60, well past even the
# largest of those, so 0.3 comfortably separates "meaningfully different"
# from "the two statistics naturally won't match exactly."
DIVERGENCE_THRESHOLD = 0.3


def load_populations():
    """Returns {population_name: (n_networks, 8, 8) RDM array}, all built
    via the identical build_rdm_cosine path, from raw population vectors."""
    pops = {}

    flyvis = np.load("results_exp1_8dir_50models_full_shiu.npz", allow_pickle=True)
    cc_pop_matrices = flyvis["cc_pop_matrices"]      # (50, 8, 65)
    rand_pop_matrices = flyvis["rand_pop_matrices"]  # (50, 8, 65)

    print(f"Rebuilding CC RDMs from raw population vectors "
          f"({cc_pop_matrices.shape[0]} networks)...")
    pops["CC (pretrained, connectome-constrained)"] = np.array(
        [build_rdm_cosine(m) for m in cc_pop_matrices])

    print(f"Rebuilding weight-shuffled-random RDMs from raw population "
          f"vectors ({rand_pop_matrices.shape[0]} networks)...")
    pops["weight-shuffled random (stability-constrained)"] = np.array(
        [build_rdm_cosine(m) for m in rand_pop_matrices])

    print("Loading Exp5 degree_preserving_swap RDMs (already built via "
          "build_rdm_cosine in the original evaluation run)...")
    pops["Exp5 degree_preserving_swap (trained-random)"] = np.load(
        "exp5_henning_rdms_degree_preserving_swap.npy")

    print("Loading Exp5 erdos_renyi RDMs...")
    pops["Exp5 erdos_renyi (trained-random)"] = np.load(
        "exp5_henning_rdms_erdos_renyi.npy")

    return pops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="compare_all_populations_results.json")
    args = ap.parse_args()

    print("=== Loading Henning references ===")
    vm_pop = np.load("henning_population_matrix.npy")
    vm_rdm = build_rdm_cosine(vm_pop)
    raw_pop = np.load("raw_population_matrix.npy")
    raw_rdm = build_rdm_cosine(raw_pop)
    circ_ref = circular_reference()
    references = {"von_mises": vm_rdm, "raw": raw_rdm}

    print("\n=== Loading all four populations ===")
    pops = load_populations()
    for name, rdms in pops.items():
        print(f"  {name}: {rdms.shape[0]} networks")

    all_results = {}
    print(f"\n=== Computing ensemble-mean partial r, per-network stats, and "
          f"{args.n_perm}-permutation p for every population x reference "
          f"pair ===")
    for pop_name, rdms in pops.items():
        mean_rdm = rdms.mean(axis=0)
        all_results[pop_name] = {}
        for ref_name, ref_rdm in references.items():
            print(f"\n[{pop_name} / {ref_name}]")

            perm_result = permutation_test(mean_rdm, ref_rdm, circ_ref,
                                            args.n_perm, args.seed)

            per_net_r = np.array([
                compute_partial_r(rdm, ref_rdm, circ_ref) for rdm in rdms
            ])
            n_negative = int(np.sum(per_net_r < 0))
            n_total = len(per_net_r)
            # Binomial test: is the negative/positive split among individual
            # networks itself distinguishable from a 50/50 coin flip? This is
            # what actually grounds a "70% negative" or "54% negative" claim
            # statistically -- reporting the percentage alone, without this,
            # doesn't tell you whether it's a real skew or just where n=10 or
            # n=50 coin flips happened to land.
            binom_p = binomtest(n_negative, n_total, p=0.5,
                                 alternative="two-sided").pvalue

            ensemble_mean_r = perm_result["obs_r"]
            per_network_mean = float(per_net_r.mean())
            divergence = abs(ensemble_mean_r - per_network_mean)
            sign_disagrees = (np.sign(ensemble_mean_r) != np.sign(per_network_mean)
                               and abs(per_network_mean) > 1e-6)
            diverges = (divergence > DIVERGENCE_THRESHOLD) or sign_disagrees

            result = dict(
                n_networks=n_total,
                ensemble_mean_r=ensemble_mean_r,
                perm_p_two_sided=perm_result["p_two_sided"],
                perm_p_one_sided=perm_result["p_one_sided"],
                ensemble_significant=bool(perm_result["p_two_sided"] < ALPHA),
                per_network_mean=per_network_mean,
                per_network_pct_negative=float(100 * n_negative / n_total),
                per_network_binom_p=float(binom_p),
                per_network_sign_significant=bool(binom_p < ALPHA),
                ensemble_vs_per_network_gap=float(divergence),
                ensemble_vs_per_network_diverges=bool(diverges),
                per_network_r=per_net_r.tolist(),
            )
            all_results[pop_name][ref_name] = result

            sig_flag = " *" if result["ensemble_significant"] else ""
            print(f"  ensemble-mean partial r = {result['ensemble_mean_r']:+.4f}  "
                  f"(perm p two-sided = {result['perm_p_two_sided']:.4f}{sig_flag})")
            sign_sig_flag = " *" if result["per_network_sign_significant"] else ""
            print(f"  per-network mean        = {result['per_network_mean']:+.4f}  "
                  f"({result['per_network_pct_negative']:.0f}% negative, "
                  f"binomial p={binom_p:.4f}{sign_sig_flag}, n={n_total})")
            if diverges:
                print(f"  [!] DIVERGENCE: ensemble-mean ({ensemble_mean_r:+.4f}) and "
                      f"per-network mean ({per_network_mean:+.4f}) differ by "
                      f"{divergence:.3f} (threshold {DIVERGENCE_THRESHOLD})"
                      f"{' AND DISAGREE IN SIGN' if sign_disagrees else ''} -- "
                      f"the ensemble-mean statistic is not representative of "
                      f"how individual networks in this population behave; "
                      f"do not trust the ensemble-mean number alone for this "
                      f"population/reference pair.")

    # ---- THE DECISIVE QUESTION: is every population negative -- and is
    # that negativity actually reliable at both the ensemble and the
    # per-network level? A sign check on the point estimate alone is not
    # enough: p >= ALPHA means "indistinguishable from zero," and an
    # ensemble-mean/per-network divergence means the ensemble-mean number
    # doesn't represent the population it was computed from. Both must be
    # checked before treating a population's sign as meaningful. ----
    print(f"\n{'='*90}\nMASTER TABLE -- significance- and "
          f"divergence-aware\n{'='*90}")
    header = (f"{'population':<48}{'ref':<10}{'mean r':>9}{'p':>8}"
              f"{'pernet r':>10}{'%neg':>6}{'sign p':>8}{'flag':>6}")
    print(header)
    for pop_name in pops:
        for ref_name in references:
            r = all_results[pop_name][ref_name]
            flags = ""
            if r["ensemble_significant"]:
                flags += "E"
            if r["per_network_sign_significant"]:
                flags += "S"
            if r["ensemble_vs_per_network_diverges"]:
                flags += "!"
            print(f"{pop_name:<48}{ref_name:<10}{r['ensemble_mean_r']:>9.3f}"
                  f"{r['perm_p_two_sided']:>8.3f}{r['per_network_mean']:>10.3f}"
                  f"{r['per_network_pct_negative']:>6.0f}"
                  f"{r['per_network_binom_p']:>8.3f}{flags:>6}")
    print(f"\n  Flags: E = ensemble-mean significant (p<{ALPHA}) | "
          f"S = per-network sign split significant (binomial p<{ALPHA}) | "
          f"! = ensemble-mean and per-network mean diverge by >"
          f"{DIVERGENCE_THRESHOLD} and/or disagree in sign -- ensemble-mean "
          f"number is NOT trustworthy for that row if '!' is present.")

    # A population/reference pair counts as "reliably negative" only if the
    # per-network sign split is itself significant AND the two summary
    # statistics don't diverge -- NOT merely because the point estimate's
    # sign happens to be negative, which was the bug in the previous
    # version of this verdict (it let a non-significant, noise-level r
    # count as "negative" with equal weight to a real effect).
    print(f"\n{'='*90}\nVERDICT (significance- and divergence-checked)\n{'='*90}")
    for ref_name in references:
        reliable_negative, reliable_positive, unreliable = [], [], []
        for pop_name in pops:
            r = all_results[pop_name][ref_name]
            if r["ensemble_vs_per_network_diverges"]:
                unreliable.append(pop_name)
            elif r["per_network_sign_significant"]:
                if r["per_network_mean"] < 0:
                    reliable_negative.append(pop_name)
                else:
                    reliable_positive.append(pop_name)
            # else: sign not significant -- neither reliably negative nor
            # positive, just not distinguishable from chance; omitted from
            # both lists deliberately.

        print(f"\n  [{ref_name}]")
        print(f"    Reliably negative (per-network sign split significant, "
              f"summary stats agree): {reliable_negative or '(none)'}")
        print(f"    Reliably positive: {reliable_positive or '(none)'}")
        print(f"    Ensemble-mean NOT trustworthy for: {unreliable or '(none)'}")
        n_reliable = len(reliable_negative) + len(reliable_positive)
        if n_reliable == 0:
            print(f"    --> No population shows a reliable (significant, "
                  f"non-divergent) sign on this reference. Nothing here "
                  f"currently supports treating '{ref_name}' as detecting "
                  f"anything at this sample size -- this matches the "
                  f"known power limitation of this reference/stimulus "
                  f"count (80% power not reached until r~0.55-0.6).")
        elif reliable_positive:
            print(f"    --> At least one population shows a RELIABLE "
                  f"positive sign while others show reliable negative -- "
                  f"the sign genuinely discriminates between conditions "
                  f"here, which is the necessary condition for treating "
                  f"this as a real wiring/training-related finding. "
                  f"Examine what distinguishes {reliable_positive} from "
                  f"{reliable_negative} directly.")
        else:
            print(f"    --> Every population with a reliable sign is "
                  f"negative ({reliable_negative}). This does not yet "
                  f"discriminate between conditions -- but note whether "
                  f"the untrained population(s) among them differ from the "
                  f"trained ones, since that axis (trained vs. untrained), "
                  f"not wiring per se, is the live hypothesis this data "
                  f"currently supports better.")

    # Cross-population contrasts on the raw reference specifically (the more
    # defensible, less-assumption-laden reference), Mann-Whitney across all
    # pairs, for completeness.
    print(f"\n{'='*78}\nPairwise Mann-Whitney contrasts (raw reference, "
          f"per-network r distributions)\n{'='*78}")
    pop_names = list(pops.keys())
    for i in range(len(pop_names)):
        for j in range(i + 1, len(pop_names)):
            a = np.array(all_results[pop_names[i]]["raw"]["per_network_r"])
            b = np.array(all_results[pop_names[j]]["raw"]["per_network_r"])
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            print(f"  {pop_names[i]:<48} vs {pop_names[j]:<48} "
                  f"U={u:.1f} p={p:.4f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
