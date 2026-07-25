"""
Analysis for the instability-analysis experiment (Frankle et al. 2020-style):
does a network's trend direction/magnitude reproduce across replicate retrains
of the SAME wiring with different training noise, or is it just as variable
as trends across DIFFERENT wiring realizations?

Design this tests: 5 "seed" networks (degree_preserving_swap/0002, 0003, 0005,
0009; erdos_renyi/0004), each retrained R>=3 times from identical wiring,
varying only the training-noise seed (minibatch order / optimizer seed --
confirm with the actual training script which knobs exist beyond wiring
itself before running this). Each replicate evaluated the same way as the
original trajectory work, producing its own per-replicate rho.

The test: is between-GROUP variance in rho (degree_preserving_swap/0002 vs
0003 vs 0005 vs 0009 vs erdos_renyi/0004) larger than within-GROUP variance
(replicates of the same wiring)? A one-way-ANOVA-style F-statistic captures
this directly, but rho values aren't guaranteed normal, so significance comes
from a PERMUTATION test (shuffle group labels, recompute F, build a null
distribution empirically) rather than the F-distribution -- same discipline
used throughout this project (Nili et al. 2014-style stimulus-label
permutation, the exact-enumeration fix for small-n Spearman p-values, etc.),
not a new methodological choice.

Interpretation:
  - Large observed F, small permutation p -> wiring realization determines
    direction (hypothesis A): replicates of the same wiring cluster tightly,
    different wiring realizations differ from each other.
  - Small F, large permutation p -> training noise determines direction
    (hypothesis B): replicates of the same wiring are just as scattered as
    replicates of different wiring -- direction has nothing to do with wiring.

Usage:
    python analyze_instability_experiment.py --in instability_results.json

Expected input JSON: a flat list of dicts, one per replicate evaluation, each
with at least {"seed_network": "degree_preserving_swap/0002", "replicate": 0,
"rho_von_mises": ..., "rho_raw": ...} -- matching the per-network rho output
already produced by analyze_training_trajectory.py, just re-keyed by which
seed network + replicate index each retrain belongs to.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def one_way_f_statistic(groups):
    """groups: list of arrays, one per wiring-group, of replicate rho values.
    Standard one-way ANOVA F-statistic -- used here only as a natural summary
    of between- vs within-group spread, NOT to look up a p-value from the
    F-distribution (that requires normality this data isn't assumed to have;
    see the permutation test below for the actual significance)."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    k = len(groups)
    n_total = len(all_vals)

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0 or ss_within == 0:
        return np.inf if ss_between > 0 else 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return ms_between / ms_within


def permutation_test(groups, n_perm=20000, seed=42):
    """Shuffles which replicate belongs to which group, recomputes F each
    time, and reports the fraction of shuffles with F >= observed -- an
    exact-in-spirit, assumption-light significance test, consistent with
    every other p-value in this project being permutation-based rather than
    relying on a parametric null it hasn't earned."""
    rng = np.random.default_rng(seed)
    observed_f = one_way_f_statistic(groups)

    group_sizes = [len(g) for g in groups]
    pooled = np.concatenate(groups)
    n_total = len(pooled)

    count_as_extreme = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        idx = 0
        perm_groups = []
        for size in group_sizes:
            perm_groups.append(shuffled[idx:idx + size])
            idx += size
        perm_f = one_way_f_statistic(perm_groups)
        if perm_f >= observed_f - 1e-12:
            count_as_extreme += 1

    p = (count_as_extreme + 1) / (n_perm + 1)  # +1 smoothing, standard
    return observed_f, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="instability_results.json")
    ap.add_argument("--scheme", default=None,
                     help="Optional: restrict to one scheme only (e.g. "
                          "'degree_preserving_swap' or 'erdos_renyi'), matched "
                          "against the prefix before '/' in seed_network. "
                          "Default: pool all schemes together, as before.")
    args = ap.parse_args()

    with open(args.infile) as f:
        rows = json.load(f)

    if args.scheme is not None:
        rows = [r for r in rows if r["seed_network"].split("/")[0] == args.scheme]
        print(f"Filtered to scheme='{args.scheme}': {len(rows)} rows remain")
        if not rows:
            print(f"No rows matched scheme='{args.scheme}' -- check the exact "
                  f"scheme name against what's actually in {args.infile}.")
            return

    for ref_key in ["rho_von_mises", "rho_raw"]:
        print(f"\n{'=' * 70}\nREFERENCE: {ref_key}\n{'=' * 70}")

        by_group = defaultdict(list)
        for row in rows:
            if row.get(ref_key) is not None:
                by_group[row["seed_network"]].append(row[ref_key])

        groups_named = sorted(by_group.items())
        print(f"Loaded {sum(len(v) for _, v in groups_named)} replicate evaluations "
              f"across {len(groups_named)} seed networks\n")

        for name, vals in groups_named:
            vals = np.array(vals)
            same_sign_as_first = (np.sign(vals) == np.sign(vals[0])).all()
            print(f"  {name}: {len(vals)} replicates, rho = {[f'{v:+.3f}' for v in vals]}, "
                  f"all same sign: {same_sign_as_first}")

        if len(groups_named) < 2:
            print("\nNeed at least 2 seed networks to run the between/within test.")
            continue

        groups = [np.array(vals) for _, vals in groups_named]
        observed_f, p = permutation_test(groups)

        print(f"\nObserved F-statistic (between-group / within-group variance): {observed_f:.3f}")
        print(f"Permutation p-value ({20000} shuffles): {p:.5f}")
        print()
        if p < 0.05:
            print("SIGNIFICANT: replicates of the same wiring cluster together more than "
                  "chance would predict -- evidence for hypothesis A (wiring realization "
                  "determines direction).")
        else:
            print("NOT SIGNIFICANT: replicates of the same wiring are about as scattered as "
                  "replicates of different wiring -- evidence for hypothesis B (training "
                  "noise determines direction, not wiring), or the design is underpowered "
                  "at this replicate count and needs more replicates per network before "
                  "concluding either way.")


if __name__ == "__main__":
    main()
