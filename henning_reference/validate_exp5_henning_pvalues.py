#!/usr/bin/env python
"""
validate_exp5_henning_pvalues.py -- replaces the analytic (asymptotic)
scipy.stats.spearmanr p-values that exp5_henning_evaluate.py reported with
proper stimulus-label permutation p-values, following the exact methodology
already established and validated elsewhere in this project (Nili et al.
2014 stimulus-label randomization test; the same technique used in
production.py's permutation_test_rdm, correct_exp5_circularity.py, and
exp5_link_and_evaluate.py's permutation_test_partial).

WHY THIS IS NECESSARY, NOT OPTIONAL
------------------------------------
exp5_henning_evaluate.py computed each ensemble-mean partial correlation's
p-value via scipy.stats.spearmanr's built-in p, which assumes a large-sample
asymptotic null distribution. Every RDM comparison in this project has
n=8 conditions -> 28 upper-triangle pairs -- far too small for that
asymptotic assumption to be trustworthy, and this project has already
been burned once by trusting an analytic p-value that a permutation test
later showed was wrong (n=403,651 RDM pairs elsewhere in the project gave
absurd analytic p-values in the other direction; small n here risks the
opposite failure mode -- an analytic p that's too optimistic). This script
does not change any observed r value; it only replaces how the p-value
attached to each r is computed.

METHODOLOGY -- must match exp5_henning_evaluate.py's statistic EXACTLY
------------------------------------------------------------------------
exp5_henning_evaluate.py's reported r is: rank-residualize the model RDM
against the circular-distance reference, separately rank-residualize the
biological RDM against the same circular reference, then Spearman-correlate
the two residual vectors. (This is subtly different from
exp5_link_and_evaluate.py's partial_spearman_rdm, which computes Pearson
directly on the rank-residuals without a second ranking step -- the two are
not numerically identical. This script reuses exp5_henning_evaluate.py's
own rank_residualize function by IMPORT, not reimplementation, specifically
so the permutation null is built under the identical statistic that
produced the r-values already reported -- a permutation test is only valid
if the null distribution is generated under the same statistic as the
observed value.)

The null: permute the stimulus-direction labels of the biological
reference AND the circular-distance control jointly (same permutation
index for both, since circular distance is itself a function of stimulus
position), recompute the identical residualize-then-correlate statistic
against the FIXED (unpermuted) model RDM, repeat 10,000 times. This is the
standard stimulus-label randomization test (Nili et al. 2014) already used
throughout this project.

Reports BOTH one-sided (in the direction actually observed, since two of
the four observed correlations are negative -- not the direction a naive
fidelity hypothesis would have predicted in advance) and two-sided p-values,
since this negative-correlation result was not anticipated going in and a
two-sided test is the more defensible default for an unpredicted-direction
finding; the one-sided value is reported alongside for direct comparison
against the original analytic p-values.

USAGE:
    python validate_exp5_henning_pvalues.py [--n_perm 10000] [--seed 42]

Requires (same directory): exp5_henning_evaluate.py (imported, not copied),
exp5_henning_rdms_degree_preserving_swap.npy, exp5_henning_rdms_erdos_renyi.npy,
henning_population_matrix.npy, raw_population_matrix.npy
"""

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from exp5_henning_evaluate import (
    build_rdm_cosine,
    rank_residualize,
    circular_reference,
    N_DIRECTIONS,
)

SCHEMES = ["degree_preserving_swap", "erdos_renyi"]


def compute_partial_r(model_rdm, ref_rdm, control_rdm):
    """Exact statistic exp5_henning_evaluate.py's summarize_scheme computes:
    rank-residualize model and reference separately against control, then
    Spearman-correlate the residuals. Recomputes model's own residualization
    too, since under permutation the control itself changes."""
    model_resid = rank_residualize(model_rdm, control_rdm)
    ref_resid = rank_residualize(ref_rdm, control_rdm)
    r, _ = spearmanr(model_resid, ref_resid)
    return r


def permutation_test(model_rdm, ref_rdm, control_rdm, n_permutations, seed):
    """Stimulus-label permutation test (Nili et al. 2014), reusing the
    identical joint-permutation approach validated in
    exp5_link_and_evaluate.py's permutation_test_partial: reference and
    control are permuted by the SAME index each draw, since circular
    distance is a function of stimulus position and must stay consistent
    with whichever positions the reference values are (under the null)
    relabeled to. Model RDM stays fixed -- only the reference's assumed
    correspondence to stimulus position is randomized."""
    rng = np.random.default_rng(seed)
    n = ref_rdm.shape[0]
    obs = compute_partial_r(model_rdm, ref_rdm, control_rdm)

    null_r = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        ref_p = ref_rdm[np.ix_(perm, perm)]
        ctl_p = control_rdm[np.ix_(perm, perm)]
        null_r[i] = compute_partial_r(model_rdm, ref_p, ctl_p)

    # Two-sided: appropriate default since the negative direction found
    # here was not predicted in advance.
    p_two_sided = (np.sum(np.abs(null_r) >= abs(obs)) + 1) / (n_permutations + 1)
    # One-sided, in the direction actually observed -- for direct
    # comparison against the original (also one-directional in effect)
    # analytic scipy p-values.
    if obs < 0:
        p_one_sided = (np.sum(null_r <= obs) + 1) / (n_permutations + 1)
    else:
        p_one_sided = (np.sum(null_r >= obs) + 1) / (n_permutations + 1)

    return dict(
        obs_r=float(obs),
        p_two_sided=float(p_two_sided),
        p_one_sided=float(p_one_sided),
        null_mean=float(null_r.mean()),
        null_sd=float(null_r.std()),
        null_r=null_r.tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="exp5_henning_permutation_results.json")
    args = ap.parse_args()

    print("=== Loading Henning references (same construction as "
          "exp5_henning_evaluate.py) ===")
    vm_pop = np.load("henning_population_matrix.npy")
    vm_rdm = build_rdm_cosine(vm_pop)
    raw_pop = np.load("raw_population_matrix.npy")
    raw_rdm = build_rdm_cosine(raw_pop)
    circ_ref = circular_reference()

    references = {"von_mises": vm_rdm, "raw": raw_rdm}

    print("\n=== Loading per-network RDMs from the Exp5 evaluation run ===")
    scheme_rdms = {}
    for scheme in SCHEMES:
        path = f"exp5_henning_rdms_{scheme}.npy"
        rdms = np.load(path)
        print(f"  {scheme}: {path} -> shape {rdms.shape}")
        scheme_rdms[scheme] = rdms

    # Load the original analytic p-values for direct side-by-side comparison.
    try:
        with open("exp5_henning_results.json") as f:
            original = json.load(f)
    except FileNotFoundError:
        print("[warn] exp5_henning_results.json not found -- proceeding "
              "without the original-vs-permutation comparison table.")
        original = {}

    all_results = {}
    print(f"\n=== Running {args.n_perm}-permutation test per "
          f"scheme x reference (this may take a few seconds per cell) ===")
    for scheme in SCHEMES:
        mean_rdm = scheme_rdms[scheme].mean(axis=0)
        all_results[scheme] = {}
        for ref_name, ref_rdm in references.items():
            print(f"\n[{scheme} / {ref_name}]")
            result = permutation_test(mean_rdm, ref_rdm, circ_ref,
                                       args.n_perm, args.seed)
            all_results[scheme][ref_name] = result

            orig_p = None
            try:
                orig_p = original[scheme][ref_name]["ensemble_mean_p"]
            except (KeyError, TypeError):
                pass

            print(f"  observed partial r        = {result['obs_r']:+.4f}")
            print(f"  null distribution          = {result['null_mean']:+.4f} "
                  f"+/- {result['null_sd']:.4f}")
            print(f"  permutation p (two-sided)  = {result['p_two_sided']:.4f}")
            print(f"  permutation p (one-sided,  "
                  f"direction observed) = {result['p_one_sided']:.4f}")
            if orig_p is not None:
                print(f"  ORIGINAL analytic p (scipy spearmanr) = {orig_p:.4f}")
                ratio = (result['p_two_sided'] / orig_p) if orig_p > 0 else float("inf")
                if ratio > 3 or ratio < 1 / 3:
                    print(f"  [!] permutation p differs from analytic p by "
                          f"{ratio:.1f}x -- the analytic p-value was NOT "
                          f"trustworthy at this sample size; use the "
                          f"permutation p going forward.")
                else:
                    print(f"  permutation p roughly agrees with the analytic "
                          f"p (within 3x) -- less cause for concern here, "
                          f"though the permutation p remains the more "
                          f"defensible number to report.")

    # Strip the large null_r arrays before printing the final summary table,
    # keep them in the saved JSON for anyone who wants to inspect the null
    # distributions directly (e.g. to check for non-normality).
    print(f"\n{'='*72}\nSUMMARY -- permutation p-values to actually report\n{'='*72}")
    print(f"{'scheme':<26}{'reference':<12}{'obs r':>10}{'p (2-sided)':>14}{'p (1-sided)':>14}")
    for scheme in SCHEMES:
        for ref_name in references:
            r = all_results[scheme][ref_name]
            print(f"{scheme:<26}{ref_name:<12}{r['obs_r']:>10.4f}"
                  f"{r['p_two_sided']:>14.4f}{r['p_one_sided']:>14.4f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
