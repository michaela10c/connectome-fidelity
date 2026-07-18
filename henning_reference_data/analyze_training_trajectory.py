#!/usr/bin/env python
"""
analyze_training_trajectory.py -- the correct statistical treatment of
training_trajectory_henning.py's output.

WHY THE NAIVE POOLED CORRELATION IS WRONG
--------------------------------------------
training_trajectory_henning.py's own printed summary (Spearman across all
network-checkpoint pairs pooled together) treats every checkpoint evaluation as
an independent observation. It isn't: checkpoints from the same network are
repeated measures on one underlying trajectory, and different networks sit at
different baseline levels (confirmed directly in the first n=4-network run --
e.g. erdos_renyi/0000 sits around -0.4 to -0.7 throughout, while
degree_preserving_swap/0001 sits around 0 to -0.3). Pooling mixes
between-network offset variance into what should be a within-network trend
test, diluting a signal that is actually more consistent than the pooled
number suggests.

THE CORRECT UNIT OF REPLICATION IS THE NETWORK, NOT THE CHECKPOINT
----------------------------------------------------------------------
This script:
  1. Computes each network's OWN Spearman(iteration, fidelity_r) trend,
     using only that network's own checkpoints (proper within-subject test).
  2. Combines across networks two ways, both appropriate for a small number
     of independent units: (a) a sign test on the per-network trend
     directions -- does the trend go the same way more often than chance?
     (b) Fisher's method combining the per-network p-values into one
     meta-analytic p-value -- more powered than the sign test when the
     individual trends are consistent but each falls short of significance
     alone, which is exactly the situation the first n=4 run showed.
  3. Reports the pre-training (iteration -1) anchor separately, since it is
     an independent cross-check (compare directly against the
     weight-shuffled-random population's per-network mean from
     compare_all_populations_results.json, if available) and should not be
     folded into the trend statistic at all.

This script is written to just work on whatever training_trajectory_results.json
exists -- rerun training_trajectory_henning.py with more --net_indices at any
point and rerun this script on the same filename with no changes needed.

USAGE:
    python analyze_training_trajectory.py [--results training_trajectory_results.json]
"""

import argparse
import json
from collections import defaultdict
from itertools import permutations

import numpy as np
from scipy.stats import spearmanr, binomtest, combine_pvalues, binom


def exact_spearman_pvalue(iters, rs, max_exact_n=8, n_monte_carlo=20000, seed=42):
    """Exact (or, above max_exact_n, Monte Carlo) permutation p-value for a
    Spearman correlation -- NOT scipy.stats.spearmanr's default p-value.

    scipy's own docstring states its default p-value 'is only accurate for
    very large samples (>500 observations); for smaller sample sizes,
    consider a permutation test.' Confirmed catastrophically wrong at the
    sample sizes used here (5-6 checkpoints per network): a PERFECT
    correlation at n=4 displays as scipy's p=0.000000, while the true exact
    one-sided p-value (only 1 of 4!=24 orderings gives perfect correlation)
    is 0.0417 -- not a rounding difference, a qualitatively different
    conclusion. This is what produced the earlier run's absurd Fisher's
    combined statistic (chi2=2792 against a chi2(28) null): several
    near-perfect per-network trends were registering as p~0 instead of
    their true ~2-8% value, and Fisher's method has no way to know the
    inputs it was given were wrong.

    At n<=8, enumerates ALL n! orderings exactly (<=40320, fast, no need
    for Monte Carlo at all at these small n). Above that, falls back to a
    large Monte Carlo permutation test, matching the project's established
    discipline of never trusting an untested small-sample p-value."""
    n = len(iters)
    obs_rho, _ = spearmanr(iters, rs)
    if np.isnan(obs_rho):
        return obs_rho, 1.0  # degenerate (constant) trajectory -- no
                              # correlation is defined; report p=1, not a
                              # spuriously small value from either method

    if n <= max_exact_n:
        rs = np.asarray(rs)
        count_extreme = 0
        total = 0
        for perm in permutations(range(n)):
            perm_rho, _ = spearmanr(iters, rs[list(perm)])
            if not np.isnan(perm_rho) and abs(perm_rho) >= abs(obs_rho) - 1e-9:
                count_extreme += 1
            total += 1
        p = count_extreme / total
    else:
        rng = np.random.default_rng(seed)
        rs = np.asarray(rs)
        count_extreme = 0
        for _ in range(n_monte_carlo):
            perm_rs = rng.permutation(rs)
            perm_rho, _ = spearmanr(iters, perm_rs)
            if not np.isnan(perm_rho) and abs(perm_rho) >= abs(obs_rho) - 1e-9:
                count_extreme += 1
        p = (count_extreme + 1) / (n_monte_carlo + 1)  # +1 smoothing, standard
                                                          # for Monte Carlo
                                                          # permutation p-values
    return obs_rho, p


def per_network_trend(traj, ref_key):
    """Spearman(iteration, r) within one network's own trajectory, using only
    the actually-trained (iteration >= 0) AND resolvable checkpoints, with
    an EXACT permutation p-value (see exact_spearman_pvalue) -- NOT
    scipy.stats.spearmanr's default, which is unreliable at this sample
    size. The 'resolvable' field (added after training_trajectory_henning.py's
    precision guard fix) marks checkpoints whose RDM span didn't clear the
    float32 round-off floor by the established 10x margin -- their r values
    are None, not real measurements, and must never be averaged or
    correlated as if they were. The pre-training anchor is a different kind
    of evidence and is excluded from the trend test itself regardless,
    reported separately by the caller."""
    trained = [row for row in traj
               if row["approx_iteration"] >= 0 and row.get("resolvable", True)
               and row.get(ref_key) is not None]
    if len(trained) < 3:
        return None, None, len(trained)
    iters = np.array([row["approx_iteration"] for row in trained])
    rs = np.array([row[ref_key] for row in trained])
    rho, p = exact_spearman_pvalue(iters, rs)
    return rho, p, len(trained)


def sign_test_power_exact(n, p_true, alpha=0.05):
    """Exact power of the two-sided binomial sign test at sample size n,
    given the TRUE probability p_true that any one network's trend is
    negative. Computed by exact enumeration (sum the binomial pmf over
    outcomes that would reach significance), not simulation -- unlike the
    RDM-based tests elsewhere in this project, the per-network sign here is a
    genuinely i.i.d. Bernoulli draw (independent networks, independent
    trainings), so an exact closed-form treatment is appropriate and more
    precise than Monte Carlo would be at no added cost."""
    power = 0.0
    for k in range(n + 1):
        p_val = binomtest(k, n, p=0.5, alternative="two-sided").pvalue
        if p_val < alpha:
            power += binom.pmf(k, n, p_true)
    return power


def required_n_for_sign_test(p_true, target_power=0.80, alpha=0.05, n_max=200):
    for n in range(4, n_max + 1):
        if sign_test_power_exact(n, p_true, alpha) >= target_power:
            return n
    return None  # not reached within n_max


def fisher_power_bootstrap(observed_ps, n_range, n_boot=5000, alpha=0.05, seed=42):
    """Power of Fisher's combined-p test at each n in n_range, estimated by
    bootstrap resampling (with replacement) from the ACTUALLY-OBSERVED
    per-network p-values, not an assumed effect size. This directly answers
    'if future networks' trends look like the ones we've already measured,
    how many would we need' -- the same simulation-based-not-textbook-formula
    discipline used earlier in this project for the Henning-reference power
    analysis, and necessary here too since Fisher's method's power depends on
    the actual p-value distribution shape, not just a single effect size
    number."""
    rng = np.random.default_rng(seed)
    observed_ps = np.clip(np.array(observed_ps), 1e-300, 1.0)
    powers = {}
    for n in n_range:
        sig_count = 0
        for _ in range(n_boot):
            sample = rng.choice(observed_ps, size=n, replace=True)
            _, p = combine_pvalues(sample, method="fisher")
            if p < alpha:
                sig_count += 1
        powers[n] = sig_count / n_boot
    return powers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="training_trajectory_results.json")
    ap.add_argument("--compare_populations",
                     default="compare_all_populations_results.json",
                     help="optional: cross-check the pre-training anchor "
                          "against the weight-shuffled-random population, "
                          "if this file is present")
    args = ap.parse_args()

    with open(args.results) as f:
        all_rows = json.load(f)

    # Group into per-network trajectories.
    trajectories = defaultdict(list)
    for row in all_rows:
        key = (row["scheme"], row["net_idx"])
        trajectories[key].append(row)
    n_networks = len(trajectories)
    print(f"Loaded {len(all_rows)} checkpoint evaluations across "
          f"{n_networks} independent networks.")

    ref_keys = [k for k in all_rows[0] if k.startswith("r_")]

    for ref_key in ref_keys:
        ref_name = ref_key[2:]
        print(f"\n{'=' * 78}\nREFERENCE: {ref_name}\n{'=' * 78}")

        per_network = {}
        for key, traj in trajectories.items():
            rho, p, n_pts = per_network_trend(traj, ref_key)
            per_network[key] = (rho, p, n_pts)
            scheme, net_idx = key
            if rho is None:
                print(f"  {scheme}/{net_idx:04d}: too few trained checkpoints "
                      f"({n_pts}) for a trend test")
            else:
                print(f"  {scheme}/{net_idx:04d}: rho={rho:+.3f}, p={p:.3f} "
                      f"(n={n_pts} checkpoints)")

        valid = {k: v for k, v in per_network.items() if v[0] is not None}
        rhos = np.array([v[0] for v in valid.values()])
        ps = np.array([v[1] for v in valid.values()])

        if len(rhos) == 0:
            print("  No networks had enough checkpoints for a trend test.")
            continue

        # --- Sign test: do the per-network trends agree in direction? ---
        n_neg = int(np.sum(rhos < 0))
        n_tot = len(rhos)
        sign_p = binomtest(n_neg, n_tot, p=0.5, alternative="two-sided").pvalue
        print(f"\n  Per-network trend sign test: {n_neg}/{n_tot} networks show "
              f"a negative (iteration, fidelity) trend, binomial p={sign_p:.4f}")
        if n_neg == n_tot or n_neg == 0:
            print(f"  -> {'ALL' if n_neg == n_tot else 'NONE'} of {n_tot} "
                  f"independent networks trend the same direction -- consistent, "
                  f"but a sign test at n={n_tot} has limited power to reach "
                  f"significance on its own regardless of how consistent it looks.")

        # --- Fisher's method: combine per-network p-values into one
        # meta-analytic p-value. More powered than the sign test when
        # individual trends are consistent but each falls short alone. ---
        # Guard against p=0 (would blow up the log-sum); floor at machine eps.
        ps_safe = np.clip(ps, 1e-300, 1.0)
        try:
            fisher_stat, fisher_p = combine_pvalues(ps_safe, method="fisher")
            print(f"  Fisher's combined p-value across {n_tot} independent "
                  f"networks' trend tests: chi2={fisher_stat:.3f}, p={fisher_p:.4f}")
            print(f"  (Note: Fisher's method combines evidence for 'a trend "
                  f"exists' regardless of direction consistency -- read this "
                  f"alongside the sign test above, not instead of it. If the "
                  f"trends point in different directions, a small Fisher's p "
                  f"does not mean they agree.)")
        except Exception as e:
            print(f"  Fisher's method failed: {e}")

        # --- Power analysis: how many networks would actually be needed? ---
        print(f"\n  --- Power analysis: how many networks to reliably detect "
              f"this? ---")
        print(f"  Sign test, exact power, at several benchmark 'true "
              f"consistency' levels (since n={n_tot} is too small to trust a "
              f"point estimate of the true consistency itself):")
        print(f"    {'true P(negative)':<20}{'n needed for 80% power':>26}")
        for p_true in [0.65, 0.75, 0.85, 0.95]:
            n_req = required_n_for_sign_test(p_true, target_power=0.80)
            n_req_str = str(n_req) if n_req else ">200"
            print(f"    {p_true:<20.2f}{n_req_str:>26}")
        observed_p_true = n_neg / n_tot if n_neg >= n_tot - n_neg else 1 - n_neg / n_tot
        n_req_observed = required_n_for_sign_test(observed_p_true, target_power=0.80)
        print(f"    (observed consistency this run: {max(n_neg, n_tot - n_neg)}/{n_tot} "
              f"= {observed_p_true:.2f} -> {n_req_observed if n_req_observed else '>200'} "
              f"networks needed for 80% power IF this holds exactly at scale -- "
              f"treat as optimistic, given how little data it's estimated from)")

        print(f"\n  Fisher's method, bootstrap power using the ACTUAL observed "
              f"p-value distribution from this run's {n_tot} networks "
              f"(more informative than the sign-test benchmarks above, since "
              f"it uses the real effect sizes rather than assumed ones):")
        n_range = [n_tot, 6, 8, 10, 15, 20, 30]
        n_range = sorted(set(n for n in n_range if n >= n_tot))
        fisher_powers = fisher_power_bootstrap(ps, n_range)
        print(f"    {'n networks':<15}{'power (Fisher)':>18}")
        for n, power in fisher_powers.items():
            print(f"    {n:<15}{power:>17.1%}")
        print(f"  (Bootstrap resamples WITH REPLACEMENT from only {n_tot} "
              f"observed p-values, so this is itself a rough estimate, not a "
              f"precise one -- but it is grounded in this run's actual effect "
              f"sizes rather than a guessed effect size, which is the more "
              f"honest number to plan around.)")

        # --- Pre-training anchor, reported separately, not part of the trend test.
        # Excludes unresolvable checkpoints explicitly -- averaging in a None/NaN
        # here is exactly the failure mode this project already learned to guard
        # against on Experiment 4. ---
        pretrain_all = [row for key, traj in trajectories.items() for row in traj
                         if row["approx_iteration"] == -1]
        pretrain_resolvable = [row for row in pretrain_all
                                if row.get("resolvable", True) and row.get(ref_key) is not None]
        n_excluded = len(pretrain_all) - len(pretrain_resolvable)
        if pretrain_resolvable:
            pretrain_vals = np.array([row[ref_key] for row in pretrain_resolvable])
            excl_note = f" ({n_excluded} unresolvable, excluded)" if n_excluded else ""
            print(f"\n  Pre-training (iteration -1) anchor: mean={pretrain_vals.mean():+.4f} "
                  f"+/- {pretrain_vals.std():.4f} (n={len(pretrain_vals)}{excl_note})")
        elif pretrain_all:
            print(f"\n  Pre-training (iteration -1) anchor: ALL {len(pretrain_all)} "
                  f"pre-training checkpoints were unresolvable for this reference -- "
                  f"no anchor value can be reported, not even an approximate one.")

    # --- Cross-check the pre-training anchor against weight-shuffled-random,
    # if that comparison data is available. ---
    try:
        with open(args.compare_populations) as f:
            pop_data = json.load(f)
        shuffled = pop_data.get("weight-shuffled random (stability-constrained)")
        if shuffled:
            print(f"\n{'=' * 78}\nCROSS-CHECK: pre-training anchor vs. weight-shuffled-random "
                  f"population\n{'=' * 78}")
            print("Two independent representations of 'untrained' -- do they agree?")
            for ref_key in ref_keys:
                ref_name = ref_key[2:]
                if ref_name not in shuffled:
                    continue
                shuffled_mean = shuffled[ref_name]["per_network_mean"]
                pretrain_vals = np.array([
                    row[ref_key] for traj in trajectories.values() for row in traj
                    if row["approx_iteration"] == -1 and row.get("resolvable", True)
                    and row.get(ref_key) is not None
                ])
                if len(pretrain_vals):
                    print(f"  [{ref_name}] pre-training anchor mean = "
                          f"{pretrain_vals.mean():+.4f} (n={len(pretrain_vals)} resolvable)  |  "
                          f"weight-shuffled-random per-network mean = {shuffled_mean:+.4f}  |  "
                          f"gap = {abs(pretrain_vals.mean() - shuffled_mean):.4f}")
    except FileNotFoundError:
        print(f"\n({args.compare_populations} not found -- skipping the "
              f"pre-training-vs-weight-shuffled cross-check. Not required, "
              f"just a nice-to-have if that file is in this directory.)")


if __name__ == "__main__":
    main()
