#!/usr/bin/env python
"""
exp5_link_and_evaluate.py — bridge tonight's stability-sweep checkpoints into
production.py's real evaluation pipeline, and add the partial-correlation
step production.py's aggregate is currently missing.

WHY THIS EXISTS (replaces the earlier exp5_rsa_pilot.py draft)
----------------------------------------------------------------
exp5_rsa_pilot.py (previous draft) reimplemented get_population_vector and
the von Mises biological reference from guesses, because production.py's
actual source wasn't available yet. It now is. production.py already has an
audited, verified evaluation pipeline (get_population_vector, build_rdm,
build_bio_rdm, evaluate_network, is_stable) -- reused verbatim from
Experiment 4. Reimplementing it a second time would just create a second
place for a subtle mismatch to hide. This script does NOT reimplement
evaluation; it (1) makes tonight's checkpoints visible to production.py's own
naming convention, then (2) calls production.py's real functions by import,
adding only the ONE missing piece: the partial correlation controlling for
circular structure, which aggregate_from_disk() does not currently compute.

THE ONLY REAL GAP: DIRECTORY NAMING
------------------------------------
production.py expects:      <results_dir>/flow/exp5_<scheme>/<net_idx:04d>
exp5_stability_sweep.py used: <results_dir>/flow/exp5_<scheme>_short/<seed:04d>

is_training_complete() checks the checkpoint's ACTUAL logged iteration
(chkpt_iter.h5) against --n_iters with a 0.99 tolerance -- it does not care
about directory naming beyond finding the right path. So symlinking
exp5_<scheme>/<seed:04d> -> exp5_<scheme>_short/<seed:04d> is sufficient for
production.py to recognize these as complete 40k-iteration runs and skip
straight to evaluation. No retraining, no code duplication.

net_idx vs seed: tonight's sweep used seed in {1..9} (skipped seed 0, which
already has its own full 250k-iteration run). production.py's own convention
is seed = seeds[net_idx % len(seeds)] -- to keep this simple and avoid ANY
risk of an off-by-one or wraparound mismatch, this script symlinks net_idx
DIRECTLY to seed (net_idx 1..9), not net_idx 0..8, and calls production.py
with --seeds 1-9 to match. Aggregation then just reads whatever rdm_net*.npy
files exist, so the exact net_idx values don't need to start at 0.

THE SECOND GAP: is_stable() vs fade_max() ARE DIFFERENT CHECKS
-----------------------------------------------------------------
exp5_stability_sweep.py's fade_max() computes ONE scalar (max|activity| after
a fade_in_state call) with no full simulate() pass. production.py's
is_stable() runs a full network.simulate() on the actual ON-edge check
stimulus and requires ALL outputs finite and < 1e6. These are related but not
identical checks -- fade_max is a cheaper proxy, is_stable is the real
Exp 1-4 stability criterion. All 18 tonight's networks had fade_max orders of
magnitude below 1e6 (max was 552), so they are extremely likely to also pass
is_stable -- but this script re-runs is_stable() for real via
evaluate_network's call path rather than assuming the fade_max result
transfers, because that assumption is exactly the kind of thing this project
has been burned by before (see: the n=1 divergence claim).

WHAT THIS SCRIPT ADDS ON TOP OF production.py (does not modify that file)
----------------------------------------------------------------------------
production.py's aggregate_from_disk() computes ONLY r_trained_random_vs_bio
(raw Spearman r vs the von Mises reference). Per Experiment 3, this reference
is 97.8% explained by circular structure on the 12-condition ON-only stimulus
set -- raw r here is mostly a circularity score, not a fidelity score. This
script additionally computes the PARTIAL correlation controlling for the
circular reference (same rank-residualization method as
biological_reference.ipynb's partial_spearman_rdm), for both the per-scheme
mean RDM and (if both schemes are present) reports the degree_swap-vs-ER
contrast on the PARTIAL r, which is the number that's actually comparable to
CC's partial r=0.145 (n.s.) and random's partial r=0.061 (n.s.) from
Experiment 3.

USAGE
-----
    # one-time: symlink tonight's checkpoints into production.py's naming
    python exp5_link_and_evaluate.py --link-only

    # then run production.py itself per network (reuses the real pipeline,
    # no retraining -- is_training_complete() will see these as done):
    for scheme in degree_preserving_swap erdos_renyi; do
      for seed in 1 2 3 4 5 6 7 8 9; do
        python production.py --scheme $scheme --only_net $seed \
            --n_networks 10 --seeds 1-9 --n_iters 40000 \
            --connectome_dir ./rand_out --out_dir ./pilot_out_40k/$scheme
      done
    done

    # then let THIS script aggregate with the partial-correlation addition:
    python exp5_link_and_evaluate.py --aggregate \
        --out_root ./pilot_out_40k --n_iters 40000
"""

import os, sys, glob, json, argparse, shutil
import numpy as np
from scipy.stats import spearmanr, rankdata, mannwhitneyu

SCHEMES = ["degree_preserving_swap", "erdos_renyi"]
SEEDS = list(range(1, 10))  # tonight's sweep: seeds 1-9, seed 0 has its own 250k run


# ── directory linking ───────────────────────────────────────────────────────
def link_checkpoints(results_dir):
    """
    Symlink flow/exp5_<scheme>/<seed:04d> -> flow/exp5_<scheme>_short/<seed:04d>
    for every scheme x seed, so production.py's own naming convention finds
    tonight's already-trained checkpoints without any retraining.
    """
    flow_dir = os.path.join(results_dir, "flow")
    made, skipped = [], []
    for scheme in SCHEMES:
        src_root = os.path.join(flow_dir, f"exp5_{scheme}_short")
        dst_root = os.path.join(flow_dir, f"exp5_{scheme}")
        os.makedirs(dst_root, exist_ok=True)
        for seed in SEEDS:
            src = os.path.join(src_root, f"{seed:04d}")
            dst = os.path.join(dst_root, f"{seed:04d}")
            if not os.path.isdir(src):
                print(f"  [skip] source missing: {src}")
                skipped.append(src)
                continue
            if os.path.exists(dst) or os.path.islink(dst):
                print(f"  [skip] destination already exists: {dst}")
                skipped.append(dst)
                continue
            os.symlink(src, dst, target_is_directory=True)
            print(f"  [link] {dst} -> {src}")
            made.append(dst)
    print(f"\nlinked {len(made)}, skipped {len(skipped)}")
    return made, skipped


# ── partial correlation (matches biological_reference.ipynb exactly) ───────
def upper(mat):
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def circular_reference(n=12):
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(d, n - d).astype(float)


def partial_spearman_rdm(rdm_model, rdm_ref, rdm_control):
    """Verbatim from biological_reference.ipynb -- do not reimplement differently."""
    a = rankdata(upper(rdm_model))
    b = rankdata(upper(rdm_ref))
    c = rankdata(upper(rdm_control))
    C = np.column_stack([np.ones_like(c), c])
    res_a = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]
    res_b = b - C @ np.linalg.lstsq(C, b, rcond=None)[0]
    den = np.linalg.norm(res_a) * np.linalg.norm(res_b)
    return np.nan if den == 0 else float(res_a @ res_b / den)


def permutation_test_partial(rdm_model, rdm_ref, rdm_control, n_permutations=10000, seed=42):
    """Verbatim from biological_reference.ipynb."""
    rng = np.random.default_rng(seed)
    n = rdm_model.shape[0]
    obs = partial_spearman_rdm(rdm_model, rdm_ref, rdm_control)
    count = 0
    for _ in range(n_permutations):
        p = rng.permutation(n)
        ref_p = rdm_ref[np.ix_(p, p)]
        ctl_p = rdm_control[np.ix_(p, p)]
        if partial_spearman_rdm(rdm_model, ref_p, ctl_p) >= obs:
            count += 1
    return obs, (count + 1) / (n_permutations + 1)


# ── aggregation with the partial-correlation addition ───────────────────────
def aggregate_with_partial(out_root, n_iters, n_perm=10000):
    """
    For each scheme's out_dir, read the rdm_net*.npy / unstable_net*.flag files
    production.py already wrote (via the same convention aggregate_from_disk()
    uses), and additionally compute the partial correlation vs the biological
    reference controlling for circular structure -- the number production.py's
    own aggregate does not currently report.
    """
    # Import production.py's build_bio_rdm so the reference is BYTE-IDENTICAL
    # to what production.py itself uses -- do not reconstruct it separately.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from production import build_bio_rdm  # noqa: E402

    bio_ref = build_bio_rdm()
    circ_ref = circular_reference()

    scheme_results = {}
    for scheme in SCHEMES:
        out_dir = os.path.join(out_root, scheme)
        if not os.path.isdir(out_dir):
            print(f"[{scheme}] no out_dir at {out_dir}, skipping")
            continue

        rdms, stable_idx, unstable_idx = [], [], []
        for k in range(max(SEEDS) + 1):
            rdm_p = os.path.join(out_dir, f"rdm_net{k:03d}.npy")
            flag_p = os.path.join(out_dir, f"unstable_net{k:03d}.flag")
            if os.path.exists(rdm_p):
                rdms.append(np.load(rdm_p))
                stable_idx.append(k)
            elif os.path.exists(flag_p):
                unstable_idx.append(k)

        n_trained = len(stable_idx) + len(unstable_idx)
        if not rdms:
            print(f"[{scheme}] no stable RDMs found in {out_dir} -- "
                  f"has production.py been run per-network yet?")
            continue

        # per-network partial r (for the Mann-Whitney contrast across schemes)
        per_net_partial = [
            partial_spearman_rdm(rdm, bio_ref, circ_ref) for rdm in rdms
        ]
        per_net_raw = [
            spearmanr(upper(rdm), upper(bio_ref)).statistic for rdm in rdms
        ]

        mean_rdm = np.mean(np.stack(rdms), axis=0)
        raw_r_mean_rdm = spearmanr(upper(mean_rdm), upper(bio_ref)).statistic
        raw_r_circ_mean_rdm = spearmanr(upper(mean_rdm), upper(circ_ref)).statistic
        partial_r_mean_rdm, p_perm_partial = permutation_test_partial(
            mean_rdm, bio_ref, circ_ref, n_permutations=n_perm)

        result = dict(
            scheme=scheme, n_stable=len(rdms), n_unstable=len(unstable_idx),
            n_trained=n_trained, stable_idx=stable_idx, unstable_idx=unstable_idx,
            raw_r_mean_rdm=float(raw_r_mean_rdm),
            raw_r_vs_circular_mean_rdm=float(raw_r_circ_mean_rdm),
            partial_r_mean_rdm=float(partial_r_mean_rdm),
            partial_r_p_perm=float(p_perm_partial),
            per_network_raw_r=per_net_raw,
            per_network_partial_r=per_net_partial,
            per_network_partial_r_mean=float(np.mean(per_net_partial)),
            per_network_partial_r_sd=float(np.std(per_net_partial, ddof=1)) if len(per_net_partial) > 1 else None,
        )
        scheme_results[scheme] = result

        print(f"\n[{scheme}] n_stable={len(rdms)}/{n_trained}")
        print(f"  raw r vs bio (mean RDM):      {raw_r_mean_rdm:.3f}")
        print(f"  raw r vs circular (mean RDM): {raw_r_circ_mean_rdm:.3f}  "
              f"<-- if this ~= raw r vs bio, raw r is mostly circularity (Exp 3 lesson)")
        print(f"  PARTIAL r vs bio | circular:  {partial_r_mean_rdm:.3f}  "
              f"(p_perm={p_perm_partial:.4f})  <-- the comparable-to-Exp-3 number")
        print(f"  per-network partial r: {np.mean(per_net_partial):.3f} +/- "
              f"{np.std(per_net_partial, ddof=1) if len(per_net_partial)>1 else float('nan'):.3f}")

    # cross-scheme contrast on the PARTIAL r -- the actual "degree-breaking gap"
    if "degree_preserving_swap" in scheme_results and "erdos_renyi" in scheme_results:
        ds_partial = scheme_results["degree_preserving_swap"]["per_network_partial_r"]
        er_partial = scheme_results["erdos_renyi"]["per_network_partial_r"]
        if len(ds_partial) >= 2 and len(er_partial) >= 2:
            u_stat, p_val = mannwhitneyu(ds_partial, er_partial, alternative="two-sided")
            print(f"\n{'='*70}")
            print("DEGREE-VS-ER CONTRAST ON PARTIAL r (the number Phase 1 was scoped to get)")
            print(f"{'='*70}")
            print(f"  degree_preserving_swap: partial r = "
                  f"{np.mean(ds_partial):.3f} +/- {np.std(ds_partial, ddof=1):.3f}  (n={len(ds_partial)})")
            print(f"  erdos_renyi:             partial r = "
                  f"{np.mean(er_partial):.3f} +/- {np.std(er_partial, ddof=1):.3f}  (n={len(er_partial)})")
            print(f"  Mann-Whitney U={u_stat:.1f}  p={p_val:.4f}")
            print(f"  Reference (Exp 3, same readout, n=50 CC ensemble): "
                  f"CC partial r = 0.145 (n.s.), random partial r = 0.061 (n.s.)")
            scheme_results["cross_scheme_contrast"] = dict(
                U=float(u_stat), p=float(p_val),
                note="Compare both scheme means against Exp 3's CC=0.145 and random=0.061 "
                     "(both n.s. at n=50) -- these Exp5 n=9 numbers are a pilot, not a "
                     "publication-ready ensemble size.")

    out_json = os.path.join(out_root, "exp5_partial_correlation_results.json")
    with open(out_json, "w") as f:
        json.dump(scheme_results, f, indent=2)
    print(f"\nwrote {out_json}")
    return scheme_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-only", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--results_dir", default=None,
                    help="flyvis results_dir; defaults to flyvis's own results_dir if omitted")
    ap.add_argument("--out_root", default="./pilot_out_40k")
    ap.add_argument("--n_iters", type=int, default=40000)
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()

    if args.link_only:
        if args.results_dir:
            rd = args.results_dir
        else:
            from flyvis import results_dir as _rd
            rd = str(_rd)
        print(f"linking checkpoints under {rd}/flow/ ...")
        link_checkpoints(rd)
        return

    if args.aggregate:
        aggregate_with_partial(args.out_root, args.n_iters, args.n_perm)
        return

    print("nothing to do -- pass --link-only or --aggregate")


if __name__ == "__main__":
    main()
