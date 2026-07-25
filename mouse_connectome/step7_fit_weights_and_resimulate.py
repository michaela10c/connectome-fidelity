"""
Step 7: per-condition weight fitting, then re-simulate and compare to biology.

WHY THIS EXISTS (read before running):
    step2/step5 use a SINGLE fixed SYN_WEIGHT_EE (=5.0), tuned by eye against
    the REAL connectome ("If mean Hz is 0 or >100, adjust SYN_WEIGHT_EE" --
    step2's own check_outputs() docstring), then applied unchanged to every
    null. check_rate_comparability.py confirmed this did NOT produce a
    confound in the existing result (real 17.6 Hz vs C1/C2/C3 all 16.5-17.1
    Hz -- all in one tight band, zero silent neurons everywhere). So this is
    not fixing a broken result. It is the next, more principled step: give
    every wiring condition its OWN independently-fitted weight, mirroring
    Billeh et al. 2020's approach of tuning synaptic weights so a model's
    firing-rate statistics match a target -- rather than sharing one
    hand-tuned constant across every condition, real included.

    This is also the natural first move toward the actual fidelity question:
    once conditions can be independently "trained" to hit a criterion, this
    machinery generalizes directly to fitting on a real objective later
    (rather than just a target rate).

TARGET, AND AN OPEN DESIGN QUESTION -- flagged, not silently decided:
    Billeh fit to MEASURED biological firing-rate distributions (Allen
    Institute electrophysiology, per cell class). Nothing in this pipeline
    currently has a verified, comparable real Hz target -- the biological
    reference here (signal_corr_matrix.npy) is calcium-imaging-derived
    signal correlation, not raw spike rates on a directly comparable scale.
    Rather than assume a biological number that hasn't been verified, this
    script fits every condition to the SAME self-consistent target: by
    default, real's own rate at the current fixed weight (TARGET_HZ=None ->
    measured from the existing real run). This is still the right fix for
    the actual problem this script addresses (equal opportunity across
    conditions, not "make everything look like biology in Hz"). If a real,
    verified biological Hz target becomes available, swap TARGET_HZ for it
    -- everything downstream is unaffected by that choice.

SEARCH DESIGN (kept cheap on purpose):
    Fitting only needs the POPULATION MEAN RATE, not the full 8-orientation
    RDM. So the search phase runs ONE orientation (theta=0) per candidate
    weight -- the same "cheap proxy, expensive final run" split already
    used elsewhere in this codebase (production.py's smoke-test/reportable
    distinction; step5's edge-hash-guarded resimulation). Only once a
    condition's weight is fitted does it get the full 8-orientation run
    needed for RSA.

    Search method: bracket-then-bisect. Rate is monotone increasing in
    weight (more synaptic drive -> more spikes) within the sane operating
    range check_rate_comparability.py already confirmed everything sits in,
    so bisection on a 1D monotone function is appropriate and simple.

SMOKE TEST FIRST (do this before the full run):
    python step7_fit_weights_and_resimulate.py --smoke-test
        Fits + resimulates real + 1 null per family (4 conditions total).
        Confirms the search converges and the pipeline runs end to end
        before committing to all 151 conditions.
    python step7_fit_weights_and_resimulate.py
        Full run: real + all 150 nulls.

Reads:  node_data_v1.csv, step4_null_manifest.json,
        signal_corr_matrix.npy, signal_corr_nucleus_ids.npy,
        structural_rdm_nucleus_ids.npy, output_pointnet/ (existing real run,
        used only to measure the default TARGET_HZ anchor)
Writes: output_pointnet_fitted_{tag}/          (fitted-weight resim per condition)
        step7_fitted_weights.json              (tag -> fitted SYN_WEIGHT_EE, achieved Hz)
        step7_results.txt, step7_results.png   (RSA-vs-biology at fitted weights,
                                                 same format as step5, for direct
                                                 comparison against the unfitted result)
"""

import os
import sys
import json
import time
import hashlib
import shutil
import argparse
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Force line-buffered stdout: when redirected to a file (as in the intended
# `nohup ... > log 2>&1 &` usage), Python fully buffers stdout by default,
# so print() output sits invisible until the buffer fills or the process
# exits -- even with flush=True on some calls but not others. This makes
# every print() flush after each newline, matching interactive-terminal
# behavior, without depending on -u being remembered at launch time or on
# flush=True being present on every individual call.
sys.stdout.reconfigure(line_buffering=True)

import step2_build_and_simulate_microns as step2

ORIENTATIONS   = step2.ORIENTATIONS
TSTOP_MS       = step2.TSTOP
GRAY_SCREEN_MS = 500.0
ANALYSIS_MS    = TSTOP_MS - GRAY_SCREEN_MS
NODE_CSV       = "node_data_v1.csv"
MANIFEST       = "step4_null_manifest.json"
REAL_OUTPUT    = "output_pointnet"          # existing unfitted real run

# -- search parameters ----------------------------------------------------------
TARGET_HZ       = None   # None -> measure from existing real run (see docstring)
RATE_TOLERANCE  = 0.05   # fractional tolerance on target (5%)
MAX_ITERS       = 12     # bisection iterations cap
WEIGHT_LO_INIT  = 0.5    # initial bracket, expanded automatically if needed
WEIGHT_HI_INIT  = 40.0


def fitted_dir_for(tag):
    return f"output_pointnet_fitted_{tag}"


class ProgressTracker:
    """Prints an 'N/total done, elapsed, ETA' line after each completed item,
    plus a milestone summary every `report_every` items. Time-per-item is a
    running average over everything completed so far, so the ETA tightens as
    more items finish."""

    def __init__(self, total, label, report_every=5):
        self.total = total
        self.label = label
        self.report_every = report_every
        self.n_done = 0
        self.t_start = time.time()

    def step(self, item_time_s=None):
        self.n_done += 1
        elapsed = time.time() - self.t_start
        avg_per_item = elapsed / self.n_done
        remaining = self.total - self.n_done
        eta_s = remaining * avg_per_item
        if self.n_done % self.report_every == 0 or self.n_done == self.total:
            print(f"    [progress] {self.label}: {self.n_done}/{self.total} done  "
                  f"elapsed={elapsed/60:.1f}min  avg={avg_per_item:.1f}s/item  "
                  f"ETA={eta_s/60:.1f}min", flush=True)


def fitted_net_dir_for(tag):
    return f"network_fitted_{tag}"


# ── cheap single-orientation rate probe (search phase) ─────────────────────────

def probe_mean_rate(edge_file, tag, weight, N_neurons, theta=0):
    """Run ONE orientation at the given SYN_WEIGHT_EE and return population
    mean Hz. Cheap: this is the search-phase workhorse, called many times per
    condition. Builds into a scratch network dir that gets overwritten each
    call -- not a reportable output, just a probe."""
    step2.SYN_WEIGHT_EE = weight   # module-global override; build_v1_recurrent
                                    # reads this fresh each call (verified: not
                                    # a captured default argument)
    probe_net_dir = f"network_probe_{tag}"
    probe_out_dir = f"output_probe_{tag}"
    if os.path.exists(probe_net_dir):
        shutil.rmtree(probe_net_dir)
    if os.path.exists(probe_out_dir):
        shutil.rmtree(probe_out_dir)

    neurons_df = step2.load_nodes()
    edges_df = step2.load_edges(edge_file)
    step2.build_v1_recurrent(neurons_df, edges_df, probe_net_dir)
    circuit_cfg = step2.make_circuit_config(f"probe_{tag}", probe_net_dir, step2.LGN_DIR)
    step2.run_pointnet_single(theta, f"probe_{tag}", probe_out_dir, circuit_cfg)

    fpath = os.path.join(probe_out_dir, f"theta{theta:03d}", "spikes.h5")
    with h5py.File(fpath, "r") as h5:
        node_ids = h5["spikes/microns_v1/node_ids"][()]
        timestamps = h5["spikes/microns_v1/timestamps"][()]
    m = timestamps >= GRAY_SCREEN_MS
    node_ids = node_ids[m]
    n_spikes_post = len(node_ids)
    mean_hz = n_spikes_post / N_neurons / (ANALYSIS_MS / 1000.0)

    shutil.rmtree(probe_net_dir, ignore_errors=True)
    shutil.rmtree(probe_out_dir, ignore_errors=True)
    return mean_hz


def fit_weight(edge_file, tag, target_hz, N_neurons):
    """Bracket-then-bisect search for the SYN_WEIGHT_EE that makes this
    condition's population mean rate (single-orientation probe) equal
    target_hz, within RATE_TOLERANCE."""
    lo, hi = WEIGHT_LO_INIT, WEIGHT_HI_INIT
    rate_lo = probe_mean_rate(edge_file, tag, lo, N_neurons)
    rate_hi = probe_mean_rate(edge_file, tag, hi, N_neurons)

    # Expand bracket if target isn't between rate_lo and rate_hi yet.
    expand_iters = 0
    while not (rate_lo <= target_hz <= rate_hi) and expand_iters < 6:
        if target_hz < rate_lo:
            lo /= 2
            rate_lo = probe_mean_rate(edge_file, tag, lo, N_neurons)
        else:
            hi *= 2
            rate_hi = probe_mean_rate(edge_file, tag, hi, N_neurons)
        expand_iters += 1

    if not (rate_lo <= target_hz <= rate_hi):
        print(f"    [{tag}] WARNING: could not bracket target_hz={target_hz:.3f} "
              f"(rate_lo={rate_lo:.3f} at w={lo:.3f}, rate_hi={rate_hi:.3f} at "
              f"w={hi:.3f}). Using closest endpoint.")
        return (lo if abs(rate_lo - target_hz) < abs(rate_hi - target_hz) else hi,
                rate_lo if abs(rate_lo - target_hz) < abs(rate_hi - target_hz) else rate_hi)

    for i in range(MAX_ITERS):
        mid = (lo + hi) / 2
        rate_mid = probe_mean_rate(edge_file, tag, mid, N_neurons)
        if abs(rate_mid - target_hz) <= RATE_TOLERANCE * target_hz:
            return mid, rate_mid
        if rate_mid < target_hz:
            lo = mid
        else:
            hi = mid

    return mid, rate_mid


# ── full 8-orientation resim at fitted weight ───────────────────────────────────

def resim_at_fitted_weight(edge_file, tag, fitted_weight):
    step2.SYN_WEIGHT_EE = fitted_weight
    fit_tag = f"fitted_{tag}"
    net_dir = fitted_net_dir_for(tag)
    out_dir = fitted_dir_for(tag)

    edge_hash = hashlib.md5(open(edge_file, "rb").read()).hexdigest() + f"_w{fitted_weight:.4f}"
    stamp_path = os.path.join(out_dir, ".fit_stamp")
    spikes_complete = all(
        os.path.exists(os.path.join(out_dir, f"theta{t:03d}", "spikes.h5"))
        for t in ORIENTATIONS)
    stamp_ok = os.path.exists(stamp_path) and open(stamp_path).read().strip() == edge_hash
    if spikes_complete and stamp_ok:
        print(f"    [{tag}] fitted resim already done at this weight, reusing")
        return out_dir

    if os.path.exists(net_dir):
        shutil.rmtree(net_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    step2.run_simulation(edge_file, fit_tag, build_lgn=False)
    # run_simulation uses network_dir_for/output_dir_for keyed on the tag it
    # was called with; redirect to our fitted_* naming so probes, fitted runs,
    # and the original unfitted runs never collide on disk.
    default_net = step2.network_dir_for(fit_tag)
    default_out = step2.output_dir_for(fit_tag)
    if default_net != net_dir and os.path.exists(default_net):
        shutil.move(default_net, net_dir)
    if default_out != out_dir and os.path.exists(default_out):
        shutil.move(default_out, out_dir)

    with open(stamp_path, "w") as sf:
        sf.write(edge_hash)
    return out_dir


# ── rate/RDM/RSA (identical recipe to step5, copied for self-containment) ──────

def rates_from_output(output_dir, N_neurons):
    rate = np.zeros((N_neurons, len(ORIENTATIONS)), dtype=np.float32)
    for col, theta in enumerate(ORIENTATIONS):
        fpath = os.path.join(output_dir, f"theta{theta:03d}", "spikes.h5")
        with h5py.File(fpath, "r") as h5:
            node_ids   = h5["spikes/microns_v1/node_ids"][()]
            timestamps = h5["spikes/microns_v1/timestamps"][()]
        m = timestamps >= GRAY_SCREEN_MS
        node_ids = node_ids[m]
        uniq, cnt = np.unique(node_ids, return_counts=True)
        for nid, c in zip(uniq, cnt):
            if nid < N_neurons:
                rate[nid, col] = c / (ANALYSIS_MS / 1000.0)
    return rate


def sim_bio_correlation(output_dir, node_df, sig, sig_idx, bio_nucleus_ids):
    from scipy.spatial.distance import pdist, squareform
    N_neurons = len(node_df)
    rate = rates_from_output(output_dir, N_neurons)
    node_id_to_nucleus = node_df['nucleus_id'].to_dict()
    common = [nid for nid in bio_nucleus_ids if nid in sig_idx]
    node_ids_common = [k for k, v in node_id_to_nucleus.items() if v in common]
    if len(node_ids_common) < 50:
        return None, len(node_ids_common)
    rate_sel = rate[node_ids_common, :]
    sim_rdm = squareform(pdist(rate_sel, metric='cosine'))
    nuc_sel = [node_id_to_nucleus[nid] for nid in node_ids_common]
    bio_idx_sel = [sig_idx[n] for n in nuc_sel]
    bio_rdm = 1.0 - sig[np.ix_(bio_idx_sel, bio_idx_sel)]
    n = len(node_ids_common)
    triu = np.triu_indices(n, k=1)
    r, _ = spearmanr(sim_rdm[triu], bio_rdm[triu])
    return r, n


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true",
                    help="Fit+resim only real + 1 null per family (4 total) "
                         "to validate the pipeline before the full 151-condition run.")
    ap.add_argument("--target-hz", type=float, default=None,
                    help="Override the fitting target. Default: measure from "
                         "the existing unfitted real run (see docstring for "
                         "why this is a self-consistent anchor, not a "
                         "verified biological rate).")
    args = ap.parse_args()

    node_df = pd.read_csv(NODE_CSV, index_col="node_id")
    N_neurons = len(node_df)
    print(f"N_neurons: {N_neurons}")

    with open(MANIFEST) as f:
        manifest = json.load(f)

    conditions = [{"tag": "real", "family": "real",
                   "edge_file": "candidate2_real_edges.json"}] + manifest

    if args.smoke_test:
        one_per_family = {}
        for e in manifest:
            one_per_family.setdefault(e["family"], e)
        conditions = [conditions[0]] + list(one_per_family.values())
        print(f"[SMOKE TEST] {len(conditions)} conditions: "
              f"{[c['tag'] for c in conditions]}")
    else:
        print(f"[FULL RUN] {len(conditions)} conditions")

    # -- determine fitting target ------------------------------------------------
    if args.target_hz is not None:
        target_hz = args.target_hz
        print(f"Target rate (user-specified): {target_hz:.3f} Hz")
    elif TARGET_HZ is not None:
        target_hz = TARGET_HZ
        print(f"Target rate (script default): {target_hz:.3f} Hz")
    else:
        print("Measuring target rate from existing unfitted real run "
              f"({REAL_OUTPUT}/theta000)...")
        rate = rates_from_output(REAL_OUTPUT, N_neurons)
        # single-orientation equivalent, theta=0, for consistency with the
        # single-orientation probes used during search
        target_hz = float(rate[:, 0].mean())
        print(f"Target rate (measured, real theta=0): {target_hz:.3f} Hz")
    print("NOTE: this is a self-consistent target (real's own rate), not a "
          "verified biological Hz target -- see script docstring.\n")

    # -- fit each condition -------------------------------------------------------
    print("Fitting per-condition SYN_WEIGHT_EE (cheap single-orientation search)...")

    # Resumability: if a previous run already produced step7_fitted_weights.json,
    # load it and skip tags already fitted. Previously this phase had no
    # persistence at all -- a crash on condition 7 of 151 meant losing 1-6 too.
    fitted = {}
    if os.path.exists("step7_fitted_weights.json"):
        with open("step7_fitted_weights.json") as f:
            fitted = json.load(f)
        print(f"  Resuming: {len(fitted)} conditions already fitted, skipping those.")

    failed_fit_tags = []
    remaining = [c for c in conditions if c["tag"] not in fitted]
    fit_progress = ProgressTracker(len(remaining), "fitting", report_every=5)
    for c in remaining:
        tag, edge_file = c["tag"], c["edge_file"]
        t0 = time.time()
        try:
            w, achieved_hz = fit_weight(edge_file, tag, target_hz, N_neurons)
        except Exception as e:
            # One condition's simulator failure (e.g. a transient filesystem
            # issue) should not take down a multi-hour, 151-condition run.
            # Log it, skip it, keep going -- report failures at the end.
            print(f"  [{tag}] FAILED during fitting: {type(e).__name__}: {e}")
            failed_fit_tags.append(tag)
            fit_progress.step()
            continue
        dt = time.time() - t0
        fitted[tag] = {"family": c["family"], "fitted_weight": w,
                       "achieved_hz": achieved_hz, "target_hz": target_hz,
                       "fit_time_s": dt}
        print(f"  [{tag}] fitted weight={w:.4f}  achieved={achieved_hz:.3f} Hz  "
              f"(target={target_hz:.3f})  [{dt:.1f}s]")
        fit_progress.step()
        # Save after every condition, not just at the end, so a later crash
        # loses at most the one in-flight condition, not everything since
        # the last successful save.
        with open("step7_fitted_weights.json", "w") as f:
            json.dump(fitted, f, indent=2)

    if failed_fit_tags:
        print(f"\n  {len(failed_fit_tags)} condition(s) FAILED during fitting "
              f"and were skipped: {failed_fit_tags}")
        with open("step7_failed_tags.json", "w") as f:
            json.dump({"failed_fitting": failed_fit_tags}, f, indent=2)
    print("\nSaved step7_fitted_weights.json")


    # -- full resim at fitted weight + RSA vs biology ------------------------------
    print("\nRe-simulating at fitted weights (full 8 orientations) and "
          "computing RSA vs biology...")
    sig = np.load("signal_corr_matrix.npy")
    sig_ids = np.load("signal_corr_nucleus_ids.npy")
    sig_idx = {int(n): i for i, n in enumerate(sig_ids)}
    bio_nucleus_ids = np.load("structural_rdm_nucleus_ids.npy").tolist()

    results = {}
    by_family = {}
    if os.path.exists("step7_bio_correlations.json"):
        with open("step7_bio_correlations.json") as f:
            results = json.load(f)
        for tag, v in results.items():
            fam = next((c["family"] for c in conditions if c["tag"] == tag), None)
            if fam is not None and v.get("r") is not None:
                by_family.setdefault(fam, []).append(v["r"])
        print(f"  Resuming: {len(results)} conditions already resimmed, skipping those.")

    failed_resim_tags = []
    fittable = [c for c in conditions if c["tag"] in fitted and c["tag"] not in results]
    resim_progress = ProgressTracker(len(fittable), "resim+RSA", report_every=5)
    for c in fittable:
        tag, edge_file = c["tag"], c["edge_file"]
        w = fitted[tag]["fitted_weight"]
        try:
            out_dir = resim_at_fitted_weight(edge_file, tag, w)
            r, n = sim_bio_correlation(out_dir, node_df, sig, sig_idx, bio_nucleus_ids)
        except Exception as e:
            print(f"  [{tag}] FAILED during resim: {type(e).__name__}: {e}")
            failed_resim_tags.append(tag)
            resim_progress.step()
            continue
        results[tag] = {"r": r, "n": n, "fitted_weight": w}
        by_family.setdefault(c["family"], []).append(r)
        print(f"  [{tag}] fitted-sim vs bio: r = {r:.4f}  (n={n})")
        resim_progress.step()
        # Save after every condition, same reasoning as the fitting loop.
        with open("step7_bio_correlations.json", "w") as f:
            json.dump(results, f, indent=2)

    if failed_resim_tags:
        print(f"\n  {len(failed_resim_tags)} condition(s) FAILED during resim "
              f"and were skipped: {failed_resim_tags}")
        prior = {}
        if os.path.exists("step7_failed_tags.json"):
            with open("step7_failed_tags.json") as f:
                prior = json.load(f)
        prior["failed_resim"] = failed_resim_tags
        with open("step7_failed_tags.json", "w") as f:
            json.dump(prior, f, indent=2)

    # -- report, matching step5's format for direct comparison --------------------
    r_real = results["real"]["r"]
    print("\n" + "=" * 72)
    print("STEP 7 RESULT -- fitted-weight real-sim vs biology, against "
          "fitted-weight null-sim-vs-bio")
    print("=" * 72)
    print(f"real-sim vs bio (fitted weight={fitted['real']['fitted_weight']:.4f}): "
          f"r = {r_real:.4f}")
    print(f"{'Null':<26} {'mean':>8} {'sd':>7} {'z':>7} {'emp p':>8} {'real>null':>10}")
    print("-" * 72)
    lines = []
    for fam in sorted(by_family):
        if fam == "real":
            continue
        arr = np.array([x for x in by_family[fam] if x is not None])
        mean, sd = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
        z = (r_real - mean) / sd if sd > 0 else float('nan')
        emp_p = (np.sum(arr >= r_real) + 1) / (len(arr) + 1)
        frac = float(np.mean(r_real > arr))
        n = len(arr)
        print(f"{fam+' (n='+str(n)+')':<26} {mean:>8.4f} {sd:>7.4f} {z:>7.2f} "
              f"{emp_p:>8.3f} {100*frac:>9.0f}%")
        lines.append((fam, mean, sd, z, emp_p, frac, n))

    with open("step7_results.txt", "w") as f:
        f.write("Step 7 -- fitted-weight real-sim vs biology, against fitted-weight "
                "null-sim-vs-bio\n")
        f.write("=" * 72 + "\n")
        f.write("Each condition (real and every null) independently fitted its own "
                "SYN_WEIGHT_EE\nto reach the SAME target rate before the RSA-vs-biology "
                "comparison -- unlike\nstep5, which shared one hand-tuned weight across "
                "every condition.\n\n")
        f.write(f"Target rate: {target_hz:.3f} Hz "
                f"({'self-consistent anchor from real' if args.target_hz is None else 'user-specified'})\n")
        f.write(f"real-sim vs bio: r = {r_real:.4f}  "
                f"(fitted weight={fitted['real']['fitted_weight']:.4f})\n\n")
        f.write(f"{'Null':<26} {'mean':>8} {'sd':>7} {'z':>7} {'emp p':>8} "
                f"{'real>null':>10}\n")
        f.write("-" * 72 + "\n")
        for fam, mean, sd, z, emp_p, frac, n in lines:
            f.write(f"{fam+' (n='+str(n)+')':<26} {mean:>8.4f} {sd:>7.4f} "
                    f"{z:>7.2f} {emp_p:>8.3f} {100*frac:>9.0f}%\n")
        f.write("\nCompare directly against step5_results.txt (same format, "
                "unfitted shared weight).\n")
        f.write("PROTOTYPE: iaf_psc_alpha LIF, excitatory-only, single free "
                "parameter (SYN_WEIGHT_EE)\nfitted per condition; SYN_WEIGHT_LGN "
                "and all other parameters still shared.\n")
    print("\nSaved step7_results.txt, step7_bio_correlations.json")

    # -- plot -----------------------------------------------------------------
    if lines:
        fig, ax = plt.subplots(figsize=(8, 5))
        fams = [l[0] for l in lines]
        for i, fam in enumerate(fams, start=1):
            arr = np.array([x for x in by_family[fam] if x is not None])
            jit = np.random.uniform(-0.06, 0.06, len(arr))
            ax.scatter(np.full(len(arr), i) + jit, arr, s=30, alpha=0.6,
                       color="tab:blue", label="null sims" if i == 1 else None)
        ax.axhline(r_real, color="tomato", lw=2,
                   label=f"real-sim vs bio (fitted) = {r_real:.3f}")
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.set_xticks(range(1, len(fams) + 1)); ax.set_xticklabels(fams)
        ax.set_ylabel("null-sim vs biology (Spearman r)")
        ax.set_title("Step 7: fitted-weight real-sim vs null-sim, vs measured biology\n"
                     "(each condition independently fitted to the same target rate)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("step7_results.png", dpi=150, bbox_inches="tight")
        print("Saved step7_results.png")

    print("\n--- STEP 7 COMPLETE ---")
    if args.smoke_test:
        print("This was a SMOKE TEST (4 conditions). Not a reportable result.")
        print("Run without --smoke-test for the full 151-condition comparison.")


if __name__ == "__main__":
    main()
