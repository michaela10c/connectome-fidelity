"""
step7_ei_fit_weights.py — two-parameter (E, I) weight fitting for the signed
MICrONS graph, extending step7_fit_weights_and_resimulate.py's design rather
than replacing it.

WHY THIS EXISTS
---------------
step7 fits a single free parameter (SYN_WEIGHT_EE) per condition, because the
E-only graph has one weight scale to fit. The signed graph has two: an
excitatory gain and an inhibitory gain (step2_ei_recurrent.py's
SYN_WEIGHT_E / SYN_WEIGHT_I), and unlike the E-only case, this pair is NOT
reducible to one free parameter -- E and I populations can have genuinely
different target rates, and the ratio between the two gains is itself a real
structural question (E/I balance), not a nuisance parameter to average away.

DESIGN: COORDINATE DESCENT, NOT A NEW ALGORITHM
-------------------------------------------------
Rather than a 2D search, this alternates step7-style 1D bisections:
    round 1: fit SYN_WEIGHT_E to hit target_hz_E, holding I fixed
             fit SYN_WEIGHT_I to hit target_hz_I, holding E (just-fitted) fixed
    round 2: repeat both, now starting from round 1's fitted values
    ... until both rates are within tolerance simultaneously, or max_rounds
Each individual bisection is step7's existing fit_weight() logic, applied to
one axis at a time -- no new search machinery, just alternation.

TARGETS: SELF-CONSISTENT, NOT VERIFIED BIOLOGICAL RATES
-----------------------------------------------------------
Same reasoning as step7: no verified biological Hz target exists in this
pipeline for E and I populations separately. Targets default to the real
signed connectome's own E-population and I-population rates at the default
(unfit) weights -- self-consistent anchors, swappable via --target-hz-e /
--target-hz-i if real targets become available. This does not change the
fitting logic, only the numbers it fits toward.

STABILITY, MADE EXPLICIT (NOT OPTIONAL HERE)
-----------------------------------------------
Unlike the pure-excitatory case, an E/I point-neuron network CAN destabilize
in dynamically distinct ways: too-weak inhibition relative to excitation can
runaway-saturate; too-strong inhibition can fully silence the network. Every
probe now checks is_stable_ei() (finite activity, within a sane rate band)
BEFORE returning a rate. A condition whose search cannot find a stable (E, I)
pair within the bracket is logged to step7_ei_failed_tags.json and skipped --
matching step7's existing resilience pattern (one condition's failure does
not kill a multi-condition run) -- rather than silently reporting an unstable
network's rate as if it were meaningful.

SMOKE TEST FIRST:
    python step7_ei_fit_weights.py --smoke-test
        Fits + resimulates only the real signed connectome (no nulls yet).
        Confirms the coordinate-descent search converges and the E/I
        simulation path runs end to end before committing to a full batch
        of null_edges_C1signed_seed*.json / null_edges_C1unrestricted_seed*.json.

Reads:  node_data_v1_ei.csv (full E+I population -- NOT node_data_v1.csv,
        which is E-only), node_types.json, signed_real_edges.json (or a null
        edge file), signal_corr_matrix.npy, signal_corr_nucleus_ids.npy,
        structural_rdm_nucleus_ids.npy, output_pointnet_real_ei/ (if it
        exists, used only to measure default self-consistent targets)
Writes: output_pointnet_fitted_ei_{tag}/         (fitted-weight resim)
        step7_ei_fitted_weights.json             (tag -> E,I fitted weights)
        step7_ei_results.txt, step7_ei_results.png
        step7_ei_failed_tags.json                (conditions that could not
                                                    reach a stable fit)
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

sys.stdout.reconfigure(line_buffering=True)

import step2_build_and_simulate_microns as step2
import step2_ei_recurrent as step2_ei

ORIENTATIONS   = step2.ORIENTATIONS
TSTOP_MS       = step2.TSTOP
GRAY_SCREEN_MS = 500.0
ANALYSIS_MS    = TSTOP_MS - GRAY_SCREEN_MS
NODE_CSV       = "node_data_v1_ei.csv"   # full E+I population, NOT
                                           # node_data_v1.csv (E-only) --
                                           # see step2_ei_recurrent.py's
                                           # load_nodes_ei() docstring
NODE_TYPES     = "node_types.json"
REAL_OUTPUT_EI = "output_pointnet_real_ei"

RATE_TOLERANCE  = 0.05
MAX_ITERS       = 12       # bisection iterations, per axis, per round
MAX_ROUNDS      = 4        # coordinate-descent rounds (E then I, repeated)
WEIGHT_LO_INIT  = 2.0    # narrowed from 0.5: both null families showed
                          # instability (I silenced, hz_i~0) and a
                          # non-monotonic peak in hz_e-vs-w_e below this
                          # point -- confirmed via direct grid/sweep on two
                          # independent nulls, not assumed from one
WEIGHT_HI_INIT  = 15.0    # narrowed from 40.0: real solutions for both
                          # nulls and the real connectome all live in
                          # roughly 3-10 for both axes; 15 gives headroom
                          # without re-entering the wide-extrapolation
                          # territory where the original bracket got lost

# Stability band: a probe outside this range (silent or saturating) is
# treated as unstable, not just "off-target". Matches the spirit of
# Experiment 5's is_stable check on the fly side -- finite, bounded,
# non-trivial activity, not a specific target value.
MIN_STABLE_HZ = 0.1
MAX_STABLE_HZ = 200.0


def is_stable_ei(hz):
    return bool(MIN_STABLE_HZ <= hz <= MAX_STABLE_HZ)


def fitted_dir_for(tag):
    return f"output_pointnet_fitted_ei_{tag}"


class ProgressTracker:
    def __init__(self, total, label, report_every=5):
        self.total = total
        self.label = label
        self.report_every = report_every
        self.n_done = 0
        self.t_start = time.time()

    def step(self):
        self.n_done += 1
        elapsed = time.time() - self.t_start
        avg = elapsed / self.n_done
        remaining = self.total - self.n_done
        eta_s = remaining * avg
        if self.n_done % self.report_every == 0 or self.n_done == self.total:
            print(f"    [progress] {self.label}: {self.n_done}/{self.total} done  "
                  f"elapsed={elapsed/60:.1f}min  avg={avg:.1f}s/item  "
                  f"ETA={eta_s/60:.1f}min", flush=True)


# ── cheap single-orientation probe, now returning (rate_E, rate_I, stable) ──

def probe_mean_rates_ei(edge_file, tag, weight_e, weight_i, N_neurons,
                        node_types, e_mask, i_mask, theta=0):
    """Run ONE orientation at the given (weight_e, weight_i) and return
    (mean_hz_E, mean_hz_I, stable). Cheap: the search-phase workhorse."""
    probe_net_dir = f"network_ei_probe_{tag}"
    probe_out_dir = f"output_ei_probe_{tag}"
    if os.path.exists(probe_net_dir):
        shutil.rmtree(probe_net_dir)
    if os.path.exists(probe_out_dir):
        shutil.rmtree(probe_out_dir)

    neurons_df = step2_ei.load_nodes_ei()
    edges_df = step2_ei.load_edges_signed(edge_file)
    step2_ei.build_v1_recurrent_ei(neurons_df, edges_df, probe_net_dir,
                                    weight_e, weight_i, node_types=node_types)
    circuit_cfg = step2.make_circuit_config(f"ei_probe_{tag}", probe_net_dir,
                                            step2.LGN_DIR)
    step2.run_pointnet_single(theta, f"ei_probe_{tag}", probe_out_dir, circuit_cfg)

    fpath = os.path.join(probe_out_dir, f"theta{theta:03d}", "spikes.h5")
    with h5py.File(fpath, "r") as h5:
        node_ids = h5["spikes/microns_v1/node_ids"][()]
        timestamps = h5["spikes/microns_v1/timestamps"][()]
    m = timestamps >= GRAY_SCREEN_MS
    node_ids = node_ids[m]

    n_e = e_mask.sum()
    n_i = i_mask.sum()
    e_spikes = np.isin(node_ids, np.where(e_mask)[0]).sum()
    i_spikes = np.isin(node_ids, np.where(i_mask)[0]).sum()
    hz_e = e_spikes / max(n_e, 1) / (ANALYSIS_MS / 1000.0)
    hz_i = i_spikes / max(n_i, 1) / (ANALYSIS_MS / 1000.0)

    shutil.rmtree(probe_net_dir, ignore_errors=True)
    shutil.rmtree(probe_out_dir, ignore_errors=True)

    stable = is_stable_ei(hz_e) and is_stable_ei(hz_i)
    return hz_e, hz_i, stable


def _bisect_one_axis(edge_file, tag, target_hz, fixed_other_weight,
                     is_e_axis, N_neurons, node_types, e_mask, i_mask):
    """One step7-style bisection, on whichever axis (E or I) is being fit
    this call, holding the other axis fixed at fixed_other_weight.

    DIRECTION IS DETECTED, NOT ASSUMED (E increases with its own weight, I
    decreases with its own weight, both confirmed empirically).

    FEASIBILITY FLOOR, added after a concrete diagnosed failure: fitting E
    in isolation against its own target can walk into a region where E is
    too low to drive I to threshold via E->I edges AT ALL -- I goes
    completely silent (hz_i=0), confirmed constant across every tested I
    weight at that E value (I's own weight only scales its OUTGOING
    synapses, which do nothing while I isn't firing). E's own isolated
    target can look BEST satisfied exactly in this dead zone, since nothing
    stops E's bisection from optimizing there -- it has no visibility into
    what it's doing to the other axis. This is not a bracket-tuning
    problem; it is encoded directly as a constraint: any candidate whose
    OTHER axis is dead is treated as infeasible and pushed away from,
    regardless of how close the axis-being-fit's own rate looks to target.
    """
    def probe(w):
        we, wi = (w, fixed_other_weight) if is_e_axis else (fixed_other_weight, w)
        hz_e, hz_i, stable = probe_mean_rates_ei(
            edge_file, tag, we, wi, N_neurons, node_types, e_mask, i_mask)
        axis_rate = hz_e if is_e_axis else hz_i
        other_rate = hz_i if is_e_axis else hz_e
        other_dead = other_rate < MIN_STABLE_HZ
        return axis_rate, other_dead, stable

    lo, hi = WEIGHT_LO_INIT, WEIGHT_HI_INIT
    rate_lo, dead_lo, stable_lo = probe(lo)
    rate_hi, dead_hi, stable_hi = probe(hi)
    increasing = rate_hi >= rate_lo   # detected, not assumed

    if dead_lo and dead_hi:
        raise RuntimeError(
            f"the OTHER axis is dead (silent) across the entire bracket "
            f"[{lo}, {hi}] for this weight -- not a search failure, a real "
            f"finding: this range cannot sustain both populations "
            f"simultaneously; the bracket itself needs rethinking for this "
            f"condition, not another bisection attempt")

    # If exactly one endpoint kills the other axis, don't treat it as a
    # legitimate bracket boundary -- pull it toward the live endpoint until
    # it escapes the dead zone, before doing anything else.
    pull_iters = 0
    while dead_lo and pull_iters < 8:
        lo = (lo + hi) / 2
        rate_lo, dead_lo, stable_lo = probe(lo)
        pull_iters += 1
    while dead_hi and pull_iters < 8:
        hi = (lo + hi) / 2
        rate_hi, dead_hi, stable_hi = probe(hi)
        pull_iters += 1

    def bracketed(rl, rh):
        return (rl <= target_hz <= rh) if increasing else (rh <= target_hz <= rl)

    expand_iters = 0
    while not bracketed(rate_lo, rate_hi) and expand_iters < 6:
        rate_at_lo_is_too_far = (target_hz < rate_lo) if increasing else (target_hz > rate_lo)
        if rate_at_lo_is_too_far:
            lo /= 2
            rate_lo, dead_lo, stable_lo = probe(lo)
            if dead_lo:
                # expansion walked back into the dead zone -- pull out again
                # rather than accepting this as progress
                lo = (lo + hi) / 2
                rate_lo, dead_lo, stable_lo = probe(lo)
        else:
            hi *= 2
            rate_hi, dead_hi, stable_hi = probe(hi)
        expand_iters += 1

    if not bracketed(rate_lo, rate_hi):
        best = lo if abs(rate_lo - target_hz) < abs(rate_hi - target_hz) else hi
        best_rate = rate_lo if best == lo else rate_hi
        best_stable = stable_lo if best == lo else stable_hi
        return best, best_rate, best_stable

    stable = stable_lo and stable_hi
    for _ in range(MAX_ITERS):
        mid = (lo + hi) / 2
        rate_mid, dead_mid, stable_mid = probe(mid)
        stable = stable_mid
        if dead_mid:
            # infeasible regardless of how close rate_mid looks to target --
            # push toward the live side rather than accepting it
            if is_e_axis:
                lo = mid   # low E starves I -- move toward higher E
            else:
                hi = mid   # symmetric handling if I's own axis ever dies
            continue
        if abs(rate_mid - target_hz) <= RATE_TOLERANCE * target_hz:
            return mid, rate_mid, stable_mid
        rate_too_low = rate_mid < target_hz
        # increasing: rate too low -> need MORE weight -> raise lo
        # decreasing: rate too low -> need LESS weight -> lower hi
        if rate_too_low == increasing:
            lo = mid
        else:
            hi = mid
    return mid, rate_mid, stable_mid


def _fit_weights_ei_single_start(edge_file, tag, target_hz_e, target_hz_i,
                                 N_neurons, node_types, e_mask, i_mask,
                                 start_e, start_i, verbose=True):
    """One coordinate-descent run from a SPECIFIC starting (E,I) pair.
    Alternate E and I via 1D bisection until both hit target simultaneously,
    MAX_ROUNDS is exhausted, or a genuine fixed point is detected (weights
    stop changing between rounds while stable) -- concretely observed on
    null_edges_C1unrestricted_seed0 from the default start: rounds 2-4
    landed on byte-identical (E=4.252, I=2.406) with I within tolerance but
    E persistently off. Bisection deterministically re-deriving the
    identical weight from the identical fixed I IS convergence
    mechanically; it just may not satisfy tolerance -- which is exactly why
    this is now wrapped by fit_weights_ei() below, trying multiple starts
    rather than trusting whichever equilibrium one arbitrary start reaches.

    Returns (weight_e, weight_i, achieved_hz_e, achieved_hz_i, stable,
    converged_type) where converged_type is 'full' or 'fixed_point'.
    Raises RuntimeError only if MAX_ROUNDS is exhausted WITHOUT reaching
    either outcome -- i.e. the search was still actually changing.
    """
    w_e, w_i = start_e, start_i
    last_stable = False
    prev_w_e, prev_w_i = None, None
    for rnd in range(MAX_ROUNDS):
        w_e, hz_e, stable_e = _bisect_one_axis(
            edge_file, tag, target_hz_e, w_i, True, N_neurons,
            node_types, e_mask, i_mask)
        w_i, hz_i, stable_i = _bisect_one_axis(
            edge_file, tag, target_hz_i, w_e, False, N_neurons,
            node_types, e_mask, i_mask)
        last_stable = stable_e and stable_i
        if verbose:
            print(f"    [{tag}] (start E={start_e:.2f},I={start_i:.2f}) "
                  f"round {rnd+1}: E->w={w_e:.3f} hz={hz_e:.2f} "
                  f"(target {target_hz_e:.2f})  I->w={w_i:.3f} hz={hz_i:.2f} "
                  f"(target {target_hz_i:.2f})  stable={last_stable}")
        e_ok = abs(hz_e - target_hz_e) <= RATE_TOLERANCE * target_hz_e
        i_ok = abs(hz_i - target_hz_i) <= RATE_TOLERANCE * target_hz_i
        if e_ok and i_ok and last_stable:
            return w_e, w_i, hz_e, hz_i, True, "full"

        if (prev_w_e is not None and last_stable
                and abs(w_e - prev_w_e) < 1e-6 and abs(w_i - prev_w_i) < 1e-6):
            missed = []
            if not e_ok:
                missed.append(f"E (hz={hz_e:.2f}/target={target_hz_e:.2f})")
            if not i_ok:
                missed.append(f"I (hz={hz_i:.2f}/target={target_hz_i:.2f})")
            if verbose:
                print(f"    [{tag}] fixed point reached from this start "
                      f"(weights unchanged, stable). Outside tolerance: "
                      f"{', '.join(missed)}")
            return w_e, w_i, hz_e, hz_i, True, "fixed_point"
        prev_w_e, prev_w_i = w_e, w_i

    at_bracket_edge = (w_i >= WEIGHT_HI_INIT * 0.999 or w_i <= WEIGHT_LO_INIT * 1.001
                       or w_e >= WEIGHT_HI_INIT * 0.999 or w_e <= WEIGHT_LO_INIT * 1.001)
    raise RuntimeError(
        f"did not converge within {MAX_ROUNDS} rounds from start "
        f"E={start_e:.2f},I={start_i:.2f} (final: E hz={hz_e:.2f}/"
        f"target={target_hz_e:.2f}, I hz={hz_i:.2f}/target={target_hz_i:.2f}, "
        f"stable={last_stable})"
        + (" -- an axis is stuck at its search bracket edge" if at_bracket_edge else ""))


# Alternate starting points tried if the default (5,5) lands on a fixed
# point outside tolerance. Not theory-derived -- a pragmatic spread across
# the [2,15]x[2,15] bracket, motivated directly by the empirical finding
# that null_edges_C1unrestricted_seed5 escaped the (4.252,2.406) trap that
# caught every other seed starting from the same (5,5) default: different
# starting points can land in different bisection intervals (the same
# mechanism that explains signed_seed2's distinct weights), so trying a
# few is a direct, mechanism-informed way to give a trapped seed the same
# chance to escape that seed5 got by accident, without needing to know in
# advance which start will work for which seed.
_ALT_STARTS = [(10.0, 10.0), (8.0, 3.0), (3.0, 8.0)]


def fit_weights_ei(edge_file, tag, target_hz_e, target_hz_i, N_neurons,
                   node_types, e_mask, i_mask, verbose=True):
    """Multi-start wrapper around _fit_weights_ei_single_start(). Tries the
    default (5,5) start first (matching all prior behavior and every
    already-fitted condition's provenance); if that lands on a fixed point
    outside tolerance, tries _ALT_STARTS in order, stopping at the first
    full-tolerance match. If none reach full tolerance, returns whichever
    fixed point came closest to target across ALL starts tried -- an
    improvement over trusting a single arbitrary start's equilibrium, even
    when no start reaches a clean match.
    """
    attempts = []   # (w_e, w_i, hz_e, hz_i, stable, converged_type, dist)
    starts = [(5.0, 5.0)] + _ALT_STARTS
    for start_e, start_i in starts:
        try:
            result = _fit_weights_ei_single_start(
                edge_file, tag, target_hz_e, target_hz_i, N_neurons,
                node_types, e_mask, i_mask, start_e, start_i, verbose=verbose)
        except RuntimeError as e:
            if verbose:
                print(f"    [{tag}] start E={start_e:.2f},I={start_i:.2f} "
                      f"did not settle: {e}")
            continue
        w_e, w_i, hz_e, hz_i, stable, ctype = result
        if ctype == "full":
            if verbose:
                print(f"    [{tag}] full convergence from start "
                      f"E={start_e:.2f},I={start_i:.2f} -- stopping search")
            return result
        dist = (abs(hz_e - target_hz_e) / target_hz_e
               + abs(hz_i - target_hz_i) / target_hz_i)
        attempts.append(result + (dist,))

    if not attempts:
        raise RuntimeError(
            f"no starting point (of {len(starts)} tried) reached either "
            f"full tolerance or a stable fixed point")

    best = min(attempts, key=lambda a: a[-1])
    if verbose:
        print(f"    [{tag}] no start reached full tolerance; using the "
              f"closest fixed point found across {len(attempts)} start(s) "
              f"tried: E={best[0]:.3f} I={best[1]:.3f}")
    return best[:6]


# ── full 8-orientation resim + RSA-vs-biology (mirrors step7's design) ─────

def _code_fingerprint():
    """Hash of every simulation-relevant source file, so cached resim
    output is correctly invalidated when the CODE changes, not just the
    edge file or weights -- this directly closes the gap that let 'real'
    go stale for most of a session (found 2026-07-12): the old stamp
    (edge-file hash + weights only) had no way to detect that
    step2_ei_recurrent.py's epsilon-perturbation fix had been added after
    real's cache was already written, so a cache from before that fix sat
    unnoticed and unrefreshed indefinitely. Any future edit to any of
    these files now invalidates every cached condition automatically.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_hash = [
        os.path.abspath(__file__),
        os.path.join(this_dir, "step2_ei_recurrent.py"),
        os.path.join(this_dir, "step2_build_and_simulate_microns.py"),
    ]
    h = hashlib.md5()
    for f in sorted(files_to_hash):
        if os.path.exists(f):
            h.update(open(f, "rb").read())
        else:
            h.update(f"MISSING:{f}".encode())
    return h.hexdigest()[:12]


def resim_at_fitted_weights_ei(edge_file, tag, w_e, w_i, node_types):
    fit_tag = f"fitted_ei_{tag}"
    out_dir = fitted_dir_for(tag)
    net_dir = f"network_fitted_ei_{tag}"

    edge_hash = (hashlib.md5(open(edge_file, "rb").read()).hexdigest()
                + f"_e{w_e:.4f}_i{w_i:.4f}"
                + f"_code{_code_fingerprint()}")
    stamp_path = os.path.join(out_dir, ".fit_stamp")
    spikes_complete = all(
        os.path.exists(os.path.join(out_dir, f"theta{t:03d}", "spikes.h5"))
        for t in ORIENTATIONS)
    stamp_ok = os.path.exists(stamp_path) and open(stamp_path).read().strip() == edge_hash
    if spikes_complete and stamp_ok:
        print(f"    [{tag}] fitted E/I resim already done, reusing")
        return out_dir

    if os.path.exists(net_dir):
        shutil.rmtree(net_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    step2_ei.run_simulation_ei(edge_file, fit_tag, w_e, w_i,
                               build_lgn=False, node_types=node_types)
    default_net = step2.network_dir_for(fit_tag)
    default_out = step2.output_dir_for(fit_tag)
    if default_net != net_dir and os.path.exists(default_net):
        shutil.move(default_net, net_dir)
    if default_out != out_dir and os.path.exists(default_out):
        shutil.move(default_out, out_dir)

    with open(stamp_path, "w") as sf:
        sf.write(edge_hash)
    return out_dir


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
                    help="Fit+resim only the real signed connectome. "
                         "Confirms the E/I path works before a full null batch.")
    ap.add_argument("--target-hz-e", type=float, default=None)
    ap.add_argument("--target-hz-i", type=float, default=None)
    ap.add_argument("--conditions", type=str, default=None,
                    help="Comma-separated edge files to process beyond 'real'. "
                         "Default: just the real signed connectome.")
    args = ap.parse_args()

    node_df = pd.read_csv(NODE_CSV, index_col="node_id")
    N_neurons = len(node_df)
    node_types = step2_ei.load_node_types(NODE_TYPES)
    node_id_to_nucleus = node_df['nucleus_id'].to_dict()
    e_mask = np.array([node_types.get(node_id_to_nucleus.get(i)) != 'I'
                       for i in range(N_neurons)])
    i_mask = ~e_mask
    print(f"N_neurons: {N_neurons}  (E={e_mask.sum()}, I={i_mask.sum()})")

    conditions = [{"tag": "real", "edge_file": "signed_real_edges.json"}]
    if args.conditions:
        for ef in args.conditions.split(","):
            tag = os.path.splitext(os.path.basename(ef))[0]
            conditions.append({"tag": tag, "edge_file": ef})
    if args.smoke_test:
        conditions = conditions[:1]
        print(f"[SMOKE TEST] {len(conditions)} condition(s): "
              f"{[c['tag'] for c in conditions]}")

    # -- determine self-consistent E/I targets ------------------------------
    if args.target_hz_e is not None and args.target_hz_i is not None:
        target_hz_e, target_hz_i = args.target_hz_e, args.target_hz_i
        print(f"Targets (user-specified): E={target_hz_e:.3f}  I={target_hz_i:.3f}")
    else:
        if not os.path.exists(REAL_OUTPUT_EI):
            sys.exit(
                f"No {REAL_OUTPUT_EI}/ found to measure self-consistent "
                f"targets from, and --target-hz-e/--target-hz-i not given. "
                f"Run once with default weights first, or pass explicit targets.")
        print(f"Measuring self-consistent targets from {REAL_OUTPUT_EI}/theta000...")
        rate = rates_from_output(REAL_OUTPUT_EI, N_neurons)
        target_hz_e = float(rate[e_mask, 0].mean())
        target_hz_i = float(rate[i_mask, 0].mean())
        print(f"Targets (measured): E={target_hz_e:.3f}  I={target_hz_i:.3f}")
    print("NOTE: self-consistent anchors, not verified biological Hz targets "
          "-- see script docstring.\n")

    # -- fit each condition -------------------------------------------------
    fitted = {}
    if os.path.exists("step7_ei_fitted_weights.json"):
        try:
            with open("step7_ei_fitted_weights.json") as f:
                fitted = json.load(f)
            print(f"  Resuming: {len(fitted)} conditions already fitted.")
        except json.JSONDecodeError as e:
            print(f"  WARNING: step7_ei_fitted_weights.json exists but is not "
                  f"valid JSON ({e}) -- almost certainly a truncated write "
                  f"from an earlier crash. Starting fresh rather than "
                  f"propagating this as a fatal error; the file will be "
                  f"overwritten by the first successful fit below.")
            fitted = {}

    failed_tags = []
    remaining = [c for c in conditions if c["tag"] not in fitted]
    fit_progress = ProgressTracker(len(remaining), "fitting", report_every=1)
    for c in remaining:
        tag, edge_file = c["tag"], c["edge_file"]
        try:
            w_e, w_i, hz_e, hz_i, stable, converged_type = fit_weights_ei(
                edge_file, tag, target_hz_e, target_hz_i, N_neurons,
                node_types, e_mask, i_mask)
        except Exception as e:
            print(f"  [{tag}] FAILED during fitting: {type(e).__name__}: {e}")
            failed_tags.append(tag)
            fit_progress.step()
            continue
        fitted[tag] = {"fitted_weight_e": w_e, "fitted_weight_i": w_i,
                       "achieved_hz_e": hz_e, "achieved_hz_i": hz_i,
                       "target_hz_e": target_hz_e, "target_hz_i": target_hz_i,
                       "stable": stable, "converged_type": converged_type}
        print(f"  [{tag}] E={w_e:.4f} (hz={hz_e:.2f})  I={w_i:.4f} (hz={hz_i:.2f})  "
              f"stable={stable}  converged={converged_type}")
        fit_progress.step()
        with open("step7_ei_fitted_weights.json", "w") as f:
            json.dump(fitted, f, indent=2)

    if failed_tags:
        with open("step7_ei_failed_tags.json", "w") as f:
            json.dump({"failed_fitting": failed_tags}, f, indent=2)
        print(f"\n  {len(failed_tags)} condition(s) FAILED: {failed_tags}")

    # -- resim + RSA vs biology ----------------------------------------------
    sig = np.load("signal_corr_matrix.npy")
    sig_ids = np.load("signal_corr_nucleus_ids.npy")
    sig_idx = {int(n): i for i, n in enumerate(sig_ids)}
    bio_nucleus_ids = np.load("structural_rdm_nucleus_ids.npy").tolist()

    results = {}
    if os.path.exists("step7_ei_bio_correlations.json"):
        try:
            with open("step7_ei_bio_correlations.json") as f:
                results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  WARNING: step7_ei_bio_correlations.json exists but is "
                  f"not valid JSON ({e}) -- starting fresh, same reasoning "
                  f"as the fitted-weights file above.")
            results = {}

    fittable = [c for c in conditions if c["tag"] in fitted and c["tag"] not in results]
    resim_progress = ProgressTracker(len(fittable), "resim+RSA", report_every=1)
    for c in fittable:
        tag, edge_file = c["tag"], c["edge_file"]
        w_e = fitted[tag]["fitted_weight_e"]
        w_i = fitted[tag]["fitted_weight_i"]
        try:
            out_dir = resim_at_fitted_weights_ei(edge_file, tag, w_e, w_i, node_types)
            r, n = sim_bio_correlation(out_dir, node_df, sig, sig_idx, bio_nucleus_ids)
        except Exception as e:
            print(f"  [{tag}] FAILED during resim: {type(e).__name__}: {e}")
            resim_progress.step()
            continue
        results[tag] = {"r": r, "n": n, "fitted_weight_e": w_e, "fitted_weight_i": w_i}
        print(f"  [{tag}] fitted-EI-sim vs bio: r = {r:.4f}  (n={n})")
        resim_progress.step()
        with open("step7_ei_bio_correlations.json", "w") as f:
            json.dump(results, f, indent=2)

    with open("step7_ei_results.txt", "w") as f:
        f.write("Step 7 EI -- fitted (E,I)-weight sim vs biology\n")
        f.write("=" * 72 + "\n")
        f.write(f"Targets: E={target_hz_e:.3f} Hz, I={target_hz_i:.3f} Hz "
                f"(self-consistent anchors)\n\n")
        for tag, res in results.items():
            f.write(f"{tag}: r={res['r']:.4f}  (n={res['n']}, "
                    f"E_w={res['fitted_weight_e']:.4f}, "
                    f"I_w={res['fitted_weight_i']:.4f})\n")
    print("\nSaved step7_ei_results.txt, step7_ei_bio_correlations.json")

    print("\n--- STEP 7 EI COMPLETE ---")
    if args.smoke_test:
        print("This was a SMOKE TEST (real connectome only). Not a null comparison yet.")


if __name__ == "__main__":
    main()
