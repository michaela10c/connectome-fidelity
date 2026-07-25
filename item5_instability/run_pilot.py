#!/usr/bin/env python3
"""
run_pilot.py — paired Exp 5 pilot orchestrator (degree_preserving_swap + erdos_renyi).

WHY THIS EXISTS
---------------
production.py trains/evaluates ONE network per invocation (via --only_net N) and
handles its own stability gate (writes unstable_net{N}.flag on failure, saves
rdm_net{N}.npy on success). It does NOT decide how many networks to train, run
multiple schemes, or stop once enough stable nets are collected. This wrapper does
exactly that and nothing else — it shells out to production.py per network so all
the scientific logic (sampling, RSA, stability check, resume-bug workaround) stays
in the one audited file.

WHAT IT DOES
------------
For each scheme, it trains networks one at a time (net_idx = 0, 1, 2, ...), after
each one checks whether production.py produced an RDM (stable) or a .flag (unstable),
and keeps going until it has TARGET_STABLE stable RDMs OR hits MAX_ATTEMPTS trainings.
It prints a running stability tally after every network so you can watch the rate
accrue in real time (and abort early if instability is pathological). When a scheme
reaches its target, it runs production.py --aggregate_only for that scheme to compute
the ensemble RSA. Fully resumable: cached RDMs and existing flags are counted on
restart, so a killed run picks up where it left off.

IMPORTANT — this is the PILOT wrapper. It intentionally does NOT parallelize across
GPUs: on a single-GPU instance networks train sequentially. Sequential is correct for
a pilot (you want to watch stability accrue and be able to stop). Parallelizing across
many instances is a separate Phase-2 concern.

PRE-FLIGHT (added 2026-07-08)
-----------------------------
This wrapper now imports exp5_preflight and (1) hard-checks seed sufficiency before
each scheme's loop via require_enough_seeds() — replacing the old soft warning, so a
run cannot silently reuse seeds — and (2) clears each network's flyvis results dir via
clean_network_results_dir() immediately before training it, so the datamate
"incompatible config" collision (rc=1 at solver init) cannot recur on reruns/retries.
exp5_preflight.py must sit in the same dir as this script.

USAGE (single-GPU instance, corrected production.py already in place)
--------------------------------------------------------------------
    # generate connectomes for all seeds you'll use first (>= max_attempts):
    for s in $(seq 0 24); do
        python randomize_connectome_schemes.py fib25-fib19_v2.2.json \
            --seed $s --outdir ./rand_out
    done

    # smoke-test the wrapper on 1 stable net each at low iters FIRST (minutes, ~free):
    python run_pilot.py --target 1 --max_attempts 2 --n_iters 2000 --smoke

    # then the real paired pilot (n=10 stable each, full recipe):
    python run_pilot.py --target 10 --max_attempts 25 --n_iters 250000

    # run it under nohup so it survives disconnects:
    nohup python run_pilot.py --target 10 --max_attempts 25 --n_iters 250000 \
        > pilot.log 2>&1 &
    tail -f pilot.log

NOTES
-----
* Each scheme writes to its own out_dir (./pilot_out/<scheme>/) so RDMs never mix.
* net_idx maps to seed = net_idx % n_seeds (production.py's own convention), so you
  need enough distinct seed connectomes generated to cover MAX_ATTEMPTS. The
  pre-flight require_enough_seeds() check now HARD-STOPS on this rather than warning.
* "stable rate" printed here is stable / trained-so-far — your first real estimate
  of the wiring-null instability rate, which the single overnight net could not give.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

# --- pre-flight helpers (seed sufficiency + results-dir cleanup) -------------
# exp5_preflight.py must be in the same directory as this script.
from exp5_preflight import require_enough_seeds, clean_network_results_dir

SCHEMES = ["degree_preserving_swap", "erdos_renyi"]

# --- resolve interpreter and production.py path relative to this file ---
HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTION = os.path.join(HERE, "production.py")
PYTHON = sys.executable


def count_state(out_dir):
    """Return (stable_indices, unstable_indices) from what's on disk for a scheme."""
    stable = sorted(int(os.path.basename(p)[len("rdm_net"):-len(".npy")])
                    for p in glob.glob(os.path.join(out_dir, "rdm_net*.npy")))
    unstable = sorted(int(os.path.basename(p)[len("unstable_net"):-len(".flag")])
                      for p in glob.glob(os.path.join(out_dir, "unstable_net*.flag")))
    return stable, unstable


def run_one_network(scheme, net_idx, connectome_dir, out_dir, n_iters, n_perm):
    """Invoke production.py for a single network. Returns the subprocess returncode."""
    # PRE-FLIGHT: clear this network's flyvis results dir so the datamate
    # "incompatible config" collision cannot fire at solver init (and the
    # network trains fresh on the resume=false path, dodging bug #23).
    clean_network_results_dir(scheme, net_idx)

    cmd = [
        PYTHON, PRODUCTION,
        "--scheme", scheme,
        "--only_net", str(net_idx),
        "--n_networks", str(net_idx + 1),   # so seeds list is long enough for wrap
        "--connectome_dir", connectome_dir,
        "--out_dir", out_dir,
        "--n_iters", str(n_iters),
        "--n_perm", str(n_perm),
    ]
    print(f"    >>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    print(f"    <<< net {net_idx} returncode={rc} wall={dt/3600:.2f}h", flush=True)
    return rc


def aggregate(scheme, connectome_dir, out_dir, n_iters, n_perm, target):
    cmd = [
        PYTHON, PRODUCTION,
        "--scheme", scheme,
        "--aggregate_only",
        "--n_networks", str(target),
        "--connectome_dir", connectome_dir,
        "--out_dir", out_dir,
        "--n_iters", str(n_iters),
        "--n_perm", str(n_perm),
    ]
    print(f"  [aggregate {scheme}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd)


def n_seeds_available(scheme, connectome_dir):
    return len(glob.glob(os.path.join(connectome_dir, f"fib25_{scheme}_seed*.json")))


def run_scheme(scheme, args):
    out_dir = os.path.join(args.out_root, scheme)
    os.makedirs(out_dir, exist_ok=True)

    # PRE-FLIGHT: hard-stop unless enough distinct seed connectomes exist for this
    # scheme to cover max_attempts (so retries are genuine draws, not exact repeats).
    # Replaces the previous soft warning.
    require_enough_seeds(scheme, args.target, args.connectome_dir,
                         max_attempts=args.max_attempts, hard=True)

    print(f"\n{'='*70}\n[{scheme}] target={args.target} stable, "
          f"max_attempts={args.max_attempts}, n_iters={args.n_iters}\n{'='*70}", flush=True)

    net_idx = 0
    while True:
        stable, unstable = count_state(out_dir)
        n_trained = len(stable) + len(unstable)

        if len(stable) >= args.target:
            print(f"[{scheme}] TARGET REACHED: {len(stable)} stable "
                  f"({len(unstable)} unstable excluded, {n_trained} trained). "
                  f"stability rate = {len(stable)}/{n_trained} "
                  f"= {100*len(stable)/max(n_trained,1):.0f}%", flush=True)
            break
        if n_trained >= args.max_attempts:
            print(f"[{scheme}] STOP: hit max_attempts={args.max_attempts} with only "
                  f"{len(stable)}/{args.target} stable ({len(unstable)} unstable). "
                  f"stability rate = {len(stable)}/{n_trained} "
                  f"= {100*len(stable)/max(n_trained,1):.0f}%. "
                  f"Instability is high for this scheme — reconsider before scaling.",
                  flush=True)
            break

        # advance net_idx past any already-done indices (resumable)
        rdm_path = os.path.join(out_dir, f"rdm_net{net_idx:03d}.npy")
        flag_path = os.path.join(out_dir, f"unstable_net{net_idx:03d}.flag")
        if os.path.exists(rdm_path) or os.path.exists(flag_path):
            net_idx += 1
            continue

        print(f"\n[{scheme}] --- net {net_idx} --- "
              f"(so far: {len(stable)} stable / {n_trained} trained, "
              f"need {args.target}) ---", flush=True)
        rc = run_one_network(scheme, net_idx, args.connectome_dir, out_dir,
                             args.n_iters, args.n_perm)
        if rc != 0:
            print(f"[{scheme}] net {net_idx} production.py exited non-zero (rc={rc}). "
                  f"This is a TRAINING/PIPELINE failure, not a stability exclusion "
                  f"(those are handled cleanly by production.py). Investigate before "
                  f"continuing — stopping this scheme to avoid burning compute on a "
                  f"systematic error.", flush=True)
            break
        net_idx += 1

    # final tally + aggregate if we have any stable nets
    stable, unstable = count_state(out_dir)
    n_trained = len(stable) + len(unstable)
    print(f"\n[{scheme}] FINAL: {len(stable)} stable, {len(unstable)} unstable, "
          f"{n_trained} trained, rate={100*len(stable)/max(n_trained,1):.0f}%", flush=True)
    if len(stable) >= 1:
        aggregate(scheme, args.connectome_dir, out_dir, args.n_iters, args.n_perm,
                  args.target)
        # surface the result inline
        rj = os.path.join(out_dir, "exp5_result.json")
        if os.path.exists(rj):
            with open(rj) as f:
                r = json.load(f)
            print(f"[{scheme}] RESULT: r_trained_random_vs_bio="
                  f"{r.get('r_trained_random_vs_bio'):.4f} "
                  f"p_perm={r.get('p_perm')} "
                  f"(n_stable={r.get('n_stable')}/{r.get('n_trained')} trained)",
                  flush=True)
    else:
        print(f"[{scheme}] no stable networks — nothing to aggregate.", flush=True)
    return dict(scheme=scheme, n_stable=len(stable), n_unstable=len(unstable),
                n_trained=n_trained)


def main():
    ap = argparse.ArgumentParser(description="Paired Exp 5 pilot orchestrator.")
    ap.add_argument("--schemes", nargs="+", default=SCHEMES,
                    help="Schemes to run (default: degree_preserving_swap erdos_renyi).")
    ap.add_argument("--target", type=int, default=10,
                    help="Stable networks wanted per scheme (default 10).")
    ap.add_argument("--max_attempts", type=int, default=25,
                    help="Max trainings per scheme before giving up (default 25). "
                         "Sizing: at ~70%% instability you'd need ~33 to get 10 "
                         "stable; set with the measured rate in mind.")
    ap.add_argument("--n_iters", type=int, default=250000,
                    help="250000 = full recipe. Use 2000 for --smoke only.")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--connectome_dir", default="./rand_out")
    ap.add_argument("--out_root", default="./pilot_out")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke-test mode: forces tiny targets to check the wrapper "
                         "and pipeline run end-to-end. Results are NOT scientific.")
    args = ap.parse_args()

    if not os.path.exists(PRODUCTION):
        sys.exit(f"production.py not found next to this script at {PRODUCTION}. "
                 f"Put run_pilot.py in the same dir as the CORRECTED production.py.")

    if args.smoke:
        print("[SMOKE] wrapper/pipeline check only — results NOT valid. "
              "Forcing target=1, max_attempts=2, n_iters=min(n_iters,2000).")
        args.target = 1
        args.max_attempts = 2
        args.n_iters = min(args.n_iters, 2000)

    if args.n_iters < 250000 and not args.smoke:
        print(f"[!] n_iters={args.n_iters} < 250000 — results are NOT the full recipe "
              f"and are not scientifically valid. Use only for testing.", flush=True)

    print(f"paired pilot: schemes={args.schemes} target={args.target} "
          f"max_attempts={args.max_attempts} n_iters={args.n_iters} "
          f"connectome_dir={args.connectome_dir} out_root={args.out_root}", flush=True)

    summary = []
    t0 = time.time()
    for scheme in args.schemes:
        summary.append(run_scheme(scheme, args))

    print(f"\n{'#'*70}\nPAIRED PILOT COMPLETE (wall {(time.time()-t0)/3600:.2f}h)\n{'#'*70}")
    for s in summary:
        rate = 100 * s["n_stable"] / max(s["n_trained"], 1)
        print(f"  {s['scheme']:28s} {s['n_stable']:2d} stable / {s['n_trained']:2d} "
              f"trained ({rate:.0f}% stable)")
    print("\nNext: compare r_trained_random_vs_bio across the two schemes.")
    print("  - degree_preserving_swap is the STRINGENT null (your MICrONS z=1.30 tie).")
    print("  - erdos_renyi BREAKS degree. If ER drops well below swap, wiring-beyond-")
    print("    degree matters in the fly. If swap~=ER~=CC, degree suffices (mouse story).")
    print("  - The overnight n=1 swap gave r=0.72; watch whether n=10 holds or regresses.")
    print("  - Read r ONLY alongside the instability rates above (different nulls,")
    print("    different stability filters — the exclude-and-report caveat).")


if __name__ == "__main__":
    main()
