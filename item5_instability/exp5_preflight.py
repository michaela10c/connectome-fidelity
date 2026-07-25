"""
exp5_preflight.py

Pre-flight helpers for the Experiment 5 (trained-random) ensemble orchestration.

Fixes two concrete failures observed in the paired-pilot run on 2026-07-08:

  1. CONFIG-COLLISION CRASH (the rc=1 at ~36s):
     flyvis/datamate aborts at solver init when a results directory
     `flow/exp5_<scheme>/<NNNN>` already exists with a config that does not
     match the new run (the "incompatible config" warning immediately preceding
     the crash). Leftover dirs from earlier runs poison fresh networks.
     FIX: remove the specific network results dir BEFORE training it, so every
     network trains truly fresh (resume=false path, which also dodges the
     known resolve_checkpoints() resume bug, flyvis issue #23).

  2. SEED REUSE (the "only 20 distinct seeds but max_attempts=25" warning):
     training is deterministic, so network index i reuses seed connectome
     `fib25_<scheme>_seed{i}.json`. If you have fewer distinct seed files than
     target networks, indices beyond the seed count are EXACT REPEATS, not new
     draws — a fake ensemble. FIX: refuse to start unless enough distinct seed
     connectomes exist for the requested target.

Neither helper needs a GPU. Import and call from the orchestrator, or run this
file directly to check/repair state before a run.

USAGE (from the orchestrator, per scheme, before the per-network loop):

    from exp5_preflight import require_enough_seeds, clean_network_results_dir

    # once per scheme, before the loop:
    require_enough_seeds(scheme, target=10, connectome_dir="./rand_out")

    # inside the loop, before training network `net_idx`:
    clean_network_results_dir(scheme, net_idx)

USAGE (standalone, to inspect/repair before spinning the instance back up):

    python exp5_preflight.py --scheme degree_preserving_swap --target 10 \
        --connectome_dir ./rand_out --clean
"""

import argparse
import glob
import os
import shutil
import sys


# ---------------------------------------------------------------------------
# Locate the flyvis results root. flyvis stores trained networks under
#   <flyvis_pkg>/data/results/flow/<network_name>/<NNNN>/
# We resolve it from the installed package rather than hardcoding the venv path,
# so this keeps working if the venv path changes across instances.
# ---------------------------------------------------------------------------
def flyvis_results_root():
    """Return absolute path to <flyvis>/data/results, or None if unresolvable."""
    try:
        import flyvis  # noqa: F401
        pkg_dir = os.path.dirname(flyvis.__file__)
    except Exception as e:  # flyvis not importable (e.g. off-instance) -> best effort
        print(f"[preflight] WARNING: could not import flyvis ({e}); "
              f"falling back to None. Results-dir cleanup will be skipped.",
              file=sys.stderr)
        return None
    return os.path.join(pkg_dir, "data", "results")


def network_results_dir(scheme, net_idx, results_root=None):
    """Path to the flyvis results dir for one Exp5 network."""
    if results_root is None:
        results_root = flyvis_results_root()
    if results_root is None:
        return None
    network_name = f"exp5_{scheme}/{net_idx:04d}"
    return os.path.join(results_root, "flow", network_name)


# ---------------------------------------------------------------------------
# FIX 1: clear a network's stale results dir before training it.
# ---------------------------------------------------------------------------
def clean_network_results_dir(scheme, net_idx, results_root=None, verbose=True):
    """
    Remove the flyvis results dir for (scheme, net_idx) if it exists, so the
    network trains fresh and cannot hit the datamate config-collision crash or
    the resume-bug path.

    Returns True if a dir was removed, False if nothing was there (or path
    unresolvable). Safe to call unconditionally before every training attempt.
    """
    d = network_results_dir(scheme, net_idx, results_root)
    if d is None:
        return False
    if os.path.isdir(d):
        shutil.rmtree(d)
        if verbose:
            print(f"[preflight] cleared stale results dir: {d}")
        return True
    return False


def clean_scheme_results(scheme, results_root=None, verbose=True):
    """
    Remove ALL results dirs for a scheme (flow/exp5_<scheme>/*). Use this once
    before a fresh ensemble run for a scheme if you want a clean slate, rather
    than per-network. Returns the number of dirs removed.
    """
    if results_root is None:
        results_root = flyvis_results_root()
    if results_root is None:
        return 0
    base = os.path.join(results_root, "flow", f"exp5_{scheme}")
    if not os.path.isdir(base):
        return 0
    n = 0
    for child in glob.glob(os.path.join(base, "*")):
        if os.path.isdir(child):
            shutil.rmtree(child)
            n += 1
    if verbose:
        print(f"[preflight] cleared {n} network dir(s) under {base}")
    return n


# ---------------------------------------------------------------------------
# FIX 2: refuse to run more networks than we have distinct seed connectomes.
# ---------------------------------------------------------------------------
def count_seed_connectomes(scheme, connectome_dir):
    """
    Count distinct seed connectome files for a scheme:
      <connectome_dir>/fib25_<scheme>_seed<N>.json
    Returns (count, sorted_seed_indices).
    """
    pattern = os.path.join(connectome_dir, f"fib25_{scheme}_seed*.json")
    seeds = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        # fib25_<scheme>_seed<N>.json  -> extract N
        try:
            n = int(base.rsplit("seed", 1)[1].split(".json")[0])
            seeds.append(n)
        except (IndexError, ValueError):
            continue
    seeds = sorted(set(seeds))
    return len(seeds), seeds


def require_enough_seeds(scheme, target, connectome_dir, max_attempts=None,
                         hard=True):
    """
    Ensure at least `target` distinct seed connectomes exist for `scheme`
    (and, if given, at least `max_attempts`, since retries also consume seeds).

    If `hard` is True, raises SystemExit when there are too few — so the
    orchestrator stops BEFORE spending GPU time on an ensemble that would
    silently reuse seeds (exact-repeat networks = fake ensemble).

    Returns (ok: bool, count: int, needed: int).
    """
    needed = target if max_attempts is None else max(target, max_attempts)
    count, seeds = count_seed_connectomes(scheme, connectome_dir)
    ok = count >= needed
    if not ok:
        msg = (
            f"[preflight] INSUFFICIENT SEEDS for scheme '{scheme}': "
            f"found {count} distinct seed connectome(s) "
            f"({seeds if seeds else 'none'}) in {connectome_dir}, "
            f"but need >= {needed} "
            f"(target={target}"
            + (f", max_attempts={max_attempts}" if max_attempts else "")
            + "). Networks beyond the seed count would REUSE seeds and "
            "reproduce identical results (deterministic training) — a fake "
            "ensemble, not genuine resampling.\n"
            f"[preflight] FIX: generate seeds 0..{needed-1}, e.g.\n"
            f"    for s in $(seq 0 {needed-1}); do \\\n"
            f"        python randomize_connectome_schemes.py "
            f"fib25-fib19_v2.2.json --seed $s --outdir {connectome_dir}; \\\n"
            f"    done\n"
            f"  (or add a --seeds/--n_seeds loop to the randomizer so it "
            f"writes all seeds in one call)."
        )
        print(msg, file=sys.stderr)
        if hard:
            raise SystemExit(2)
    else:
        print(f"[preflight] seed check OK for '{scheme}': "
              f"{count} distinct seeds >= {needed} needed.")
    return ok, count, needed


# ---------------------------------------------------------------------------
# Standalone entry point: inspect (and optionally repair) state before a run.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pre-flight check/repair for the Exp5 ensemble.")
    ap.add_argument("--scheme", required=True,
                    help="e.g. degree_preserving_swap, erdos_renyi, rf_shuffle")
    ap.add_argument("--target", type=int, default=10,
                    help="target number of STABLE networks for the ensemble")
    ap.add_argument("--max_attempts", type=int, default=None,
                    help="max training attempts (retries also consume seeds)")
    ap.add_argument("--connectome_dir", default="./rand_out")
    ap.add_argument("--clean", action="store_true",
                    help="remove ALL flow/exp5_<scheme>/* results dirs "
                         "(clean slate for this scheme)")
    args = ap.parse_args()

    print(f"[preflight] scheme={args.scheme} target={args.target} "
          f"max_attempts={args.max_attempts} connectome_dir={args.connectome_dir}")

    # 1. Seed sufficiency (non-hard here so we can also do cleanup and report all).
    ok, count, needed = require_enough_seeds(
        args.scheme, args.target, args.connectome_dir,
        max_attempts=args.max_attempts, hard=False)

    # 2. Report / optionally clear stale results dirs.
    root = flyvis_results_root()
    if root is None:
        print("[preflight] flyvis results root unresolvable "
              "(flyvis not importable here) — skipping results-dir inspection. "
              "Run this ON the instance to inspect/clear stale dirs.")
    else:
        base = os.path.join(root, "flow", f"exp5_{args.scheme}")
        existing = sorted(glob.glob(os.path.join(base, "*"))) if os.path.isdir(base) else []
        print(f"[preflight] existing results dirs for '{args.scheme}': "
              f"{len(existing)}")
        for d in existing:
            print(f"    {d}")
        if args.clean and existing:
            n = clean_scheme_results(args.scheme, results_root=root)
            print(f"[preflight] --clean removed {n} dir(s).")
        elif existing and not args.clean:
            print("[preflight] NOTE: these stale dirs will cause the "
                  "datamate config-collision crash on rerun. Re-run with "
                  "--clean, or ensure the orchestrator calls "
                  "clean_network_results_dir() before each network.")

    # Exit non-zero if seeds are insufficient, so this can gate a shell pipeline.
    if not ok:
        print("[preflight] RESULT: NOT READY (insufficient seeds).")
        sys.exit(2)
    print("[preflight] RESULT: ready (seeds sufficient; ensure results-dir "
          "cleanup is wired into the per-network loop).")


if __name__ == "__main__":
    main()
