#!/usr/bin/env python3
"""
eigenspectrum_stability.py

Spectral-radius stability analysis for the connectome vs. its randomization nulls.

WHAT THIS DOES (and, importantly, what it does NOT do):

Makes "the real connectome sits in a stable region" a COMPUTABLE statement — in
the spectral-radius sense. For a linear/linearized recurrent operator x_{t+1} = W x_t,
dynamics are stable iff the spectral radius rho(W) = max|eigenvalue(W)| < 1 (discrete
time) or max Re(eigenvalue) < 0 (continuous time). We compute rho(W) for the real
connectome's cell-type connectivity matrix and for each randomization scheme, and ask
whether the empirical instability the trained-random baselines showed (66-80% unstable
in Exp 1-2) tracks a larger spectral radius than the real connectome's.

This is the HONEST, well-defined version of the "is the connectome in a stable region?"
question raised for the Choi meeting. It is NOT the same as placing the connectome in
Choi's mean-field sigma-parameterized stability space: that would require fitting her
distance-dependent Gaussian projection-width formalism (sigma_e, sigma_p, sigma_s) to an
irregular measured connectome, which is a cross-formalism modeling step that may not be
well-defined (a real connectome has no intrinsic Gaussian projection width). This script
deliberately stays in the eigenspectrum formalism, where "stable region" is unambiguous
(eigenvalues of a matrix are unique), and does not claim to compute her sigma-space
placement. Any bridge to her framework is a conversation for the meeting, not a claim here.

CAVEATS (state these if the numbers are shown to anyone):
  [1] W here is the 65x65 CELL-TYPE connectivity operator (signed, optionally
      synapse-count-weighted), NOT the full neuron-by-neuron Jacobian of the trained
      Flyvis network. So rho(W) is a property of the WIRING operator, not of the trained
      nonlinear dynamical system. It is the right object for "does wiring structure sit
      in a stable regime," but it is a linear proxy, not the trained network's true
      stability (which depends on learned gains, time constants, nonlinearity).
  [2] Flyvis is graded-potential with per-edge time constants and lambda_mult scaling;
      a one-matrix linear operator is a simplification. Treat rho(W) as a structural
      indicator, not a substitute for the forward-pass is_stable() check the production
      pipeline already applies to trained nets.
  [3] The absolute threshold (rho<1) is formalism-dependent; what is robust is the
      RELATIVE comparison (real vs. nulls) under one consistent construction.

USAGE:
  python eigenspectrum_stability.py CONNECTOME.json \
      [--rand_dir ./rand_out] [--schemes degree_preserving_swap,erdos_renyi] \
      [--seeds 0-9] [--weight signed_count|sign_only] [--out eig_out]
"""
import os, json, argparse
import numpy as np


def load_spec(path):
    with open(path) as f:
        return json.load(f)


def build_W(spec, weight="signed_count"):
    """Build the cell-type connectivity operator W (n_types x n_types).

    Row = target, col = source, so (W x)_target = sum_source W[target,source] x[source]
    — standard recurrent convention x_{t+1} = W x_t.

    weight:
      'signed_count' : W[t,s] = alpha * (total synapses on that src->tar edge)
                       (matches the synapse-count-proportional weighting the production
                        run uses; the biologically weighted operator)
      'sign_only'    : W[t,s] = alpha (unit-magnitude signed adjacency; isolates
                        topology+sign from synapse-count magnitude)
    """
    edges = spec["edges"]
    nodes = sorted(set([e["src"] for e in edges] + [e["tar"] for e in edges]))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n), dtype=float)
    for e in edges:
        s, t = idx[e["src"]], idx[e["tar"]]
        alpha = float(e.get("alpha", 1))
        if weight == "sign_only":
            w = alpha
        else:  # signed_count
            nsyn = sum(off[1] for off in e["offsets"]) if e.get("offsets") else 1.0
            w = alpha * float(nsyn)
        W[t, s] += w   # accumulate (a cell-type pair is one edge here, but be safe)
    return W, nodes


def spectral_stats(W):
    """Eigenvalues and stability-relevant summaries of W."""
    eig = np.linalg.eigvals(W)
    rho = float(np.max(np.abs(eig)))           # spectral radius (discrete-time stability: <1)
    max_re = float(np.max(eig.real))           # continuous-time stability: <0
    # abscissa/gap style summaries
    return {
        "spectral_radius": rho,
        "max_real_part": max_re,
        "n_eigs_mag_gt_1": int(np.sum(np.abs(eig) > 1.0)),
        "n_eigs_re_gt_0": int(np.sum(eig.real > 0.0)),
        "eigs": eig,
    }


def summarize(name, W, verbose=True):
    s = spectral_stats(W)
    if verbose:
        print(f"  {name:28s} rho={s['spectral_radius']:.4f}  "
              f"max_Re={s['max_real_part']:+.4f}  "
              f"|eig|>1: {s['n_eigs_mag_gt_1']:3d}  Re>0: {s['n_eigs_re_gt_0']:3d}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("connectome", help="real connectome JSON")
    ap.add_argument("--rand_dir", default=None,
                    help="dir with randomized JSONs (fib25_<scheme>_seed<k>.json)")
    ap.add_argument("--schemes", default="degree_preserving_swap,sign_preserving_target_perm,erdos_renyi",
                    help="comma-separated scheme names to compare (must exist in rand_dir)")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--weight", default="signed_count",
                    choices=["signed_count", "sign_only"])
    ap.add_argument("--out", default="eig_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def parse_seeds(x):
        if "-" in x:
            a, b = x.split("-"); return list(range(int(a), int(b) + 1))
        return [int(v) for v in x.split(",")]
    seeds = parse_seeds(args.seeds)

    print(f"weighting: {args.weight}")
    print("=" * 74)
    print("SPECTRAL-RADIUS STABILITY (structural / linear proxy — see caveats in header)")
    print("=" * 74)

    real = load_spec(args.connectome)
    W_real, nodes = build_W(real, args.weight)
    print(f"\nreal connectome: n_types={len(nodes)}  n_edges={len(real['edges'])}")
    real_stats = summarize("REAL", W_real)
    np.save(os.path.join(args.out, "eigs_real.npy"), real_stats["eigs"])

    results = {"real": {k: v for k, v in real_stats.items() if k != "eigs"},
               "weight": args.weight, "n_types": len(nodes)}

    if args.rand_dir:
        for scheme in [s.strip() for s in args.schemes.split(",")]:
            print(f"\nscheme: {scheme}")
            rhos, maxres, found = [], [], 0
            for k in seeds:
                p = os.path.join(args.rand_dir, f"fib25_{scheme}_seed{k}.json")
                if not os.path.exists(p):
                    continue
                found += 1
                Wk, _ = build_W(load_spec(p), args.weight)
                st = summarize(f"  seed{k}", Wk, verbose=(found <= 3))
                rhos.append(st["spectral_radius"]); maxres.append(st["max_real_part"])
            if not rhos:
                print(f"    (no files found for {scheme} in {args.rand_dir}; skipping)")
                continue
            rhos = np.array(rhos); maxres = np.array(maxres)
            # where does the REAL connectome sit relative to the null distribution?
            frac_below = float(np.mean(rhos <= real_stats["spectral_radius"]))
            z = ((real_stats["spectral_radius"] - rhos.mean()) / rhos.std()
                 if rhos.std() > 0 else float("nan"))
            print(f"    --- {scheme}: n={len(rhos)} nulls ---")
            print(f"    null rho: mean={rhos.mean():.4f} sd={rhos.std():.4f} "
                  f"range=[{rhos.min():.4f},{rhos.max():.4f}]")
            print(f"    REAL rho={real_stats['spectral_radius']:.4f}  "
                  f"-> z={z:+.2f}; {100*frac_below:.0f}% of nulls have rho >= real")
            results[scheme] = {
                "n_nulls": len(rhos),
                "null_rho_mean": float(rhos.mean()),
                "null_rho_sd": float(rhos.std()),
                "null_rho_min": float(rhos.min()),
                "null_rho_max": float(rhos.max()),
                "real_rho": real_stats["spectral_radius"],
                "real_vs_null_z": (float(z) if z == z else None),
                "frac_nulls_rho_ge_real": frac_below,
            }

    with open(os.path.join(args.out, "eigenspectrum_stability.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {os.path.join(args.out, 'eigenspectrum_stability.json')}")
    print("\nINTERPRETATION (honest):")
    print("  If REAL rho is LOWER than the null distribution (real sits below), that is")
    print("  structural evidence the connectome's wiring operator is in a more stable")
    print("  regime than random rewirings — the computable version of 'stable region,'")
    print("  in the spectral-radius sense ONLY. It does NOT place the connectome in")
    print("  Choi's sigma-space, and rho(W) is a linear proxy, not the trained network's")
    print("  true stability. Pair with the forward-pass is_stable rates from production.")


if __name__ == "__main__":
    main()
