"""
signed_null_c2.py — signed E/I analog of 04_null_candidate2_distance_constrained.py.

WHY THIS EXISTS
---------------
C2 tests a different question than C1 or C3: does spatial proximity alone
explain the connection pattern, independent of degree (C1) or cell-type
statistics (C3)? Original mechanism: for each neuron, resample its output
targets by drawing from a distance-decay connection-probability profile
estimated from the real graph, rather than uniformly at random.

FUNDAMENTALLY DIFFERENT CHARACTER FROM C1/C3 -- READ BEFORE USING
-----------------------------------------------------------------------
C1 preserves individual degree exactly (swaps). C3 preserves cell-type
block counts exactly (resampling within a fixed count). C2 preserves
NEITHER exactly -- edge count and degree are only approximately matched
by construction, because target sampling caps at
min(real_out_degree_for_this_class, number_of_targets_with_nonzero_weight),
and because the probability profile itself is estimated from a random
sample, not computed exactly. The original script's own validation
reports "Edge count: real=X, random=Y (Z% of real)" -- this was never
expected to be 100%. Consequently, verify_c2() below does NOT do the
OK/FAIL exact checks signed_null_c1.py and signed_null_c3.py use --
it reports achieved-vs-target distance-profile agreement and
achieved-vs-real edge-count ratio as continuous diagnostics, and asks you
to judge whether the approximation is close enough, not a binary pass/fail.

DESIGN DECISION, STATED EXPLICITLY: FOUR SEPARATE PROFILES, NOT ONE
-------------------------------------------------------------------------
The original (E-only) script computes ONE distance-probability profile
for the whole graph. For the signed graph, this script computes FOUR
profiles separately -- one per (source_type, target_type) pair (E->E,
E->I, I->E, I->I) -- following the same principle already load-bearing
elsewhere in this project (assign_synapse_counts() in signed_null_c1.py,
C3's per-block synapse sampling): never pool statistics across classes
with different typical behavior. I->E connections are denser and shorter-
range on average than E->E in this dataset (see the block-density table
in signed_null_c1.py's diagnostics); a single pooled profile would blur
that difference and let it leak into what should be a purely spatial
null.

Each neuron's real out-edges are split by target type (E-target vs
I-target) BEFORE resampling; each sub-group's targets are independently
redrawn according to the matching class's spatial profile, preserving
that sub-group's count approximately, the same way the original preserves
total out-degree approximately.

SYNAPSE COUNTS: per (source_type, target_type) class, matching
signed_null_c1.py's granularity (not C3's finer per-cell-type-block
granularity, since C2's own defining structure operates at the E/I class
level, not the cell-type level).

INPUTS:
    signed_real_edges.json, node_types.json  (as signed_null_c1.py)
    node_data_v1_ei.csv                      (x, y, z columns, NEW)

USAGE:
    python signed_null_c2.py                 # seed 0, verify, write
    python signed_null_c2.py --seed 3
    python signed_null_c2.py --n_seeds 10
    python signed_null_c2.py --verify_only
"""

import argparse
import sys

import numpy as np
import pandas as pd

from signed_null_c1 import load_signed, graph_stats, write_null

NODE_CSV = "node_data_v1_ei.csv"
N_BINS = 20
N_SAMPLE_PAIRS = 80_000   # matches the original script's sample size


def load_positions(ids, path=NODE_CSV):
    """Return (N,3) array of x,y,z in the same nucleus_id order as `ids`."""
    df = pd.read_csv(path)
    nuc_to_pos = {row.nucleus_id: (row.x, row.y, row.z) for row in df.itertuples()}
    missing = [nid for nid in ids if nid not in nuc_to_pos]
    if missing:
        raise ValueError(
            f"{len(missing)} nucleus_ids have no position in {path} -- "
            f"e.g. {missing[:5]}")
    return np.array([nuc_to_pos[nid] for nid in ids], dtype=float)


def compute_distance_profile(A_real, pos, src_idx, tgt_idx, seed,
                             n_bins=N_BINS, n_samples=N_SAMPLE_PAIRS,
                             verbose=True):
    """Distance-connection-probability profile for ONE (source_type,
    target_type) class, estimated the same way the original script does:
    random-sample candidate pairs, bin by distance, compute empirical
    connection probability per bin. Returns (bin_edges, conn_prob).
    """
    rng = np.random.default_rng(seed)
    sample_src = rng.choice(src_idx, size=n_samples, replace=True)
    sample_tgt = rng.choice(tgt_idx, size=n_samples, replace=True)
    mask_self = sample_src != sample_tgt
    sample_src, sample_tgt = sample_src[mask_self], sample_tgt[mask_self]

    dists = np.linalg.norm(pos[sample_src] - pos[sample_tgt], axis=1)
    connected = A_real[sample_src, sample_tgt] > 0

    bin_edges = np.linspace(0, np.percentile(dists, 99), n_bins + 1)
    conn_prob = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (dists >= bin_edges[i]) & (dists < bin_edges[i + 1])
        n_in_bin = in_bin.sum()
        conn_prob[i] = connected[in_bin].sum() / n_in_bin if n_in_bin > 0 else 0.0
    if verbose:
        print(f"    profile: n_pairs_sampled={len(dists)}  "
              f"short-range p={conn_prob[0]:.4f}  long-range p={conn_prob[-1]:.4f}")
    return bin_edges, conn_prob


def make_c2_signed(A_real, syn_real, pos, types, seed, verbose=True):
    """For each of the 4 (source_type, target_type) classes: compute that
    class's distance-probability profile, then for each source neuron in
    that class, resample its same-class out-edge targets according to the
    profile, preserving that sub-group's count approximately (capped at
    however many candidate targets have nonzero weight, same as the
    original script's k_actual logic).
    """
    rng = np.random.default_rng(seed)
    N = A_real.shape[0]
    is_I = (types == "I")
    A_out = np.zeros_like(A_real)
    syn_out = np.zeros_like(syn_real)
    profiles = {}
    edge_count_report = []

    for src_I in (False, True):
        src_idx = np.where(is_I == src_I)[0]
        for tgt_I in (False, True):
            tgt_idx = np.where(is_I == tgt_I)[0]
            label = f"{'I' if src_I else 'E'}->{'I' if tgt_I else 'E'}"
            real_count = int(A_real[np.ix_(src_idx, tgt_idx)].sum())
            if real_count == 0:
                continue

            if verbose:
                print(f"  [{label}] computing distance profile...")
            bin_edges, conn_prob = compute_distance_profile(
                A_real, pos, src_idx, tgt_idx, seed=seed, verbose=verbose)
            profiles[label] = (bin_edges, conn_prob)

            syn_block_real = syn_real[np.ix_(src_idx, tgt_idx)]
            syn_pool = syn_block_real[syn_block_real > 0].ravel()

            achieved_this_class = 0
            for pi in src_idx:
                # this source neuron's REAL out-edges within this class
                real_out_mask = A_real[pi, tgt_idx] > 0
                k = int(real_out_mask.sum())
                if k == 0:
                    continue

                d = np.linalg.norm(pos[tgt_idx] - pos[pi], axis=1)
                bin_idx = np.clip(np.digitize(d, bin_edges) - 1, 0, N_BINS - 1)
                weights = conn_prob[bin_idx].copy()
                # zero out self (only relevant when src_I == tgt_I and pi in tgt_idx)
                self_pos = np.where(tgt_idx == pi)[0]
                if len(self_pos):
                    weights[self_pos[0]] = 0.0
                total_w = weights.sum()
                if total_w == 0:
                    continue
                probs = weights / total_w

                k_actual = min(k, int((weights > 0).sum()))
                if k_actual == 0:
                    continue
                chosen_local = rng.choice(len(tgt_idx), size=k_actual,
                                          replace=False, p=probs)
                chosen_global = tgt_idx[chosen_local]

                syn_vals = rng.choice(
                    syn_pool, size=k_actual,
                    replace=(k_actual > len(syn_pool)))
                A_out[pi, chosen_global] = 1.0
                syn_out[pi, chosen_global] = syn_vals
                achieved_this_class += k_actual

            edge_count_report.append((label, real_count, achieved_this_class))
            if verbose:
                print(f"    [{label}] real={real_count}  achieved={achieved_this_class}  "
                      f"({100*achieved_this_class/real_count:.1f}% of real)")

    return A_out, syn_out, profiles, edge_count_report


def verify_c2(A_real, A_out, syn_real, syn_out, edge_count_report, types):
    """Distributional diagnostics, NOT exact OK/FAIL invariant checks --
    C2 does not preserve degree or edge count exactly by design. Reports
    continuous agreement measures; judgment on "close enough" is left to
    whoever is reading this, same as the original script's own validation
    plot did (never asserted exact match).
    """
    print("\n  Diagnostics (NOT exact invariants -- C2 is approximate by design):")
    total_real = sum(r for _, r, _ in edge_count_report)
    total_out = sum(a for _, _, a in edge_count_report)
    print(f"    Overall edge count: real={total_real}  null={total_out}  "
          f"({100*total_out/total_real:.1f}% of real)")
    for label, real_count, achieved in edge_count_report:
        print(f"    [{label}] {achieved}/{real_count} "
              f"({100*achieved/real_count:.1f}%)")

    overlap = int((A_real.astype(bool) & A_out.astype(bool)).sum())
    print(f"    Edge overlap real ∩ null: {overlap} "
          f"({100*overlap/int(A_real.sum()):.2f}% of real edges)")

    check(int(A_out.diagonal().sum()) == 0, "no self-loops")
    print(f"    total synapse count: real={int(syn_real.sum())}  "
          f"null={int(syn_out.sum())} (approximate by design, not rescaled)")

    print("\n  NOTE: no pass/fail verdict is printed here on purpose -- "
          "compare the per-class edge-count percentages and overlap above "
          "against what you consider an acceptable approximation before "
          "trusting this null, the same judgment the original script's "
          "validation plot required rather than asserted.")


def check(cond, msg):
    print(f"    {'OK  ' if cond else 'FAIL'} {msg}")


def main():
    ap = argparse.ArgumentParser(description="Signed distance-constrained null (C2).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()

    print("Loading signed graph ...")
    A, syn, types, ids, _ = load_signed()
    graph_stats(A, types, "real")
    print(f"  [real] synapses={int(syn.sum())}")

    print("Loading positions ...")
    pos = load_positions(ids)
    print(f"  x range: {pos[:,0].min()/1000:.1f}..{pos[:,0].max()/1000:.1f} um")
    print(f"  y range: {pos[:,1].min()/1000:.1f}..{pos[:,1].max()/1000:.1f} um")
    print(f"  z range: {pos[:,2].min()/1000:.1f}..{pos[:,2].max()/1000:.1f} um")

    if args.verify_only:
        print("\n--verify_only: real graph and positions loaded. No null generated.")
        return

    for s in range(args.seed, args.seed + args.n_seeds):
        print(f"\n=== C2-signed null, seed {s} ===")
        A_c2, syn_c2, profiles, report = make_c2_signed(A, syn, pos, types, seed=s)
        graph_stats(A_c2, types, "null")
        verify_c2(A, A_c2, syn, syn_c2, report, types)
        if not args.no_write:
            write_null(A_c2, syn_c2, ids, f"null_edges_C2signed_seed{s}.json")


if __name__ == "__main__":
    main()
