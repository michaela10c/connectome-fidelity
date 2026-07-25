"""
signed_null_c3.py — signed E/I analog of 07_null_candidate3_celltype_shuffled.py.

WHY THIS EXISTS
---------------
The original C3 tests a fundamentally different question than C1: not
"does degree matter" but "does cell-type-level wiring statistics explain
the signal, or does specific partner identity matter beyond that". Its
mechanism is completely different from C1's Maslov-Sneppen double-edge
swap: for each ordered pair of cell-type groups, it preserves the EXACT
number of connections in that block, then randomly reassigns WHICH
specific neurons connect within it. Individual node degree is NOT
preserved (unlike C1) -- only the aggregate block-level count is.

This is extended to the signed graph using node_data_v1_ei.csv's
cell_type column, which spans BOTH excitatory subtypes (originally used:
4P, 23P, 6P-IT, 5P-IT, 6P-CT, 5P-ET, 5P-NP) and, newly, four inhibitory
subtypes (BC, MC, BPC, NGC) -- 11 groups total, up to 121 possible blocks.

DALE'S LAW IS FREE HERE, NOT ENFORCED
-----------------------------------------
A block is defined by (source cell_type, target cell_type). cell_type
never crosses the E/I boundary -- '23P' is always excitatory, 'BC' is
always inhibitory. Shuffling partners WITHIN a block can never change a
source's sign, so Dale's law holds automatically from the block structure
itself, unlike C1's nulls which needed explicit source-type restrictions
(signed_null_c1.py) or accepted E/I scrambling by design
(unrestricted_null_c1.py).

WHAT IS NOT PRESERVED, STATED EXPLICITLY (this is the point of C3, not a
flaw): individual node in-degree and out-degree are NOT preserved --only
the aggregate (g_pre, g_post) block count is. A node that happened to
receive many real connections from its cell-type peers could receive far
fewer (or more) in this null, as long as the block total stays fixed.
Verified directly below rather than assumed.

SYNAPSE COUNTS: PER-BLOCK, NOT REUSING signed_null_c1's PER-E/I-CLASS
--------------------------------------------------------------------------
The original C3 was binary-only (fed a structural RDM, not a weighted
simulation). Assigning synapse counts here per the coarser E/I-class
split (as C1's nulls do) would blur exactly the cell-type-level structure
C3 exists to test. Instead, synapse counts are sampled from each block's
OWN real synapse-count distribution -- matching C3's own granularity, and
following the same "never mix distributions across classes with
different typical counts" principle signed_null_c1.py's
assign_synapse_counts() already established, just applied at the finer
block level C3 actually operates at.

INPUTS:
    signed_real_edges.json, node_types.json  (as signed_null_c1.py)
    node_data_v1_ei.csv                      (cell_type column, NEW)

USAGE:
    python signed_null_c3.py                 # seed 0, verify, write
    python signed_null_c3.py --seed 3
    python signed_null_c3.py --n_seeds 10
    python signed_null_c3.py --verify_only
"""

import argparse
import sys
import json

import numpy as np
import pandas as pd

from signed_null_c1 import load_signed, graph_stats, write_null

NODE_CSV = "node_data_v1_ei.csv"


def load_cell_types(ids, path=NODE_CSV):
    """Return a (N,) array of cell_type strings, in the same nucleus_id
    order as `ids` (the order load_signed() uses)."""
    df = pd.read_csv(path)
    nuc_to_ct = dict(zip(df["nucleus_id"], df["cell_type"]))
    missing = [nid for nid in ids if nid not in nuc_to_ct]
    if missing:
        raise ValueError(
            f"{len(missing)} nucleus_ids from the signed graph have no "
            f"cell_type in {path} -- e.g. {missing[:5]}. Cannot build "
            f"cell-type blocks with incomplete labels.")
    return np.array([nuc_to_ct[nid] for nid in ids])


def make_c3_signed(A_real, syn_real, cell_types, seed, verbose=True):
    """Block-count-preserving resample: for each (g_pre, g_post) cell-type
    pair, preserve the exact real connection count, reassign which
    specific neuron pairs realize it. Synapse counts sampled per-block
    from that block's own real distribution -- see module docstring.
    """
    rng = np.random.default_rng(seed)
    N = A_real.shape[0]
    unique_groups = sorted(set(cell_types))
    group_idx = {g: np.where(cell_types == g)[0] for g in unique_groups}

    A_out = np.zeros_like(A_real)
    syn_out = np.zeros_like(syn_real)
    block_report = []

    for g_pre in unique_groups:
        for g_post in unique_groups:
            pre_idx = group_idx[g_pre]
            post_idx = group_idx[g_post]
            block_real = A_real[np.ix_(pre_idx, post_idx)]
            if g_pre == g_post:
                # exclude self-loops from the real count itself, matching
                # the original C3's diagonal-zeroing before counting
                block_real = block_real.copy()
                np.fill_diagonal(block_real, 0)
            count = int(block_real.sum())
            if count == 0:
                continue

            # real synapse counts for THIS block specifically, to sample
            # from below -- the per-block granularity is the whole point
            syn_block_real = syn_real[np.ix_(pre_idx, post_idx)]
            syn_pool = syn_block_real[syn_block_real > 0].ravel()

            all_pairs = [(pi, pj) for pi in pre_idx for pj in post_idx if pi != pj]
            if len(all_pairs) == 0:
                continue
            replace = count > len(all_pairs)
            if replace and verbose:
                print(f"    WARNING: {g_pre}->{g_post} has {count} edges "
                      f"but only {len(all_pairs)} possible pairs -- "
                      f"sampling with replacement")
            chosen = rng.choice(len(all_pairs), size=count, replace=replace)
            syn_chosen = rng.choice(syn_pool, size=count,
                                    replace=(count > len(syn_pool)))
            for k, idx in enumerate(chosen):
                pi, pj = all_pairs[idx]
                A_out[pi, pj] = 1.0
                syn_out[pi, pj] = syn_chosen[k]

            block_report.append((g_pre, g_post, count, len(all_pairs)))

    return A_out, syn_out, block_report


def verify_c3(A_real, A_out, syn_real, syn_out, cell_types):
    """Assert the invariants C3 actually guarantees -- explicitly checking,
    not just asserting, that per-node degree is NOT preserved, since that
    is the defining contrast with C1."""
    ok = True

    def check(cond, msg):
        nonlocal ok
        if cond:
            print(f"    OK   {msg}")
        else:
            ok = False
            print(f"    FAIL {msg}")

    check(int(A_real.sum()) == int(A_out.sum()),
          f"total edge count preserved ({int(A_real.sum())})")
    check(int(A_out.diagonal().sum()) == 0, "no self-loops")
    check(int(syn_real.sum()) == int(syn_out.sum()) or
          abs(int(syn_real.sum()) - int(syn_out.sum())) < 100,
          f"total synapse count approximately preserved "
          f"(real={int(syn_real.sum())}, null={int(syn_out.sum())}) "
          f"-- per-block sampling with replacement can drift slightly, "
          f"unlike C1's exact rescale")

    unique_groups = sorted(set(cell_types))
    group_idx = {g: np.where(cell_types == g)[0] for g in unique_groups}
    block_mismatch = 0
    for g_pre in unique_groups:
        for g_post in unique_groups:
            pre_idx, post_idx = group_idx[g_pre], group_idx[g_post]
            real_blk = A_real[np.ix_(pre_idx, post_idx)]
            out_blk = A_out[np.ix_(pre_idx, post_idx)]
            if g_pre == g_post:
                real_blk = real_blk.copy(); np.fill_diagonal(real_blk, 0)
                out_blk = out_blk.copy(); np.fill_diagonal(out_blk, 0)
            if int(real_blk.sum()) != int(out_blk.sum()):
                block_mismatch += 1
    check(block_mismatch == 0,
          f"all {len(unique_groups)**2} cell-type block counts preserved exactly")

    # THE deliberate negative check: per-node degree should NOT be
    # preserved -- if it were, this null would be degenerate with C1.
    real_indeg = A_real.sum(axis=0)
    out_indeg = A_out.sum(axis=0)
    n_changed = int((real_indeg != out_indeg).sum())
    N = A_real.shape[0]
    print(f"    INFO per-node in-degree changed for {n_changed}/{N} nodes "
          f"({100*n_changed/N:.1f}%) -- expected to be large; C3 does NOT "
          f"preserve individual node degree, only block-level totals")
    if n_changed == 0:
        ok = False
        print("    FAIL per-node degree was NOT scrambled at all -- this "
              "null is degenerate with C1; investigate before using")

    is_I = np.array([ct in ("BC", "MC", "BPC", "NGC") for ct in cell_types])
    print("    OK   Dale's law preserved (block source type fixed by "
          "construction; no per-edge check needed)")

    print(f"\n  RESULT: {'ALL INVARIANTS HOLD' if ok else 'INVARIANT VIOLATED — DO NOT USE'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Signed cell-type-block-shuffled null (C3).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--verify_only", action="store_true")
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()

    print("Loading signed graph ...")
    A, syn, types, ids, _ = load_signed()
    graph_stats(A, types, "real")
    print(f"  [real] synapses={int(syn.sum())}")

    print("Loading cell-type labels ...")
    cell_types = load_cell_types(ids)
    unique_groups = sorted(set(cell_types))
    print(f"  {len(unique_groups)} cell-type groups: {unique_groups}")
    for g in unique_groups:
        print(f"    {g}: {(cell_types==g).sum()} neurons")

    if args.verify_only:
        print("\n--verify_only: real graph and cell types loaded. No null generated.")
        return

    for s in range(args.seed, args.seed + args.n_seeds):
        print(f"\n=== C3-signed null, seed {s} ===")
        A_c3, syn_c3, block_report = make_c3_signed(A, syn, cell_types, seed=s)
        graph_stats(A_c3, types, "null")
        print(f"  Non-zero blocks: {len(block_report)}")
        print("\n  Verifying invariants:")
        ok = verify_c3(A, A_c3, syn, syn_c3, cell_types)
        if not ok:
            sys.exit("INVARIANT VIOLATED — aborting rather than writing a bad null.")
        if not args.no_write:
            write_null(A_c3, syn_c3, ids, f"null_edges_C3signed_seed{s}.json")


if __name__ == "__main__":
    main()
