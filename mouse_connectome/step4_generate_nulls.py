"""
Null-through-simulation, part 1 (step 4) — generate NULL CONNECTOME EDGE FILES for simulation.

Produces null connectomes on the FULL simulated neuron set (node_data_v1.csv,
1,764 neurons) so each can be run through the same BMTK pipeline as the real
connectome. Each null is written as a [pre, post, n_syn] JSON in nucleus-id
space — the exact format step2's run_simulation() consumes.

NULL FAMILIES (same logic as the validated step3 structural test, but generated
on the FULL graph rather than the 899-neuron intersection, because the
simulation runs on the full graph):
    C1  degree-preserving (configuration model)
    C2  distance-constrained (resample targets by real distance-decay profile)
    C3  cell-type(layer)-block-shuffled (preserve per-block edge counts)

SMOKE-TEST SCOPE (read this):
    WEIGHT_MODE_NULL = "binary"  -> every null edge gets n_syn = 1.
    This is the SMOKE TEST setting: it proves the null-through-simulation
    plumbing works, but it is CONFOUNDED against the real run, which uses
    synapse-count weighting. DO NOT REPORT binary-null results as the answer.
    For the real experiment set WEIGHT_MODE_NULL = "preserve_counts" (shuffles
    the real synapse-count multiset onto the null edges) — a methodological
    choice to confirm with Choi before the reported run.

USAGE:
    python step4_generate_nulls.py            # N=1 per family (smoke test)
    python step4_generate_nulls.py 10         # N=10 per family (real run)

Writes:   null_edges_{C1,C2,C3}_seed{S}.json   (one per family per seed)
Reads:    node_data_v1.csv, candidate2_real_edges.json
"""

import sys
import json
import numpy as np
import pandas as pd
import networkx as nx

# Must be the edge file WITH synapse counts ([pre,post,n_syn]). candidate1 has
# real counts (median 1, max 16); candidate2 is len-2 (no counts) and would make
# preserve_counts a no-op (all-1 pool). step2/step3 also use candidate1 as the
# real graph, so this keeps the null topology + count pool consistent with the
# real run.
REAL_EDGES = "candidate1_real_edges.json"
NODE_CSV   = "node_data_v1.csv"

# "binary" (smoke test) or "preserve_counts" (reported run; confirm with Choi)
# Set to preserve_counts: nulls now carry the real synapse-count distribution
# (shuffled onto null edges), removing the weighting confound vs the
# synapse-count-weighted real run. This is the REPORTABLE configuration.
WEIGHT_MODE_NULL = "preserve_counts"


def load_full_graph():
    node_df = pd.read_csv(NODE_CSV, index_col="node_id")
    nucleus_ids = node_df["nucleus_id"].astype(int).to_numpy()
    N = len(nucleus_ids)
    id_to_idx = {int(n): i for i, n in enumerate(nucleus_ids)}
    idx_to_id = {i: int(n) for i, n in enumerate(nucleus_ids)}

    with open(REAL_EDGES) as f:
        raw = json.load(f)

    A = np.zeros((N, N), dtype=np.float32)
    counts = []  # real synapse-count multiset (for preserve_counts mode)
    for e in raw:
        pre, post = int(e[0]), int(e[1])
        c = int(e[2]) if len(e) == 3 else 1
        if pre in id_to_idx and post in id_to_idx:
            A[id_to_idx[pre], id_to_idx[post]] = 1.0
            counts.append(c)
    counts = np.array(counts, dtype=int)
    print(f"  Full graph: N={N}, edges={int(A.sum())}, "
          f"syn_count median={int(np.median(counts))} max={int(counts.max())}")

    # positions (microns) for C2
    pos = np.zeros((N, 3), dtype=float)
    for i in range(N):
        row = node_df.iloc[i]
        pos[i] = [float(row["x"]) / 1000.0,
                  float(row["y"]) / 1000.0,
                  float(row["z"]) / 1000.0]

    # layer groups for C3
    def ct_to_layer(ct):
        ct = str(ct)
        if ct.startswith("23"): return "L2/3"
        if ct.startswith("4"):  return "L4"
        if ct.startswith("5"):  return "L5"
        if ct.startswith("6"):  return "L6"
        return "other"
    groups = np.array([ct_to_layer(node_df.iloc[i]["cell_type"]) for i in range(N)])

    return A, pos, groups, idx_to_id, counts, N


def adjacency_to_edge_file(A, idx_to_id, counts_pool, out_path, seed):
    """Write a null adjacency as [pre, post, n_syn] JSON in nucleus-id space.

    binary          : n_syn = 1 for every edge.
    preserve_counts : the real synapse-count multiset is shuffled and assigned
                      across the null edges (same count distribution as real).
    """
    rows, cols = np.where(A > 0)
    n_edges = len(rows)
    if WEIGHT_MODE_NULL == "preserve_counts":
        rng = np.random.default_rng(seed + 999983)  # distinct stream
        pool = counts_pool.copy()
        if len(pool) >= n_edges:
            chosen = rng.choice(pool, size=n_edges, replace=False)
        else:
            chosen = rng.choice(pool, size=n_edges, replace=True)
        syn = chosen.astype(int)
    else:  # binary
        syn = np.ones(n_edges, dtype=int)

    edges = [[idx_to_id[int(r)], idx_to_id[int(c)], int(s)]
             for r, c, s in zip(rows, cols, syn)]
    with open(out_path, "w") as f:
        json.dump(edges, f)
    print(f"    wrote {out_path}  ({n_edges} edges, weight_mode={WEIGHT_MODE_NULL})")


# ── null generators on the FULL graph ─────────────────────────────────────────

def make_C1(A_real, seed, **_):
    """Degree-preserving null via Maslov-Sneppen double-edge swaps.

    Why not the configuration model: nx.directed_configuration_model returns a
    MultiDiGraph whose parallel edges collapse when cast to DiGraph, and its
    self-loops must be removed — both DELETE edges, so the null ends up with
    ~2.5% fewer edges (and therefore fewer total synapses) than the real graph,
    and the per-node degrees are perturbed by the collapse. That breaks the
    synapse-count matching with the real run and makes C1 not-quite
    degree-preserving — exactly the confound to avoid in the most stringent null.

    Double-edge swap fixes both: it starts FROM the real edge set and repeatedly
    picks two directed edges (a->b, c->d) and rewires them to (a->d, c->b). This
    preserves every node's in- and out-degree EXACTLY and preserves the total
    edge count EXACTLY (a swap is rejected and retried, never dropped, if it
    would create a self-loop or a duplicate edge). The result is a graph with
    identical degree sequence and identical edge count to real, with partner
    identity randomized — a true degree-preserving null.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.where(A_real > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    n_edges = len(edges)
    edge_set = set(edges)

    # Maslov-Sneppen: aim for ~10 successful swaps per edge for thorough mixing.
    target_swaps = 10 * n_edges
    max_attempts = 100 * n_edges  # generous cap so we reach target_swaps
    swaps = 0
    attempts = 0
    while swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        i = rng.integers(n_edges)
        j = rng.integers(n_edges)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        # rewire (a->b, c->d) -> (a->d, c->b)
        if a == d or c == b:
            continue                      # would create a self-loop
        if (a, d) in edge_set or (c, b) in edge_set:
            continue                      # would create a duplicate edge
        # accept the swap
        edge_set.discard((a, b)); edge_set.discard((c, d))
        edge_set.add((a, d)); edge_set.add((c, b))
        edges[i] = (a, d); edges[j] = (c, b)
        swaps += 1

    if swaps < target_swaps:
        print(f"    [C1] WARNING: only {swaps}/{target_swaps} swaps reached "
              f"({attempts} attempts) — null may be under-randomized")

    A = np.zeros_like(A_real)
    for (r, c) in edge_set:
        A[r, c] = 1.0
    return A


def _distance_profile(A_real, pos, N, rng, n_bins=20):
    idx = np.arange(N)
    sp = rng.choice(idx, 60000); tp = rng.choice(idx, 60000)
    keep = sp != tp; sp, tp = sp[keep], tp[keep]
    d = np.linalg.norm(pos[sp] - pos[tp], axis=1)
    conn = A_real[sp, tp] > 0
    edges = np.linspace(0, np.percentile(d, 99), n_bins + 1)
    prob = np.zeros(n_bins)
    for b in range(n_bins):
        m = (d >= edges[b]) & (d < edges[b+1])
        if m.sum() > 0:
            prob[b] = conn[m].sum() / m.sum()
    return edges, prob


def make_C2(A_real, seed, pos=None, N=None, **_):
    rng = np.random.default_rng(seed)
    edges, prob = _distance_profile(A_real, pos, N, rng)
    n_bins = len(prob)
    out_deg = A_real.sum(axis=1).astype(int)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        k = out_deg[i]
        if k == 0:
            continue
        d = np.linalg.norm(pos - pos[i], axis=1)
        b = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
        w = prob[b].copy(); w[i] = 0
        tot = w.sum()
        if tot == 0:
            continue
        p = w / tot
        kk = min(k, int((w > 0).sum()))
        if kk == 0:
            continue
        tgt = rng.choice(N, size=kk, replace=False, p=p)
        A[i, tgt] = 1.0
    return A


def make_C3(A_real, seed, groups=None, N=None, **_):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(groups))
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    A = np.zeros((N, N), dtype=np.float32)
    for gp in uniq:
        for gq in uniq:
            blk = A_real[np.ix_(gidx[gp], gidx[gq])].copy()
            if gp == gq:
                np.fill_diagonal(blk, 0)
            cnt = int(blk.sum())
            if cnt == 0:
                continue
            pairs = [(pi, pj) for pi in gidx[gp] for pj in gidx[gq] if pi != pj]
            if not pairs:
                continue
            chosen = rng.choice(len(pairs), size=cnt, replace=cnt > len(pairs))
            for c in chosen:
                pi, pj = pairs[c]
                A[pi, pj] = 1.0
    return A


GENERATORS = {"C1": make_C1, "C2": make_C2, "C3": make_C3}


def main(n_per_family):
    print(f"Generating null connectome edge files: N={n_per_family} per family, "
          f"weight_mode={WEIGHT_MODE_NULL}")
    if WEIGHT_MODE_NULL == "binary":
        print("  *** SMOKE TEST (binary nulls): confounded vs synapse-count real. "
              "Do not report. Switch to 'preserve_counts' for the real run. ***")
    A_real, pos, groups, idx_to_id, counts, N = load_full_graph()

    manifest = []
    for fam, fn in GENERATORS.items():
        for s in range(n_per_family):
            print(f"  [{fam}] seed {s} ...")
            A_null = fn(A_real, s, pos=pos, groups=groups, N=N)
            out = f"null_edges_{fam}_seed{s}.json"
            adjacency_to_edge_file(A_null, idx_to_id, counts, out, s)
            manifest.append({"family": fam, "seed": s, "edge_file": out,
                             "tag": f"{fam}_seed{s}"})

    with open("step4_null_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} null edge files + step4_null_manifest.json")
    print("Next: step5_run_null_simulation.py")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(n)
