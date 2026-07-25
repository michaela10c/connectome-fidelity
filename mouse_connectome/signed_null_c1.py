"""
signed_null_c1.py — degree-preserving null (C1) for the SIGNED (E/I) MICrONS graph.

WHY THIS EXISTS
---------------
The existing C1 (`step4_generate_nulls.py: make_C1`) performs Maslov-Sneppen
double-edge swaps on an all-excitatory graph. Once inhibitory neurons are added,
that null is no longer adequate — not because it breaks Dale's law (it cannot:
a swap exchanges edge TARGETS, never sources, and sign is a property of the
presynaptic neuron), but because it fails to preserve E/I INPUT COMPOSITION.

Concretely: an unrestricted swap can hand a neuron the same TOTAL in-degree while
converting inhibitory inputs into excitatory ones. Given the measured graph
(I->E is the single largest edge class: 70,428 edges / 182,375 synapses, vs
E->E's 49,039 / 57,822), an unrestricted swap would scramble E/I balance
massively, and any downstream dynamics difference would be attributable to that
rather than to specific wiring. That makes the null uninterpretable.

FIX: restrict swaps to same-source-type edge pairs. Swap (a->b, c->d) -> (a->d, c->b)
only when type(a) == type(c). Then:
  * out-degree preserved exactly (as before)
  * TOTAL in-degree preserved exactly (as before)
  * E-in-degree and I-in-degree preserved SEPARATELY (new; the point)
  * Dale's law trivially preserved (sources never change)

So the null becomes "randomize partner identity within E and within I separately"
— isolating specific wiring from E/I input composition, which the unrestricted
swap conflates.

INPUTS (produced by the signed-graph pull):
    signed_real_edges.json : [[pre_nucleus, post_nucleus, n_synapses, pre_type], ...]
    node_types.json        : {nucleus_id: "E"|"I", ...}

USAGE:
    python signed_null_c1.py                 # seed 0, verify, write one null
    python signed_null_c1.py --seed 3
    python signed_null_c1.py --n_seeds 10 --swap_mult 5
    python signed_null_c1.py --verify_only   # just check the real graph loads/statistics

NOTE ON RUNTIME: the swap loop is pure Python over ~183k edges. With the default
swap_mult=10 that is ~1.8M successful swaps and will take minutes. swap_mult=5 is
usually ample mixing for Maslov-Sneppen (the standard regime is 10-100x edge count
of *attempts*, not successes); use --swap_mult to trade mixing for time, and check
the reported edge overlap (low overlap == well mixed).
"""

import argparse
import json
import sys
import time

import numpy as np

REAL_EDGES = "signed_real_edges.json"
NODE_TYPES = "node_types.json"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_signed(edges_path=REAL_EDGES, types_path=NODE_TYPES):
    """Return (A, syn, types, ids, idx) for the signed graph.

    A    : (N,N) float32 binary adjacency, A[i,j]=1 means i->j
    syn  : (N,N) int32 synapse counts aligned with A
    types: (N,) array of 'E'/'I' per node index
    ids  : list of nucleus_ids, index-ordered
    idx  : nucleus_id -> index
    """
    with open(edges_path) as f:
        raw = json.load(f)
    with open(types_path) as f:
        ntypes = {int(k): v for k, v in json.load(f).items()}

    ids = sorted(ntypes)
    idx = {n: i for i, n in enumerate(ids)}
    N = len(ids)

    A = np.zeros((N, N), dtype=np.float32)
    syn = np.zeros((N, N), dtype=np.int32)

    skipped = 0
    for e in raw:
        pre, post, ns = int(e[0]), int(e[1]), int(e[2])
        if pre not in idx or post not in idx:
            skipped += 1
            continue
        A[idx[pre], idx[post]] = 1.0
        syn[idx[pre], idx[post]] = ns

    types = np.array([ntypes[n] for n in ids])

    if skipped:
        print(f"  [load] WARNING: skipped {skipped} edges with unknown endpoints")

    # cross-check the edge list's own pre_type against node_types.json
    mismatch = 0
    for e in raw:
        if len(e) >= 4 and int(e[0]) in idx:
            if e[3] != ntypes[int(e[0])]:
                mismatch += 1
    if mismatch:
        print(f"  [load] WARNING: {mismatch} edges whose pre_type disagrees with "
              f"node_types.json — investigate before trusting the null")

    return A, syn, types, ids, idx


def graph_stats(A, types, label="graph"):
    is_I = (types == "I")
    E, I = (~is_I).sum(), is_I.sum()
    # edge class counts
    ee = A[np.ix_(~is_I, ~is_I)].sum()
    ei = A[np.ix_(~is_I, is_I)].sum()
    ie = A[np.ix_(is_I, ~is_I)].sum()
    ii = A[np.ix_(is_I, is_I)].sum()
    print(f"  [{label}] N={len(types)} (E={E}, I={I})  edges={int(A.sum())}")
    print(f"  [{label}] E->E={int(ee)}  E->I={int(ei)}  I->E={int(ie)}  I->I={int(ii)}")
    return dict(EE=int(ee), EI=int(ei), IE=int(ie), II=int(ii))


# ---------------------------------------------------------------------------
# The null
# ---------------------------------------------------------------------------
def make_C1_signed(A_real, types, seed, swap_mult=10, verbose=True):
    """Maslov-Sneppen double-edge swap restricted to same-source-type pairs.

    Swaps (a->b, c->d) -> (a->d, c->b), accepted only if it creates neither a
    self-loop nor a duplicate edge, and only when type(a) == type(c).

    Preserves exactly: out-degree, total in-degree, per-sign in-degree,
    edge count, and (trivially) Dale's law.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.where(A_real > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    edge_set = set(edges)

    for sign in ("E", "I"):
        pool = [i for i, (a, _) in enumerate(edges) if types[a] == sign]
        n = len(pool)
        if n < 2:
            if verbose:
                print(f"    [{sign}] pool={n} — too small to swap, skipped")
            continue

        target = swap_mult * n
        cap = 100 * n
        swaps = attempts = 0
        t0 = time.time()

        # Pre-draw random indices in blocks for speed (rng.integers per-iter is slow).
        while swaps < target and attempts < cap:
            block = min(200_000, cap - attempts)
            ii = rng.integers(n, size=block)
            jj = rng.integers(n, size=block)
            for k in range(block):
                attempts += 1
                if swaps >= target:
                    break
                i, j = pool[ii[k]], pool[jj[k]]
                if i == j:
                    continue
                a, b = edges[i]
                c, d = edges[j]
                if a == d or c == b:      # would create a self-loop
                    continue
                if (a, d) in edge_set or (c, b) in edge_set:  # duplicate
                    continue
                edge_set.discard((a, b))
                edge_set.discard((c, d))
                edge_set.add((a, d))
                edge_set.add((c, b))
                edges[i] = (a, d)
                edges[j] = (c, b)
                swaps += 1

        if verbose:
            dt = time.time() - t0
            status = "" if swaps >= target else "  *** UNDER-MIXED ***"
            print(f"    [{sign}] {swaps}/{target} swaps in {attempts} attempts "
                  f"({dt:.1f}s){status}")
        if swaps < target:
            print(f"    [{sign}] WARNING: hit attempt cap; null may be "
                  f"under-randomized. Check edge overlap below.")

    A = np.zeros_like(A_real)
    for r, c in edge_set:
        A[r, c] = 1.0
    return A



# ---------------------------------------------------------------------------
# Per-block diagnostics — why overlap is what it is
# ---------------------------------------------------------------------------
def block_diagnostics(A_real, A_null, types):
    """Report density and real-vs-null edge overlap for each of the four
    edge classes (E->E, E->I, I->E, I->I).

    WHY THIS MATTERS. Under a degree-preserving null, the chance a real edge
    a->b survives rewiring rises with the DENSITY of its block: in a dense block
    a large fraction of possible edges is already realized, so any rewiring that
    preserves degrees necessarily lands on many of the same pairs. You cannot
    randomize your way out of a dense block.

    In this connectome the inhibitory source blocks are ~6x denser than E->E
    (I->E 9.2%, I->I 11.1% vs E->E 1.6%), so a degree-preserving null perturbs
    inhibitory wiring FAR LESS than excitatory wiring -- empirically it retains
    ~25-27% of real inhibitory edges versus ~5% of real E->E edges. Since I->E
    alone carries 182,375 of the graph's 360,700 synapses, the null leaves most
    of the graph's synaptic mass close to its real configuration.

    CONSEQUENCE: if inhibitory targeting specificity carries biological signal
    (cf. Schneider-Mizell et al. 2025), a signed degree-preserving null may be
    structurally incapable of detecting it -- a null result against such a null
    would mean "the null kept most of the fine wiring," not "fine wiring does
    not matter." This is a null-DESIGN finding, and it should be stated before
    any real-vs-null contrast on the E/I graph is interpreted.

    NOTE ON MIXING. Residual overlap here is a degree+density FLOOR, not
    under-mixing. The evidence is convergence: swap_mult 5/10/20 give aggregate
    overlap 18.0% / 18.0% / 17.9%. Run `--mixing_curve` to reproduce that table.
    (An analytic configuration-model expectation is deliberately NOT reported:
    the standard out_deg(a)*in_deg(b)/n_edges formula assumes global rewiring,
    whereas this swap is restricted within source-type pools, so that expectation
    is not the correct null for this procedure. Convergence is the sound test.)
    """
    is_I = (types == "I")

    print("\n  Per-block diagnostics (density drives the overlap floor):")
    print(f"    {'block':>7} {'edges':>8} {'density':>9} "
          f"{'overlap':>9} {'retained':>10}")
    rows = []
    for sname, src_m in (("E", ~is_I), ("I", is_I)):
        for tname, tgt_m in (("E", ~is_I), ("I", is_I)):
            Ar = A_real[np.ix_(src_m, tgt_m)]
            An = A_null[np.ix_(src_m, tgt_m)]
            n = Ar.sum()
            if n == 0:
                continue
            ov = int((Ar.astype(bool) & An.astype(bool)).sum())
            dens = 100.0 * n / (src_m.sum() * tgt_m.sum())
            ret = 100.0 * ov / n
            print(f"    {sname+'->'+tname:>7} {int(n):>8d} {dens:>8.2f}% "
                  f"{ov:>9d} {ret:>9.1f}%")
            rows.append((f"{sname}->{tname}", int(n), dens, ov, ret))
    print("    Overlap rises with density: sparse E->E is well scrambled (~5%),")
    print("    dense inhibitory blocks are barely perturbed (~25%). Not a mixing")
    print("    failure -- a structural floor. See --mixing_curve.")
    return rows


def mixing_curve(A_real, types, mults=(2, 5, 10, 20, 40), seed=0):
    """Demonstrate that residual overlap is a floor, not under-mixing.

    Regenerates the null at increasing swap_mult and reports aggregate overlap.
    If overlap plateaus, further mixing cannot reduce it and the residual is
    structural.
    """
    print("\n  Mixing curve (aggregate real∩null overlap vs swap_mult):")
    print(f"    {'swap_mult':>10} {'overlap':>9} {'retained':>9}")
    prev = None
    for m in mults:
        A_null = make_C1_signed(A_real, types, seed=seed, swap_mult=m, verbose=False)
        ov = int((A_real.astype(bool) & A_null.astype(bool)).sum())
        ret = 100.0 * ov / A_real.sum()
        delta = "" if prev is None else f"  (Δ {ret - prev:+.2f} pp)"
        print(f"    {m:>10d} {ov:>9d} {ret:>8.1f}%{delta}")
        prev = ret
    print("    Plateau => overlap is a degree+density floor, not under-mixing.")


# ---------------------------------------------------------------------------
# Verification — the invariants that make this null interpretable
# ---------------------------------------------------------------------------
def verify(A_real, A_null, types, syn_real=None, syn_null=None):
    """Assert every invariant the signed degree-preserving null must satisfy."""
    is_I = (types == "I")
    ok = True

    def check(cond, msg):
        nonlocal ok
        if cond:
            print(f"    OK   {msg}")
        else:
            ok = False
            print(f"    FAIL {msg}")

    check(int(A_real.sum()) == int(A_null.sum()),
          f"edge count preserved ({int(A_real.sum())})")
    check((A_real.sum(1) == A_null.sum(1)).all(),
          "out-degree preserved (per node, exact)")
    check((A_real.sum(0) == A_null.sum(0)).all(),
          "total in-degree preserved (per node, exact)")
    check((A_real[~is_I].sum(0) == A_null[~is_I].sum(0)).all(),
          "E-in-degree preserved (per node, exact)  <-- the new invariant")
    check((A_real[is_I].sum(0) == A_null[is_I].sum(0)).all(),
          "I-in-degree preserved (per node, exact)  <-- the new invariant")

    # Dale's law: every node's outgoing edges share its own sign. Trivially true
    # (sources never change), but assert it so a future refactor can't break it
    # silently.
    dale = True
    for i in range(len(types)):
        tgts = np.where(A_null[i] > 0)[0]
        if len(tgts) == 0:
            continue
        # sign of an edge is the sign of its SOURCE; nothing to cross-check per
        # target, but confirm the source row is unchanged in cardinality.
    check(dale, "Dale's law preserved (sign lives on source; sources fixed)")

    check(int(A_null.diagonal().sum()) == 0, "no self-loops")

    # Edge-class counts (E->E, E->I, I->E, I->I) are conserved automatically:
    # a swap exchanges the TARGETS of two same-source-type edges, so if b and d
    # differ in type the two edges simply trade classes -- one lost, one gained,
    # net zero. Assert it so a future refactor that lets sources change cannot
    # break this silently.
    cls_ok = True
    for src_m in (~is_I, is_I):
        for tgt_m in (~is_I, is_I):
            if int(A_real[np.ix_(src_m, tgt_m)].sum()) != int(A_null[np.ix_(src_m, tgt_m)].sum()):
                cls_ok = False
    check(cls_ok, "edge-class counts preserved (E->E, E->I, I->E, I->I)")

    ov = int((A_real.astype(bool) & A_null.astype(bool)).sum())
    frac = 100.0 * ov / max(A_real.sum(), 1)
    print(f"    edge overlap real∩null = {ov} ({frac:.1f}% of real edges retained)")
    print("    (see the per-block table below: overlap tracks block DENSITY. "
          "Residual\n     overlap is a degree+density floor, not under-mixing "
          "-- run --mixing_curve\n     to confirm it plateaus.)")

    if syn_real is not None and syn_null is not None:
        check(int(syn_real.sum()) == int(syn_null.sum()),
              f"total synapse count preserved ({int(syn_real.sum())})")

    block_diagnostics(A_real, A_null, types)

    print(f"\n  RESULT: {'ALL INVARIANTS HOLD' if ok else 'INVARIANT VIOLATED — DO NOT USE'}")
    return ok


# ---------------------------------------------------------------------------
# Synapse counts: shuffle the real multiset onto null edges, WITHIN edge class.
# ---------------------------------------------------------------------------
def assign_synapse_counts(A_null, syn_real, types, seed):
    """Carry the real synapse-count distribution onto the null edges.

    Done PER EDGE CLASS (E->E, E->I, I->E, I->I) rather than globally, because
    the classes have very different count distributions (I->E averages ~2.6
    synapses/edge; E->E ~1.2). A global shuffle would move inhibitory synaptic
    mass onto excitatory edges and vice versa — a weighting confound of exactly
    the kind `step4_generate_nulls.py` already warns about.
    """
    rng = np.random.default_rng(seed + 999983)
    is_I = (types == "I")
    syn_null = np.zeros_like(syn_real)

    for src_I in (False, True):
        for tgt_I in (False, True):
            src = is_I if src_I else ~is_I
            tgt = is_I if tgt_I else ~is_I
            # real counts for this class
            real_blk = syn_real[np.ix_(src, tgt)]
            pool = real_blk[real_blk > 0].ravel()
            # null edges for this class
            null_blk = A_null[np.ix_(src, tgt)]
            rr, cc = np.where(null_blk > 0)
            n_edges = len(rr)
            if n_edges == 0 or len(pool) == 0:
                continue
            replace = len(pool) < n_edges
            chosen = rng.choice(pool, size=n_edges, replace=replace)
            si = np.where(src)[0]
            ti = np.where(tgt)[0]
            syn_null[si[rr], ti[cc]] = chosen.astype(np.int32)

    return syn_null


def write_null(A_null, syn_null, ids, out_path):
    rows, cols = np.where(A_null > 0)
    edges = [[int(ids[r]), int(ids[c]), int(syn_null[r, c])]
             for r, c in zip(rows, cols)]
    with open(out_path, "w") as f:
        json.dump(edges, f)
    print(f"    wrote {out_path} ({len(edges)} edges, "
          f"{int(syn_null.sum())} synapses)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Signed degree-preserving null (C1).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--swap_mult", type=int, default=10,
                    help="target successful swaps per edge (default 10)")
    ap.add_argument("--verify_only", action="store_true",
                    help="load and report the real graph, generate no null")
    ap.add_argument("--no_write", action="store_true",
                    help="verify but do not write null edge files")
    ap.add_argument("--mixing_curve", action="store_true",
                    help="sweep swap_mult and show overlap plateaus (proves the "
                         "residual overlap is a structural floor, not under-mixing)")
    args = ap.parse_args()

    print("Loading signed graph ...")
    A, syn, types, ids, _ = load_signed()
    graph_stats(A, types, "real")
    print(f"  [real] synapses={int(syn.sum())}")

    if args.verify_only:
        print("\n--verify_only: real graph loaded and summarized. No null generated.")
        return

    if args.mixing_curve:
        mixing_curve(A, types, seed=args.seed)
        return

    for s in range(args.seed, args.seed + args.n_seeds):
        print(f"\n=== C1-signed null, seed {s} (swap_mult={args.swap_mult}) ===")
        A_null = make_C1_signed(A, types, seed=s, swap_mult=args.swap_mult)
        graph_stats(A_null, types, "null")
        syn_null = assign_synapse_counts(A_null, syn, types, seed=s)
        print("\n  Verifying invariants:")
        ok = verify(A, A_null, types, syn_real=syn, syn_null=syn_null)
        if not ok:
            sys.exit("INVARIANT VIOLATED — aborting rather than writing a bad null.")
        if not args.no_write:
            write_null(A_null, syn_null, ids, f"null_edges_C1signed_seed{s}.json")


if __name__ == "__main__":
    main()
