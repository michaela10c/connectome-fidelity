#!/usr/bin/env python3
"""
randomize_connectome_schemes.py

Randomization schemes for the Experiment 5 trained-random baseline, all operating
on the Flyvis connectome JSON (fib25-fib19_v2.2.json). INSPECTION ONLY — this
generates randomized JSONs and reports what each preserves/scrambles. No training.
The scheme to actually train at scale is a design decision for the Choi meeting.

Schemes:
  1. degree_preserving        — configuration-model stub reshuffle. Keeps in/out
                                degree sequence and scrambles partner identity, BUT
                                can collapse multiple rewired edges onto the same
                                (src,tar) pair, which reduces the Flyvis free-
                                parameter budget (one param group per unique
                                (src,tar) pair). Retained for parity/comparison.
  2. degree_preserving_swap   — Maslov-Sneppen degree-preserving EDGE SWAP. Keeps
                                in/out degree EXACTLY and enforces a simple graph
                                (no duplicate (src,tar) pairs), so the parameter
                                budget is preserved exactly (no collapse). This is
                                the budget-matched version of scheme 1: it is both
                                the aggressive topology null AND capacity-matched.
  3. rf_shuffle               — keep the connectivity graph (same src->tar pairs and
                                signs), randomize each edge's spatial receptive-field
                                'offsets' (permute spatial coords within each edge).
  4. sign_preserving_target_permutation
                              — keep each source's out-edge count and offsets,
                                permute the *targets* among non-fixed edges while
                                respecting the source's sign structure. Scrambles
                                the least.

All schemes preserve: node set, total edge count, total synapse count, and (by
default) the signs of alpha_fixed==True edges. Schemes 2-4 additionally preserve
the parameter budget exactly (605 unique pairs -> 605). Scheme 1 may not.

USAGE:
    python randomize_connectome_schemes.py INPUT.json --seed 0 --outdir ./rand_out
"""

import json, argparse, random, copy
from collections import Counter, defaultdict


def load_spec(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scheme 1: degree-preserving rewire (configuration-model, stub reshuffle)
#   NOTE: can collapse edges onto shared (src,tar) pairs, reducing the parameter
#   budget. Kept for comparison against the swap version below.
# ---------------------------------------------------------------------------
def degree_preserving(spec, seed=0, preserve_fixed=True, allow_self=True, max_tries=200):
    rng = random.Random(seed)
    edges = copy.deepcopy(spec["edges"])
    fixed = [e for e in edges if preserve_fixed and e.get("alpha_fixed", False)]
    rewireable = [e for e in edges if not (preserve_fixed and e.get("alpha_fixed", False))]
    occupied = set((e["src"], e["tar"]) for e in fixed)
    out_stubs = [e["src"] for e in rewireable]
    in_stubs = [e["tar"] for e in rewireable]
    payloads = [{k: v for k, v in e.items() if k not in ("src", "tar")} for e in rewireable]

    best = None
    for _ in range(max_tries):
        order = list(range(len(in_stubs))); rng.shuffle(order)
        used = set(occupied); new_pairs = []; ok = True
        for i, oi in enumerate(order):
            s, t = out_stubs[i], in_stubs[oi]
            if (not allow_self and s == t) or ((s, t) in used):
                ok = False; break
            used.add((s, t)); new_pairs.append((i, s, t))
        if ok and len(new_pairs) == len(rewireable):
            best = new_pairs; break
    if best is None:
        order = list(range(len(in_stubs))); rng.shuffle(order)
        best = [(i, out_stubs[i], in_stubs[oi]) for i, oi in enumerate(order)]

    rewired = []
    for (i, s, t) in sorted(best, key=lambda x: x[0]):
        e = dict(payloads[i]); e["src"], e["tar"] = s, t; rewired.append(e)
    out = copy.deepcopy(spec); out["edges"] = fixed + rewired
    return out


# ---------------------------------------------------------------------------
# Scheme 2: degree-preserving EDGE SWAP (Maslov-Sneppen)
#   Preserves in/out degree EXACTLY and keeps the graph simple (no duplicate
#   (src,tar) pairs), so the Flyvis free-parameter budget (one param group per
#   unique (src,tar) pair) is preserved exactly. This removes the capacity
#   confound of scheme 1 while remaining an aggressive topology null.
#
#   Algorithm: repeatedly pick two rewireable edges a->b and c->d and swap their
#   targets to a->d and c->b. Accept the swap only if it (i) creates no duplicate
#   (src,tar) pair (among rewireable or fixed), and (ii) respects the self-loop
#   policy. Every accepted swap preserves each node's in- and out-degree by
#   construction. alpha_fixed edges are held out of the swap pool entirely, so
#   their signs and positions never change. Payloads (offsets/alpha/...) ride
#   with the source edge; only the target endpoint moves, exactly as in scheme 1.
#
#   n_swaps_factor: target number of accepted swaps = factor * (#rewireable edges).
#     ~10-20x the edge count is the standard Maslov-Sneppen mixing regime; the
#     default of 20 produces a well-mixed graph (empirically ~39% of pairs
#     changed on fib25) while preserving all invariants.
# ---------------------------------------------------------------------------
def degree_preserving_swap(spec, seed=0, preserve_fixed=True, allow_self=True,
                           n_swaps_factor=20, max_attempts_factor=200):
    rng = random.Random(seed)
    edges = copy.deepcopy(spec["edges"])
    fixed = [e for e in edges if preserve_fixed and e.get("alpha_fixed", False)]
    rewireable = [e for e in edges if not (preserve_fixed and e.get("alpha_fixed", False))]

    fixed_pairs = set((e["src"], e["tar"]) for e in fixed)
    pairs = set((e["src"], e["tar"]) for e in rewireable)

    m = len(rewireable)
    if m < 2:
        out = copy.deepcopy(spec); out["edges"] = fixed + rewireable
        return out

    target_swaps = n_swaps_factor * m
    max_attempts = max_attempts_factor * m
    swaps_done = 0
    attempts = 0

    while swaps_done < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        a, b = rewireable[i]["src"], rewireable[i]["tar"]
        c, d = rewireable[j]["src"], rewireable[j]["tar"]
        if b == d:
            continue
        new1, new2 = (a, d), (c, b)
        if not allow_self and (a == d or c == b):
            continue
        if new1 == new2:
            continue
        if new1 in pairs or new2 in pairs:
            continue
        if new1 in fixed_pairs or new2 in fixed_pairs:
            continue
        # commit
        pairs.discard((a, b)); pairs.discard((c, d))
        pairs.add(new1); pairs.add(new2)
        rewireable[i]["tar"] = d
        rewireable[j]["tar"] = b
        swaps_done += 1

    out = copy.deepcopy(spec); out["edges"] = fixed + rewireable
    # stash diagnostics for the fingerprint/report (not used by Flyvis loader)
    out["_swap_diagnostics"] = {"swaps_committed": swaps_done, "attempts": attempts,
                                "target_swaps": target_swaps}
    return out


# ---------------------------------------------------------------------------
# Scheme 3: receptive-field (offsets) shuffle
#   keep src->tar graph and signs intact; randomize each edge's spatial filter
#   WITHOUT changing the synapse total. We permute the spatial coordinates
#   [du,dv] of an edge's offsets while keeping that edge's own n_syn values
#   attached to their slots — same synapse counts redistributed to shuffled
#   spatial positions within the same edge. Destroys RF structure but conserves
#   per-edge (and total) synapses.
# ---------------------------------------------------------------------------
def rf_shuffle(spec, seed=0, within_sign=True):
    rng = random.Random(seed)
    edges = copy.deepcopy(spec["edges"])
    for e in edges:
        offs = e["offsets"]
        if len(offs) <= 1:
            continue
        coords = [o[0] for o in offs]      # [[du,dv], ...]
        counts = [o[1] for o in offs]      # n_syn per slot (preserved)
        perm = list(range(len(coords)))
        rng.shuffle(perm)
        new_offs = [[coords[perm[i]], counts[i]] for i in range(len(offs))]
        e["offsets"] = new_offs
    out = copy.deepcopy(spec); out["edges"] = edges
    return out


# ---------------------------------------------------------------------------
# Scheme 4: sign-preserving target permutation
#   permute the targets among non-fixed edges, but only within blocks that share
#   the same source sign, so the source's sign structure is preserved. Offsets
#   ride along with the edge's source. Scrambles the least.
# ---------------------------------------------------------------------------
def sign_preserving_target_perm(spec, seed=0, preserve_fixed=True, allow_self=True,
                                max_tries=200):
    rng = random.Random(seed)
    edges = copy.deepcopy(spec["edges"])
    fixed = [e for e in edges if preserve_fixed and e.get("alpha_fixed", False)]
    rewireable = [e for e in edges if not (preserve_fixed and e.get("alpha_fixed", False))]
    occupied = set((e["src"], e["tar"]) for e in fixed)

    by_sign = defaultdict(list)
    for e in rewireable:
        by_sign[e["alpha"]].append(e)

    new_rewired = []
    for sign, block in by_sign.items():
        targets = [e["tar"] for e in block]
        best = None
        for _ in range(max_tries):
            perm = targets[:]; rng.shuffle(perm)
            used = set(occupied); ok = True; trial = []
            for e, t in zip(block, perm):
                s = e["src"]
                if (not allow_self and s == t) or ((s, t) in used):
                    ok = False; break
                used.add((s, t)); trial.append(t)
            if ok:
                best = trial; break
        if best is None:
            best = targets[:]  # fallback: identity within block
        for e, t in zip(block, best):
            ne = dict(e); ne["tar"] = t; new_rewired.append(ne)

    out = copy.deepcopy(spec); out["edges"] = fixed + new_rewired
    return out


# ---------------------------------------------------------------------------
# Scheme 5: Erdos-Renyi floor (matched node and edge counts ONLY)
# The crudest null (cf. FlyGM 2026, which uses an ER graph with matched node
# and edge counts as its "beats no structure at all" floor). Draws a random
# simple digraph on the SAME node set with the SAME number of rewireable edges.
#
# WHAT IT PRESERVES: node set, total edge count, total synapse count, and the
#   payload multiset (offsets/alpha/n_syn ride with the source edge's payload,
#   randomly reassigned).
# WHAT IT DELIBERATELY DOES NOT PRESERVE (this is the point of a floor):
#   - degree sequence (in/out degrees change)
#   - the 605-unique-pair budget -> a network built on this is NOT
#     parameter-matched to the CC network. Unlike schemes 2-4, param_budget_matched
#     will be False. This is expected for a floor and MUST be reported, not hidden.
#   - per-source sign structure / Dale consistency may break (signs ride with the
#     randomly-placed payloads, so a source can end up projecting a different sign
#     multiset than in the real connectome).
# Fixed-sign (alpha_fixed) edges are held out and kept in place by default, so at
# least those literature-backed signs are preserved; everything else is a floor.
#
# Because ER is NOT budget-matched, if you run it, frame it explicitly as the
# crude floor, and pair any geometry comparison with a budget-matched scheme
# (degree_preserving_swap) so a reviewer cannot attribute a difference to the
# parameter-count change rather than the loss of wiring structure.
# ---------------------------------------------------------------------------
def erdos_renyi(spec, seed=0, preserve_fixed=True, allow_self=True,
                max_tries_factor=200):
    rng = random.Random(seed)
    edges = copy.deepcopy(spec["edges"])
    fixed = [e for e in edges if preserve_fixed and e.get("alpha_fixed", False)]
    rewireable = [e for e in edges if not (preserve_fixed and e.get("alpha_fixed", False))]
    fixed_pairs = set((e["src"], e["tar"]) for e in fixed)

    nodes = sorted(set([e["src"] for e in edges] + [e["tar"] for e in edges]))
    m = len(rewireable)
    payloads = [{k: v for k, v in e.items() if k not in ("src", "tar")}
                for e in rewireable]

    # Draw m distinct random (src,tar) pairs, disjoint from the fixed edges,
    # forming a simple digraph (no duplicate pairs).
    new_pairs = set()
    attempts = 0
    max_attempts = max_tries_factor * max(m, 1)
    while len(new_pairs) < m and attempts < max_attempts:
        attempts += 1
        s = nodes[rng.randrange(len(nodes))]
        t = nodes[rng.randrange(len(nodes))]
        if not allow_self and s == t:
            continue
        if (s, t) in fixed_pairs or (s, t) in new_pairs:
            continue
        new_pairs.add((s, t))

    new_pairs = list(new_pairs)
    # Randomly assign the (shuffled) real payloads to the random edges, so the
    # payload multiset (and total synapse count) is preserved but placement is random.
    rng.shuffle(payloads)
    rewired = []
    for (s, t), pl in zip(new_pairs, payloads):
        e = dict(pl); e["src"], e["tar"] = s, t
        rewired.append(e)

    out = copy.deepcopy(spec); out["edges"] = fixed + rewired
    out["_er_diagnostics"] = {"edges_drawn": len(new_pairs), "target": m,
                              "attempts": attempts,
                              "note": "ER floor: NOT degree- or budget-matched"}
    return out


# ---------------------------------------------------------------------------
# Preservation fingerprint
# ---------------------------------------------------------------------------
def fingerprint(orig, new, label):
    o_e, n_e = orig["edges"], new["edges"]
    def outd(E): return Counter(e["src"] for e in E)
    def ind(E):  return Counter(e["tar"] for e in E)
    o_pairs = set((e["src"], e["tar"]) for e in o_e)
    n_pairs = set((e["src"], e["tar"]) for e in n_e)
    changed = len(o_pairs - n_pairs)

    def n_pair_groups(E):  # SynapseSign / SynapseCountScaling -> free params
        return len(set((e["src"], e["tar"]) for e in E))
    def n_count_groups(E):  # SynapseCount groups by (src,tar,du,dv) -> fixed params
        keys = set()
        for e in E:
            for off in e["offsets"]:
                du, dv = off[0]
                keys.add((e["src"], e["tar"], du, dv))
        return len(keys)

    o_fixed = {(e["src"], e["tar"]): e["alpha"] for e in o_e if e.get("alpha_fixed")}
    n_signs = {(e["src"], e["tar"]): e["alpha"] for e in n_e}
    fixed_ok = all(n_signs.get(k) == v for k, v in o_fixed.items())

    # --- Dale's-law consistency, with special attention to dual-sign cell types ---
    # A cell type "violates Dale's law" if, in the ORIGINAL connectome, it acts as a
    # source with both excitatory and inhibitory out-edges (on fib25: R8, Am, C2, C3).
    # A randomization should not change the SET of signs any source projects: the sign
    # rides with the source edge, so swapping targets must leave each source's multiset
    # of out-edge signs identical. We verify that here, and separately report the
    # dual-sign types so their handling is explicit and auditable.
    def src_sign_multiset(E):
        d = defaultdict(Counter)
        for e in E:
            d[e["src"]][e["alpha"]] += 1
        return d
    o_srcsign = src_sign_multiset(o_e)
    n_srcsign = src_sign_multiset(n_e)
    # every source must project the same multiset of signs after randomization
    dale_consistent = (o_srcsign == n_srcsign)
    # identify the original dual-sign (Dale-violating) source types
    dual_sign_types = sorted(s for s, c in o_srcsign.items() if len(c) > 1)
    # confirm each dual-sign type's sign multiset is unchanged
    dual_ok = all(o_srcsign[s] == n_srcsign.get(s) for s in dual_sign_types)

    syn_o = sum(sum(o[1] for o in e['offsets']) for e in o_e)
    syn_n = sum(sum(o[1] for o in e['offsets']) for e in n_e)

    n_collapsed = len(n_e) - len(n_pairs)
    budget_matched = (n_pair_groups(o_e) == n_pair_groups(n_e))

    row = {
        "scheme": label,
        "edges": f"{len(o_e)} -> {len(n_e)}",
        "pairs_changed": f"{changed}/{len(o_pairs)} ({100*changed/len(o_pairs):.0f}%)",
        "collapsed_pairs": n_collapsed,
        "outdeg_preserved": outd(o_e) == outd(n_e),
        "indeg_preserved": ind(o_e) == ind(n_e),
        "fixed_signs_ok": fixed_ok,
        "dale_consistent": dale_consistent,
        "dual_sign_types": ",".join(dual_sign_types) if dual_sign_types else "(none)",
        "dual_sign_ok": dual_ok,
        "pair_groups (~free)": f"{n_pair_groups(o_e)} -> {n_pair_groups(n_e)}",
        "param_budget_matched": budget_matched,
        "count_groups (~fixed)": f"{n_count_groups(o_e)} -> {n_count_groups(n_e)}",
        "total_syn": f"{syn_o:.0f} -> {syn_n:.0f}",
    }
    if "_er_diagnostics" in new:
        d = new["_er_diagnostics"]
        row["er_info"] = (f"{d['edges_drawn']}/{d['target']} random edges "
                          f"in {d['attempts']} attempts — {d['note']}")
    if "_swap_diagnostics" in new:
        d = new["_swap_diagnostics"]
        row["swap_info"] = (f"{d['swaps_committed']}/{d['target_swaps']} swaps "
                            f"in {d['attempts']} attempts")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    spec = load_spec(args.input)

    schemes = {
        "degree_preserving": degree_preserving(spec, seed=args.seed),
        "degree_preserving_swap": degree_preserving_swap(spec, seed=args.seed),
        "rf_shuffle": rf_shuffle(spec, seed=args.seed),
        "sign_preserving_target_perm": sign_preserving_target_perm(spec, seed=args.seed),
        "erdos_renyi": erdos_renyi(spec, seed=args.seed),
    }

    rows = []
    for name, newspec in schemes.items():
        diag = newspec.pop("_swap_diagnostics", None)
        er_diag = newspec.pop("_er_diagnostics", None)
        path = os.path.join(args.outdir, f"fib25_{name}_seed{args.seed}.json")
        with open(path, "w") as f:
            json.dump(newspec, f)
        if diag is not None:
            newspec["_swap_diagnostics"] = diag  # restore for fingerprint
        if er_diag is not None:
            newspec["_er_diagnostics"] = er_diag  # restore for fingerprint
        rows.append(fingerprint(spec, newspec, name))
        print(f"wrote {path}")

    print("\n=== PRESERVATION FINGERPRINTS ===")
    keys = ["scheme","edges","pairs_changed","collapsed_pairs","outdeg_preserved",
            "indeg_preserved","fixed_signs_ok","dale_consistent","dual_sign_types",
            "dual_sign_ok","pair_groups (~free)",
            "param_budget_matched","count_groups (~fixed)","total_syn","swap_info","er_info"]
    for k in keys:
        print(f"\n{k}:")
        for r in rows:
            if k in r:
                print(f"  {r['scheme']:32s} {r[k]}")


if __name__ == "__main__":
    main()
