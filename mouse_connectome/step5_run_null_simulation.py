"""
Null-through-simulation, part 2 (step 5) — run each NULL connectome through simulation and compare each
null-sim's BIOLOGICAL correlation against the real-sim's.

This is the experiment that fills the missing 2x2 cell: it asks whether the
connectome-constrained simulation matches measured biology MORE than
structured-random connectomes do, when each is run through the IDENTICAL
simulation pipeline (same shared LGN drive, same model, same weighting mode).

PIPELINE PER CONNECTOME:
    edge_file --> step2.run_simulation(edge_file, tag, build_lgn=False)
              --> output_pointnet_{tag}/theta*/spikes.h5
              --> rates (8 orientations) --> cosine RDM on the 899-neuron
                  biology-intersection set --> Spearman r vs biological ref RDM

HEADLINE:
    real-sim-vs-bio  (from your validated real run)  compared against the
    DISTRIBUTION of null-sim-vs-bio across all null instances.

SCOPE / HONESTY:
    - Reuses the SHARED LGN scaffold (build_lgn=False): identical input drive
      for real and every null. Only the recurrent connectome differs.
    - Smoke test (N=1, binary nulls): proves the path end to end. Not a
      reportable result until preserve_counts nulls + an N agreed with Choi.
    - Real-sim spikes are read from output_pointnet/ (your existing real run).

USAGE:
    python step4_generate_nulls.py 1          # generate nulls first
    python step5_run_null_simulation.py           # simulate + compare

Reads:  step4_null_manifest.json, node_data_v1.csv,
        signal_corr_matrix.npy, signal_corr_nucleus_ids.npy,
        output_pointnet/ (real sim, already run)
Writes: step5_results.txt, step5_results.png, step5_bio_correlations.json,
        step5_resource_log.json (resources-framing seed: per-condition
        fidelity + resource budget)
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import step2_build_and_simulate_microns as step2

ORIENTATIONS   = step2.ORIENTATIONS
TSTOP_MS       = step2.TSTOP
GRAY_SCREEN_MS = 500.0
ANALYSIS_MS    = TSTOP_MS - GRAY_SCREEN_MS
NODE_CSV       = "node_data_v1.csv"
REAL_OUTPUT    = "output_pointnet"     # real run (legacy dir)


# ── rate extraction + RDM (same recipe as step3) ──────────────────────────────

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


def edge_resource_budget(edge_file):
    """Resource accounting for one connectome, for the resources-framing seed.

    An emulation is a budget of resources (wiring structure, free parameters,
    training data, compute). For the *mouse prototype* most of these are matched
    across the real run and all nulls by construction:
      - synapse-count multiset is preserved (nulls carry the real counts shuffled
        onto null edges), so edge_count and total_synapses are matched -> NOT the
        resource axis here; logged to DOCUMENT that they are controlled.
      - free parameters (fixed LIF params, synapse-count weighting) are identical
        across conditions -> matched, not varying.
      - training iterations are zero (untrained) for every condition -> matched.
    The one resource that actually VARIES across conditions is how much wiring
    *structure* each connectome retains relative to the real one. That is the
    real x-axis of the resources question for the mouse, and it is a property of
    the null family (degree- vs distance- vs cell-type-preserving), not of the
    edge file's counts. We therefore log the matched budgets here for
    transparency and attach the family-level structure label downstream; we do
    NOT fabricate a single-number "bits retained" we cannot yet justify.

    Returns a dict, or None if the edge file can't be read.
    """
    import time
    for attempt in range(6):
        try:
            with open(edge_file) as ef:
                edges = json.load(ef)
            break
        except (TimeoutError, OSError) as e:
            if attempt == 5:
                print(f"  [io] resource read of {edge_file} failed ({e}); skipping budget")
                return None
            time.sleep(5 * (attempt + 1))
    # edges are [pre, post, n_synapses]; be defensive about shape
    edge_count = len(edges)
    total_synapses = None
    try:
        if edge_count and len(edges[0]) >= 3:
            total_synapses = int(sum(int(e[2]) for e in edges))
    except (TypeError, IndexError, ValueError):
        total_synapses = None
    nodes = set()
    try:
        for e in edges:
            nodes.add(e[0]); nodes.add(e[1])
    except (TypeError, IndexError):
        pass
    return {
        "edge_count": edge_count,
        "total_synapses": total_synapses,   # matched across conditions (preserve_counts)
        "n_nodes_with_edges": len(nodes) if nodes else None,
    }


def _load_edge_set(edge_file):
    """Load an edge file as a set of (pre, post) directed pairs, or None on
    read failure. Retries on transient cloud-sync timeouts (see budget helper)."""
    import time
    for attempt in range(6):
        try:
            with open(edge_file) as ef:
                edges = json.load(ef)
            break
        except (TimeoutError, OSError):
            if attempt == 5:
                return None
            time.sleep(5 * (attempt + 1))
    try:
        return {(int(e[0]), int(e[1])) for e in edges}
    except (TypeError, IndexError, ValueError):
        return None


def edge_overlap_with_real(edge_file, real_edge_set):
    """Resources-framing seed: how much of the REAL wiring this null retains.

    Returns the fraction of real directed edges also present in the null (edge
    overlap with real). This is the quantitative x-axis for the structure-vs-
    fidelity relationship: nulls that retain more real structure should, if
    structure buys biological fidelity, land closer to real on the biological
    axis. It turns the ORDINAL null ordering (degree > distance > cell-type) into
    a numeric one at matched budget.

    HONEST SCOPE: this is edge-overlap-with-real, a crude structural-content
    proxy, NOT an information-theoretic 'bits retained'. A null could retain few
    edges but the fidelity-relevant ones, so overlap correlates with — but is not
    identical to — 'how much fidelity-relevant structure survived'. Adequate for
    a seed; do not report as a rigorous information measure. Because all families
    are edge-count-matched to real, this fraction is directly comparable across
    families (same denominator).

    Returns a float in [0, 1], or None if either edge set is unavailable.
    """
    if real_edge_set is None:
        return None
    null_set = _load_edge_set(edge_file)
    if null_set is None:
        return None
    n_real = len(real_edge_set)
    if n_real == 0:
        return None
    return len(real_edge_set & null_set) / n_real


def sim_bio_correlation(output_dir, node_df, sig, sig_idx, bio_nucleus_ids):
    """Build sim RDM on the biology-intersection neuron set and correlate it
    against the biological signal-correlation reference RDM. Returns (r, n)."""
    N_neurons = len(node_df)
    nucleus_to_node = {int(v): k for k, v in node_df["nucleus_id"].to_dict().items()}

    rate = rates_from_output(output_dir, N_neurons)

    # restrict to neurons present in both the sim and the biological reference
    common_nuc, common_node = [], []
    for nid in bio_nucleus_ids:
        node_id = nucleus_to_node.get(int(nid))
        if node_id is not None:
            common_nuc.append(int(nid)); common_node.append(node_id)
    common_nuc = np.array(common_nuc); common_node = np.array(common_node)

    sim_rates = rate[common_node, :]
    sim_rdm = squareform(pdist(sim_rates, metric="cosine"))

    bio_sel = [sig_idx[int(n)] for n in common_nuc]
    bio_rdm = 1.0 - sig[np.ix_(bio_sel, bio_sel)]

    nb = len(common_nuc); tb = np.triu_indices(nb, k=1)
    va, vb = sim_rdm[tb], bio_rdm[tb]
    mask = np.isfinite(va) & np.isfinite(vb)
    r = spearmanr(va[mask], vb[mask])[0]
    return r, nb


def main():
    node_df = pd.read_csv(NODE_CSV, index_col="node_id")
    sig = np.load("signal_corr_matrix.npy")
    sig_ids = np.load("signal_corr_nucleus_ids.npy")
    sig_idx = {int(n): i for i, n in enumerate(sig_ids)}
    bio_nucleus_ids = sig_ids

    # 1) real-sim vs bio (from existing real run)
    print("Computing real-sim vs biology (from output_pointnet/)...")
    r_real, n_real = sim_bio_correlation(REAL_OUTPUT, node_df, sig, sig_idx,
                                         bio_nucleus_ids)
    print(f"  real-sim vs bio: r = {r_real:.4f} (n={n_real})")

    # 2) simulate each null, then null-sim vs bio
    with open("step4_null_manifest.json") as f:
        manifest = json.load(f)
    print(f"\nSimulating {len(manifest)} null connectomes "
          f"(shared LGN, build_lgn=False)...")

    results = {"real": r_real}
    # Resources-framing seed: log (fidelity, resource-budget) per condition.
    # Cheap, additive, rides alongside the fidelity result. The real connectome's
    # edge file is candidate1_real_edges.json (what step2 built the real run from).
    REAL_EDGE_FILE = "candidate1_real_edges.json"
    real_edge_set = _load_edge_set(REAL_EDGE_FILE)  # for edge-overlap-with-real
    resource_log = {
        "real": {
            "bio_correlation": r_real,
            "family": "real",
            "structure_retained": "full (real connectome)",
            "structure_retained_fraction": 1.0,   # real overlaps itself fully
            "training_iterations": 0,            # mouse prototype is untrained
            "budget": edge_resource_budget(REAL_EDGE_FILE),
        }
    }
    by_family = {}
    import shutil, hashlib
    for entry in manifest:
        tag = entry["tag"]; fam = entry["family"]; edge_file = entry["edge_file"]
        out_dir = step2.output_dir_for(tag)
        net_dir = step2.network_dir_for(tag)
        # Resim guard keyed to the EDGE FILE, not just existence of spikes.
        # We stamp each null output dir with a hash of the edge file it was
        # simulated from. If the current edge file's hash matches the stamp AND
        # all 8 spike files exist, reuse them (lets you re-run reporting without
        # redoing an hour of sims). If the connectome changed (e.g. binary ->
        # preserve_counts, or new seeds), the hash differs and we re-simulate.
        # Hash the edge file to decide reuse-vs-resimulate. Wrapped in a retry
        # because on cloud-synced filesystems (iCloud/Dropbox) a read can
        # transiently time out when a file has been evicted to cloud; retrying
        # gives the sync daemon a moment to materialize it rather than killing
        # a multi-hour run. (Best fix is to run off a non-synced path.)
        import time
        edge_hash = None
        for attempt in range(6):
            try:
                with open(edge_file, "rb") as ef:
                    edge_hash = hashlib.md5(ef.read()).hexdigest()
                break
            except (TimeoutError, OSError) as e:
                if attempt == 5:
                    raise
                wait = 5 * (attempt + 1)
                print(f"  [io] read of {edge_file} failed ({e}); "
                      f"retry {attempt+1}/5 in {wait}s")
                time.sleep(wait)
        stamp_path = os.path.join(out_dir, ".edge_hash")
        spikes_complete = all(
            os.path.exists(os.path.join(out_dir, f"theta{t:03d}", "spikes.h5"))
            for t in ORIENTATIONS)
        stamp_ok = (os.path.exists(stamp_path)
                    and open(stamp_path).read().strip() == edge_hash)
        if spikes_complete and stamp_ok:
            print(f"\n=== {tag}: spikes match current connectome, reusing ===")
        else:
            for d in (out_dir, net_dir):
                if os.path.exists(d):
                    shutil.rmtree(d)
            print(f"\n=== simulating null {tag} ({edge_file}) [fresh] ===")
            step2.run_simulation(edge_file, tag, build_lgn=False)
            with open(stamp_path, "w") as sf:
                sf.write(edge_hash)
        r_null, _ = sim_bio_correlation(out_dir, node_df, sig, sig_idx,
                                        bio_nucleus_ids)
        results[tag] = r_null
        by_family.setdefault(fam, []).append(r_null)
        # resources-framing seed: log this null's (fidelity, budget, structure)
        _fam_structure = {
            "C1": "degree-preserving (retains in/out-degree sequence)",
            "C2": "distance-constrained (retains spatial connection profile)",
            "C3": "cell-type-shuffled (retains cell-type block structure)",
        }
        resource_log[tag] = {
            "bio_correlation": r_null,
            "family": fam,
            "structure_retained": _fam_structure.get(fam, fam),
            "structure_retained_fraction": edge_overlap_with_real(edge_file, real_edge_set),
            "training_iterations": 0,
            "budget": edge_resource_budget(edge_file),
        }
        print(f"  {tag}: null-sim vs bio r = {r_null:.4f}")

    # 3) report — z-score + empirical p per family (matches structural test format)
    def fam_stats(arr):
        arr = np.array(arr, dtype=float)
        mean = arr.mean()
        sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
        z = (r_real - mean) / sd if sd > 0 else float('nan')
        # one-tailed empirical p: fraction of nulls >= real, +1 smoothing
        emp_p = (np.sum(arr >= r_real) + 1) / (len(arr) + 1)
        frac_real_gt = float(np.mean(r_real > arr))
        return mean, sd, z, emp_p, frac_real_gt

    print("\n" + "=" * 72)
    print("OPTION A RESULT — real-sim vs biology, against null-sim-vs-bio")
    print("=" * 72)
    print(f"real-sim vs bio: r = {r_real:.4f}  (n=899, fixed; shared LGN)")
    print(f"{'Null':<26} {'mean':>8} {'sd':>7} {'z':>7} {'emp p':>8} {'real>null':>10}")
    print("-" * 72)
    lines = []
    for fam in sorted(by_family):
        mean, sd, z, emp_p, frac = fam_stats(by_family[fam])
        n = len(by_family[fam])
        print(f"{fam+' (n='+str(n)+')':<26} {mean:>8.4f} {sd:>7.4f} {z:>7.2f} "
              f"{emp_p:>8.3f} {100*frac:>9.0f}%")
        lines.append((fam, mean, sd, z, emp_p, frac, n))

    with open("step5_results.txt", "w") as f:
        f.write("Null-through-simulation — real-sim vs biology, against null-sim-vs-bio "
                "(null-through-simulation)\n")
        f.write("=" * 72 + "\n")
        f.write("Each null connectome is re-simulated through the IDENTICAL "
                "pipeline as real\n(shared seeded LGN drive, same model, matched "
                "synapse-count weighting).\nOnly the recurrent connectome differs "
                "between conditions.\n\n")
        f.write(f"real-sim vs bio: r = {r_real:.4f}  (n=899, fixed)\n\n")
        f.write(f"{'Null':<26} {'mean':>8} {'sd':>7} {'z':>7} {'emp p':>8} "
                f"{'real>null':>10}\n")
        f.write("-" * 72 + "\n")
        for fam, mean, sd, z, emp_p, frac, n in lines:
            f.write(f"{fam+' (n='+str(n)+')':<26} {mean:>8.4f} {sd:>7.4f} "
                    f"{z:>7.2f} {emp_p:>8.3f} {100*frac:>9.0f}%\n")
        f.write("\n  z = (real - null_mean) / null_sd ; "
                "emp p = fraction of null draws >= real (+1 smoothing)\n")
        f.write("\nNOTE: real is a single fixed value (one connectome, one shared\n"
                "LGN seed). The comparison is real vs the null WIRING distribution\n"
                "under matched input. Putting error bars on real (varying the LGN\n"
                "seed) is a separate analysis, deferred pending design discussion.\n")
        f.write("\nPROTOTYPE: iaf_psc_alpha LIF, excitatory-only, 1,762 of 75k "
                "neurons,\nlinear synapse-count weighting.\n")
    with open("step5_bio_correlations.json", "w") as f:
        json.dump(results, f, indent=2)
    # Resources-framing seed (additive; does not change any existing output).
    # Per-condition (fidelity, resource-budget, structure-retained) so the data
    # is in hand when the resources question becomes active. For the mouse
    # prototype the budgets are matched by construction (preserve_counts,
    # untrained, fixed params); the varying axis is family-level structure. The
    # quantitative resource trade-offs (e.g. training-data-vs-wiring exchange
    # rate) require a TRAINABLE model and are first measurable on the fly side
    # (Experiment 5), then here once a trained mouse model exists.
    with open("step5_resource_log.json", "w") as f:
        json.dump({
            "note": ("Resources-framing seed. Budgets (edge_count, "
                     "total_synapses, params, training) are matched across real "
                     "and all nulls by construction; the resource that varies is "
                     "wiring structure. structure_retained_fraction is "
                     "edge-overlap-with-real (fraction of real edges the null "
                     "keeps) - a crude structural-content proxy giving a numeric "
                     "x-axis for structure-vs-fidelity, NOT an information-"
                     "theoretic measure. Not yet a resource-worth measurement - "
                     "that needs a trainable model."),
            "conditions": resource_log,
        }, f, indent=2)
    print("\nSaved step5_results.txt, step5_bio_correlations.json, "
          "step5_resource_log.json")

    # 4) plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fams = sorted(by_family)
    for i, fam in enumerate(fams, start=1):
        arr = np.array(by_family[fam])
        jit = np.random.uniform(-0.06, 0.06, len(arr))
        ax.scatter(np.full(len(arr), i) + jit, arr, s=30, alpha=0.6,
                   color="tab:blue", label="null sims" if i == 1 else None)
    ax.axhline(r_real, color="tomato", lw=2, label=f"real-sim vs bio = {r_real:.3f}")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.set_xticks(range(1, len(fams) + 1)); ax.set_xticklabels(fams)
    ax.set_ylabel("null-sim vs biology (Spearman r)")
    ax.set_title("Null-through-simulation: real-sim vs null-sim, correlation to measured biology\n"
                 "(null-through-simulation, shared LGN)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("step5_results.png", dpi=150,
                                    bbox_inches="tight")
    print("Saved step5_results.png")


if __name__ == "__main__":
    main()
