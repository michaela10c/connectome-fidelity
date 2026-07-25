"""
Structural-property preservation vs. biological fidelity — post-hoc analysis.

Purpose (strengthens the interpretation of the null-through-simulation result):
the null families differ 16-fold in biological fidelity (C1 degree 0.045,
C2 distance 0.028, C3 cell-type 0.003) while retaining nearly the same low
fraction of the real connectome's *specific* edges (5.1 / 4.6 / 2.1 %). This
script quantifies WHY: it measures, per family, how well each null preserves
several structural properties of the real connectome, so we can show that
fidelity tracks *degree-sequence preservation* specifically — not edge identity,
and not the other structural features. This converts the categorical null design
into a quantitative structure -> fidelity mapping.

Three analyses:
  (1) Edge-overlap vs fidelity, across all instances — show overlap does NOT
      predict fidelity (near-constant overlap, wide fidelity spread).
  (2) Degree-sequence preservation per family — the positive claim: fidelity
      tracks how well the null preserves each node's in/out degree.
  (3) A battery of structural-property preservations per family (degree, spatial
      distance profile, cell-type block structure, reciprocity, clustering) —
      each family should preserve its OWN target property and scramble others;
      the property whose preservation tracks fidelity is the fidelity-carrying one.

NO new simulations. Reads existing null edge files + the fidelity values already
in step5_resource_log.json (or step5_bio_correlations.json).

Reads:
    step4_null_manifest.json         (which null files exist, family/tag)
    candidate1_real_edges.json       (real connectome, [pre,post] or [pre,post,nsyn])
    null_edges_{fam}_seed{s}.json    (the nulls)
    node_data_v1.csv                 (positions + cell types, for #3)
    step5_bio_correlations.json      (fidelity per condition)
Writes:
    structure_vs_fidelity.txt        (summary tables)
    structure_vs_fidelity.png        (overlap-vs-fidelity + degree-preservation plots)
    structure_vs_fidelity.json       (per-family metrics, machine-readable)

Usage: python analyze_structure_vs_fidelity.py
"""

import os
import json
import numpy as np
import pandas as pd

REAL_EDGES = "candidate1_real_edges.json"
NODE_CSV   = "node_data_v1.csv"
MANIFEST   = "step4_null_manifest.json"
FIDELITY   = "step5_bio_correlations.json"


# ── loaders ───────────────────────────────────────────────────────────────────

def load_edge_pairs(path):
    """Return list of (pre, post) int tuples, tolerating [pre,post] or
    [pre,post,nsyn]."""
    with open(path) as f:
        raw = json.load(f)
    return [(int(e[0]), int(e[1])) for e in raw]


def degree_maps(edge_pairs):
    """Return (in_degree, out_degree) dicts keyed by node id."""
    ind, outd = {}, {}
    for pre, post in edge_pairs:
        outd[pre] = outd.get(pre, 0) + 1
        ind[post] = ind.get(post, 0) + 1
    return ind, outd


def degree_vector(deg_map, node_ids):
    """Dense degree vector over a fixed node ordering (0 for absent nodes)."""
    return np.array([deg_map.get(n, 0) for n in node_ids], dtype=float)


# ── structural-property preservation metrics ──────────────────────────────────

def degree_preservation(real_pairs, null_pairs, node_ids):
    """Correlation between real and null degree sequences (in and out).
    C1 should be ~1.0 by construction; C2/C3 whatever they happen to preserve."""
    ri, ro = degree_maps(real_pairs)
    ni, no = degree_maps(null_pairs)
    rin, nin = degree_vector(ri, node_ids), degree_vector(ni, node_ids)
    rout, nout = degree_vector(ro, node_ids), degree_vector(no, node_ids)
    # Pearson r; guard against zero variance
    def safe_corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "in_degree_r":  safe_corr(rin, nin),
        "out_degree_r": safe_corr(rout, nout),
    }


def edge_overlap(real_set, null_pairs):
    null_set = set(null_pairs)
    return len(real_set & null_set) / len(real_set) if real_set else float("nan")


def reciprocity(edge_pairs):
    """Fraction of edges whose reverse also exists."""
    s = set(edge_pairs)
    if not s:
        return float("nan")
    recip = sum(1 for (a, b) in s if (b, a) in s)
    return recip / len(s)


def distance_profile_preservation(real_pairs, null_pairs, pos_by_id, n_bins=20):
    """Compare the edge-length distributions (histogram over Euclidean distance)
    of real vs null via correlation of their binned densities. Captures how well
    the null preserves the spatial/distance connection profile (C2's target)."""
    def edge_lengths(pairs):
        L = []
        for a, b in pairs:
            if a in pos_by_id and b in pos_by_id:
                L.append(np.linalg.norm(pos_by_id[a] - pos_by_id[b]))
        return np.array(L)
    rl, nl = edge_lengths(real_pairs), edge_lengths(null_pairs)
    if len(rl) == 0 or len(nl) == 0:
        return float("nan")
    lo, hi = 0.0, np.percentile(rl, 99)
    edges = np.linspace(lo, hi, n_bins + 1)
    rh, _ = np.histogram(rl, bins=edges, density=True)
    nh, _ = np.histogram(nl, bins=edges, density=True)
    if rh.std() == 0 or nh.std() == 0:
        return float("nan")
    return float(np.corrcoef(rh, nh)[0, 1])


def celltype_block_preservation(real_pairs, null_pairs, layer_by_id):
    """Compare the layer-by-layer edge-count matrices (C3's target) via
    correlation of their flattened block-count matrices."""
    layers = sorted(set(layer_by_id.values()))
    idx = {l: i for i, l in enumerate(layers)}
    def block_counts(pairs):
        M = np.zeros((len(layers), len(layers)), dtype=float)
        for a, b in pairs:
            la, lb = layer_by_id.get(a), layer_by_id.get(b)
            if la is not None and lb is not None:
                M[idx[la], idx[lb]] += 1
        return M.flatten()
    rb, nb = block_counts(real_pairs), block_counts(null_pairs)
    if rb.std() == 0 or nb.std() == 0:
        return float("nan")
    return float(np.corrcoef(rb, nb)[0, 1])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    real_pairs = load_edge_pairs(REAL_EDGES)
    real_set = set(real_pairs)
    node_df = pd.read_csv(NODE_CSV)
    node_ids = node_df["nucleus_id"].astype(int).tolist()

    # positions + layers keyed by nucleus_id (for #3)
    pos_by_id, layer_by_id = {}, {}
    def ct_to_layer(ct):
        ct = str(ct)
        if ct.startswith("23"): return "L2/3"
        if ct.startswith("4"):  return "L4"
        if ct.startswith("5"):  return "L5"
        if ct.startswith("6"):  return "L6"
        return "other"
    for _, row in node_df.iterrows():
        nid = int(row["nucleus_id"])
        pos_by_id[nid] = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        layer_by_id[nid] = ct_to_layer(row["cell_type"])

    real_recip = reciprocity(real_pairs)

    # fidelity per condition
    with open(FIDELITY) as f:
        fid = json.load(f)

    # which null files exist
    with open(MANIFEST) as f:
        manifest = json.load(f)

    # accumulate per-family metrics across seeds
    fam_metrics = {}
    for entry in manifest:
        fam = entry["family"]; tag = entry["tag"]; ef = entry["edge_file"]
        if not os.path.exists(ef):
            print(f"  [skip] {ef} not found")
            continue
        npairs = load_edge_pairs(ef)
        dp = degree_preservation(real_pairs, npairs, node_ids)
        m = {
            "fidelity": fid.get(tag, float("nan")),
            "edge_overlap": edge_overlap(real_set, npairs),
            "in_degree_r": dp["in_degree_r"],
            "out_degree_r": dp["out_degree_r"],
            "distance_profile_r": distance_profile_preservation(real_pairs, npairs, pos_by_id),
            "celltype_block_r": celltype_block_preservation(real_pairs, npairs, layer_by_id),
            "reciprocity": reciprocity(npairs),
        }
        fam_metrics.setdefault(fam, []).append(m)

    # ---- aggregate + report ----
    def agg(vals):
        a = np.array([v for v in vals if not (isinstance(v, float) and np.isnan(v))], dtype=float)
        return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), 0.0)

    families = sorted(fam_metrics)
    metrics = ["fidelity", "edge_overlap", "in_degree_r", "out_degree_r",
               "distance_profile_r", "celltype_block_r", "reciprocity"]

    summary = {"real": {"reciprocity": real_recip}, "families": {}}
    lines = []
    lines.append("Structural-property preservation vs. biological fidelity")
    lines.append("=" * 72)
    lines.append(f"real connectome reciprocity: {real_recip:.3f}\n")
    header = f"{'metric':<22}" + "".join(f"{fam:>14}" for fam in families)
    lines.append(header)
    lines.append("-" * len(header))
    for met in metrics:
        row = f"{met:<22}"
        for fam in families:
            mean, sd = agg([d[met] for d in fam_metrics[fam]])
            row += f"{mean:>10.3f}±{sd:<3.2f}" if not np.isnan(mean) else f"{'n/a':>14}"
            summary["families"].setdefault(fam, {})[met] = {"mean": mean, "sd": sd}
        lines.append(row)

    lines.append("")
    lines.append("READING:")
    lines.append("  - edge_overlap is low and similar across families, but fidelity")
    lines.append("    spans a wide range -> edge identity does not drive fidelity.")
    lines.append("  - in/out_degree_r ~1.0 only for the family that preserves degree;")
    lines.append("    if fidelity tracks degree_r (not overlap, not the other property")
    lines.append("    correlations), the fidelity-carrying structure is the degree")
    lines.append("    sequence specifically.")
    lines.append("  - each family should score highest on ITS OWN target property")
    lines.append("    (C1 degree, C2 distance_profile, C3 celltype_block); the one whose")
    lines.append("    preservation co-varies with fidelity is the fidelity-relevant one.")

    report = "\n".join(lines)
    print(report)
    with open("structure_vs_fidelity.txt", "w") as f:
        f.write(report + "\n")
    with open("structure_vs_fidelity.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- plots ----
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # (1) overlap vs fidelity, all instances
        colors = {"C1": "tab:blue", "C2": "tab:orange", "C3": "tab:green"}
        for fam in families:
            ov = [d["edge_overlap"] for d in fam_metrics[fam]]
            fd = [d["fidelity"] for d in fam_metrics[fam]]
            ax1.scatter(ov, fd, s=25, alpha=0.6, label=fam,
                        color=colors.get(fam, "gray"))
        ax1.set_xlabel("edge overlap with real")
        ax1.set_ylabel("biological fidelity (Spearman r)")
        ax1.set_title("Edge overlap does NOT predict fidelity\n(low, similar overlap; wide fidelity range)")
        ax1.legend(); ax1.grid(alpha=0.3)

        # (2) degree preservation vs fidelity, per-family means
        for fam in families:
            dr = agg([d["in_degree_r"] for d in fam_metrics[fam]])[0]
            fd = agg([d["fidelity"] for d in fam_metrics[fam]])[0]
            ax2.scatter(dr, fd, s=120, color=colors.get(fam, "gray"))
            ax2.annotate(fam, (dr, fd), textcoords="offset points", xytext=(8, 4))
        ax2.set_xlabel("in-degree-sequence preservation (r with real)")
        ax2.set_ylabel("biological fidelity (Spearman r)")
        ax2.set_title("Fidelity tracks degree-sequence preservation")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("structure_vs_fidelity.png", dpi=150, bbox_inches="tight")
        print("\nSaved structure_vs_fidelity.png")
    except Exception as e:
        print(f"\n[plot skipped: {e}]")

    print("Saved structure_vs_fidelity.txt, structure_vs_fidelity.json")


if __name__ == "__main__":
    main()
