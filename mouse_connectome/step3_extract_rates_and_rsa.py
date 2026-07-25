"""
Step 3: Extract per-neuron firing rates from PointNet spike output, build a
simulation-based RDM, and run RSA against structural RDMs — reporting results
against the GENERATIVE NULL DISTRIBUTION (the scientifically meaningful test).

WHY THIS DESIGN (read before interpreting results):
  The question is NOT "is the sim-vs-real correlation non-zero" (a label
  permutation test answers that — weak). The question IS "does the REAL
  connectome correlate with the simulation better than structured-random
  null graphs do" — the Brunton/Eon fidelity question. The correct null is
  therefore the distribution of sim-vs-null RSA over many null redraws
  (C1 degree-preserving, C2 distance-constrained, C3 cell-type-shuffled),
  NOT a single null draw (which can land in its tail) and NOT label shuffling.

  Primary result:   real connectome RSA vs each null DISTRIBUTION
                    (z-score + empirical p = fraction of nulls >= real).
  Secondary check:  label-permutation p on the real correlation
                    (tests only "is it non-zero"; reported but not the headline).

  This folds in the former step3b diagnostic; step3b can be deleted after this.

SCOPE / HONESTY:
  - Sim RDM is built from the REAL-connectome simulation and held fixed.
  - Nulls vary the STRUCTURAL graph (C1/C2/C3), regenerated per seed.
  - This does NOT re-simulate on each null connectome (a separate, costlier
    "Question B" test). It tests whether simulated geometry aligns with real
    wiring more than with structured-random wiring.
  - Prototype: iaf_psc_alpha LIF, excitatory-only, 1,764 of 75k neurons,
    linear synapse-count weighting.

Inputs (from previous steps):
    output_pointnet/theta{000..315}/spikes.h5   PointNet spikes (step2)
    node_data_v1.csv                            neuron metadata (step00)
    structural_rdm_real.npy                     real connectome structural RDM (step6)
    structural_rdm_nucleus_ids.npy              nucleus IDs for structural RDMs (step6)
    candidate2_real_edges.json                  real edges, nucleus space (step4)
    signal_corr_matrix.npy                      biological reference RDM (step5)
    signal_corr_nucleus_ids.npy                 nucleus IDs for signal corr (step5)

Outputs:
    simulation_rdm.npy / simulation_rdm_nucleus_ids.npy
    simulation_rate_matrix.npy
    step3_rsa_results.txt        full result table (generative null + secondary)
    step3_rsa_results.png        real-vs-null-distribution plot
"""

import os
import json
import numpy as np
import pandas as pd
import h5py
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ── parameters ────────────────────────────────────────────────────────────────

ORIENTATIONS   = [0, 45, 90, 135, 180, 225, 270, 315]
OUTPUT_DIR     = "output_pointnet"
TSTOP_MS       = 3000.0
GRAY_SCREEN_MS = 500.0
ANALYSIS_MS    = TSTOP_MS - GRAY_SCREEN_MS

N_NULL_SEEDS   = 100     # generative-null redraws per null family (primary test)
N_LABEL_PERMS  = 1000    # label-shuffle permutations (secondary non-zero check)
RNG_SEED       = 42

# ── Step 3a: load neuron metadata ─────────────────────────────────────────────

print("Step 3a: Loading neuron metadata...")
node_df = pd.read_csv("node_data_v1.csv", index_col="node_id")
node_id_to_nucleus = node_df['nucleus_id'].to_dict()
nucleus_to_node_id = {v: k for k, v in node_id_to_nucleus.items()}
N_neurons = len(node_df)
print(f"  {N_neurons} neurons, node_ids 0..{N_neurons-1}")

# ── Step 3b: extract firing rates per neuron per orientation ──────────────────

print("\nStep 3b: Extracting firing rates from spike files...")
rate_matrix = np.zeros((N_neurons, len(ORIENTATIONS)), dtype=np.float32)

for col, theta in enumerate(ORIENTATIONS):
    fpath = os.path.join(OUTPUT_DIR, f"theta{theta:03d}", "spikes.h5")
    with h5py.File(fpath, 'r') as h5:
        node_ids   = h5['spikes/microns_v1/node_ids'][()]
        timestamps = h5['spikes/microns_v1/timestamps'][()]
    mask = timestamps >= GRAY_SCREEN_MS
    node_ids = node_ids[mask]
    unique_ids, counts = np.unique(node_ids, return_counts=True)
    for nid, count in zip(unique_ids, counts):
        if nid < N_neurons:
            rate_matrix[nid, col] = count / (ANALYSIS_MS / 1000.0)
    print(f"  theta={theta:>3}deg: {mask.sum():>7} spikes (after gray screen), "
          f"mean rate = {rate_matrix[:, col].mean():.2f} Hz")

np.save("simulation_rate_matrix.npy", rate_matrix)
print(f"\n  Saved simulation_rate_matrix.npy: {rate_matrix.shape}")

# ── Step 3c: intersection with structural RDM neuron set ──────────────────────

print("\nStep 3c: Finding intersection with structural RDM neuron set...")
structural_nucleus_ids = np.load("structural_rdm_nucleus_ids.npy")
intersection_nucleus_ids, intersection_node_ids = [], []
for nid in structural_nucleus_ids:
    node_id = nucleus_to_node_id.get(int(nid))
    if node_id is not None:
        intersection_nucleus_ids.append(int(nid))
        intersection_node_ids.append(node_id)
intersection_nucleus_ids = np.array(intersection_nucleus_ids)
intersection_node_ids    = np.array(intersection_node_ids)
N = len(intersection_nucleus_ids)
print(f"  Structural RDM neurons: {len(structural_nucleus_ids)}")
print(f"  Intersection with simulation: {N}")

# ── Step 3d: build simulation RDM over intersection neurons ───────────────────

print("\nStep 3d: Building simulation RDM...")
intersect_rates = rate_matrix[intersection_node_ids, :]   # (N, 8)
sim_rdm_full = squareform(pdist(intersect_rates, metric='cosine'))
np.save("simulation_rdm.npy", sim_rdm_full)
np.save("simulation_rdm_nucleus_ids.npy", intersection_nucleus_ids)
print(f"  Saved simulation_rdm.npy: {sim_rdm_full.shape}")

triu = np.triu_indices(N, k=1)
vec_sim = sim_rdm_full[triu]

def rsa(vec_a, vec_b):
    m = np.isfinite(vec_a) & np.isfinite(vec_b)
    return spearmanr(vec_a[m], vec_b[m])[0]

# ── Step 3e: fixed REAL connectome structural RDM (aligned) ───────────────────

print("\nStep 3e: Loading real structural RDM and building adjacency for nulls...")
real_rdm = np.load("structural_rdm_real.npy")
# align real_rdm to intersection order
sid_to_idx = {int(n): i for i, n in enumerate(structural_nucleus_ids)}
real_sel = [sid_to_idx[int(n)] for n in intersection_nucleus_ids]
real_rdm_a = real_rdm[np.ix_(real_sel, real_sel)]
r_real = rsa(vec_sim, real_rdm_a[triu])
print(f"  sim vs REAL connectome (fixed): r = {r_real:.4f}")

# Build real adjacency over the intersection set (for null regeneration)
id_to_idx = {nid: i for i, nid in enumerate(intersection_nucleus_ids.tolist())}
with open("candidate2_real_edges.json") as f:
    real_edges = json.load(f)
A_real = np.zeros((N, N), dtype=np.float32)
for e in real_edges:
    pre, post = int(e[0]), int(e[1])
    if pre in id_to_idx and post in id_to_idx:
        A_real[id_to_idx[pre], id_to_idx[post]] = 1.0
print(f"  real edges within intersection set: {int(A_real.sum())}")

# positions (microns) and layer groups for C2/C3
node_by_nuc = node_df.set_index("nucleus_id")
pos = np.zeros((N, 3), dtype=float)
for nid, i in id_to_idx.items():
    if nid in node_by_nuc.index:
        row = node_by_nuc.loc[nid]
        pos[i] = [float(row["x"]) / 1000.0, float(row["y"]) / 1000.0, float(row["z"]) / 1000.0]

def cell_type_to_layer(ct):
    ct = str(ct)
    if ct.startswith("23"): return "L2/3"
    if ct.startswith("4"):  return "L4"
    if ct.startswith("5"):  return "L5"
    if ct.startswith("6"):  return "L6"
    return "other"

groups = np.array([
    cell_type_to_layer(node_by_nuc.loc[nid]["cell_type"]) if nid in node_by_nuc.index else "other"
    for nid in intersection_nucleus_ids
])

# ── null generators (seeded) ──────────────────────────────────────────────────

def connectivity_rdm(A):
    A = A.copy()
    rn = np.linalg.norm(A, axis=1, keepdims=True)
    z = (rn == 0).flatten()
    if z.any():
        A[z] = 1.0 / A.shape[1]
        rn = np.linalg.norm(A, axis=1, keepdims=True)
    An = A / rn
    cs = np.clip(An @ An.T, -1, 1)
    rdm = (1 - cs).astype(np.float32)
    np.fill_diagonal(rdm, 0)
    return rdm

def make_C1(seed):
    G = nx.from_numpy_array(A_real, create_using=nx.DiGraph())
    Gc = nx.DiGraph(nx.directed_configuration_model(
        [d for _, d in G.in_degree()], [d for _, d in G.out_degree()], seed=seed))
    Gc.remove_edges_from(nx.selfloop_edges(Gc))
    return connectivity_rdm(nx.to_numpy_array(Gc, dtype=np.float32))

def _distance_profile(rng, n_bins=20):
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

def make_C2(seed):
    rng = np.random.default_rng(seed)
    edges, prob = _distance_profile(rng)
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
    return connectivity_rdm(A)

def make_C3(seed):
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
    return connectivity_rdm(A)

# ── Step 3f: PRIMARY test — real vs generative-null DISTRIBUTIONS ──────────────

print(f"\nStep 3f: PRIMARY test — real vs generative-null distributions "
      f"({N_NULL_SEEDS} seeds per null)...")
null_rs = {"C1 (degree-preserving)": [],
           "C2 (distance-constrained)": [],
           "C3 (cell-type-shuffled)": []}
gen = {"C1 (degree-preserving)": make_C1,
       "C2 (distance-constrained)": make_C2,
       "C3 (cell-type-shuffled)": make_C3}

for s in range(N_NULL_SEEDS):
    for name, fn in gen.items():
        null_rs[name].append(rsa(vec_sim, fn(s)[triu]))
    if (s + 1) % 20 == 0:
        print(f"  {s+1}/{N_NULL_SEEDS} null seeds done...")

def null_stats(name):
    arr = np.array(null_rs[name])
    mean, sd = arr.mean(), arr.std()
    z = (r_real - mean) / sd if sd > 0 else float('inf')
    emp_p = (np.sum(arr >= r_real) + 1) / (len(arr) + 1)  # one-tailed, +1 smoothing
    return arr, mean, sd, z, emp_p

# ── Step 3g: SECONDARY check — label permutation on real correlation ──────────

print(f"\nStep 3g: SECONDARY check — label-permutation test on real correlation "
      f"({N_LABEL_PERMS} perms)...")
rng = np.random.default_rng(RNG_SEED)
vec_real = real_rdm_a[triu]
null_label = np.zeros(N_LABEL_PERMS)
for i in range(N_LABEL_PERMS):
    perm = rng.permutation(N)
    null_label[i] = rsa(vec_sim, real_rdm_a[np.ix_(perm, perm)][triu])
label_p = (np.sum(null_label >= r_real) + 1) / (N_LABEL_PERMS + 1)
print(f"  label-permutation p (is correlation non-zero?): {label_p:.4f}")

# ── Step 3h: optional sim-vs-biological-reference (signal correlation) ─────────

bio_line = None
try:
    sig = np.load("signal_corr_matrix.npy")
    sig_ids = np.load("signal_corr_nucleus_ids.npy")
    sig_idx = {int(n): i for i, n in enumerate(sig_ids)}
    common = [n for n in intersection_nucleus_ids if int(n) in sig_idx]
    if len(common) >= 50:
        sim_sel = [list(intersection_nucleus_ids).index(n) for n in common]
        bio_sel = [sig_idx[int(n)] for n in common]
        sim_c = sim_rdm_full[np.ix_(sim_sel, sim_sel)]
        bio_c = 1.0 - sig[np.ix_(bio_sel, bio_sel)]
        nb = len(common); tb = np.triu_indices(nb, k=1)
        r_bio = rsa(sim_c[tb], bio_c[tb])
        # label-perm p for bio
        rngb = np.random.default_rng(RNG_SEED)
        nlb = np.zeros(N_LABEL_PERMS)
        for i in range(N_LABEL_PERMS):
            p = rngb.permutation(nb)
            nlb[i] = rsa(sim_c[tb], bio_c[np.ix_(p, p)][tb])
        bio_p = (np.sum(nlb >= r_bio) + 1) / (N_LABEL_PERMS + 1)
        bio_line = (len(common), r_bio, bio_p)
        print(f"\n  sim vs biological signal-correlation reference: "
              f"r = {r_bio:.4f}, label-perm p = {bio_p:.4f} (n={nb})")
except FileNotFoundError:
    print("\n  (signal_corr_matrix.npy not found — skipping bio reference comparison)")

# ── Step 3i: report ───────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("PRIMARY RESULT — real connectome vs generative-null distributions")
print("=" * 72)
print(f"sim vs REAL connectome (fixed): r = {r_real:.4f}")
print(f"{'Null':<26} {'mean':>8} {'sd':>7} {'z':>7} {'emp p':>8} {'real>null':>10}")
print("-" * 72)
lines = []
for name in null_rs:
    arr, mean, sd, z, emp_p = null_stats(name)
    frac = np.mean(r_real > arr)
    print(f"{name:<26} {mean:>8.4f} {sd:>7.4f} {z:>7.2f} {emp_p:>8.3f} {100*frac:>9.0f}%")
    lines.append((name, mean, sd, z, emp_p, frac))

with open("step3_rsa_results.txt", "w") as f:
    f.write("Step 3 — Simulation RSA: real connectome vs generative-null distributions\n")
    f.write("=" * 72 + "\n")
    f.write(f"N neurons (intersection): {N}\n")
    f.write(f"Null seeds per family: {N_NULL_SEEDS}\n")
    f.write(f"Simulation: real-connectome PointNet, synapse-count weighting "
            f"(prototype: iaf_psc_alpha, excitatory-only, {N_neurons} of 75k)\n\n")
    f.write("PRIMARY TEST (the Brunton/Eon question): does the REAL connectome\n")
    f.write("correlate with the simulated geometry better than STRUCTURED-RANDOM\n")
    f.write("null graphs? Null = distribution of sim-vs-null RSA over redraws.\n\n")
    f.write(f"sim vs REAL connectome (fixed): r = {r_real:.4f}\n\n")
    f.write(f"{'Null':<26} {'mean':>8} {'sd':>7} {'z':>7} {'emp p':>8} {'real>null':>10}\n")
    f.write("-" * 72 + "\n")
    for name, mean, sd, z, emp_p, frac in lines:
        f.write(f"{name:<26} {mean:>8.4f} {sd:>7.4f} {z:>7.2f} {emp_p:>8.3f} "
                f"{100*frac:>9.0f}%\n")
    f.write("\n  z = (real - null_mean) / null_sd ; "
            "emp p = fraction of null draws >= real (+1 smoothing)\n")
    f.write("\nSECONDARY CHECK (weaker; 'is the correlation non-zero?'):\n")
    f.write(f"  label-permutation p on real correlation: {label_p:.4f} "
            f"({N_LABEL_PERMS} perms)\n")
    if bio_line is not None:
        nb, r_bio, bio_p = bio_line
        f.write("\nSIM vs BIOLOGICAL REFERENCE (signal correlation RDM):\n")
        f.write(f"  r = {r_bio:.4f}, label-perm p = {bio_p:.4f} (n={nb} neurons)\n")
    f.write("\nSCOPE: sim held fixed on the real connectome; nulls vary STRUCTURE\n")
    f.write("(not re-simulation on null graphs). Prototype caveats apply.\n")
print("\nSaved step3_rsa_results.txt")

# ── Step 3j: plot — real vs null distributions ────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
names = list(null_rs.keys())
data = [np.array(null_rs[n]) for n in names]
bp = ax.boxplot(data, tick_labels=[n.split(" (")[0] + "\n(" + n.split(" (")[1]
                                    for n in names],
                showfliers=True, widths=0.5)
for i, d in enumerate(data, start=1):
    jitter = np.random.uniform(-0.08, 0.08, len(d))
    ax.scatter(np.full(len(d), i) + jitter, d, s=10, alpha=0.4,
               color="tab:blue", zorder=3)
ax.axhline(r_real, color="tomato", lw=2,
           label=f"real connectome (fixed) = {r_real:.3f}")
ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
ax.set_ylabel("RSA: sim RDM vs structural RDM (Spearman r)")
ax.set_title(f"Simulation RSA — real connectome vs generative-null distributions\n"
             f"({N_NULL_SEEDS} null seeds each; synapse-count-weighted prototype, N={N})")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("step3_rsa_results.png", dpi=150, bbox_inches="tight")
print("Saved step3_rsa_results.png")

print("\n" + "=" * 72)
print("Step 3 complete.")
print(f"  sim vs REAL: r = {r_real:.4f}")
for name, mean, sd, z, emp_p, frac in lines:
    print(f"  vs {name}: null {mean:.4f}±{sd:.4f}, z={z:.1f}, emp p={emp_p:.3f}, "
          f"real wins {100*frac:.0f}%")
if bio_line is not None:
    print(f"  sim vs biological reference: r = {bio_line[1]:.4f}")
print("\n(step3b diagnostic is now folded into step3 — safe to delete step3b.)")
