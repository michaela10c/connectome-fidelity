"""
MICrONS connectome fidelity — Step 9
Goal: CKA comparison between structural connectivity profiles and
biological neural responses (in silico from digital twin model).

CKA asks: do neurons that are similarly connected also respond similarly
to natural stimuli? This is a direct structure-function alignment test.

Two matrices compared:
  X: structural connectivity profile matrix (N x M)
     each row = binary connectivity profile of one neuron
     (who it connects to among the 899 shared neurons)
  Y: biological response matrix (N x S)
     each row = in_silico_resp from node_data_v1.pkl
     (digital twin model responses to 4,999 stimulus timepoints)
     validated against in vivo benchmarks in Ding et al. Extended Data Fig. 2

CKA = tr(K_X H K_Y H) / sqrt(tr(K_X H K_X H) * tr(K_Y H K_Y H))
where K = X @ X.T (linear kernel), H = centering matrix

Run this comparison for:
  - Real connectome vs. biological responses
  - Candidate 1 (degree-preserving) vs. biological responses
  - Candidate 2 (distance-constrained) vs. biological responses
  - Candidate 3 (cell-type-block-shuffled) vs. biological responses

Significance via permutation test (neuron label shuffling, n=1000).

Requires:
  node_data_v1.pkl
  signal_corr_nucleus_ids.npy
  structural_rdm_nucleus_ids.npy
  candidate2_real_edges.json
  candidate2_random_edges.json
  candidate3 — regenerated from structural_rdm_candidate3.npy

Outputs:
  cka_results.txt        — CKA values and permutation p-values
  cka_comparison.png     — bar chart comparable to rsa_comparison_full.png
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import networkx as nx

# -------------------------------------------------------------------
# Step 1: Load shared neuron set and biological response matrix
# -------------------------------------------------------------------
print("Loading neuron sets and response data...")

struct_ids = np.load('structural_rdm_nucleus_ids.npy').tolist()
sig_corr_ids = np.load('signal_corr_nucleus_ids.npy').tolist()

# Use the same shared neuron set as RSA comparison
sig_id_set = set(sig_corr_ids)
shared_ids = [nid for nid in struct_ids if nid in sig_id_set]
N = len(shared_ids)
id_to_idx = {nid: i for i, nid in enumerate(shared_ids)}
print(f"Shared neurons: {N}")

# Load node_data and extract in_silico_resp for shared neurons
node_data = pd.read_pickle('node_data_v1.pkl')
subset = node_data[node_data['nucleus_id'].isin(shared_ids)].copy()
subset = subset.set_index('nucleus_id').loc[shared_ids]

# Biological response matrix Y: (N, 4999)
Y = np.stack(subset['in_silico_resp'].values).astype(np.float32)
print(f"Biological response matrix shape: {Y.shape}")

# Center Y (zero mean per neuron across stimuli)
Y = Y - Y.mean(axis=1, keepdims=True)

# -------------------------------------------------------------------
# Step 2: Build structural connectivity matrices
# -------------------------------------------------------------------
print("\nBuilding structural connectivity matrices...")

def build_adjacency(edge_list, id_to_idx, N):
    A = np.zeros((N, N), dtype=np.float32)
    for pre, post in edge_list:
        if pre in id_to_idx and post in id_to_idx:
            A[id_to_idx[pre], id_to_idx[post]] = 1.0
    return A

# Real graph
with open('candidate2_real_edges.json') as f:
    real_edges = json.load(f)
A_real = build_adjacency(real_edges, id_to_idx, N)
print(f"Real graph: {int(A_real.sum())} edges")

# Candidate 1 — regenerate from real degree sequence
G_real_sub = nx.from_numpy_array(A_real, create_using=nx.DiGraph())
in_seq  = [d for _, d in G_real_sub.in_degree()]
out_seq = [d for _, d in G_real_sub.out_degree()]
G_c1 = nx.DiGraph(nx.directed_configuration_model(in_seq, out_seq, seed=42))
G_c1.remove_edges_from(nx.selfloop_edges(G_c1))
A_c1 = nx.to_numpy_array(G_c1, dtype=np.float32)
print(f"Candidate 1: {int(A_c1.sum())} edges")

# Candidate 2
with open('candidate2_random_edges.json') as f:
    c2_edges = json.load(f)
A_c2 = build_adjacency(c2_edges, id_to_idx, N)
print(f"Candidate 2: {int(A_c2.sum())} edges")

# Candidate 3 — load RDM and reconstruct won't work; regenerate
# Use structural_rdm_candidate3 adjacency by reloading from saved json if present,
# otherwise fall back to recomputing from node_data
try:
    with open('candidate3_random_edges.json') as f:
        c3_edges = json.load(f)
    A_c3 = build_adjacency(c3_edges, id_to_idx, N)
    print(f"Candidate 3: {int(A_c3.sum())} edges (from saved JSON)")
except FileNotFoundError:
    # Regenerate Candidate 3 from scratch
    print("Regenerating Candidate 3 from node_data...")
    node_data_c3 = pd.read_pickle('node_data_v1.pkl')
    subset_c3 = node_data_c3[node_data_c3['nucleus_id'].isin(shared_ids)].copy()
    subset_c3 = subset_c3.set_index('nucleus_id').loc[shared_ids]
    subset_c3['group'] = subset_c3['brain_area'].astype(str) + '_' + subset_c3['layer'].astype(str)
    groups = subset_c3['group'].values
    unique_groups = sorted(set(groups))
    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    block_counts = {}
    for g_pre in unique_groups:
        for g_post in unique_groups:
            pre_idx = group_indices[g_pre]
            post_idx = group_indices[g_post]
            block = A_real[np.ix_(pre_idx, post_idx)].copy()
            if g_pre == g_post:
                np.fill_diagonal(block, 0)
            count = int(block.sum())
            if count > 0:
                block_counts[(g_pre, g_post)] = count

    np.random.seed(42)
    A_c3 = np.zeros((N, N), dtype=np.float32)
    for (g_pre, g_post), count in block_counts.items():
        pre_idx = group_indices[g_pre]
        post_idx = group_indices[g_post]
        all_pairs = [(pi, pj) for pi in pre_idx for pj in post_idx if pi != pj]
        if not all_pairs:
            continue
        replace = count > len(all_pairs)
        chosen = np.random.choice(len(all_pairs), size=count, replace=replace)
        for idx in chosen:
            pi, pj = all_pairs[idx]
            A_c3[pi, pj] = 1.0
    print(f"Candidate 3: {int(A_c3.sum())} edges (regenerated)")

# -------------------------------------------------------------------
# Step 3: CKA computation
# Linear CKA: CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X,K_X) * HSIC(K_Y,K_Y))
# where K = X @ X.T and HSIC uses centered kernels
# -------------------------------------------------------------------
def linear_cka(X, Y):
    """
    Compute linear CKA between X (N x d1) and Y (N x d2).
    Both should be mean-centered across samples (rows).
    """
    n = X.shape[0]
    K_X = X @ X.T
    K_Y = Y @ Y.T

    def center_kernel(K):
        row_mean = K.mean(axis=1, keepdims=True)
        col_mean = K.mean(axis=0, keepdims=True)
        grand_mean = K.mean()
        return K - row_mean - col_mean + grand_mean

    K_X_c = center_kernel(K_X)
    K_Y_c = center_kernel(K_Y)

    hsic_xy = np.sum(K_X_c * K_Y_c) / (n - 1) ** 2
    hsic_xx = np.sum(K_X_c * K_X_c) / (n - 1) ** 2
    hsic_yy = np.sum(K_Y_c * K_Y_c) / (n - 1) ** 2

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def rbf_kernel(X, sigma_frac=0.5):
    """
    Compute RBF kernel matrix for X (N x d).
    sigma = sigma_frac * median pairwise distance (Kornblith et al. convention).
    """
    # Pairwise squared distances
    sq_dists = np.sum(X ** 2, axis=1, keepdims=True) \
             + np.sum(X ** 2, axis=1) \
             - 2 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0)  # numerical safety
    median_dist = np.sqrt(np.median(sq_dists[sq_dists > 0]))
    sigma = sigma_frac * median_dist
    return np.exp(-sq_dists / (2 * sigma ** 2))


def rbf_cka(X, Y, sigma_frac=0.5):
    """
    Compute RBF CKA between X and Y using Kornblith et al. convention:
    sigma = sigma_frac * median pairwise Euclidean distance.
    """
    n = X.shape[0]
    K_X = rbf_kernel(X, sigma_frac)
    K_Y = rbf_kernel(Y, sigma_frac)

    def center_kernel(K):
        row_mean = K.mean(axis=1, keepdims=True)
        col_mean = K.mean(axis=0, keepdims=True)
        grand_mean = K.mean()
        return K - row_mean - col_mean + grand_mean

    K_X_c = center_kernel(K_X)
    K_Y_c = center_kernel(K_Y)

    hsic_xy = np.sum(K_X_c * K_Y_c) / (n - 1) ** 2
    hsic_xx = np.sum(K_X_c * K_X_c) / (n - 1) ** 2
    hsic_yy = np.sum(K_Y_c * K_Y_c) / (n - 1) ** 2

    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))

print("\nComputing Linear CKA values...")

# Center structural matrices
def center_rows(A):
    return A - A.mean(axis=1, keepdims=True)

cka_real = linear_cka(center_rows(A_real), Y)
cka_c1   = linear_cka(center_rows(A_c1),   Y)
cka_c2   = linear_cka(center_rows(A_c2),   Y)
cka_c3   = linear_cka(center_rows(A_c3),   Y)

print(f"Real connectome:  Linear CKA = {cka_real:.4f}")
print(f"Candidate 1:      Linear CKA = {cka_c1:.4f}")
print(f"Candidate 2:      Linear CKA = {cka_c2:.4f}")
print(f"Candidate 3:      Linear CKA = {cka_c3:.4f}")

print("\nComputing RBF CKA values (sigma = 0.5 * median distance)...")
rbf_real = rbf_cka(center_rows(A_real), Y)
rbf_c1   = rbf_cka(center_rows(A_c1),   Y)
rbf_c2   = rbf_cka(center_rows(A_c2),   Y)
rbf_c3   = rbf_cka(center_rows(A_c3),   Y)

print(f"Real connectome:  RBF CKA = {rbf_real:.4f}")
print(f"Candidate 1:      RBF CKA = {rbf_c1:.4f}")
print(f"Candidate 2:      RBF CKA = {rbf_c2:.4f}")
print(f"Candidate 3:      RBF CKA = {rbf_c3:.4f}")

# -------------------------------------------------------------------
# Step 4: Permutation test — shuffle neuron labels in Y
# -------------------------------------------------------------------
print(f"\nRunning permutation tests (n=1000)...")
np.random.seed(42)
n_perms = 1000

perm_cka = {k: np.zeros(n_perms) for k in ['real', 'c1', 'c2', 'c3']}
perm_rbf = {k: np.zeros(n_perms) for k in ['real', 'c1', 'c2', 'c3']}

A_real_c = center_rows(A_real)
A_c1_c   = center_rows(A_c1)
A_c2_c   = center_rows(A_c2)
A_c3_c   = center_rows(A_c3)

for i in range(n_perms):
    perm_idx = np.random.permutation(N)
    Y_perm = Y[perm_idx]
    perm_cka['real'][i] = linear_cka(A_real_c, Y_perm)
    perm_cka['c1'][i]   = linear_cka(A_c1_c,   Y_perm)
    perm_cka['c2'][i]   = linear_cka(A_c2_c,   Y_perm)
    perm_cka['c3'][i]   = linear_cka(A_c3_c,   Y_perm)
    perm_rbf['real'][i] = rbf_cka(A_real_c, Y_perm)
    perm_rbf['c1'][i]   = rbf_cka(A_c1_c,   Y_perm)
    perm_rbf['c2'][i]   = rbf_cka(A_c2_c,   Y_perm)
    perm_rbf['c3'][i]   = rbf_cka(A_c3_c,   Y_perm)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n_perms} done...")

perm_p_real = (perm_cka['real'] >= cka_real).mean()
perm_p_c1   = (perm_cka['c1']   >= cka_c1).mean()
perm_p_c2   = (perm_cka['c2']   >= cka_c2).mean()
perm_p_c3   = (perm_cka['c3']   >= cka_c3).mean()

rbf_p_real  = (perm_rbf['real'] >= rbf_real).mean()
rbf_p_c1    = (perm_rbf['c1']   >= rbf_c1).mean()
rbf_p_c2    = (perm_rbf['c2']   >= rbf_c2).mean()
rbf_p_c3    = (perm_rbf['c3']   >= rbf_c3).mean()

print(f"\nLinear CKA permutation p-values:")
print(f"Real: {perm_p_real:.4f}  C1: {perm_p_c1:.4f}  C2: {perm_p_c2:.4f}  C3: {perm_p_c3:.4f}")
print(f"\nRBF CKA permutation p-values:")
print(f"Real: {rbf_p_real:.4f}  C1: {rbf_p_c1:.4f}  C2: {rbf_p_c2:.4f}  C3: {rbf_p_c3:.4f}")

# -------------------------------------------------------------------
# Step 5: Save results
# -------------------------------------------------------------------
results_text = f"""CKA Comparison Results — MICrONS Connectome Fidelity
=====================================================
N neurons: {N}
Structural representation: binary connectivity profile (N x {N})
Biological representation: in silico responses to naturalistic stimuli
  from digital twin model (N x 4999), validated against in vivo
  benchmarks in Ding et al. 2025 Extended Data Fig. 2
Kernels: linear CKA and RBF CKA (sigma = 0.5 * median pairwise distance,
  per Kornblith et al. 2019 convention)

Linear CKA Results
------------------
Real connectome:  CKA = {cka_real:.4f}, perm p = {perm_p_real:.4f}
Candidate 1:      CKA = {cka_c1:.4f}, perm p = {perm_p_c1:.4f}
Candidate 2:      CKA = {cka_c2:.4f}, perm p = {perm_p_c2:.4f}
Candidate 3:      CKA = {cka_c3:.4f}, perm p = {perm_p_c3:.4f}

RBF CKA Results (sigma = 0.5 * median distance)
-------------------------------------------------
Real connectome:  CKA = {rbf_real:.4f}, perm p = {rbf_p_real:.4f}
Candidate 1:      CKA = {rbf_c1:.4f}, perm p = {rbf_p_c1:.4f}
Candidate 2:      CKA = {rbf_c2:.4f}, perm p = {rbf_p_c2:.4f}
Candidate 3:      CKA = {rbf_c3:.4f}, perm p = {rbf_p_c3:.4f}

Cross-validation with RSA
-------------------------
RSA (Spearman r): real=0.026***, C1=-0.006 n.s., C2=0.003 n.s., C3=0.013**
Linear CKA: real={cka_real:.4f}, C1={cka_c1:.4f}, C2={cka_c2:.4f}, C3={cka_c3:.4f}
RBF CKA:    real={rbf_real:.4f}, C1={rbf_c1:.4f}, C2={rbf_c2:.4f}, C3={rbf_c3:.4f}

Note: If linear CKA is flat across conditions but RBF CKA differentiates them,
this suggests the fine-grained wiring differences are captured in local geometry
but washed out by the dominant large principal components (see Kornblith et al. 2019,
Eq. 14 — linear CKA is eigenvalue-weighted, dominated by large PCs).
"""

with open('cka_results.txt', 'w') as f:
    f.write(results_text)
print("\nSaved cka_results.txt")

# -------------------------------------------------------------------
# Step 6: Plot — side by side linear and RBF
# -------------------------------------------------------------------
labels  = ['Real\nconnectome', 'Candidate 1\n(degree)',
           'Candidate 2\n(distance)', 'Candidate 3\n(cell type)']
colors   = ['steelblue', 'lightgray', 'tomato', 'mediumpurple']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, vals, pps, title in [
    (axes[0], [cka_real, cka_c1, cka_c2, cka_c3],
     [perm_p_real, perm_p_c1, perm_p_c2, perm_p_c3], 'Linear CKA'),
    (axes[1], [rbf_real, rbf_c1, rbf_c2, rbf_c3],
     [rbf_p_real, rbf_p_c1, rbf_p_c2, rbf_p_c3], 'RBF CKA (σ = 0.5 × median dist)')
]:
    ax.bar(labels, vals, color=colors, edgecolor='black',
           linewidth=0.8, width=0.5)
    for i, (v, pp) in enumerate(zip(vals, pps)):
        marker = '***' if pp < 0.001 else '**' if pp < 0.01 else \
                 '*' if pp < 0.05 else 'n.s.'
        ypos = v + 0.0002 if v >= 0 else v - 0.001
        ax.text(i, ypos, marker, ha='center', va='bottom', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("CKA\n(connectivity profiles vs. neural responses)", fontsize=10)
    ax.set_title(
        f"{title}\nConnectome geometry vs. functional responses (N=899)",
        fontsize=10
    )
    ax.set_ylim(min(0, min(vals)) - 0.002, max(vals) + 0.006)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cka_comparison.png', dpi=150, bbox_inches='tight')
print("Saved cka_comparison.png")

print("\n--- STEP 9 COMPLETE ---")
print(f"Linear CKA: real={cka_real:.4f} (p={perm_p_real:.4f})  "
      f"C1={cka_c1:.4f} (p={perm_p_c1:.4f})  "
      f"C2={cka_c2:.4f} (p={perm_p_c2:.4f})  "
      f"C3={cka_c3:.4f} (p={perm_p_c3:.4f})")
print(f"RBF CKA:    real={rbf_real:.4f} (p={rbf_p_real:.4f})  "
      f"C1={rbf_c1:.4f} (p={rbf_p_c1:.4f})  "
      f"C2={rbf_c2:.4f} (p={rbf_p_c2:.4f})  "
      f"C3={rbf_c3:.4f} (p={rbf_p_c3:.4f})")
