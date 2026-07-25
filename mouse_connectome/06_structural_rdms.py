"""
MICrONS orientation script — Step 6
Goal: build structural RDMs for the real connectome and both null models,
restricted to the 899 neurons that have signal correlation data.

Each neuron is represented by its connectivity profile vector — its row
in the adjacency matrix (outgoing connections to all other neurons in
the population). Pairwise cosine distances between these vectors form
the structural RDM.

Requires:
- candidate1_random_edges.json, candidate1_nucleus_ids.json
  (from 02_null_candidate1_degree_preserving.py)
- candidate2_real_edges.json, candidate2_random_edges.json, candidate2_nucleus_ids.json
  (from 04_null_candidate2_distance_constrained.py; candidate2_real_edges.json is the authoritative real graph)
- signal_corr_nucleus_ids.npy (from 05_biological_reference_signal_correlation.py)

Outputs:
- structural_rdm_real.npy       : (N, N) cosine distance RDM, real connectome
- structural_rdm_candidate1.npy : (N, N) cosine distance RDM, degree-preserving null
- structural_rdm_candidate2.npy : (N, N) cosine distance RDM, distance-constrained null
- structural_rdm_nucleus_ids.npy: (N,) nucleus_ids in RDM row/column order
  (subset of signal_corr_nucleus_ids that also appear in the connectivity graph)
"""

import numpy as np
import json
from scipy.spatial.distance import cdist

# -------------------------------------------------------------------
# Step 1: Load nucleus_ids from both sources and find intersection
# -------------------------------------------------------------------
print("Loading nucleus ID sets...")

sig_corr_ids = np.load('signal_corr_nucleus_ids.npy')
sig_corr_id_set = set(sig_corr_ids.tolist())
print(f"Neurons with signal correlation data: {len(sig_corr_id_set)}")

with open('candidate1_nucleus_ids.json') as f:
    conn_ids = set(json.load(f))
print(f"Neurons in connectivity graph: {len(conn_ids)}")

# Intersection: neurons with both connectivity and functional data
shared_ids = sorted(sig_corr_id_set & conn_ids)
N = len(shared_ids)
id_to_idx = {nid: i for i, nid in enumerate(shared_ids)}
print(f"Neurons with both connectivity and signal correlation: {N}")

# -------------------------------------------------------------------
# Step 2: Build adjacency matrices
# Use candidate2_real_edges.json as authoritative real graph —
# candidate1 and candidate2 use the same real synapses, but candidate2's
# nucleus_id mapping is more reliable (it includes positions).
# candidate1_random_edges.json is the degree-preserving null.
# candidate2_random_edges.json is the distance-constrained null.
# -------------------------------------------------------------------
def build_adjacency(edge_list, id_to_idx, N):
    """Build NxN binary adjacency matrix from edge list."""
    A = np.zeros((N, N), dtype=np.float32)
    n_included = 0
    for pre, post in edge_list:
        if pre in id_to_idx and post in id_to_idx:
            A[id_to_idx[pre], id_to_idx[post]] = 1.0
            n_included += 1
    return A, n_included

print("\nBuilding adjacency matrices...")

# Real graph — use candidate2's real edges (same synapse data, better mapping)
with open('candidate2_real_edges.json') as f:
    real_edges = json.load(f)
A_real, n_real = build_adjacency(real_edges, id_to_idx, N)
print(f"Real graph: {n_real} edges within shared neuron set "
      f"(density: {n_real / (N*(N-1)):.4f})")

# Candidate 1 null (degree-preserving) — regenerate directly from A_real
# This avoids dependency on step 2's nucleus_id mapping
import networkx as nx

print("Generating Candidate 1 (degree-preserving rewiring) from real graph...")
G_real_sub = nx.from_numpy_array(A_real, create_using=nx.DiGraph())
in_seq = [d for _, d in G_real_sub.in_degree()]
out_seq = [d for _, d in G_real_sub.out_degree()]
G_c1 = nx.directed_configuration_model(in_seq, out_seq, seed=42)
G_c1 = nx.DiGraph(G_c1)
G_c1.remove_edges_from(nx.selfloop_edges(G_c1))
A_c1 = nx.to_numpy_array(G_c1, dtype=np.float32)
n_c1 = int(A_c1.sum())
print(f"Candidate 1 random: {n_c1} edges within shared neuron set")

# Candidate 2 null (distance-constrained)
with open('candidate2_random_edges.json') as f:
    c2_edges = json.load(f)
A_c2, n_c2 = build_adjacency(c2_edges, id_to_idx, N)
print(f"Candidate 2 random: {n_c2} edges within shared neuron set")

# -------------------------------------------------------------------
# Step 3: Compute pairwise cosine distance RDMs
# Each neuron's connectivity profile = its row in the adjacency matrix
# (who it sends connections to)
# Cosine distance = 1 - cosine similarity
# -------------------------------------------------------------------
print("\nComputing structural RDMs (pairwise cosine distance)...")

def connectivity_rdm(A):
    """Compute NxN cosine distance RDM from adjacency matrix rows."""
    # Handle zero-vector rows (neurons with no outgoing connections
    # in this subset) by replacing with uniform vector to avoid NaN
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    zero_mask = (row_norms == 0).flatten()
    if zero_mask.any():
        print(f"  Warning: {zero_mask.sum()} neurons have no outgoing "
              f"connections in shared subset — using uniform profile")
        A[zero_mask] = 1.0 / A.shape[1]
        row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_norm = A / row_norms
    # Cosine similarity matrix, then convert to distance
    cos_sim = A_norm @ A_norm.T
    cos_sim = np.clip(cos_sim, -1, 1)
    rdm = 1 - cos_sim
    np.fill_diagonal(rdm, 0)
    return rdm.astype(np.float32)

rdm_real = connectivity_rdm(A_real.copy())
print(f"Real RDM: mean distance = {rdm_real[np.triu_indices(N, k=1)].mean():.4f}")

rdm_c1 = connectivity_rdm(A_c1.copy())
print(f"Candidate 1 RDM: mean distance = {rdm_c1[np.triu_indices(N, k=1)].mean():.4f}")

rdm_c2 = connectivity_rdm(A_c2.copy())
print(f"Candidate 2 RDM: mean distance = {rdm_c2[np.triu_indices(N, k=1)].mean():.4f}")

# -------------------------------------------------------------------
# Step 4: Save outputs
# -------------------------------------------------------------------
shared_ids_array = np.array(shared_ids, dtype=np.int64)

np.save('structural_rdm_real.npy', rdm_real)
np.save('structural_rdm_candidate1.npy', rdm_c1)
np.save('structural_rdm_candidate2.npy', rdm_c2)
np.save('structural_rdm_nucleus_ids.npy', shared_ids_array)

print(f"\nSaved structural RDMs for {N} neurons:")
print("  structural_rdm_real.npy")
print("  structural_rdm_candidate1.npy")
print("  structural_rdm_candidate2.npy")
print("  structural_rdm_nucleus_ids.npy")
print("\nReady for step 7: RSA comparison against signal correlation RDM.")
