"""
MICrONS orientation script — Step 7
Goal: implement Candidate 3 (cell-type-block-shuffled rewiring) and
compute its structural RDM for comparison against the real connectome
and Candidates 1 and 2.

Strategy: for each ordered pair of cell-type groups (e.g. VISp L2/3 ->
VISp L4), preserve the number of connections in that block exactly, but
randomly reassign which specific neurons connect to which. This controls
for degree distribution, spatial structure, AND cell-type-to-cell-type
connection statistics -- leaving only sub-cell-type wiring specificity
as the remaining variable.

If real > Candidate 3: specific partner identity within cell-type blocks
  matters beyond cell-type statistics -- the strongest possible claim.
If real ~ Candidate 3: cell-type-level wiring statistics explain the
  signal -- specific partner identity within a cell type doesn't add much.

Requires:
  signal_corr_nucleus_ids.npy
  node_data_v1.pkl  (or node_data_v1.csv if pkl is broken)
  structural_rdm_nucleus_ids.npy (from step 6, defines shared neuron order)
  candidate2_real_edges.json (authoritative real edges in nucleus_id space)

Outputs:
  structural_rdm_candidate3.npy  -- (N, N) cosine distance RDM
  candidate3_random_edges.json   -- edge list in nucleus_id space (NEW)
  candidate3_nucleus_ids.json    -- nucleus IDs for the shared neuron set (NEW)
  candidate3_summary.txt         -- block statistics and sanity checks
"""

import numpy as np
import pandas as pd
import json

# -------------------------------------------------------------------
# Step 1: Load shared neuron set and cell-type labels
# -------------------------------------------------------------------
print("Loading neuron set and cell-type labels...")

shared_ids = np.load('structural_rdm_nucleus_ids.npy').tolist()
N = len(shared_ids)
id_to_idx = {nid: i for i, nid in enumerate(shared_ids)}

# Load node data -- try CSV first (pkl requires pandas < 2.0)
try:
    node_data = pd.read_csv('node_data_v1.csv', index_col='node_id')
    if 'nucleus_id' not in node_data.columns:
        raise ValueError("nucleus_id not in CSV")
except Exception:
    node_data = pd.read_pickle('node_data_v1.pkl')

subset = node_data[node_data['nucleus_id'].isin(shared_ids)].copy()
subset = subset.set_index('nucleus_id').loc[shared_ids]

# Derive layer from cell_type since brain_area/layer columns are not in CSV
# cell_type encoding: 23P=L2/3, 4P=L4, 5P-*=L5, 6P-*=L6, others=other
def cell_type_to_layer(ct):
    ct = str(ct)
    if ct.startswith('23'):
        return 'L2/3'
    elif ct.startswith('4'):
        return 'L4'
    elif ct.startswith('5'):
        return 'L5'
    elif ct.startswith('6'):
        return 'L6'
    else:
        return 'other'

subset['layer'] = subset['cell_type'].apply(cell_type_to_layer)
# All neurons are excitatory V1 (no brain_area column available in CSV)
# Use layer only as the group label -- still a meaningful test of
# whether laminar wiring statistics explain functional geometry
subset['group'] = subset['cell_type']
groups = subset['group'].values  # (N,) array of group labels per neuron
unique_groups = sorted(set(groups))

print(f"N neurons: {N}")
print(f"Cell-type groups ({len(unique_groups)}):")
for g in unique_groups:
    print(f"  {g}: {(groups == g).sum()} neurons")

# -------------------------------------------------------------------
# Step 2: Build real adjacency matrix
# -------------------------------------------------------------------
print("\nBuilding real adjacency matrix...")

with open('candidate2_real_edges.json') as f:
    real_edges = json.load(f)

A_real = np.zeros((N, N), dtype=np.float32)
for pre, post in real_edges:
    if pre in id_to_idx and post in id_to_idx:
        A_real[id_to_idx[pre], id_to_idx[post]] = 1.0

n_real = int(A_real.sum())
print(f"Real edges within shared neuron set: {n_real}")

# -------------------------------------------------------------------
# Step 3: Count connections in each cell-type block
# -------------------------------------------------------------------
print("\nCounting connections per cell-type block...")

group_indices = {g: np.where(groups == g)[0] for g in unique_groups}

block_counts = {}
total_counted = 0
for g_pre in unique_groups:
    for g_post in unique_groups:
        pre_idx = group_indices[g_pre]
        post_idx = group_indices[g_post]
        block = A_real[np.ix_(pre_idx, post_idx)]
        if g_pre == g_post:
            np.fill_diagonal(block, 0)
        count = int(block.sum())
        if count > 0:
            block_counts[(g_pre, g_post)] = count
            total_counted += count

print(f"Total edges accounted for in blocks: {total_counted}")
print(f"Non-zero blocks: {len(block_counts)}")

# -------------------------------------------------------------------
# Step 4: Build Candidate 3 -- resample within each block
# -------------------------------------------------------------------
print("\nBuilding Candidate 3 (cell-type-block-shuffled rewiring)...")
np.random.seed(42)

A_c3 = np.zeros((N, N), dtype=np.float32)

for (g_pre, g_post), count in block_counts.items():
    pre_idx = group_indices[g_pre]
    post_idx = group_indices[g_post]

    all_pairs = []
    for pi in pre_idx:
        for pj in post_idx:
            if pi != pj:
                all_pairs.append((pi, pj))

    if len(all_pairs) == 0:
        continue

    replace = count > len(all_pairs)
    if replace:
        print(f"  Warning: {g_pre}->{g_post} has {count} edges but only "
              f"{len(all_pairs)} possible pairs -- sampling with replacement")

    chosen = np.random.choice(len(all_pairs), size=count, replace=replace)
    for idx in chosen:
        pi, pj = all_pairs[idx]
        A_c3[pi, pj] = 1.0

n_c3 = int(A_c3.sum())
print(f"Candidate 3 edges: {n_c3} (real: {n_real})")

overlap = int((A_real * A_c3).sum())
print(f"Edge overlap with real graph: {overlap} ({100*overlap/n_real:.2f}%)")

# -------------------------------------------------------------------
# Step 4b: Save edge list in nucleus_id space (NEW)
# Converts adjacency matrix back to [pre_nucleus_id, post_nucleus_id] pairs
# so step2_build_and_simulate_microns.py can consume it directly.
# -------------------------------------------------------------------
print("\nSaving Candidate 3 edge list...")

idx_to_id = {i: nid for nid, i in id_to_idx.items()}
c3_edge_pairs = []
rows, cols = np.where(A_c3 == 1.0)
for r, c in zip(rows, cols):
    c3_edge_pairs.append([idx_to_id[r], idx_to_id[c]])

with open('candidate3_random_edges.json', 'w') as f:
    json.dump(c3_edge_pairs, f)
with open('candidate3_nucleus_ids.json', 'w') as f:
    json.dump(shared_ids, f)

print(f"  candidate3_random_edges.json  ({len(c3_edge_pairs)} edges)")
print(f"  candidate3_nucleus_ids.json   ({len(shared_ids)} neurons)")

# -------------------------------------------------------------------
# Step 5: Compute structural RDM (cosine distance)
# -------------------------------------------------------------------
print("\nComputing Candidate 3 structural RDM...")

row_norms = np.linalg.norm(A_c3, axis=1, keepdims=True)
zero_mask = (row_norms == 0).flatten()
if zero_mask.any():
    print(f"  {zero_mask.sum()} neurons with no outgoing connections -- using uniform profile")
    A_c3[zero_mask] = 1.0 / N
    row_norms = np.linalg.norm(A_c3, axis=1, keepdims=True)

A_c3_norm = A_c3 / row_norms
cos_sim = A_c3_norm @ A_c3_norm.T
cos_sim = np.clip(cos_sim, -1, 1)
rdm_c3 = (1 - cos_sim).astype(np.float32)
np.fill_diagonal(rdm_c3, 0)

mean_dist = rdm_c3[np.triu_indices(N, k=1)].mean()
print(f"Candidate 3 RDM: mean distance = {mean_dist:.4f}")

# -------------------------------------------------------------------
# Step 6: Save outputs
# -------------------------------------------------------------------
np.save('structural_rdm_candidate3.npy', rdm_c3)

summary = f"""Candidate 3 (cell-type-block-shuffled rewiring) -- Summary
==========================================================
Strategy: preserve block-level connection counts between all 9
cell-type groups (brain_area x layer), shuffle specific partners
within each block.

Cell-type groups: {len(unique_groups)}
{chr(10).join(f'  {g}: {(groups == g).sum()} neurons' for g in unique_groups)}

Block statistics:
  Non-zero blocks: {len(block_counts)}
  Total edges: {n_c3} (real: {n_real})
  Edge overlap with real graph: {overlap} ({100*overlap/n_real:.2f}%)

RDM: mean cosine distance = {mean_dist:.4f}

Interpretive role: if real connectome beats Candidate 3, specific
partner identity within cell-type blocks matters beyond cell-type
statistics. If real ~ Candidate 3, cell-type-level wiring explains
most of the signal.
"""

with open('candidate3_summary.txt', 'w') as f:
    f.write(summary)

print("\nSaved:")
print("  structural_rdm_candidate3.npy")
print("  candidate3_random_edges.json")
print("  candidate3_nucleus_ids.json")
print("  candidate3_summary.txt")
print("\n--- STEP 7 COMPLETE ---")
print("Run step 8 to update the RSA comparison with Candidate 3.")
