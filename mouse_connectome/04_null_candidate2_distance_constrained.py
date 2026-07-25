"""
MICrONS orientation script — Step 4
Goal: implement Candidate 2 (distance-constrained rewiring) and
sanity-check the output. Confirm that the randomized graph:
1. Has similar edge count to the real graph
2. Preserves the distance-connection-probability profile
3. Has zero (or near-zero) edge overlap with the real graph

This is NOT a full experiment — no geometry comparison, no biological
reference. Just: does this randomization run, and does it produce a
null graph that respects spatial wiring structure?

Run this LOCALLY in the `microns` environment, after step3.py has
been run successfully.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caveclient import CAVEclient

client = CAVEclient('minnie65_public')
client.version = 1718

# ---------------------------------------------------------------------
# Step 1: Get proofread excitatory neurons with positions
# (same as step3.py)
# ---------------------------------------------------------------------
proof_df = client.materialize.tables.proofreading_status_and_strategy(
    status_axon='t'
).query(select_columns=['pt_root_id', 'status_axon'])

cell_type_df = client.materialize.tables.aibs_metamodel_celltypes_v661(
    pt_root_id=proof_df.pt_root_id
).query(
    select_columns={
        'nucleus_detection_v0': ['pt_root_id', 'id', 'pt_position'],
        'aibs_metamodel_celltypes_v661': ['classification_system', 'cell_type'],
    },
    split_positions=True,
    desired_resolution=[1, 1, 1]
).rename(columns={'id': 'nucleus_id'})

proof_ct_df = proof_df.merge(cell_type_df, on='pt_root_id', how='inner')
exc_df = proof_ct_df.query("classification_system == 'excitatory_neuron'").copy()
exc_root_ids = exc_df.pt_root_id.unique().astype('int64').tolist()

pos = exc_df.drop_duplicates(subset='pt_root_id').set_index('pt_root_id')[
    ['pt_position_x', 'pt_position_y', 'pt_position_z']
].copy().astype(float)
pos = pos / 1000  # nm -> microns

print(f"Proofread excitatory neurons: {len(exc_df)} ({len(pos)} unique pt_root_ids)")

# ---------------------------------------------------------------------
# Step 2: Pull real synapses (same as step2.py and step3.py)
# ---------------------------------------------------------------------
print("\nQuerying synapses between proofread excitatory neurons...")
syn_df = client.materialize.synapse_query(
    pre_ids=exc_root_ids,
    post_ids=exc_root_ids,
    remove_autapses=True,
)

edge_counts = (
    syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id'])
    .size()
    .reset_index(name='n_synapses')
)

real_edges = set(zip(edge_counts.pre_pt_root_id, edge_counts.post_pt_root_id))
root_ids = np.array(exc_df.pt_root_id.values, dtype=np.int64)
n = len(root_ids)

print(f"Real graph: {n} nodes, {len(real_edges)} directed edges")

# ---------------------------------------------------------------------
# Step 3: Compute the real distance-connection-probability profile
# (same logic as step3.py, used to define the rewiring target)
# ---------------------------------------------------------------------
print("\nComputing real distance-connection-probability profile...")

np.random.seed(42)
sample_pre = np.random.choice(root_ids, size=80000, replace=True)
sample_post = np.random.choice(root_ids, size=80000, replace=True)
mask_self = sample_pre != sample_post
sample_pre = sample_pre[mask_self]
sample_post = sample_post[mask_self]

sample_distances = []
sample_connected = []
for pre, post in zip(sample_pre[:60000], sample_post[:60000]):
    if pre in pos.index and post in pos.index:
        d = np.linalg.norm(pos.loc[pre].values - pos.loc[post].values)
        sample_distances.append(d)
        sample_connected.append((pre, post) in real_edges)

sample_distances = np.array(sample_distances)
sample_connected = np.array(sample_connected)

# Bin the real distance-connection-probability profile
n_bins = 20
bin_edges = np.linspace(0, np.percentile(sample_distances, 99), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

real_conn_prob = []
for i in range(n_bins):
    mask = (sample_distances >= bin_edges[i]) & (sample_distances < bin_edges[i+1])
    n_in_bin = mask.sum()
    prob = sample_connected[mask].sum() / n_in_bin if n_in_bin > 0 else 0
    real_conn_prob.append(prob)

real_conn_prob = np.array(real_conn_prob)
print(f"Real connection prob: short-range={real_conn_prob[0]:.4f}, "
      f"long-range={real_conn_prob[-1]:.4f}")

# ---------------------------------------------------------------------
# Step 4: Build the distance-constrained random graph
# Strategy: for each neuron, sample its connection targets by drawing
# from the real distance-dependent probability distribution rather than
# uniformly at random. Each neuron's out-degree is preserved (same
# number of outputs as in the real graph), but targets are resampled
# according to the real distance-decay profile.
# ---------------------------------------------------------------------
print("\nBuilding Candidate 2 (distance-constrained rewiring)...")

# Build out-degree sequence from real graph
out_degree = {}
for pre, post in real_edges:
    out_degree[pre] = out_degree.get(pre, 0) + 1

# For each neuron, compute distance to all other neurons
# then sample targets according to the real distance-prob profile
random_edges = set()
np.random.seed(42)

root_id_list = sorted(
    set(exc_root_ids) & set(pos.index) & set([pre for pre, _ in real_edges] + [post for _, post in real_edges])
)
pos_array = pos.loc[root_id_list].values.astype(float)  # guaranteed same size as root_id_list
root_id_to_idx = {rid: i for i, rid in enumerate(root_id_list)}
n_nodes = len(root_id_list)
print(f"Neurons with positions available for rewiring: {n_nodes}")
assert pos_array.shape[0] == n_nodes, f"pos_array rows {pos_array.shape[0]} != n_nodes {n_nodes}"

processed = 0
for pre_id, k in out_degree.items():
    if pre_id not in root_id_to_idx:
        continue

    pre_idx = root_id_to_idx[pre_id]
    pre_pos = pos_array[pre_idx]

    # Compute distance from this neuron to all others in pos_array
    diffs = pos_array - pre_pos
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    # Convert distances to connection weights using the real prob profile
    bin_indices = np.digitize(dists, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    weights = real_conn_prob[bin_indices].copy()

    # Zero out self-connection
    weights[pre_idx] = 0

    # Normalize to get sampling probabilities
    total_weight = weights.sum()
    if total_weight == 0:
        continue
    probs = weights / total_weight

    # Explicit size check before sampling
    assert len(probs) == n_nodes, f"probs size {len(probs)} != n_nodes {n_nodes}"

    k_actual = min(k, int((weights > 0).sum()))
    if k_actual == 0:
        continue

    sampled_indices = np.random.choice(
        n_nodes, size=k_actual, replace=False, p=probs
    )
    for idx in sampled_indices:
        post_id = root_id_list[idx]
        if post_id != pre_id:
            random_edges.add((pre_id, post_id))

    processed += 1
    if processed % 200 == 0:
        print(f"  Processed {processed}/{len(out_degree)} neurons...")

print(f"\nCandidate 2 graph: {len(random_edges)} directed edges "
      f"(real: {len(real_edges)})")

# ---------------------------------------------------------------------
# Step 5: Sanity checks
# ---------------------------------------------------------------------
print("\n--- Sanity checks ---")

# Edge count
print(f"Edge count: real={len(real_edges)}, random={len(random_edges)} "
      f"({100*len(random_edges)/len(real_edges):.1f}% of real)")

# Edge overlap
overlap = len(real_edges & random_edges)
print(f"Edge overlap: {overlap} ({100*overlap/len(real_edges):.2f}% of real edges)")

# Verify the random graph preserves the distance-connection-probability profile
rand_connected = np.array([(pre, post) in random_edges
                           for pre, post in zip(sample_pre[:60000],
                                                sample_post[:60000])
                           if pre in pos.index and post in pos.index])

# Recompute using same samples
rand_conn_prob = []
for i in range(n_bins):
    mask = (sample_distances >= bin_edges[i]) & (sample_distances < bin_edges[i+1])
    n_in_bin = mask.sum()
    if n_in_bin > 0 and i < len(rand_connected):
        prob = rand_connected[mask[:len(rand_connected)]].sum() / n_in_bin
    else:
        prob = 0
    rand_conn_prob.append(prob)

rand_conn_prob = np.array(rand_conn_prob)

# ---------------------------------------------------------------------
# Step 6: Plot comparison of real vs random distance-prob profiles
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bin_centers, real_conn_prob, 'o-', color='steelblue',
        linewidth=2, label='Real connectome')
ax.plot(bin_centers, rand_conn_prob, 's--', color='tomato',
        linewidth=2, label='Candidate 2 (distance-constrained random)')
ax.set_xlabel('Distance between neuron soma centers (μm)')
ax.set_ylabel('Connection probability')
ax.set_title('Distance-Connection-Probability Profile\nReal vs Candidate 2 Null Model')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('candidate2_validation.png', dpi=150, bbox_inches='tight')
print("\nSaved validation figure to candidate2_validation.png")

print("\n--- STEP 4 COMPLETE ---")
print("Candidate 2 (distance-constrained rewiring) implemented and sanity-checked.")

# ---------------------------------------------------------------------
# Save edge lists for use in step 6 (structural RDM construction)
# Use nucleus_id (not pt_root_id) to match signal_corr_nucleus_ids.npy
# ---------------------------------------------------------------------
import json

exc_df_dedup = exc_df.drop_duplicates(subset='pt_root_id')
root_to_nucleus = dict(zip(
    exc_df_dedup.pt_root_id.astype(int),
    exc_df_dedup.nucleus_id.astype(int)
))

def edges_to_nucleus(edge_set, root_to_nucleus):
    result = []
    for u, v in edge_set:
        nu = root_to_nucleus.get(int(u))
        nv = root_to_nucleus.get(int(v))
        if nu is not None and nv is not None:
            result.append([nu, nv])
    return result

real_edges_nuc = edges_to_nucleus(real_edges, root_to_nucleus)
random_edges_nuc = edges_to_nucleus(random_edges, root_to_nucleus)
nucleus_ids_list = [root_to_nucleus[int(n)] for n in root_id_list
                    if int(n) in root_to_nucleus]

with open('candidate2_real_edges.json', 'w') as f:
    json.dump(real_edges_nuc, f)
with open('candidate2_random_edges.json', 'w') as f:
    json.dump(random_edges_nuc, f)
with open('candidate2_nucleus_ids.json', 'w') as f:
    json.dump(nucleus_ids_list, f)

print(f"\nSaved edge lists (nucleus_id space, {len(nucleus_ids_list)} neurons):")
print("  candidate2_real_edges.json")
print("  candidate2_random_edges.json")
print("  candidate2_nucleus_ids.json")
