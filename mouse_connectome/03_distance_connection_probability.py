"""
MICrONS orientation script — Step 3
Goal: compute the distance-connection-probability relationship in the
real proofread excitatory connectivity graph, to determine empirically
whether spatial structure is strong enough to warrant Candidate 2
(distance-constrained rewiring) as the null model.

This does NOT implement any null model or run any RSA/CKA analysis.
It just answers one empirical question: does connection probability
decay strongly with physical distance in this specific dataset?

Run this LOCALLY in the `microns` environment, after step2.py has
been run successfully at least once.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caveclient import CAVEclient

client = CAVEclient('minnie65_public')
client.version = 1718

# ---------------------------------------------------------------------
# Step 1: Get the same proofread excitatory neuron set as before,
# but this time also pull their spatial positions (nucleus centroids)
# ---------------------------------------------------------------------
proof_df = client.materialize.tables.proofreading_status_and_strategy(
    status_axon='t'
).query(
    select_columns=['pt_root_id', 'status_axon']
)

cell_type_df = client.materialize.tables.aibs_metamodel_celltypes_v661(
    pt_root_id=proof_df.pt_root_id
).query(
    select_columns={
        'nucleus_detection_v0': ['pt_root_id', 'id', 'pt_position'],
        'aibs_metamodel_celltypes_v661': ['classification_system', 'cell_type'],
    },
    split_positions=True,
    desired_resolution=[1, 1, 1]  # convert to nanometers
).rename(columns={'id': 'nucleus_id'})

proof_ct_df = proof_df.merge(cell_type_df, on='pt_root_id', how='inner')
exc_df = proof_ct_df.query("classification_system == 'excitatory_neuron'").copy()
exc_root_ids = exc_df.pt_root_id.unique().astype('int64').tolist()

print(f"Proofread excitatory neurons with positions: {len(exc_df)}")

# ---------------------------------------------------------------------
# Step 2: Pull synapses within the proofread excitatory population
# (same as step2.py)
# ---------------------------------------------------------------------
print("\nQuerying synapses between proofread excitatory neurons...")
syn_df = client.materialize.synapse_query(
    pre_ids=exc_root_ids,
    post_ids=exc_root_ids,
    remove_autapses=True,
)
print(f"Synapses found: {len(syn_df)}")

# Binarize to unique directed connections
edge_counts = (
    syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id'])
    .size()
    .reset_index(name='n_synapses')
)
connected_pairs = set(
    zip(edge_counts.pre_pt_root_id, edge_counts.post_pt_root_id)
)
print(f"Unique directed connections: {len(connected_pairs)}")

# ---------------------------------------------------------------------
# Step 3: Build position lookup (in micrometers, for readability)
# MICrONS voxel resolution: 4nm x 4nm x 40nm
# After desired_resolution=[1,1,1], positions are already in nm
# Divide by 1000 to get microns
# ---------------------------------------------------------------------
pos = exc_df.set_index('pt_root_id')[
    ['pt_position_x', 'pt_position_y', 'pt_position_z']
].copy().astype(float)
pos = pos / 1000  # nm -> microns

# ---------------------------------------------------------------------
# Step 4: Sample pairs and compute distance-connection-probability
# Computing all N^2 pairs for N=1762 is ~3M pairs — feasible but slow.
# We'll compute a random sample of non-connected pairs to compare
# against connected pairs, then bin by distance.
# ---------------------------------------------------------------------
print("\nComputing pairwise distances and connection probabilities...")

root_ids = exc_df.pt_root_id.values
n = len(root_ids)

# Get distances for all CONNECTED pairs
connected_distances = []
for pre, post in connected_pairs:
    if pre in pos.index and post in pos.index:
        d = np.linalg.norm(pos.loc[pre].values - pos.loc[post].values)
        connected_distances.append(d)

connected_distances = np.array(connected_distances)
print(f"Connected pairs with positions: {len(connected_distances)}")
print(f"Connected pair distance: mean={connected_distances.mean():.1f} um, "
      f"median={np.median(connected_distances):.1f} um, "
      f"max={connected_distances.max():.1f} um")

# Sample random pairs (non-self) to get distance distribution of all possible pairs
np.random.seed(42)
n_sample = min(100000, n * (n - 1))
sample_idx = np.random.choice(n, size=(n_sample, 2), replace=True)
# Remove self-pairs
sample_idx = sample_idx[sample_idx[:, 0] != sample_idx[:, 1]]

all_sample_distances = []
all_sample_connected = []
for i, j in sample_idx[:50000]:  # cap at 50k for speed
    pre = root_ids[i]
    post = root_ids[j]
    if pre in pos.index and post in pos.index:
        d = np.linalg.norm(pos.loc[pre].values - pos.loc[post].values)
        is_connected = (pre, post) in connected_pairs
        all_sample_distances.append(d)
        all_sample_connected.append(is_connected)

all_sample_distances = np.array(all_sample_distances)
all_sample_connected = np.array(all_sample_connected)

# ---------------------------------------------------------------------
# Step 5: Bin by distance and compute connection probability per bin
# ---------------------------------------------------------------------
bin_edges = np.linspace(0, all_sample_distances.max(), 21)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
conn_prob = []
n_pairs_per_bin = []

for i in range(len(bin_edges) - 1):
    mask = (all_sample_distances >= bin_edges[i]) & \
           (all_sample_distances < bin_edges[i+1])
    n_in_bin = mask.sum()
    n_connected = all_sample_connected[mask].sum()
    prob = n_connected / n_in_bin if n_in_bin > 0 else 0
    conn_prob.append(prob)
    n_pairs_per_bin.append(n_in_bin)

conn_prob = np.array(conn_prob)
n_pairs_per_bin = np.array(n_pairs_per_bin)

print("\n--- Distance-Connection-Probability Summary ---")
print(f"{'Distance bin (um)':<25} {'Connection prob':<20} {'N pairs'}")
for i in range(len(bin_centers)):
    print(f"{bin_centers[i]:<25.1f} {conn_prob[i]:<20.4f} {n_pairs_per_bin[i]}")

# ---------------------------------------------------------------------
# Step 6: Plot and save
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: connection probability vs distance
axes[0].plot(bin_centers, conn_prob, 'o-', color='steelblue', linewidth=2)
axes[0].set_xlabel('Distance between neuron soma centers (μm)')
axes[0].set_ylabel('Connection probability')
axes[0].set_title('Distance-Connection-Probability\n(Proofread excitatory neurons, MICrONS)')
axes[0].grid(True, alpha=0.3)

# Right: distribution of connected vs all-pair distances
axes[1].hist(all_sample_distances, bins=40, alpha=0.5, color='gray',
             density=True, label='All sampled pairs')
axes[1].hist(connected_distances, bins=40, alpha=0.7, color='steelblue',
             density=True, label='Connected pairs')
axes[1].set_xlabel('Distance between neuron soma centers (μm)')
axes[1].set_ylabel('Density')
axes[1].set_title('Distance Distribution\nConnected vs All Pairs')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distance_connection_probability.png', dpi=150, bbox_inches='tight')
print("\nSaved figure to distance_connection_probability.png")

# ---------------------------------------------------------------------
# Key diagnostic output: does connection probability decay with distance?
# ---------------------------------------------------------------------
first_bin = conn_prob[n_pairs_per_bin > 10][0]
last_bin = conn_prob[n_pairs_per_bin > 10][-1]
ratio = first_bin / last_bin if last_bin > 0 else float('inf')

print(f"\n--- KEY DIAGNOSTIC ---")
print(f"Connection probability at short range: {first_bin:.4f}")
print(f"Connection probability at long range:  {last_bin:.4f}")
print(f"Ratio (short/long):                    {ratio:.1f}x")
print()
if ratio > 3:
    print("STRONG distance dependence detected.")
    print("Candidate 2 (distance-constrained rewiring) is likely warranted.")
    print("Candidate 1 alone may produce a null model that is trivially")
    print("distinguishable from real cortex due to ignored spatial structure.")
elif ratio > 1.5:
    print("MODERATE distance dependence detected.")
    print("Candidate 2 adds rigor but Candidate 1 may still be defensible.")
    print("Worth discussing with Choi which is more appropriate.")
else:
    print("WEAK distance dependence detected.")
    print("Candidate 1 (degree-preserving rewiring) may be sufficient.")
    print("Spatial structure does not appear to be a dominant confound here.")

