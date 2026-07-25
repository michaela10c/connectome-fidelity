"""
MICrONS orientation script — Step 2
Goal: implement Candidate 1 (degree-preserving rewiring) and sanity-check
the output. This is NOT a full experiment — no geometry comparison,
no biological reference. Just: does this randomization run, and does
it produce a sensible null graph?

Run this LOCALLY in the `microns` environment, after step1.py has
been run successfully at least once.
"""

import numpy as np
import pandas as pd
import networkx as nx
from caveclient import CAVEclient

client = CAVEclient('minnie65_public')
client.version = 1718

# ---------------------------------------------------------------------
# Step 1: Get the same proofread excitatory neuron set as step1.py
# ---------------------------------------------------------------------
proof_df = client.materialize.tables.proofreading_status_and_strategy(
    status_axon='t'
).query(
    select_columns=['pt_root_id', 'status_axon', 'status_dendrite']
)

cell_type_df = client.materialize.tables.aibs_metamodel_celltypes_v661(
    pt_root_id=proof_df.pt_root_id
).query(
    select_columns={
        'nucleus_detection_v0': ['pt_root_id', 'id'],
        'aibs_metamodel_celltypes_v661': ['classification_system', 'cell_type'],
    }
).rename(columns={'id': 'nucleus_id'})

proof_ct_df = proof_df.merge(cell_type_df, on='pt_root_id', how='inner')
exc_df = proof_ct_df.query("classification_system == 'excitatory_neuron'")
exc_root_ids = exc_df.pt_root_id.unique().astype('int64').tolist()

print(f"Proofread excitatory neurons (same set as step1.py): {len(exc_root_ids)}")

# ---------------------------------------------------------------------
# Step 2: Pull synapses WITHIN this proofread excitatory population
# ---------------------------------------------------------------------
print("\nQuerying synapses between proofread excitatory neurons...")
print("(This may take a few minutes.)")

syn_df = client.materialize.synapse_query(
    pre_ids=exc_root_ids,
    post_ids=exc_root_ids,
    remove_autapses=True,
)

print(f"Synapses found within proofread excitatory population: {len(syn_df)}")

# ---------------------------------------------------------------------
# Step 3: Build the REAL connectivity graph (binarized: connected or not)
# ---------------------------------------------------------------------
edge_counts = (
    syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id'])
    .size()
    .reset_index(name='n_synapses')
)

G_real = nx.DiGraph()
G_real.add_nodes_from(exc_root_ids)
for _, row in edge_counts.iterrows():
    G_real.add_edge(row['pre_pt_root_id'], row['post_pt_root_id'])

print(f"\nReal graph: {G_real.number_of_nodes()} nodes, "
      f"{G_real.number_of_edges()} edges")
print(f"Real graph density: {nx.density(G_real):.4f}")

in_degrees_real = [d for _, d in G_real.in_degree()]
out_degrees_real = [d for _, d in G_real.out_degree()]
print(f"Real in-degree:  mean={np.mean(in_degrees_real):.2f}, "
      f"max={np.max(in_degrees_real)}")
print(f"Real out-degree: mean={np.mean(out_degrees_real):.2f}, "
      f"max={np.max(out_degrees_real)}")

# ---------------------------------------------------------------------
# Step 4: CANDIDATE 1 — degree-preserving rewiring (configuration model)
# ---------------------------------------------------------------------
print("\n--- Generating Candidate 1: degree-preserving rewiring ---")

# networkx directed_configuration_model takes in/out degree sequences
# and builds a random graph matching them (as a multigraph, so we
# simplify after)
in_seq = [d for _, d in G_real.in_degree()]
out_seq = [d for _, d in G_real.out_degree()]

G_random_multi = nx.directed_configuration_model(
    in_seq, out_seq, seed=42
)
G_random = nx.DiGraph(G_random_multi)  # collapse multi-edges
G_random.remove_edges_from(nx.selfloop_edges(G_random))  # remove self-loops

# FIX: directed_configuration_model assigns integer indices 0..N-1
# matching the order of in_seq/out_seq, which came from G_real.in_degree()
# Remap back to root IDs using that same ordering so edges_to_nucleus works
node_order = list(G_real.nodes())
G_random = nx.relabel_nodes(G_random, {i: node_order[i] for i in range(len(node_order))})

print(f"Random graph: {G_random.number_of_nodes()} nodes, "
      f"{G_random.number_of_edges()} edges")
print(f"Random graph density: {nx.density(G_random):.4f}")

in_degrees_rand = [d for _, d in G_random.in_degree()]
out_degrees_rand = [d for _, d in G_random.out_degree()]
print(f"Random in-degree:  mean={np.mean(in_degrees_rand):.2f}, "
      f"max={np.max(in_degrees_rand)}")
print(f"Random out-degree: mean={np.mean(out_degrees_rand):.2f}, "
      f"max={np.max(out_degrees_rand)}")

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------
print("\n--- Sanity checks ---")
print(f"Edge count preserved (real vs random, expect close): "
      f"{G_real.number_of_edges()} vs {G_random.number_of_edges()}")
print(f"Degree distribution preserved (compare means above): "
      f"{'OK' if abs(np.mean(in_degrees_real) - np.mean(in_degrees_rand)) < 0.5 else 'MISMATCH'}")

real_edges = set(G_real.edges())
random_edges = set(G_random.edges())
overlap = len(real_edges & random_edges)
print(f"Edge overlap between real and random graph: {overlap} "
      f"({100*overlap/len(real_edges):.1f}% of real edges also appear in random — "
      f"should be low, confirming the randomization actually changed who connects to whom)")

# ---------------------------------------------------------------------
# Save edge lists for use in step 6 (structural RDM construction)
# Use nucleus_id (not pt_root_id) to match signal_corr_nucleus_ids.npy
# ---------------------------------------------------------------------
import json

# Build pt_root_id -> nucleus_id mapping from exc_df
exc_df_dedup = exc_df.drop_duplicates(subset='pt_root_id')
root_to_nucleus = dict(zip(
    exc_df_dedup.pt_root_id.astype(int),
    exc_df_dedup.nucleus_id.astype(int)
))

def edges_to_nucleus(edge_list, root_to_nucleus):
    result = []
    for u, v in edge_list:
        nu = root_to_nucleus.get(int(u))
        nv = root_to_nucleus.get(int(v))
        if nu is not None and nv is not None:
            result.append([nu, nv])
    return result

# Carry synapse counts into the real edge list: [pre_nucleus, post_nucleus, n_synapses]
# (random/null edges remain [pre, post] — they have no biological synapse count)
real_edges_nuc = []
for _, row in edge_counts.iterrows():
    nu = root_to_nucleus.get(int(row['pre_pt_root_id']))
    nv = root_to_nucleus.get(int(row['post_pt_root_id']))
    if nu is not None and nv is not None:
        real_edges_nuc.append([nu, nv, int(row['n_synapses'])])
random_edges_nuc = edges_to_nucleus(G_random.edges(), root_to_nucleus)
nucleus_ids_list = [root_to_nucleus[int(n)] for n in G_real.nodes()
                    if int(n) in root_to_nucleus]

with open('candidate1_real_edges.json', 'w') as f:
    json.dump(real_edges_nuc, f)
with open('candidate1_random_edges.json', 'w') as f:
    json.dump(random_edges_nuc, f)
with open('candidate1_nucleus_ids.json', 'w') as f:
    json.dump(nucleus_ids_list, f)

print(f"\nSaved edge lists (nucleus_id space, {len(nucleus_ids_list)} neurons):")
print(f"  candidate1_real_edges.json    ({len(real_edges_nuc)} edges, now with synapse counts)")
print(f"  candidate1_random_edges.json  ({len(random_edges_nuc)} edges)")
print("  candidate1_nucleus_ids.json")

print("\n--- STEP 2 COMPLETE ---")
print("Candidate 1 (degree-preserving rewiring) implemented and sanity-checked.")
print("No geometry analysis or biological reference comparison has been run.")
