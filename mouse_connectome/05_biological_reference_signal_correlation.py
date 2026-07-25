"""
MICrONS orientation script — Step 5
Goal: compute the pairwise in vivo signal correlation RDM for the
coregistered proofread excitatory neuron subset, using trial-averaged
calcium trace responses from node_data_v1.pkl (Ding et al. 2025), and
visualize it.

This resolves the biological reference question: instead of relying on
the digital twin model's pref_dir/pref_ori (model-mediated), we use
directly-measured functional responses from raw calcium imaging.

Signal correlation = Pearson r between two neurons' 120-element
trial-averaged response vectors (6 oracle clips x 20 time bins).

NOTE ON DATA SOURCES (do not "modernize" this to read the .csv):
  This script reads node_data_v1.pkl, NOT node_data_v1.csv. The two are
  not interchangeable. The .csv (from 00_save_node_positions.py) is the
  deduplicated 1,762-neuron SIMULATION node table (node_id, pt_root_id,
  nucleus_id, cell_type, x/y/z). The .pkl is the Ding et al. release and
  is the ONLY source of the calcium response vectors (in_vivo_mean_resp)
  and the area/layer/hva metadata used here. The biological reference
  must come from the .pkl.

Neuron set: this script defines its own neuron set (the coregistered
proofread excitatory subset with measured calcium responses, ~899),
independently of the simulation's 1,762-neuron set. Downstream RSA
(08, step3, step5) aligns the two sets by nucleus_id, so the ordering
saved here need not match the simulation order.

Outputs:
  signal_corr_matrix.npy        : (N, N) pairwise signal correlation matrix
  signal_corr_nucleus_ids.npy   : (N,) nucleus_ids in matrix row/column order
  signal_corr_summary.txt       : key statistics
  signal_corr_distribution.png  : histogram of pairwise correlations
                                  (comparable to Ding et al. 2025 ED Fig. 6)
  signal_corr_heatmap.png       : N x N heatmap sorted by brain area / layer

Run in the microns environment after step4.py.
node_data_v1.pkl must be present in the working directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caveclient import CAVEclient

# -------------------------------------------------------------------
# Step 1: Get coregistered proofread excitatory nucleus_ids
# -------------------------------------------------------------------
client = CAVEclient('minnie65_public')
client.version = 1718

print("Querying proofread excitatory neurons...")
proof_df = client.materialize.tables.proofreading_status_and_strategy(
    status_axon='t'
).query(select_columns=['pt_root_id', 'status_axon'])

cell_type_df = client.materialize.tables.aibs_metamodel_celltypes_v661(
    pt_root_id=proof_df.pt_root_id
).query(
    select_columns={
        'nucleus_detection_v0': ['pt_root_id', 'id'],
        'aibs_metamodel_celltypes_v661': ['classification_system', 'cell_type'],
    },
).rename(columns={'id': 'nucleus_id'})

proof_ct_df = proof_df.merge(cell_type_df, on='pt_root_id', how='inner')
exc_df = proof_ct_df.query("classification_system == 'excitatory_neuron'").copy()
exc_df = exc_df.drop_duplicates(subset='pt_root_id')
exc_root_ids = exc_df.pt_root_id.unique().astype('int64').tolist()
print(f"Proofread excitatory neurons: {len(exc_root_ids)}")

print("Querying coregistration...")
coreg_df = client.materialize.tables.coregistration_manual_v4(
    pt_root_id=exc_root_ids
).query(
    select_columns=['pt_root_id', 'target_id', 'session', 'scan_idx', 'unit_id']
)
my_nucleus_ids = set(coreg_df['target_id'].unique())
print(f"Coregistered neurons: {len(my_nucleus_ids)}")

# -------------------------------------------------------------------
# Step 2: Load node_data and filter to coregistered subset
# -------------------------------------------------------------------
print("\nLoading node_data_v1.pkl...")
node_data = pd.read_pickle('node_data_v1.pkl')

subset = node_data[node_data['nucleus_id'].isin(my_nucleus_ids)].copy()
subset = subset.dropna(subset=['in_vivo_mean_resp'])
print(f"Neurons with in_vivo_mean_resp in node_data: {len(subset)}")

# Stack response vectors into matrix: shape (N, 120)
nucleus_ids = subset['nucleus_id'].values
resp_matrix = np.stack(subset['in_vivo_mean_resp'].values)  # (N, 120)
N = len(nucleus_ids)
print(f"Response matrix shape: {resp_matrix.shape}")

# -------------------------------------------------------------------
# Step 3: Compute pairwise signal correlations
# Pearson r between every pair of 120-element response vectors.
# Z-score each row, then (resp_z @ resp_z.T) / n_timepoints.
# -------------------------------------------------------------------
print(f"\nComputing {N}x{N} pairwise signal correlation matrix...")

mean = resp_matrix.mean(axis=1, keepdims=True)
std = resp_matrix.std(axis=1, keepdims=True)
std[std == 0] = 1  # avoid division by zero for flat responses
resp_z = (resp_matrix - mean) / std  # (N, 120)

sig_corr_matrix = (resp_z @ resp_z.T) / resp_matrix.shape[1]  # (N, N)
sig_corr_matrix = np.clip(sig_corr_matrix, -1, 1)  # handle FP edge cases

tri = np.triu_indices(N, k=1)
print(f"Signal correlation matrix shape: {sig_corr_matrix.shape}")
print(f"Mean signal correlation (off-diagonal): {sig_corr_matrix[tri].mean():.4f}")
print(f"Std signal correlation (off-diagonal):  {sig_corr_matrix[tri].std():.4f}")
print(f"Min: {sig_corr_matrix[tri].min():.4f}")
print(f"Max: {sig_corr_matrix[tri].max():.4f}")

# -------------------------------------------------------------------
# Step 4: Save matrix outputs
# -------------------------------------------------------------------
np.save('signal_corr_matrix.npy', sig_corr_matrix)
np.save('signal_corr_nucleus_ids.npy', nucleus_ids)

summary = f"""Signal Correlation RDM - MICrONS Proofread Excitatory Subset
=============================================================
Source: node_data_v1.pkl (Ding et al. 2025)
Biological reference: in vivo signal correlation from trial-averaged
  calcium responses to oracle clips (6 clips x 20 time bins = 120 dims)
  Validated in Ding et al. Extended Data Fig. 2-3 as faithful proxy
  for in vivo ground truth.

N neurons: {N}
Response vector length: {resp_matrix.shape[1]} (6 oracle clips x 20 time bins)

Pairwise signal correlation summary (off-diagonal):
  Mean:   {sig_corr_matrix[tri].mean():.4f}
  Std:    {sig_corr_matrix[tri].std():.4f}
  Min:    {sig_corr_matrix[tri].min():.4f}
  Max:    {sig_corr_matrix[tri].max():.4f}

Output files:
  signal_corr_matrix.npy      - ({N}, {N}) float64 pairwise correlation matrix
  signal_corr_nucleus_ids.npy - ({N},) nucleus_ids in matrix row/column order
  signal_corr_distribution.png - histogram of pairwise correlations
  signal_corr_heatmap.png      - heatmap sorted by brain area / layer
"""

with open('signal_corr_summary.txt', 'w') as f:
    f.write(summary)

print("\nSaved:")
print("  signal_corr_matrix.npy")
print("  signal_corr_nucleus_ids.npy")
print("  signal_corr_summary.txt")

# -------------------------------------------------------------------
# Step 5: Visualization (folded in from former 05b)
# -------------------------------------------------------------------

# --- Figure 1: pairwise signal correlation distribution ---
print("\nPlotting signal correlation distribution...")
corr_vals = sig_corr_matrix[tri]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(corr_vals, bins=80, color='steelblue', alpha=0.85, density=True,
        edgecolor='none')
ax.axvline(corr_vals.mean(), color='tomato', linewidth=1.5,
           label=f'Mean = {corr_vals.mean():.3f}')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Pairwise signal correlation (in vivo)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(
    'Signal correlation distribution\n'
    f'Proofread excitatory neurons, MICrONS (N={N}, {N*(N-1)//2:,} pairs)',
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
stats_text = (
    f'Mean = {corr_vals.mean():.3f}\n'
    f'Std = {corr_vals.std():.3f}\n'
    f'Min = {corr_vals.min():.3f}\n'
    f'Max = {corr_vals.max():.3f}'
)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('signal_corr_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved signal_corr_distribution.png")

# --- Figure 2: heatmap sorted by brain area and layer ---
# Uses brain_area / layer / hva columns from the pkl. If the pkl lacks
# these columns, skip the heatmap rather than failing the whole script.
heatmap_cols = ['brain_area', 'layer', 'hva']
have_meta = all(c in node_data.columns for c in heatmap_cols)

if not have_meta:
    print("\nSkipping heatmap: node_data_v1.pkl lacks brain_area/layer/hva columns.")
else:
    print("\nPlotting signal correlation heatmap...")
    meta = node_data.set_index('nucleus_id').loc[nucleus_ids][heatmap_cols].copy()

    area_order = {'VISp': 0, 'VISrl': 1, 'VISal': 2, 'VISlm': 3}
    layer_order = {'L2/3': 0, 'L4': 1, 'L5': 2, 'L6': 3}
    meta['area_rank'] = meta['brain_area'].astype(str).map(area_order).fillna(99)
    meta['layer_rank'] = meta['layer'].astype(str).map(layer_order).fillna(99)
    meta['sort_key'] = meta['area_rank'] * 10 + meta['layer_rank']

    sort_idx = meta['sort_key'].argsort().values
    sig_corr_sorted = sig_corr_matrix[np.ix_(sort_idx, sort_idx)]
    meta_sorted = meta.iloc[sort_idx]

    # boundary lines between brain areas
    area_labels = meta_sorted['brain_area'].values
    boundaries = []
    current = area_labels[0]
    for i, a in enumerate(area_labels):
        if a != current:
            boundaries.append(i)
            current = a

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sig_corr_sorted, aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3, interpolation='none')

    for b in boundaries:
        ax.axhline(b - 0.5, color='black', linewidth=0.8, alpha=0.7)
        ax.axvline(b - 0.5, color='black', linewidth=0.8, alpha=0.7)

    prev = 0
    unique_areas = []
    for b in boundaries + [N]:
        mid = (prev + b) / 2
        unique_areas.append((mid, area_labels[prev]))
        prev = b

    ax.set_xticks([m for m, _ in unique_areas])
    ax.set_xticklabels([a for _, a in unique_areas], fontsize=9, rotation=45)
    ax.set_yticks([m for m, _ in unique_areas])
    ax.set_yticklabels([a for _, a in unique_areas], fontsize=9)

    plt.colorbar(im, ax=ax, label='Signal correlation (in vivo)', shrink=0.8)
    ax.set_title(
        'Signal correlation RDM - biological reference\n'
        f'Sorted by brain area and layer (N={N} neurons)',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig('signal_corr_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved signal_corr_heatmap.png")

print("\nBiological reference RDM complete.")
print("This resolves Open Question 2 - signal correlation from raw calcium")
print("traces is the biological reference, not the digital twin model output.")
print("Distribution is comparable to Ding et al. Extended Data Fig. 6;")
print("heatmap shows whether area/layer predicts functional similarity.")
