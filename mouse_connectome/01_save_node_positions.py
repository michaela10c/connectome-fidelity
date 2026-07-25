"""
01_save_node_positions.py
One-time data prep: rebuild node_data_v1.csv from CAVEclient.

Replaces node_data_v1.pkl which cannot be loaded under pandas >= 2.0.
Run once before any simulation scripts.

POSITION SOURCE FIX (2026-06):
    Positions now come from nucleus_detection_v0 (ANATOMICAL soma centroids in
    the EM volume), NOT coregistration_manual_v4 (functional EM<->calcium match).
    Coregistration is depth-biased and was missing for 718/1764 neurons (41%,
    concentrated in L5/L6), which corrupted the distance-constrained (C2) null.
    Anatomical nucleus positions exist for ~all proofread neurons and are the
    correct source for a structural wiring-distance null anyway.

Output: node_data_v1.csv (de-duplicated by nucleus_id; node_id is contiguous 0..N-1)
    node_id      -- 0-indexed integer (row position; used by the simulation)
    pt_root_id   -- 18-digit MICrONS root ID
    nucleus_id   -- nucleus_detection_v0 ID (unique after built-in de-duplication)
    cell_type    -- e.g. '23P', '5P-IT', etc.
    x, y, z      -- soma position in nm (voxel coords * voxel size)

DE-DUPLICATION (BUILT IN): nucleus_detection_v0 can return duplicate nucleus_id
rows; these are dropped here before saving so the node_id -> rate-row mapping is
never corrupted downstream. No separate dedupe step is required.
"""

import numpy as np
import pandas as pd
from caveclient import CAVEclient

client = CAVEclient('minnie65_public')
client.materialize.version = 1718

# -- Step 1: proofread neurons -------------------------------------------------

print('Step 1: proofread neurons (axon=TRUE)...')
proof_df = client.materialize.tables.proofreading_status_and_strategy(
    status_axon='t'
).query(select_columns=['pt_root_id', 'status_axon'])
print(f'  {len(proof_df)} proofread neurons')

# -- Step 2: cell types + ANATOMICAL position (nucleus_detection_v0) ------------
# Pull pt_position from nucleus_detection_v0 here (same table that gives the
# nucleus_id), so position coverage matches the proofread set, not the
# depth-biased coregistration set.

print('Step 2: cell types + anatomical nucleus positions...')
cell_type_df = client.materialize.tables.aibs_metamodel_celltypes_v661(
    pt_root_id=proof_df.pt_root_id
).query(
    select_columns={
        'nucleus_detection_v0': ['pt_root_id', 'id', 'pt_position'],
        'aibs_metamodel_celltypes_v661': ['classification_system', 'cell_type'],
    }
).rename(columns={'id': 'nucleus_id'})

proof_ct_df = proof_df.merge(cell_type_df, on='pt_root_id', how='inner')
exc_df = proof_ct_df.query(
    "classification_system == 'excitatory_neuron'"
).reset_index(drop=True)
print(f'  {len(exc_df)} proofread excitatory neurons')

# -- Step 3: extract x/y/z from nucleus_detection_v0 pt_position ----------------
# pt_position is [x_voxel, y_voxel, z_voxel]. Minnie65 voxel size is 4/4/40 nm.
# (Confirm scaling if coordinate ranges look wrong after running.)

print('Step 3: extracting soma positions from nucleus_detection_v0...')
positions = np.stack(
    exc_df['pt_position'].apply(
        lambda p: p if isinstance(p, (list, np.ndarray)) else [0, 0, 0]
    ).values
)
exc_df['x'] = positions[:, 0] * 4.0    # voxel -> nm (x)
exc_df['y'] = positions[:, 1] * 4.0    # voxel -> nm (y)
exc_df['z'] = positions[:, 2] * 40.0   # voxel -> nm (z)

n_missing = int((exc_df[['x', 'y', 'z']].abs().sum(axis=1) == 0).sum())
if n_missing > 0:
    print(f'  WARNING: {n_missing} neurons still have no position (0,0,0) -- '
          f'these lack a nucleus_detection_v0 centroid.')
else:
    print('  All neurons have anatomical positions.')

# quick range sanity check (microns)
for ax in ['x', 'y', 'z']:
    lo, hi = exc_df[ax].min() / 1000.0, exc_df[ax].max() / 1000.0
    print(f'  {ax} range: {lo:.1f} .. {hi:.1f} um')

# -- De-duplicate by nucleus_id (BUILT IN, always runs) ------------------------
# nucleus_detection_v0 occasionally returns two rows with the same nucleus_id
# (byte-identical duplicate detections). If these survive into node_data_v1.csv,
# the node_id -> rate-matrix-row mapping shifts and silently corrupts every
# downstream comparison that maps neurons by node_id (e.g. the simulation-vs-
# biology RSA). De-duplicating here -- rather than in a separate script that
# must be remembered -- guarantees the CSV is always on the unique-nucleus_id
# set with a contiguous node_id, regardless of how this script is invoked.
n_before = len(exc_df)
exc_df = exc_df.drop_duplicates(subset='nucleus_id', keep='first').reset_index(drop=True)
n_after = len(exc_df)
if n_after != n_before:
    print(f'  De-duplicated nucleus_id: {n_before} -> {n_after} rows '
          f'({n_before - n_after} duplicate(s) removed)')
else:
    print(f'  No duplicate nucleus_ids ({n_after} rows)')

# -- Save ----------------------------------------------------------------------
# index=True with the post-dedup reset_index gives a contiguous node_id 0..N-1
# that matches the de-duplicated row order. Never write node_id from a
# pre-dedup index, or the mapping will be off.

out_cols = ['pt_root_id', 'nucleus_id', 'cell_type', 'x', 'y', 'z']
exc_df[out_cols].to_csv('node_data_v1.csv', index=True, index_label='node_id')
print(f'\nSaved node_data_v1.csv: {len(exc_df)} rows '
      f'({exc_df["nucleus_id"].nunique()} unique nucleus_ids, '
      f'node_id 0..{len(exc_df)-1})')
print(exc_df[out_cols].head(5).to_string())
print('\nDone. node_data_v1.csv is de-duplicated and ready.')
print('After running this, rebuild the downstream chain in the same generation:')
print('  rm -rf network_lgn_shared/ network/ output_pointnet/')
print('  python step2_build_and_simulate_microns.py')
print('  python step3_extract_rates_and_rsa.py')
print('  python step5_run_null_simulation.py   # if the null-through-sim result is needed')
