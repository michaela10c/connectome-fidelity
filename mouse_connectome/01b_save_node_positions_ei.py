"""
01b_save_node_positions_ei.py
Extends node_data_v1.csv to the full E+I population.

WHY THIS EXISTS
---------------
node_data_v1.csv (from 01_save_node_positions.py) contains only the 1,762
excitatory neurons -- confirmed directly: its nucleus_ids are exactly the
1,762 E-type IDs in node_types.json, with 0 overlap with the 434 I-type IDs.
This was fine for the original E-only pipeline, but step2_ei_recurrent.py's
node lookup silently drops every edge touching an inhibitory neuron (source
OR target) when matched against this file -- confirmed empirically: a real
run against the full signed graph loaded 49,039 E-sourced edges (exactly the
real E->E count) and 0 I-sourced edges, with 133,783 edges (exactly
E->I + I->E + I->I) skipped as "nucleus IDs not in node set".

This script pulls positions for the 434 missing inhibitory neurons using the
IDENTICAL CAVEclient pattern as 01_save_node_positions.py -- same
materialize version, same anatomical-position source (nucleus_detection_v0,
not the depth-biased coregistration table), same voxel->nm scaling -- so the
two populations are sourced consistently, not by two different methods that
happen to share a schema.

DOES NOT MODIFY node_data_v1.csv. That file is load-bearing for the entire
existing E-only pipeline (steps 2-9, and the just-completed 151-condition
step7 run). This writes a NEW file, node_data_v1_ei.csv, combining the
existing 1,762 E-rows UNCHANGED (same node_id 0..1761) with 434 new I-rows
appended (node_id 1762..2195).

VERIFICATION, NOT ASSUMED: after pulling, cross-checks the pulled
inhibitory neurons' nucleus_ids against node_types.json's existing 434
I-type IDs. If they don't match exactly, something is inconsistent between
this pull and the earlier signed-graph pull that produced node_types.json,
and that must be resolved before trusting node_data_v1_ei.csv for anything.

ONE THING NOT VERIFIED FROM OUTSIDE THIS RUN: the exact string
'inhibitory_neuron' for classification_system is the natural complement to
the confirmed 'excitatory_neuron' filter in 01_save_node_positions.py, but
was not independently confirmed against the live table schema. This script
prints the unique classification_system values it actually sees before
filtering, specifically so a wrong guess here is visible immediately rather
than silently returning zero rows.

Output: node_data_v1_ei.csv (2,196 rows: 1,762 E unchanged + 434 I appended;
    same schema as node_data_v1.csv plus a cell_class column so E/I identity
    doesn't require a second file to determine)
"""

import numpy as np
import pandas as pd
from caveclient import CAVEclient
import json

EXISTING_E_CSV = "node_data_v1.csv"
NODE_TYPES_JSON = "node_types.json"
OUT_CSV = "node_data_v1_ei.csv"

client = CAVEclient('minnie65_public')
client.materialize.version = 1718   # same pin as 01_save_node_positions.py

# -- Step 1: proofread neurons (same filter as the E-only pull) ----------------

print('Step 1: proofread neurons (axon=TRUE)...')
proof_df = client.materialize.tables.proofreading_status_and_strategy(
    status_axon='t'
).query(select_columns=['pt_root_id', 'status_axon'])
print(f'  {len(proof_df)} proofread neurons')

# -- Step 2: cell types + anatomical position, INHIBITORY this time -----------

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

# Print what's actually in this column before filtering -- so a wrong guess
# at the inhibitory label is visible immediately, not a silent empty result.
print(f'  classification_system values seen: '
      f'{sorted(proof_ct_df["classification_system"].dropna().unique().tolist())}')

INHIBITORY_LABEL = 'inhibitory_neuron'
inh_df = proof_ct_df.query(
    "classification_system == @INHIBITORY_LABEL"
).reset_index(drop=True)
print(f'  {len(inh_df)} proofread inhibitory neurons '
      f'(filtered on classification_system == {INHIBITORY_LABEL!r})')

if len(inh_df) == 0:
    raise SystemExit(
        f"Got 0 inhibitory neurons with classification_system == "
        f"{INHIBITORY_LABEL!r}. Check the printed list of actual values "
        f"above and fix INHIBITORY_LABEL to match -- do not proceed with "
        f"an empty pull.")

# -- Step 3: extract x/y/z, identical scaling to the E-only pull --------------

print('Step 3: extracting soma positions from nucleus_detection_v0...')
positions = np.stack(
    inh_df['pt_position'].apply(
        lambda p: p if isinstance(p, (list, np.ndarray)) else [0, 0, 0]
    ).values
)
inh_df['x'] = positions[:, 0] * 4.0
inh_df['y'] = positions[:, 1] * 4.0
inh_df['z'] = positions[:, 2] * 40.0

n_missing = int((inh_df[['x', 'y', 'z']].abs().sum(axis=1) == 0).sum())
if n_missing > 0:
    print(f'  WARNING: {n_missing} inhibitory neurons have no position '
          f'(0,0,0) -- lack a nucleus_detection_v0 centroid.')
else:
    print('  All inhibitory neurons have anatomical positions.')

for ax in ['x', 'y', 'z']:
    lo, hi = inh_df[ax].min() / 1000.0, inh_df[ax].max() / 1000.0
    print(f'  {ax} range: {lo:.1f} .. {hi:.1f} um')

# -- De-duplicate (same discipline as the E-only pull) -------------------------

n_before = len(inh_df)
inh_df = inh_df.drop_duplicates(subset='nucleus_id', keep='first').reset_index(drop=True)
n_after = len(inh_df)
if n_after != n_before:
    print(f'  De-duplicated nucleus_id: {n_before} -> {n_after} rows '
          f'({n_before - n_after} duplicate(s) removed)')
else:
    print(f'  No duplicate nucleus_ids ({n_after} rows)')

# -- Cross-check against node_types.json, not assumed to match ----------------

print('\nCross-checking pulled inhibitory nucleus_ids against node_types.json...')
with open(NODE_TYPES_JSON) as f:
    ntypes = {int(k): v for k, v in json.load(f).items()}
expected_i_ids = {n for n, t in ntypes.items() if t == 'I'}
pulled_i_ids = set(inh_df['nucleus_id'].tolist())

missing_from_pull = expected_i_ids - pulled_i_ids
extra_in_pull = pulled_i_ids - expected_i_ids
print(f'  Expected I-type nucleus_ids (node_types.json): {len(expected_i_ids)}')
print(f'  Pulled inhibitory nucleus_ids (this query):    {len(pulled_i_ids)}')
print(f'  Missing from this pull (in node_types.json, not here): '
      f'{len(missing_from_pull)}')
print(f'  Extra in this pull (here, not in node_types.json): '
      f'{len(extra_in_pull)}')
if missing_from_pull or extra_in_pull:
    print('  MISMATCH -- this pull and the earlier signed-graph pull disagree '
          'on which neurons are inhibitory. Investigate before trusting '
          'node_data_v1_ei.csv: likely a materialization-version or '
          'proofreading-filter difference between the two pulls.')
else:
    print('  EXACT MATCH -- consistent with node_types.json.')

# -- Combine with the existing E-only file, E rows unchanged -------------------

print('\nLoading existing E-only node_data_v1.csv...')
e_df = pd.read_csv(EXISTING_E_CSV)
print(f'  {len(e_df)} existing E rows (node_id {e_df["node_id"].min()}..'
      f'{e_df["node_id"].max()}, unchanged)')

e_df = e_df.copy()
e_df['cell_class'] = 'E'
inh_df = inh_df.rename(columns={'cell_type': 'cell_type'})  # already correct name
inh_out = inh_df[['pt_root_id', 'nucleus_id', 'cell_type', 'x', 'y', 'z']].copy()
inh_out['cell_class'] = 'I'
inh_out['node_id'] = range(len(e_df), len(e_df) + len(inh_out))   # 1762..2195

combined = pd.concat(
    [e_df[['node_id', 'pt_root_id', 'nucleus_id', 'cell_type', 'x', 'y', 'z', 'cell_class']],
     inh_out[['node_id', 'pt_root_id', 'nucleus_id', 'cell_type', 'x', 'y', 'z', 'cell_class']]],
    ignore_index=True,
)

# Final sanity check before writing: exactly 2,196 rows, exactly matching
# node_types.json's combined E+I population, contiguous node_id.
assert combined['node_id'].tolist() == list(range(len(combined))), \
    "node_id is not contiguous 0..N-1 after combining -- do not save"
combined_nucleus_ids = set(combined['nucleus_id'].tolist())
expected_all_ids = set(ntypes.keys())
if combined_nucleus_ids != expected_all_ids:
    print(f'  WARNING: combined file has {len(combined_nucleus_ids)} nucleus_ids, '
          f'node_types.json has {len(expected_all_ids)} -- '
          f'symmetric difference: {len(combined_nucleus_ids ^ expected_all_ids)}')
else:
    print(f'  Combined file exactly matches node_types.json\'s {len(expected_all_ids)} '
          f'neurons.')

combined.to_csv(OUT_CSV, index=False)
print(f'\nSaved {OUT_CSV}: {len(combined)} rows '
      f'({(combined.cell_class=="E").sum()} E + {(combined.cell_class=="I").sum()} I, '
      f'node_id 0..{len(combined)-1})')
print(combined.head(3).to_string())
print(combined.tail(3).to_string())
print(f'\nDone. {EXISTING_E_CSV} was NOT modified. Use {OUT_CSV} for the E/I pipeline:')
print('  step2_ei_recurrent.py and step7_ei_fit_weights.py need to be pointed at')
print(f'  {OUT_CSV} instead of node_data_v1.csv -- see the follow-up patch.')
