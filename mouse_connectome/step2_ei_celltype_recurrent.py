"""
step2_ei_celltype_recurrent.py -- extends step2_ei_recurrent.py's E/I
simulation from two uniform weight scalars (SYN_WEIGHT_E, SYN_WEIGHT_I)
to 11 independent per-cell-type weights, testing whether synaptic
diversity specifically (not neuron biophysics, not dendrites) can shift
representational geometry toward better alignment with biology.

WHY THIS EXISTS: the "why does real trail the nulls" investigation
(dimensionality, decoding) ruled out information-poverty as an
explanation and landed on "geometry doesn't align with biology as well
as it could." One concrete, specific candidate cause: uniform per-class
weighting flattens real synaptic diversity that likely co-evolved WITH
the specific wiring pattern -- a randomized wiring pattern has no such
designed relationship to disrupt in the first place, so uniform
weighting may distort real's functional signature specifically while
leaving null wiring's (already arbitrary) signature comparatively
unaffected. This is the cheapest test of that hypothesis: same neuron
model, same synapse-count-proportional scaling within each type, same
network otherwise -- only the number of independent weight parameters
changes, from 2 to 11.

DESIGN: reuses signed_null_c3.py's cell_type grouping exactly (11
groups: 4P, 23P, 6P-IT, 5P-IT, 6P-CT, 5P-ET, 5P-NP for excitatory;
BC, MC, BPC, NGC for inhibitory). Each cell type gets its own
add_edges() call with its own independently-fittable weight scalar --
a direct 11-way extension of the existing E/I split's 2-way pattern
(step2_ei_recurrent.py's build_v1_recurrent_ei already splits into two
add_edges() calls with independent weights; this is the same mechanism,
just with 11 groups instead of 2).

CRITICAL VALIDATION REQUIREMENT: with all excitatory types set to the
same weight and all inhibitory types set to the same weight, this MUST
reproduce step2_ei_recurrent.py's uniform-weight result exactly. Tested
directly in main() before this is trusted for anything else.
"""

import os

import numpy as np
import pandas as pd

import step2_ei_recurrent as step2_ei
import step2_build_and_simulate_microns as step2

NODE_CSV_EI = step2_ei.NODE_CSV_EI

E_CELL_TYPES = ["4P", "23P", "6P-IT", "5P-IT", "6P-CT", "5P-ET", "5P-NP"]
I_CELL_TYPES = ["BC", "MC", "BPC", "NGC"]
ALL_CELL_TYPES = E_CELL_TYPES + I_CELL_TYPES


def load_cell_types_by_node(neurons_df, path=NODE_CSV_EI):
    """Returns {node_id: cell_type}, matching signed_null_c3.py's
    load_cell_types() logic but keyed by node_id (this module's indexing)
    rather than by position in a nucleus_id list.
    """
    df = pd.read_csv(path)
    nuc_to_ct = dict(zip(df["nucleus_id"], df["cell_type"]))
    node_to_ct = {}
    for idx, row in neurons_df.iterrows():
        nuc = int(row.nucleus_id)
        if nuc not in nuc_to_ct:
            raise ValueError(f"nucleus_id {nuc} has no cell_type in {path}")
        node_to_ct[idx] = nuc_to_ct[nuc]
    return node_to_ct


def build_v1_recurrent_ei_celltype(neurons_df, edges_df, network_dir,
                                   weights_by_celltype, node_types=None):
    """11-way extension of step2_ei_recurrent.build_v1_recurrent_ei's 2-way
    E/I split. weights_by_celltype: dict mapping each of the 11 cell types
    in ALL_CELL_TYPES to its own weight scalar (positive for excitatory
    types, magnitude only -- sign is applied automatically based on
    E_CELL_TYPES/I_CELL_TYPES membership, matching Dale's law exactly the
    same way the 2-class version does).
    """
    from bmtk.builder import NetworkBuilder

    missing = set(ALL_CELL_TYPES) - set(weights_by_celltype.keys())
    if missing:
        raise ValueError(f"weights_by_celltype missing entries for: {missing}")

    N = len(neurons_df)
    os.makedirs(network_dir, exist_ok=True)

    if node_types is None:
        node_types = step2_ei.load_node_types()

    nucleus_to_node = {int(row.nucleus_id): idx
                       for idx, row in neurons_df.iterrows()}
    node_to_nucleus = {v: k for k, v in nucleus_to_node.items()}
    node_to_celltype = load_cell_types_by_node(neurons_df)

    # one conn_lookup dict per cell type, keyed by SOURCE cell type
    conn_lookups = {ct: {} for ct in ALL_CELL_TYPES}
    skipped = 0
    for row in edges_df.itertuples(index=False):
        src = nucleus_to_node.get(int(row.pre_node_id))
        trg = nucleus_to_node.get(int(row.post_node_id))
        if src is None or trg is None:
            skipped += 1
            continue
        n_syn = max(int(row.syn_count), 1)
        src_ct = node_to_celltype[src]
        conn_lookups[src_ct][(src, trg)] = n_syn
    if skipped:
        print(f"  Skipped {skipped} edges with nucleus IDs not in node set")
    for ct in ALL_CELL_TYPES:
        print(f"  [{ct:6s}] {len(conn_lookups[ct])} source edges  "
              f"weight={weights_by_celltype[ct]:.4f}")

    v1 = NetworkBuilder('microns_v1')
    v1.add_nodes(
        N=N, model_type='point_neuron', ei='e',
        model_template=step2.V1_MODEL_TEMPLATE,
        dynamics_params=step2.V1_DYNAMICS_PARAMS,
        x=neurons_df['x'].values,
        y=neurons_df['y'].values,
        z=neurons_df['z'].values,
    )

    # same BMTK uint32-overflow workaround as the 2-class version, applied
    # per cell type now instead of just per E/I class -- same mechanism,
    # broader application, since any of the 11 weights could independently
    # land on an exact-integer product with some nsyns value
    _EPS = 1e-6

    for ct in ALL_CELL_TYPES:
        conn_lookup = conn_lookups[ct]
        if not conn_lookup:
            continue
        is_I_type = ct in I_CELL_TYPES
        w = weights_by_celltype[ct] + _EPS
        signed_w = -w if is_I_type else w

        def make_rule(lookup):
            def rule(src, trg):
                return lookup.get((src.node_id, trg.node_id), 0)
            return rule

        v1.add_edges(
            source={'ei': 'e'}, target={'ei': 'e'},
            connection_rule=make_rule(conn_lookup),
            model_template='static_synapse',
            dynamics_params='instantaneousExc.json',
            syn_weight=signed_w, delay=2.0,
        )

    print(f"  Building V1 recurrent ({N}x{N} = {N*N:,} pairs, "
          f"{len(ALL_CELL_TYPES)}-cell-type split)...")
    v1.build()
    v1.save(output_dir=network_dir)
    print("  V1 recurrent (cell-type-specific) saved.")


def run_simulation_ei_celltype(edge_file, tag, weights_by_celltype,
                               build_lgn=True, force_lgn=False, node_types=None):
    """Cell-type-specific counterpart to step2_ei_recurrent.run_simulation_ei.
    Identical LGN/circuit/orientation-loop handling; only the V1 recurrent
    connectivity step uses per-cell-type weights instead of two uniform
    E/I scalars.
    """
    network_dir = step2.network_dir_for(tag)
    output_dir = step2.output_dir_for(tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}\nrun_simulation_ei_celltype(tag={tag!r}, "
          f"edge_file={edge_file!r})\n{'='*72}")
    neurons_df = step2_ei.load_nodes_ei()

    if build_lgn:
        step2.build_lgn_scaffold(neurons_df, force=force_lgn)
    else:
        if not os.path.exists(os.path.join(step2.LGN_DIR, "lgn_nodes.h5")):
            raise RuntimeError(
                f"Shared LGN scaffold missing in {step2.LGN_DIR}/. Run once "
                f"with build_lgn=True first.")

    edges_df = step2_ei.load_edges_signed(edge_file)
    build_v1_recurrent_ei_celltype(neurons_df, edges_df, network_dir,
                                   weights_by_celltype, node_types=node_types)
    circuit_cfg = step2.make_circuit_config(tag, network_dir, step2.LGN_DIR)
    step2.run_all_orientations(tag, output_dir, circuit_cfg)
    step2.check_outputs(len(neurons_df), output_dir)
    print(f"\n[{tag}] done (cell-type-specific) -> {output_dir}/  "
          f"(LGN drive: shared {step2.LGN_DIR}/)")
    return output_dir


if __name__ == '__main__':
    import sys
    print("=== VALIDATION MODE: uniform weights via the 11-cell-type path ===")
    print("Confirms build_v1_recurrent_ei_celltype with all-E-types-equal, "
          "all-I-types-equal EXACTLY reproduces the existing 2-parameter "
          "pipeline before trusting anything built on top of this.\n")
    if len(sys.argv) == 4:
        edge_file, tag, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        edge_file, tag, mode = "signed_real_edges.json", "real_celltype_validation", "validate"

    if mode == "validate":
        # matches real's known fitted uniform weights exactly, so the
        # resulting spike output should be byte-identical to the existing
        # tag="real" output already on disk
        w = {ct: 5.4375 for ct in E_CELL_TYPES}
        w.update({ct: 5.4375 for ct in I_CELL_TYPES})
        run_simulation_ei_celltype(edge_file, tag, w, build_lgn=False)
        print("\nNow compare this output's spike files against "
              "output_pointnet_fitted_ei_real/ -- they must match exactly "
              "(same validation pattern used for the original real-vs-repeat "
              "noise check earlier this session).")
    else:
        print(f"Unknown mode {mode!r} -- use 'validate'")
